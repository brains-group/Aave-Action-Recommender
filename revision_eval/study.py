"""Generate matched policy recommendations and call performSimulations for each.

Run in a frozen source checkout with a fresh output directory. No API calls.
"""
import argparse,os,json,pickle,hashlib,copy,contextlib,time,traceback
from pathlib import Path
from multiprocessing import Pool
from .policies import propose,POLICIES

def file_hash(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()

def dump(path,value):
 temp=path.with_suffix('.tmp')
 with temp.open('wb') as f:pickle.dump(value,f,protocol=4)
 temp.replace(path)
def js(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,default=str));temp.replace(path)
def history(user_id,up_to_timestamp):
 import pandas as pd
 x=HISTORY.get(user_id)
 if x is None:return pd.DataFrame()
 x=x[x.timestamp<=up_to_timestamp].sort_values('timestamp')
 return x.tail(10000).reset_index(drop=True) if len(x)>1000 else x

def align_simulation(profile, recommendation_timestamp, horizon, funding_multiplier):
    if horizon <= 0 or funding_multiplier < 0:
        raise ValueError('Invalid follow-up/funding setting')
    prof=copy.deepcopy(profile)
    prof['initial_wallet']={k:float(v)*funding_multiplier for k,v in prof.get('initial_wallet',{}).items()}
    base=max(int(t['timestamp']) for t in prof['transactions'])
    end=float(recommendation_timestamp)+horizon
    if base>end:raise ValueError('Action after fixed follow-up end')
    return prof,max(1,int(end-base))

def task(item):
 index,variant=item;row=ROWS[index];case=str(row['case_id']);folder=RUN/'cases'/f'{index:04d}-{variant}';folder.mkdir(exist_ok=True)
 import actionAgentTraining as agent
 import performSimulations as evaluator
 import utils.simulations as simulation
 from tools.run_single_simulation import run_simulation
 from utils.logger import set_log_file
 set_log_file(f'worker-{os.getpid()}.log',False,str(RUN/'logs'),file_level='WARNING')
 if variant=='indexed':os.environ.update(AAVE_INDEXED_COVERAGE=CONFIG['coverage'],AAVE_INDEXED_ASSETS=CONFIG['assets'])
 else:os.environ.pop('AAVE_INDEXED_COVERAGE',None)
 agent.get_user_history=history
 import pandas as pd
 agent.get_date_ranges=lambda:(pd.DatetimeIndex([pd.Timestamp(CONFIG['model_date'])]),pd.DatetimeIndex([]))
 start=time.time();generation=folder/'recommendations.pkl'
 try:
  if generation.exists():pack=pickle.load(generation.open('rb'))
  else:
   placeholder=agent.generate_next_transaction(row,'Deposit',amount=0);placeholder['abstained']=True
   profile,*_=simulation.get_limited_user_profile(placeholder,return_extras=True)
   # Funding sensitivity changes initial funds identically for every policy.
   profile['initial_wallet']={k:float(v)*CONFIG['funding_multiplier'] for k,v in profile.get('initial_wallet',{}).items()}
   checkpoint=run_simulation(profile,checkpoint_only=True)
   if checkpoint['stable_debt_present']:raise ValueError('Policy generation requires stable-debt asset adapter')
   state=dict(checkpoint['checkpoint_state'],case_id=case,state_timestamp=checkpoint['checkpoint_state']['timestamp'],timestamp=float(placeholder['timestamp']))
   recent=history(row['user'],row['timestamp'])
   state['activity_30d']=int(len(recent[recent.timestamp>=row['timestamp']-30*86400].drop_duplicates(['timestamp','pool','Index Event']))) if len(recent) else 0
   proposals={policy:propose(state,policy,budget_usd=CONFIG['budget']) for policy in POLICIES}
   recs={}
   for policy,proposal in proposals.items():
    rec=placeholder.copy()
    if proposal['action']:
     rec=agent.generate_next_transaction(row,proposal['action'],proposal['amount'],reserve=proposal['asset'])
     rec['symbol']=proposal['asset'];rec['priceInUSD']=state['assets'][proposal['asset']]['price_usd'];rec['amountUSD']=proposal['capital_usd']
    rec['abstained']=proposal['action'] is None;rec['case_id']=case
    recs[policy]=(rec,{'is_at_risk':proposal['action'] is not None,'policy':policy})
   # The agent uses precisely the same available funding and gross capital cap.
   original=agent.get_simulation_outcome
   from utils.recommendation_funding import funding_choice
   original_funding=agent.funding_choice
   original_price=agent.get_price_history_value
   agent.get_price_history_value=lambda symbol,timestamp:state['assets'][symbol]['price_usd']
   def budget_funding(s,action,timestamp,price,minimum_usd=50):
    asset,minimum,maximum=funding_choice(s,action,timestamp,price,minimum_usd)
    maximum=min(maximum,CONFIG['budget']/price(asset))
    if maximum<=0:raise agent.NoFeasibleAction('No capital budget')
    return asset,min(minimum,maximum),maximum
   agent.funding_choice=budget_funding
   agent.get_simulation_outcome=lambda *a,**kw:checkpoint
   try:
    rec,info=agent.recommend_action(row);rec['case_id']=case;recs['agent']=(rec,info)
   except Exception as error:
    rec=placeholder.copy();rec['generation_error']=str(error);rec['case_id']=case;recs['agent']=(rec,{})
   finally:agent.get_simulation_outcome=original;agent.funding_choice=original_funding;agent.get_price_history_value=original_price
   # Synthetic interventions do not inherit a historical transaction identity.
   for policy,(rec,info) in list(recs.items()):
    rec=rec.drop(labels=['id','transactionHash','blockNumber','logIndex','pool','asset_address','on_behalf_of','payer','recipient','rate_mode'],errors='ignore')
    recs[policy]=(rec,info)
   pack=dict(case_id=case,cohort=row['cohort'],variant=variant,checkpoint=checkpoint,state=state,recommendations=recs,proposals=proposals,observed_liquidation_timestamp=float(row['timestamp']+row['timeDiff']) if row['status']==1 else None)
   if not recs['agent'][0].get('generation_error'):
    pack['prediction_history']=agent.get_transaction_history_predictions(row)
   dump(generation,pack)
  if CONFIG['generate_only']:
   return dict(index=index,variant=variant,seconds=time.time()-start,status='generated')
  for policy,pair in pack['recommendations'].items():
   dest=folder/(policy+'.pkl')
   if dest.exists():continue
   rec,info=pair;arms={};actual={};original=evaluator.get_simulation_outcome
   end=float(rec['timestamp'])+CONFIG['horizon']
   def simulate(recommendation,suffix,**kwargs):
    if suffix=='with':actual.update(dict(recommendation))
    kwargs['profile'],kwargs['lookahead_seconds']=align_simulation(kwargs['profile'],rec['timestamp'],CONFIG['horizon'],CONFIG['funding_multiplier'])
    # Shared strict HF configuration; warning-margin arms are separate scenarios.
    kwargs.update(use_enhanced_hf=False,margin_threshold=CONFIG['margin'],oracle_delay_seconds=0,volatility_discount=0,liquidation_policy=CONFIG['detection_policy'])
    result=simulation.get_simulation_outcome(recommendation,suffix,**kwargs)
    arms[suffix]=result;return result
   evaluator.get_simulation_outcome=simulate;evaluator.outputFile=str(RUN/'logs'/f'strategies-{os.getpid()}.log')
   try:result=evaluator.process_recommendation(pair)
   finally:evaluator.get_simulation_outcome=original
   record=dict(case_id=case,cohort=pack['cohort'],policy=policy,variant=variant,recommendation=rec,applied_recommendation=actual,info=info,result=result,arms=arms,followup_end=end,horizon=CONFIG['horizon'])
   dump(dest,record)
  return dict(index=index,variant=variant,seconds=time.time()-start,status='complete')
 except Exception as error:
  if not generation.exists():
   rec=row.copy();rec['timestamp']=float(row['timestamp'])+600;rec['Index Event']='deposit';rec['type']='deposit';rec['amount']=0.;rec['amountUSD']=0.;rec['abstained']=True;rec['generation_error']=str(error)
   dump(generation,dict(case_id=case,cohort=row['cohort'],variant=variant,checkpoint=None,state={},proposals={},recommendations={p:(rec.copy(),{}) for p in (*POLICIES,'agent')}))
  js(folder/'error.json',dict(case_id=case,cohort=row['cohort'],error=str(error),traceback=traceback.format_exc()))
  return dict(index=index,variant=variant,status='failed',error=str(error))

def main():
 global RUN,ROWS,HISTORY,CONFIG
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['run','cohorts','data','models','coverage','assets']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--prepare-only',action='store_true');p.add_argument('--generate-only',action='store_true');p.add_argument('--history-file',type=Path);p.add_argument('--workers',type=int,default=2);p.add_argument('--budget',type=float,default=1000);p.add_argument('--horizon',type=int,default=604800);p.add_argument('--funding-multiplier',type=float,default=1);p.add_argument('--margin',type=float,default=1);p.add_argument('--detection-policy',choices=['protocol','static_warning','dynamic_warning'],default='protocol');p.add_argument('--limit',type=int);p.add_argument('--variants',nargs='+',choices=['core','indexed'],default=['core','indexed']);a=p.parse_args();RUN=a.run.resolve();RUN.mkdir(parents=True,exist_ok=True)
 for n in ['cases','logs','cache']:(RUN/n).mkdir(exist_ok=True)
 CONFIG={'coverage':str(a.coverage.resolve()),'assets':str(a.assets.resolve()),'budget':a.budget,'horizon':a.horizon,'funding_multiplier':a.funding_multiplier,'margin':a.margin,'detection_policy':a.detection_policy,'generate_only':a.generate_only}
 # Model artifacts are read-only links; all prediction and simulation caches are new.
 for n in ['models','data','date_ranges.pkl']:
  dest=RUN/'cache'/n
  if not dest.exists():dest.symlink_to((a.models/n).resolve(),target_is_directory=n!='date_ranges.pkl')
 dummy=RUN/'empty-recommendations.pkl'
 if not dummy.exists():dump(dummy,{})
 os.environ.update(AAVE_EVALUATION_CACHE=str(RUN/'cache'),AAVE_RECOMMENDATIONS_FILE=str(dummy),AAVE_FROZEN_MODELS='1')
 import pandas as pd
 # Freeze the latest common fitted model timestamp before the held-out period.
 cutoff=json.loads((a.cohorts/'manifest.json').read_text())['test_start']
 common=None
 for ie in ['Borrow','Deposit','Repay','Withdraw','Liquidated']:
  for oe in ['Borrow','Deposit','Repay','Withdraw','Liquidated']:
   if ie==oe=='Liquidated':continue
   prefix=f'xgboost_cox_{ie}_{oe}_'
   dates={f.name[len(prefix):-4].replace('_',' ') for f in (a.models/'models').glob(prefix+'*.pkl') if not f.name.endswith('_baseline.pkl')}
   dates={d for d in dates if d!='latest' and pd.Timestamp(d).timestamp()<cutoff and (a.models/'models'/(prefix+d.replace(' ','_')+'_baseline.pkl')).exists()}
   common=dates if common is None else common&dates
 if not common:raise ValueError('No common historical fitted model date before test period')
 CONFIG['model_date']=max(common,key=pd.Timestamp)
 import utils.simulations as simulation
 ROWS=[]
 for cohort in ['high_risk','normal']:
  rows=pd.read_csv(a.cohorts/(cohort+'.csv'));rows['cohort']=cohort
  if a.limit:rows=rows.head(a.limit)
  ROWS.extend(row for _,row in rows.iterrows())
 cutoff={r.user:r.timestamp for r in ROWS};hpath=a.history_file or RUN/'histories.pkl'
 if hpath.exists():HISTORY=pickle.load(hpath.open('rb'))
 else:
  parts=[]
  for source in sorted(a.data.glob('*/*/data.csv')):
   for chunk in pd.read_csv(source,chunksize=100000):
    x=chunk[chunk.timestamp<=chunk.user.map(cutoff)]
    if len(x):parts.append(x)
   print('history',source.parent.parent.name,source.parent.name,flush=True)
  frame=pd.concat(parts) if parts else pd.DataFrame();HISTORY={u:g for u,g in frame.groupby('user',sort=False)};dump(hpath,HISTORY)
 artifacts={str(f.relative_to(a.models)):file_hash(f) for d in ['models','data'] for f in (a.models/d).iterdir() if f.is_file() and CONFIG['model_date'][:10] in f.name}
 profile_hashes={}
 for row in ROWS:
  matches=list(Path('profiles').glob('*/profiles/user_'+row['user']+'.json'))
  profile_hashes[row['user']]={str(f.resolve()):file_hash(f) for f in matches}
 code_root=Path(__file__).resolve().parents[1]
 research_code={str(f.relative_to(code_root)):file_hash(f) for f in [*Path(__file__).parent.glob('*.py'),code_root/'actionAgentTraining.py',code_root/'performSimulations.py']}
 identity=dict(model_artifacts=artifacts,profile_hashes=profile_hashes,research_code=research_code,histories_sha256=file_hash(hpath),config={k:v for k,v in CONFIG.items() if k!='generate_only'},code=simulation.simulation_code_identity(),study_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cohort_manifest=json.loads((a.cohorts/'manifest.json').read_text()),case_ids=[r.case_id for r in ROWS],cohort_membership={r.case_id:r.cohort for r in ROWS},variants=a.variants)
 manifest=RUN/'manifest.json'
 if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('Changed study inputs/code; fresh directory required')
 js(manifest,identity)
 if a.prepare_only:return
 from simulator.utils import setup_protocol
 setup_protocol()
 tasks=[(i,v) for i in range(len(ROWS)) for v in a.variants];statuses=[];start=time.time()
 with Pool(a.workers) as pool:
  for done,status in enumerate(pool.imap_unordered(task,tasks),1):
   statuses.append(status);js(RUN/'progress.json',dict(done=done,total=len(tasks),seconds=time.time()-start,last=status));print(done,len(tasks),status,flush=True)
 # Export evaluator-compatible dictionaries with retained case identity.
 for cohort in ['high_risk','normal']:
  for variant in a.variants:
   for policy in (*POLICIES,'agent'):
    rows={}
    for f in sorted((RUN/'cases').glob('*-'+variant+'/recommendations.pkl')):
     pack=pickle.load(f.open('rb'))
     if pack['cohort']==cohort:rows[pack['case_id']]=pack['recommendations'][policy]
    dump(RUN/f'{cohort}-{variant}-{policy}-recommendations.pkl',rows)
 js(RUN/('generation-complete.json' if a.generate_only else 'complete.json'),dict(tasks=len(tasks),failures=sum(s['status']=='failed' for s in statuses),seconds=time.time()-start))
if __name__=='__main__':main()
