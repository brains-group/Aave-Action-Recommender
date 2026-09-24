"""Baselines on existing agent cases; completed-parent gate and read-only reuse."""
import argparse,os,sys,json,pickle,hashlib,time,copy,fcntl,contextlib,importlib.util
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def save(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 if path.suffix=='.json':tmp.write_text(json.dumps(obj,indent=2,default=str))
 else:
  with tmp.open('wb') as f:pickle.dump(obj,f,protocol=4)
 tmp.replace(path)
def key(rec,suffix,args,identity):
 args={k:v for k,v in args.items() if k!='output_file'}
 digest=hashlib.sha256(pickle.dumps(args,protocol=4)).hexdigest()[:16]
 return Path(identity)/digest/f"{rec['user']}_{int(rec.get('timestamp',0))}_{suffix}.pkl"
class StopCheckpoint(BaseException):pass

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['source','parent-run','recommendations','coverage','assets','policies','run']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--start-index',type=int,default=0);p.add_argument('--limit',type=int);p.add_argument('--once',action='store_true');p.add_argument('--generate-only',action='store_true');a=p.parse_args()
 for name in ['source','parent_run','recommendations','coverage','assets','policies','run']:setattr(a,name,getattr(a,name).resolve())
 run=a.run;run.mkdir(parents=True,exist_ok=True)
 for name in ['data','profiles']:
  link=run/name
  if not link.exists():link.symlink_to((a.source/name).resolve(),target_is_directory=True)
 os.chdir(run)
 for n in ['cache','logs','cases','simulations','locks']:(run/n).mkdir(exist_ok=True)
 guard=(run/'run.lock').open('w');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 os.environ.update(AAVE_EVALUATION_CACHE=str(run/'cache'),AAVE_RECOMMENDATIONS_FILE=str(a.recommendations.resolve()),AAVE_SIMULATION_CACHE=str(run/'simulations'))
 for n in ['AAVE_SURVIVAL_DATA','AAVE_CORE_TRANSACTIONS','AAVE_HISTORICAL_EVIDENCE','AAVE_INDEXED_COVERAGE','AAVE_INDEXED_ASSETS']:os.environ.pop(n,None)
 sys.path.insert(0,str(a.source.resolve()))
 import performSimulations as evaluator
 import utils.simulations as u
 import tools.run_single_simulation as runner
 from utils.logger import set_log_file
 set_log_file('baseline.log',False,str(run/'logs'),file_level='WARNING')
 spec=importlib.util.spec_from_file_location('baseline_policies',a.policies);policy_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy_module)
 parent=json.loads((a.parent_run/'manifest.json').read_text());identity=u.simulation_code_identity()
 assert identity==parent['code_identity'],'Simulator/input identity differs from ongoing run'
 assert sha(a.recommendations)==parent['recommendations_sha256']
 assert sha(a.coverage)==parent['coverage_sha256'] and sha(a.assets)==parent['assets_sha256']
 items=list(evaluator.recommendations.items());count=min(a.limit or len(items),len(items))
 manifest=dict(parent=str(a.parent_run.resolve()),code_identity=identity,recommendations_sha256=sha(a.recommendations),policies_sha256=sha(a.policies),runner_sha256=sha(__file__),selected=count,start_index=a.start_index,total_existing=len(items),trigger=1.10,target=1.20,funding='existing checkpoint wallet; no added capital cap or topups',timing='unchanged performSimulations evaluator',control_cache='parent read-only; no missing-control computation')
 if (run/'manifest.json').exists():assert json.loads((run/'manifest.json').read_text())==manifest,'Changed inputs; fresh run required'
 save(run/'manifest.json',manifest)
 oldcache=a.parent_run/'cache/simulation_results';stats={'parent_hits':0,'local_hits':0,'new_simulations':0,'identical_control_reuses':0}
 current_control={}
 def cache(rec,suffix,**args):
  rel=key(rec,suffix,args,identity);dest=run/'simulations'/rel
  lock=run/'locks'/(hashlib.sha256(str(rel).encode()).hexdigest()+'.lock')
  with lock.open('a') as f:
   fcntl.flock(f,fcntl.LOCK_EX)
   for origin,label in [(oldcache,'parent_hits'),(run/'simulations','local_hits')]:
    file=origin/rel
    if file.exists():
     value=pickle.load(file.open('rb'));stats[label]+=1;return value
   if suffix=='without':raise RuntimeError('Completed parent case has no matching control cache; refusing duplicate computation')
   canonical=pickle.dumps({k:v for k,v in args.items() if k!='output_file'},protocol=4)
   if canonical==current_control.get('args'):
    stats['identical_control_reuses']+=1;return current_control['value']
   value=u.run_simulation(**args);save(dest,value);stats['new_simulations']+=1;return value
 def checkpoint(profile):
  captured={};original=runner.run_all_strategies
  def stop(**kw):
   sim=kw['sim'];user=sim.get_user(kw['user_address']);account=sim.get_user_account_data(kw['user_address'])
   if user.get('stable_debt'):raise ValueError('Stable-debt policy adapter unavailable')
   captured.update(account,state_timestamp=kw['base_timestamp'],assets={})
   captured['weighted_collateral_usd']=account['total_collateral_usd']*account['effective_liquidation_threshold']
   for asset in sorted(set(user['wallet'])|set(user['debt'])|set(user['collateral'])):
    captured['assets'][asset]=dict(price_usd=sim.prices[sim.price_key_for_user(user,asset)],wallet=user['wallet'].get(asset,0),debt=user['debt'].get(asset,0)*sim.reserves[asset].variable_borrow_index,liquidation_threshold=sim.risk_parameters(user,asset)[1],collateral_enabled=sim.collateral_enabled(user,asset))
   raise StopCheckpoint()
  runner.run_all_strategies=stop
  try:
   try:runner.run_simulation(profile=copy.deepcopy(profile),lookahead_seconds=1)
   except StopCheckpoint:pass
  finally:runner.run_all_strategies=original
  assert captured,'Checkpoint capture failed'
  return captured
 def task(index,variant):
  folder=run/'cases'/f'{index:05d}-{variant}';folder.mkdir(exist_ok=True)
  source=a.parent_run/'cases'/f'{index:05d}-{variant}.json'
  parent_case=json.loads(source.read_text());rec,info=evaluator.normalize_recommendation(copy.deepcopy(items[index][1]));case_id=str(items[index][0])
  if variant=='indexed':os.environ.update(AAVE_INDEXED_COVERAGE=str(a.coverage.resolve()),AAVE_INDEXED_ASSETS=str(a.assets.resolve()))
  else:os.environ.pop('AAVE_INDEXED_COVERAGE',None);os.environ.pop('AAVE_INDEXED_ASSETS',None)
  if 'without' not in parent_case.get('arms',{}):
   save(folder/'unavailable.json',dict(reason='Existing agent case has no completed control arm',parent_result=parent_case['result'],case_id=case_id));return
  packpath=folder/'recommendations.pkl'
  try:
   profile,_,_,_,_,horizon=u.get_limited_user_profile(rec,return_extras=True)
   args=dict(profile=profile,lookahead_seconds=horizon,output_file=str(run/'logs/simulation.log'))
   control=cache(rec,'without',**args);current_control.update(args=pickle.dumps({k:v for k,v in args.items() if k!='output_file'},protocol=4),value=control)
   if packpath.exists():pack=pickle.load(packpath.open('rb'))
   else:
    state=checkpoint(profile);state.update(case_id=case_id,timestamp=rec['timestamp'])
    # Validate replayed checkpoint balances against the existing control.
    for symbol,balance in control['checkpoint_state']['wallet_balances'].items():
     assert abs(state['assets'][symbol]['wallet']-balance)<=1e-8*max(1,abs(balance)),'Checkpoint wallet mismatch'
    funds=sum(max(0,x['wallet'])*x['price_usd'] for x in state['assets'].values())
    proposals={};recs={}
    for policy in policy_module.POLICIES:
     proposal=policy_module.propose(state,policy,budget_usd=funds);proposals[policy]=proposal
     new=dict(rec)
     for k in ['id','transactionHash','blockNumber','logIndex','pool','asset_address','on_behalf_of','payer','recipient','rate_mode','generation_error']:new.pop(k,None)
     new.update(case_id=case_id,abstained=proposal['action'] is None)
     new['Index Event']=(proposal['action'] or 'Deposit').lower();new['amount']=proposal.get('amount',0);new['amountUSD']=proposal['capital_usd']
     if proposal['action']:
      new['reserve']=proposal['asset'];new['symbol']=proposal['asset'];new['priceInUSD']=state['assets'][proposal['asset']]['price_usd']
     import math
     new.update(logAmount=math.log1p(new['amount']),logAmountUSD=math.log1p(new['amountUSD']))
     recs[policy]=(new,{'is_at_risk':proposal['action'] is not None,'policy':policy})
    pack=dict(case_id=case_id,index=index,variant=variant,state=state,proposals=proposals,recommendations=recs,parent_case_sha256=sha(source));save(packpath,pack)
   if a.generate_only:return
   for policy,pair in pack['recommendations'].items():
    dest=folder/(policy+'.pkl')
    if dest.exists():continue
    arms={};applied={};original=evaluator.get_simulation_outcome
    def capture(r,suffix,**kw):
     if suffix=='with':applied.update(r)
     value=cache(r,suffix,**kw);arms[suffix]=value;return value
    evaluator.get_simulation_outcome=capture;evaluator.outputFile=str(run/'logs/simulation.log')
    try:result=evaluator.process_recommendation(copy.deepcopy(pair))
    finally:evaluator.get_simulation_outcome=original
    save(dest,dict(case_id=case_id,index=index,variant=variant,policy=policy,recommendation=pair,result=result,arms=arms,applied_recommendation=applied))
   save(folder/'complete.json',dict(case_id=case_id))
  except Exception as error:
   import traceback
   save(folder/'error.json',dict(case_id=case_id,error=str(error),traceback=traceback.format_exc()))
 started=time.time()
 while True:
  ready=0;pending=0
  for i in range(a.start_index,count):
   for variant in ['core','indexed']:
    folder=run/'cases'/f'{i:05d}-{variant}'
    if any((folder/n).exists() for n in ['complete.json','unavailable.json','error.json']):continue
    if a.generate_only and (folder/'recommendations.pkl').exists():continue
    if not (a.parent_run/'cases'/f'{i:05d}-{variant}.json').exists():pending+=1;continue
    with (run/'logs/worker.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):task(i,variant)
    ready+=1
    save(run/'progress.json',dict(last_index=i,variant=variant,seconds=time.time()-started,cache=stats,completed=len(list((run/'cases').glob('*/complete.json'))),generated=len(list((run/'cases').glob('*/recommendations.pkl'))),errors=len(list((run/'cases').glob('*/error.json'))),unavailable=len(list((run/'cases').glob('*/unavailable.json'))),total=count*2))
  for variant in ['core','indexed']:
   for policy in policy_module.POLICIES:
    output={}
    for file in sorted((run/'cases').glob('*-'+variant+'/recommendations.pkl')):
     pack=pickle.load(file.open('rb'));output[pack['case_id']]=pack['recommendations'][policy]
    save(run/f'{variant}-{policy}-recommendations.pkl',output)
  if a.once or not pending:break
  save(run/'waiting.json',dict(pending_parent_cases=pending));time.sleep(30)
 save(run/'pass-complete.json',dict(pending_parent_cases=pending,cache=stats,generate_only=a.generate_only))
if __name__=='__main__':main()
