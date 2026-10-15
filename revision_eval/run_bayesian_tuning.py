"""Adaptive GP EI MCMC calibration, then frozen-winner testing on existing cases."""
import argparse,concurrent.futures as cf,contextlib,copy,fcntl,hashlib,json,multiprocessing as mp,os,pickle,sys,time
from collections import Counter,OrderedDict
from pathlib import Path
from . import audit_baseline_tuning as audit
from . import baseline_tuning_pilot as pilot
from . import run_baseline_tuning as common
from . import run_tuned_stress as stress
from . import gp_ei_mcmc as gp
from .policies import propose
from .combine_completed_stress import control_state,same_value
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'outputs/baseline-tuning-expanded-20261012'
STRESS=ROOT/'outputs/tuned-rule-stress-20261013'
DEFAULT=ROOT/'outputs/bayesian-tuned-bot-20261013'
ENGINE=None


def proposal(state,trigger,target):
    funds=sum(max(0,a['wallet'])*a['price_usd'] for a in state['assets'].values())
    return propose(state,'static_hf',trigger=trigger,target=target,budget_usd=funds)


def abstention(base,control,p,state):
    rec=audit.recommendation(base,p,state)
    return dict(control=common.outcome(control),intervention=common.outcome(control),
        execution_failed=False,proposed_capital_usd=0.,proposal=p,
        recommendation=(rec,dict(policy='bayesian_tuned_bot',is_at_risk=False)),
        evaluation_status='abstention_reuses_existing_control',transaction=None,
        arms={'without':control,'with':control},
        result=dict(success=True,reuse='Exact same no-action state and horizon; no new intervention'))


def prepare(run,workers):
    run.mkdir(parents=True,exist_ok=True)
    if (run/'manifest.json').exists():raise ValueError('Already prepared; resume run')
    old=json.loads((OLD/'manifest.json').read_text());summary=json.loads((OLD/'calibration-summary.json').read_text())
    frozen=json.loads((OLD/'frozen-setting.json').read_text())
    assert audit.sha(OLD/'calibration-summary.json')==frozen['calibration_summary_sha256']
    assert audit.sha(OLD/'calibration-cases.json')==frozen['cases_sha256']
    cases=json.loads((OLD/'calibration-cases.json').read_text());bundle={};original_records={n:{v:[] for v in ('core','indexed')} for n in summary}
    receipts={}
    for c in cases:
        assert c['followup_end']<old['calibration_boundary']
        folder=OLD/'calibration'/c['case_id'];statefile=folder/'state.pkl';state=pickle.load(statefile.open('rb'))['state']
        receipts[str(statefile)]=audit.sha(statefile)
        for name in summary:
            path=folder/(name.replace('/','-')+'.pkl');value=pickle.load(path.open('rb'));original_records[name][c['variant']].append(value)
            if name==frozen['configuration']:
                receipts[str(path)]=audit.sha(path)
                bundle[c['case_id']]=dict(case=c,state=state,base=value['recommendation'][0],control=value['arms']['without'])
    for name,variants in original_records.items():
        for variant,records in variants.items():assert common.count_outcomes(records)==summary[name][variant],(name,variant)
    inputs=dict(old['original_inputs'])
    for p in [OLD/'manifest.json',OLD/'calibration-cases.json',OLD/'calibration-summary.json',OLD/'frozen-setting.json',
              OLD/'core-matched-cases.jsonl',OLD/'indexed-matched-cases.jsonl',STRESS/'manifest.json',STRESS/'jobs.json',
              STRESS/'matched-cases.jsonl',STRESS/'matched-results.json']:
        inputs[str(p)]=audit.sha(p)
    inputs.update(receipts)
    sources={str(ROOT/'revision_eval'/name):audit.sha(ROOT/'revision_eval'/name) for name in
        ['run_bayesian_tuning.py','gp_ei_mcmc.py','run_baseline_tuning.py','run_tuned_stress.py','policies.py','audit_baseline_tuning.py','baseline_tuning_pilot.py']}
    manifest=dict(workers=workers,rounds=6,seed=20261013,method='Snoek et al. NeurIPS 2012 GP EI MCMC, equations 2/5/6',
        bounds=gp.BOUNDS.tolist(),prior_mean=gp.PRIOR_MEAN.tolist(),prior_sd=gp.PRIOR_SD.tolist(),hyper_limits=gp.HYPER_LIMITS.tolist(),
        mcmc=dict(chains=2,burn=128,draws=48,thin=3),objective='Loss=1-mean prevention if W=0, otherwise 1+W; select zero worsening, capital tie-breaker',
        cases=dict(Counter(c['variant'] for c in cases)),calibration_boundary=old['calibration_boundary'],
        code_identity_main=old['code_identity'],code_identity_stress=json.loads((STRESS/'manifest.json').read_text())['code_identity'],
        inputs=inputs,sources=sources,created=time.time(),network_requests=0)
    common.save(run/'calibration-bundle.pkl',bundle)
    manifest['bundle_sha256']=audit.sha(run/'calibration-bundle.pkl')
    seeds=[dict(configuration=n,trigger=float(n.split('/')[0]),target=float(n.split('/')[1]),summary=s,origin='original_grid') for n,s in summary.items()]
    common.save(run/'observations.json',seeds);common.save(run/'manifest.json',manifest)
    for stage,original in [('cal-engine',old),('main-engine',old),('stress-engine',json.loads((STRESS/'manifest.json').read_text()))]:
        engine=run/stage;engine.mkdir()
        for name in ('simulations','locks','simulation-timings'):
            (run/name).mkdir(exist_ok=True);(engine/name).symlink_to(run/name,target_is_directory=True)
        for name in ('failures','active','cases','results'):(engine/name).mkdir()
        copied=copy.deepcopy(original)
        copied['cache_roots']=[str(OLD/'simulations'),str(STRESS/'simulations'),*original['cache_roots']]
        common.save(engine/'manifest.json',copied)
    common.save(run/'progress.json',dict(phase='prepared',completed_rounds=0,workers=workers,heartbeat=time.time()))
    print(json.dumps(dict(prepared=str(run),cases=manifest['cases'],initial_observations=len(seeds),additional_rounds=6,workers=workers)),flush=True)


def typed_equal(left,right):
    """Exact values/types, without treating pickle reference sharing as input."""
    if type(left) is not type(right):return False
    if isinstance(left,dict):return left.keys()==right.keys() and all(typed_equal(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):return len(left)==len(right) and all(typed_equal(a,b) for a,b in zip(left,right))
    if isinstance(left,float):return left.hex()==right.hex()
    if isinstance(left,(str,int,bool,type(None))):return left==right
    return pickle.dumps(left,protocol=4)==pickle.dumps(right,protocol=4)


class CalibrationWorker(common.Worker):
    def __init__(self,run):
        super().__init__(run)
        self.bundle=pickle.load((self.run.parent/'calibration-bundle.pkl').open('rb'));self.contexts=OrderedDict()
    def cached(self,rec,suffix,kwargs,allow_control=False):
        if suffix!='with':return super().cached(rec,suffix,kwargs,allow_control)
        item=self.bundle[self.current_case];ctx=self.current_context;profile=kwargs['profile']
        if set(kwargs)-{'profile','lookahead_seconds','output_file'}:raise ValueError('Unknown simulation input')
        txs=profile['transactions'];prefix=ctx['profile']['transactions']
        if len(txs)!=len(prefix)+1 or not typed_equal(txs[:-1],prefix):raise ValueError('Counterfactual changed history')
        if not typed_equal({k:v for k,v in profile.items() if k!='transactions'},
                           {k:v for k,v in ctx['profile'].items() if k!='transactions'}):
            raise ValueError('Counterfactual changed nontransaction state')
        tx=txs[-1];c=item['case']
        expected_followup=max(1,ctx['horizon']-(rec['timestamp']-(rec['timestamp']-600))*2)
        if not typed_equal(kwargs['lookahead_seconds'],expected_followup):raise ValueError('Counterfactual horizon changed')
        base_key=str(audit.cache_key(rec,'without',dict(profile=ctx['profile'],lookahead_seconds=ctx['horizon']),self.identity))
        semantic=common.job_id(pickle.dumps((base_key,tx,expected_followup),protocol=4).hex())
        folder=self.run.parent/'calibration-semantics';folder.mkdir(exist_ok=True)
        dest=folder/(semantic+'.pkl');guard=(folder/(semantic+'.lock')).open('a')
        with guard:
            fcntl.flock(guard,fcntl.LOCK_EX)
            if dest.exists():
                with self.lock:self.status['cache_hits']+=1
                return pickle.load(dest.open('rb'))
            known=pickle.load((OLD/'calibration'/c['case_id']/'1.25-1.60.pkl').open('rb'))
            if typed_equal(known['transaction'],tx):
                # Same verified control input, exact converted action/types,
                # unchanged frozen code and same historical follow-up. Object
                # sharing in pickle does not change the economic inputs.
                value=known['arms']['with']
                common.save(folder/(semantic+'.json'),dict(source=str(OLD/'calibration'/c['case_id']/'1.25-1.60.pkl'),
                    source_sha256=audit.sha(OLD/'calibration'/c['case_id']/'1.25-1.60.pkl'),
                    control_key=base_key,transaction=tx,followup=expected_followup,code_identity=self.identity,
                    verification='Exact typed input values; pickle object-sharing differences ignored'))
                with self.lock:self.status['cache_hits']+=1
            else:value=super().cached(rec,suffix,kwargs,allow_control)
            common.save(dest,value);return value

    def main_job(self,job):
        dest=Path(job['destination'])
        if dest.exists():return dict(kind='main',case_id=job['case_id'],reused=True)
        item=self.bundle[job['original_case_id']];c=item['case']
        with self.lock:self.status['phase']='calibration'
        if c['case_id'] not in self.contexts:
            if audit.sha(c['profile'])!=c['profile_sha256']:raise ValueError('Calibration profile changed')
            profile=json.loads(Path(c['profile']).read_text());original=profile['transactions']
            profile['transactions']=[t for t in original if float(t.get('timestamp',0))<=c['cutoff']]
            profile.pop('indexed_checkpoint',None)
            wallet,provenance=common.prefix_funding(profile,c['cutoff'],self.inferencer)
            profile['initial_wallet']=wallet;profile['wallet_inference']=provenance
            if c['variant']=='indexed':self.attach(profile)
            horizon=common.HORIZON+600
            control=self.cached(item['base'],'without',dict(profile=profile,lookahead_seconds=horizon))
            if not same_value(control_state(control),control_state(item['control'])):raise ValueError('Calibration control changed')
            if common.outcome(control)!=common.outcome(item['control']):raise ValueError('Calibration control label changed')
            future=sorted((t for t in original if c['cutoff']<float(t.get('timestamp',0))<self.manifest['calibration_boundary']),key=lambda t:t['timestamp'])
            self.contexts[c['case_id']]=dict(profile=profile,horizon=horizon,control=control,future=future)
            if len(self.contexts)>64:self.contexts.popitem(last=False)
        ctx=self.contexts[c['case_id']]
        self.current_case=c['case_id'];self.current_context=ctx
        rec=audit.recommendation(item['base'],job['proposal'],item['state'])
        # The control was verified in the exact existing cache above. The flag
        # retains completed rapid pairs; it never authorizes a missing control.
        value=self.evaluate(ctx,(rec,dict(policy='bayesian_tuned_bot',is_at_risk=True)),allow_control=True)
        value.update(proposal=job['proposal'],proposed_capital_usd=job['proposal']['capital_usd'],
                     case=c,configuration=job['configuration'])
        common.save(dest,value)
        return dict(kind='main',case_id=job['case_id'],reused=False)


class MainWorker(common.Worker):
    def main_job(self,job):
        dest=Path(job['destination'])
        if dest.exists():return dict(kind='main',case_id=job['case_id'],reused=True)
        ctx=pilot.context(job['row'],self.source,self.identity,self.attach)
        rec=audit.recommendation(ctx['base'],job['proposal'],ctx['pack']['state'])
        value=self.evaluate(ctx,(rec,dict(policy='bayesian_tuned_bot',is_at_risk=True)))
        value.update(proposal=job['proposal'],proposed_capital_usd=job['proposal']['capital_usd'],configuration=job['configuration'])
        common.save(dest,value)
        return dict(kind='main',case_id=job['case_id'],reused=False)


def initialize(stage,engine):
    global ENGINE
    if stage=='calibration':ENGINE=CalibrationWorker(engine)
    elif stage=='main':ENGINE=MainWorker(engine)
    else:ENGINE=stress.StressWorker(engine)


def work(job):
    return ENGINE.work(('main',job))


def batch(pool,jobs,run,phase,round_number=0):
    todo=[j for j in jobs if not Path(j['destination']).exists()]
    futures={pool.submit(work,j):j for j in todo};receipts=[];started=time.time()
    while futures:
        done,_=cf.wait(futures,timeout=5,return_when=cf.FIRST_COMPLETED)
        for f in done:
            job=futures.pop(f);receipt=f.result();receipts.append(receipt)
            print(f'{phase} {len(receipts)}/{len(todo)} {job["case_id"]} success={receipt["success"]}',flush=True)
        errors=[r for r in receipts if not r['success']]
        common.save(run/'progress.json',dict(phase=phase,round=round_number,rounds=6,workers=pool._max_workers,
            started=started,heartbeat=time.time(),completed=len(jobs)-len(todo)+sum(r['success'] for r in receipts),
            total=len(jobs),remaining=len(futures),failures=len(errors),elapsed_seconds=time.time()-started,
            rate_eta_seconds=(time.time()-started)*len(futures)/len(receipts) if len(receipts)>=8 else None))
        if errors:
            common.save(run/f'{phase}-failure-receipts.json',errors)
            for f in futures:f.cancel()
            raise RuntimeError('Evaluation failed; no case is dropped. See stage failures.')
    common.save(run/f'{phase}-receipts.json',receipts)


def new_calibration_jobs(run,bundle,trial):
    folder=run/'calibration'/f'round-{trial["round"]:02d}';folder.mkdir(parents=True,exist_ok=True)
    jobs=[];counts=Counter()
    for case_id,item in bundle.items():
        p=proposal(item['state'],trial['trigger'],trial['target']);dest=folder/(case_id+'.pkl')
        if p['action'] is None:
            if not dest.exists():common.save(dest,abstention(item['base'],item['control'],p,item['state']))
            counts['abstentions']+=1
        else:
            counts['action_proposals']+=1;jobs.append(dict(case_id=f'r{trial["round"]:02d}:{case_id}',
                original_case_id=case_id,key=f'round-{trial["round"]}:{case_id}',destination=str(dest),
                proposal=p,configuration=trial['configuration']))
    trial['workload']=dict(counts);return jobs


def calibration_summary(run,bundle,trial):
    folder=run/'calibration'/f'round-{trial["round"]:02d}';by_variant={v:[] for v in ('core','indexed')}
    for case_id,item in bundle.items():
        value=pickle.load((folder/(case_id+'.pkl')).open('rb'))
        if value['control']!=common.outcome(item['control']):raise ValueError('Control label differs')
        by_variant[item['case']['variant']].append(value)
    summary={v:common.count_outcomes(records) for v,records in by_variant.items()}
    assert summary['core']['control_liquidations']==10 and summary['indexed']['control_liquidations']==23
    return summary


def prepare_main(run,selected):
    dest=run/'main-engine';rows=list(map(json.loads,(ROOT/'outputs/baseline-tuning-cache-audit-20261012/proposals.jsonl').open()))
    jobs=[]
    def load(row):return row,pickle.load(Path(row['pack']).open('rb'))
    with cf.ThreadPoolExecutor(max_workers=16) as pool:
        for row,pack in pool.map(load,rows):
            p=proposal(pack['state'],selected['trigger'],selected['target'])
            output=dest/'cases'/(common.job_id(row['case_id'])+'.pkl')
            if p['action'] is None:
                if not output.exists():
                    control=pickle.load(Path(row['control_cache']).open('rb'))
                    common.save(output,abstention(pack['recommendations']['static_hf'][0],control,p,pack['state']))
            else:jobs.append(dict(case_id=row['case_id'],key=row['case_id'],row=row,proposal=p,destination=str(output),configuration=selected['configuration']))
    common.save(run/'main-jobs.pkl',dict(rows=rows,jobs=jobs));return rows,jobs


def summarize_main(run,rows,selected):
    old={}
    for v in ('core','indexed'):
        for r in map(json.loads,(OLD/(v+'-matched-cases.jsonl')).open()):old[(v,r['case_id'])]=r
    output={v:[] for v in ('core','indexed')}
    for row in rows:
        result=pickle.load((run/'main-engine/cases'/(common.job_id(row['case_id'])+'.pkl')).open('rb'))
        record=copy.deepcopy(old[(row['variant'],row['case_id'])])
        if record['methods']['agent']['control']!=result['control']:raise ValueError('Main control differs')
        record['methods']['original_tuned_bot']=record['methods'].pop('tuned_static_hf')
        record['methods']['tuned_rule_bot']={k:result[k] for k in ['control','intervention','proposed_capital_usd','execution_failed']}
        record['configuration']=selected['configuration'];output[row['variant']].append(record)
    final={}
    for v,records in output.items():
        assert len(records)==(11848 if v=='core' else 5471)
        final[v]={p:common.count_outcomes([r['methods'][p] for r in records]) for p in records[0]['methods']}
        (run/(v+'-main-matched-cases.jsonl')).write_text(''.join(json.dumps(r,default=str)+'\n' for r in records))
    common.save(run/'main-matched-results.json',dict(configuration=selected['configuration'],by_variant=final))


def prepare_stress(run,selected):
    engine=run/'stress-engine';m=json.loads((engine/'manifest.json').read_text());m['configuration']=selected['configuration'];common.save(engine/'manifest.json',m)
    jobs=[]
    for old in json.loads((STRESS/'jobs.json').read_text()):
        pack=pickle.load(Path(old['pack']).open('rb'));p=proposal(pack['state'],selected['trigger'],selected['target'])
        job=copy.deepcopy(old);job['proposal']=p;job['recommendation']=(audit.recommendation(pack['recommendations']['static_hf'][0],p,pack['state']),dict(policy='bayesian_tuned_bot',is_at_risk=p['action'] is not None));job['destination']=str(engine/'cases'/(job['case_id']+'.pkl'))
        if p['action'] is None:
            if not Path(job['destination']).exists():
                control=pickle.load((STRESS/'cases'/(job['case_id']+'.pkl')).open('rb'))['arms']['without']
                value=abstention(pack['recommendations']['static_hf'][0],control,p,pack['state']);value.update(configuration=selected['configuration']);common.save(job['destination'],value)
        jobs.append(job)
    common.save(run/'stress-jobs.pkl',jobs);return jobs


def summarize_stress(run,jobs,selected):
    original={(r['index'],r['variant']):r for r in map(json.loads,(STRESS/'matched-cases.jsonl').open())};rows=[]
    for job in jobs:
        value=pickle.load(Path(job['destination']).open('rb'));old=copy.deepcopy(original[(job['previous']['index'],job['variant'])])
        assert value['control']==old['methods']['agent']['control']
        assert same_value(control_state(value['arms']['without']),old['methods']['agent']['control_checkpoint'])
        old['methods']['original_tuned_bot']=old['methods'].pop('tuned_rule_bot')
        old['methods']['tuned_rule_bot']={k:value[k] for k in ['control','intervention','proposed_capital_usd','execution_failed']};rows.append(old)
    final={v:{p:common.count_outcomes([r['methods'][p] for r in rows if r['variant']==v]) for p in rows[0]['methods']} for v in ('core','indexed')}
    assert len(rows)==1695
    (run/'stress-matched-cases.jsonl').write_text(''.join(json.dumps(r,default=str)+'\n' for r in rows))
    common.save(run/'stress-matched-results.json',dict(configuration=selected['configuration'],by_variant=final))


def check_inputs(run,m):
    for file,digest in {**m['inputs'],**m['sources']}.items():
        path=Path(file);path=path if path.is_absolute() else ROOT/path
        if audit.sha(path)!=digest:raise ValueError('Frozen input/source changed: '+str(path))
    if audit.sha(run/'calibration-bundle.pkl')!=m['bundle_sha256']:raise ValueError('Calibration bundle changed')
    old=json.loads((OLD/'manifest.json').read_text())
    assert audit.frozen_identity(Path(old['source']))==m['code_identity_main']
    assert audit.frozen_identity(ROOT)==m['code_identity_stress']


def execute(run,m,smoke=False):
    guard=(run/'coordinator.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);check_inputs(run,m)
    bundle=pickle.load((run/'calibration-bundle.pkl').open('rb'));observations=json.loads((run/'observations.json').read_text())
    if smoke:
        selected={};jobs=[]
        for cid,item in bundle.items():
            p=proposal(item['state'],1.25,1.6)
            if p['action'] and item['case']['variant'] not in selected:
                selected[item['case']['variant']]=cid;jobs.append(dict(case_id='smoke:'+cid,key='smoke:'+cid,original_case_id=cid,proposal=p,configuration='1.25/1.60',destination=str(run/'smoke'/(cid+'.pkl'))))
        assert len(jobs)==2
        with cf.ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=('calibration',str(run/'cal-engine'))) as pool:batch(pool,jobs,run,'smoke')
        common.save(run/'smoke-verified.json',dict(cases=selected,completed=True));return
    with cf.ProcessPoolExecutor(max_workers=m['workers'],mp_context=mp.get_context('spawn'),initializer=initialize,initargs=('calibration',str(run/'cal-engine'))) as pool:
        for number in range(1,m['rounds']+1):
            name=run/'trials'/f'round-{number:02d}.json'
            existing=next((o for o in observations if o.get('round')==number),None)
            if existing:continue
            if name.exists():trial=json.loads(name.read_text())
            else:
                common.save(run/'progress.json',dict(phase='selecting_configuration',round=number,heartbeat=time.time(),workers=m['workers'],completed_rounds=number-1))
                started=time.time();trial=gp.propose_next(observations,m['seed']+number);trial.update(round=number,selection_seconds=time.time()-started,observation_sha256=audit.sha(run/'observations.json'));common.save(name,trial)
            print(f'Round {number}: adaptive choice {trial["configuration"]}, EI={trial["expected_improvement"]:.6g}',flush=True)
            jobs=new_calibration_jobs(run,bundle,trial);common.save(name,trial)
            batch(pool,jobs,run,'calibration',number)
            trial['summary']=calibration_summary(run,bundle,trial);trial['origin']='GP EI MCMC';common.save(name,trial)
            observations.append({k:trial[k] for k in ['configuration','trigger','target','summary','origin','round']});common.save(run/'observations.json',observations)
    frozen=run/'frozen-setting.json'
    if frozen.exists():selected=json.loads(frozen.read_text())
    else:
        selected=copy.deepcopy(gp.winner(observations));selected.update(frozen=time.time(),observations_sha256=audit.sha(run/'observations.json'));common.save(frozen,selected)
    print('Frozen winner: '+selected['configuration'],flush=True)
    if (selected['trigger'],selected['target'])==(1.25,1.6):
        for source,dest in [(OLD/'matched-results.json','main-matched-results.json'),(STRESS/'matched-results.json','stress-matched-results.json')]:
            common.save(run/dest,json.loads(source.read_text()))
        common.save(run/'complete.json',dict(completed=time.time(),winner=selected['configuration'],testing_reused=True));common.save(run/'progress.json',dict(phase='complete',heartbeat=time.time(),completed_rounds=6,testing_reused=True));return
    if not (run/'main-matched-results.json').exists():
        if (run/'main-jobs.pkl').exists():queue=pickle.load((run/'main-jobs.pkl').open('rb'));rows,jobs=queue['rows'],queue['jobs']
        else:
            common.save(run/'progress.json',dict(phase='preparing_main_test',heartbeat=time.time(),winner=selected['configuration']))
            rows,jobs=prepare_main(run,selected)
        with cf.ProcessPoolExecutor(max_workers=m['workers'],mp_context=mp.get_context('spawn'),initializer=initialize,initargs=('main',str(run/'main-engine'))) as pool:batch(pool,jobs,run,'main_testing')
        summarize_main(run,rows,selected)
    if not (run/'stress-matched-results.json').exists():
        jobs=pickle.load((run/'stress-jobs.pkl').open('rb')) if (run/'stress-jobs.pkl').exists() else prepare_stress(run,selected)
        with cf.ProcessPoolExecutor(max_workers=m['workers'],mp_context=mp.get_context('spawn'),initializer=initialize,initargs=('stress',str(run/'stress-engine'))) as pool:batch(pool,jobs,run,'stress_testing')
        summarize_stress(run,jobs,selected)
    check_inputs(run,m)
    common.save(run/'complete.json',dict(completed=time.time(),winner=selected['configuration'],testing_reused=False,
        main_cases=17319,stress_cases=1695,inputs_reverified=True))
    common.save(run/'progress.json',dict(phase='complete',heartbeat=time.time(),completed_rounds=6,winner=selected['configuration']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','smoke','run'));p.add_argument('--run',type=Path,default=DEFAULT);p.add_argument('--workers',type=int,default=16);args=p.parse_args()
    if not 1<=args.workers<=len(os.sched_getaffinity(0)):p.error('Worker count must fit available CPUs')
    run=args.run.resolve()
    if args.mode=='prepare':return prepare(run,args.workers)
    try:execute(run,json.loads((run/'manifest.json').read_text()),args.mode=='smoke')
    except Exception as error:
        common.save(run/'progress.json',dict(phase='failed',heartbeat=time.time(),error=str(error)))
        raise
if __name__=='__main__':main()
