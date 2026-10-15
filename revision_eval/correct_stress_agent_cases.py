"""Regenerate only 13 failed saved cases with matching reconstruction; no tuning."""
import argparse,concurrent.futures as cf,copy,fcntl,json,multiprocessing as mp,os,pickle,shutil,subprocess,sys,time
from pathlib import Path
from . import audit_baseline_tuning as audit
from . import run_baseline_tuning as common
from . import run_tuned_stress as stress
from .combine_completed_stress import arm_summary,control_state,same_value
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/stress-agent-state-correction-20261013'
PARENT=ROOT/'outputs/high-risk-shortest-20260925'
STUDY=PARENT/'studies/fixed-2592000-seconds-trained-v1'
BOT=ROOT/'outputs/bayesian-tuned-bot-20261013'
ENGINE=None


def prepare():
    import hashlib,pandas as pd
    if (RUN/'manifest.json').exists():raise ValueError('Already prepared; resume run')
    traces=json.loads((ROOT/'outputs/bayesian-agent-gap-audit-20261013/missed-stress-traces.json').read_text())
    original=json.loads((STUDY/'generation-core/manifest.json').read_text())
    cohort=PARENT/'cohort/high_risk.csv';histories=PARENT/'histories';models=Path(original['model_cache'])
    assert audit.sha(cohort)==original['cohort_sha256']
    assert audit.sha(histories/'manifest.json')==original['history_manifest_sha256']
    sm=json.loads((stress.RUN/'manifest.json').read_text());assert audit.frozen_identity(ROOT)==sm['code_identity']
    snapshots=RUN/'source';snapshots.mkdir(exist_ok=True)
    sources={}
    for name in ['actionAgentTraining.py','revision_eval/generate_shortest_recommendations.py','revision_eval/correct_stress_agent_cases.py','revision_eval/run_tuned_stress.py','revision_eval/run_baseline_tuning.py','revision_eval/followup.py']:
        path=ROOT/name;sources[str(path)]=audit.sha(path);shutil.copyfile(path,snapshots/path.name)
    assert sources[str(ROOT/'actionAgentTraining.py')]==original['agent_sha256']
    inputs={str(p):audit.sha(p) for p in [cohort,histories/'manifest.json',STUDY/'generation-core/recommendations.pkl',STUDY/'generation-core/manifest.json',BOT/'stress-matched-cases.jsonl',BOT/'verification.json',Path(sm['coverage']),Path(sm['assets']),ROOT/'outputs/bayesian-agent-gap-audit-20261013/missed-stress-traces.json']}
    for v in ('core','indexed'):inputs[str(RUN/(v+'-case-ids.json'))]=audit.sha(RUN/(v+'-case-ids.json'))
    frame=pd.read_csv(cohort);dates=pickle.load((models/'date_ranges.pkl').open('rb'));dates=dates[0].union(dates[1]);old_results=STUDY/'generation-core/cache/results'
    # Copy only compatible predictions/history prefixes needed by these IDs.
    # Copies keep old cache files protected from later writes.
    seeded={}
    saved=pickle.load((STUDY/'generation-core/recommendations.pkl').open('rb'))
    for variant in ('core','indexed'):
        selected=set(json.loads((RUN/(variant+'-case-ids.json')).read_text()));dest=RUN/('generation-'+variant)/'cache/results';dest.mkdir(parents=True,exist_ok=True);count=0
        for _,row in frame.loc[frame.case_id.isin(selected)].iterrows():
            base=row.drop(labels=['stress_origin','source_row','case_id'],errors='ignore')
            final=saved[row['case_id']][0].drop(labels=['case_id'],errors='ignore')
            names=[hashlib.sha256(pickle.dumps(x.to_dict(),protocol=4)).hexdigest()+'.pkl' for x in (base,final)]
            for timestamp in (float(base['timestamp']),float(base['timestamp'])+600):
                date=dates[dates<=pd.to_datetime(timestamp,unit='s')].max();key=hashlib.sha256(pickle.dumps((base['user'],timestamp,str(date)),protocol=4)).hexdigest();names.append('history_'+key+'.pkl')
            for name in names:
                source=old_results/name
                if source.is_file():shutil.copyfile(source,dest/name);count+=1
        seeded[variant]=count
    engine=RUN/'evaluation';engine.mkdir(exist_ok=True)
    m=copy.deepcopy(sm);m['configuration']='agent_matching_checkpoint_state';m['cache_roots']=json.loads((RUN/'read-cache-roots.json').read_text());common.save(engine/'manifest.json',m)
    for n in ('cases','active','failures','locks','simulations','simulation-timings'):(engine/n).mkdir(exist_ok=True)
    manifest=dict(sources=sources,inputs=inputs,code_identity=sm['code_identity'],model_listing=original['model_listing'],model_cache=str(models),cohort=str(cohort),histories=str(histories),coverage=sm['coverage'],assets=sm['assets'],horizon_seconds=2592000,workers=13,targets=[dict(index=x['index'],variant=x['variant'],case_id=x['case_id']) for x in traces],seeded_prediction_files=seeded,created=time.time(),policy_changes=False,training=False)
    common.save(RUN/'manifest.json',manifest);common.save(RUN/'progress.json',dict(phase='prepared',total=13,heartbeat=time.time()));print(json.dumps(dict(targets=13,seeded=seeded)),flush=True)


def check(m):
    for p,h in {**m['inputs'],**m['sources']}.items():assert audit.sha(p)==h,'Input/source changed: '+p
    assert audit.frozen_identity(ROOT)==m['code_identity']
    for name,listing in m['model_listing'].items():
        stat=(Path(m['model_cache'])/'models'/name).stat();assert [stat.st_size,stat.st_mtime_ns]==listing,'Frozen model changed: '+name


def initialize():
    global ENGINE
    ENGINE=stress.StressWorker(RUN/'evaluation')


def evaluate(job):return ENGINE.work(('main',job))


def execute():
    lock=(RUN/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=json.loads((RUN/'manifest.json').read_text());check(m);children={};started=time.time()
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    for name in ['AAVE_INDEXED_COVERAGE','AAVE_INDEXED_ASSETS','AAVE_HISTORICAL_EVIDENCE','AAVE_SURVIVAL_DATA','AAVE_CORE_TRANSACTIONS']:env.pop(name,None)
    for v in ('core','indexed'):
        folder=RUN/('generation-'+v)
        if (folder/'complete.json').exists():continue
        cmd=[sys.executable,'-u','-m','revision_eval.generate_shortest_recommendations','--cohort',m['cohort'],'--histories',m['histories'],'--model-cache',m['model_cache'],'--coverage',m['coverage'],'--assets',m['assets'],'--run',str(folder),'--workers','1' if v=='core' else '12','--checkpoint-mode',v,'--horizon-seconds',str(m['horizon_seconds']),'--case-ids',str(RUN/(v+'-case-ids.json')),'--read-cache-roots',str(RUN/'read-cache-roots.json')]
        log=(RUN/(v+'-generation.log')).open('a');children[v]=(subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log)
    try:
        while children:
            counts={v:len(list((RUN/('generation-'+v)/'cases').glob('*.pkl'))) for v in ('core','indexed')}
            common.save(RUN/'progress.json',dict(phase='generation',done=sum(counts.values()),total=13,by_variant=counts,heartbeat=time.time(),elapsed=time.time()-started))
            for v,(child,log) in list(children.items()):
                if child.poll() is not None:
                    log.close();del children[v]
                    if child.returncode:raise ValueError(v+' generation failed; see log')
            if children:time.sleep(5)
    except BaseException:
        for child,log in children.values():child.terminate();child.wait();log.close()
        raise
    pairs={}
    for v in ('core','indexed'):
        folder=RUN/('generation-'+v);done=json.loads((folder/'complete.json').read_text());assert done['errors']==0
        pairs[v]=pickle.load((folder/'recommendations.pkl').open('rb'))
        assert set(pairs[v])==set(json.loads((RUN/(v+'-case-ids.json')).read_text()))
    original={j['case_id']:j for j in json.loads((stress.RUN/'jobs.json').read_text())};jobs=[]
    for t in m['targets']:
        key=f'{t["index"]:05d}-{t["variant"]}';job=copy.deepcopy(original[key]);assert job['economic_case_id']==t['case_id']
        rec,info=pairs[t['variant']][t['case_id']];rec=rec.to_dict() if hasattr(rec,'to_dict') else dict(rec)
        # Agent actions omit this field; the policy adapter requires an explicit
        # boolean. False matches performSimulations' existing default exactly.
        rec.setdefault('abstained',False)
        job['recommendation']=(rec,info);job['proposal']=dict(action=None if rec.get('abstained') else str(rec['Index Event']).title(),capital_usd=float(rec['amountUSD']))
        jobs.append(job)
    common.save(RUN/'jobs.pkl',jobs);pending=[j for j in jobs if not (RUN/'evaluation/cases'/(j['case_id']+'.pkl')).exists()]
    with cf.ProcessPoolExecutor(max_workers=min(13,len(pending) or 1),mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        futures={pool.submit(evaluate,j):j for j in pending};processed=13-len(pending)
        while futures:
            done,_=cf.wait(futures,timeout=5,return_when=cf.FIRST_COMPLETED)
            for f in done:
                j=futures.pop(f);receipt=f.result();print(j['case_id'],receipt,flush=True)
                if not receipt['success']:raise ValueError('Evaluation failed: '+j['case_id'])
                processed+=1
            common.save(RUN/'progress.json',dict(phase='evaluation',done=processed,total=13,remaining=len(futures),heartbeat=time.time(),elapsed=time.time()-started))
    old=list(map(json.loads,(BOT/'stress-matched-cases.jsonl').open()));changed={};details=[]
    for job in jobs:
        value=pickle.load((RUN/'evaluation/cases'/(job['case_id']+'.pkl')).open('rb'));assert value['result']['success']
        previous=next(r for r in old if r['index']==job['previous']['index'] and r['variant']==job['variant'])
        assert same_value(control_state(value['arms']['without']),previous['methods']['agent']['control_checkpoint'])
        current=arm_summary(value['arms']);assert current['control']==previous['methods']['agent']['control']
        changed[(previous['index'],previous['variant'])]=current
        details.append(dict(index=previous['index'],variant=previous['variant'],case_id=previous['case_id'],old_agent=previous['methods']['agent'],new_agent=current,recommendation=job['recommendation']))
    assert len(changed)==13
    rows=copy.deepcopy(old)
    for r in rows:
        key=(r['index'],r['variant'])
        if key in changed:r['methods']['agent']=changed[key]
    for a,b in zip(old,rows):
        assert a['case_id']==b['case_id']
        for method in a['methods']:
            if method!='agent':assert a['methods'][method]==b['methods'][method]
        if (a['index'],a['variant']) not in changed:assert a==b
    summary={v:{p:common.count_outcomes([r['methods'][p] for r in rows if r['variant']==v]) for p in rows[0]['methods']} for v in ('core','indexed')}
    common.save(RUN/'matched-results.json',dict(by_variant=summary));common.save(RUN/'changed-cases.json',details)
    (RUN/'matched-cases.jsonl').write_text(''.join(json.dumps(r,default=str)+'\n' for r in rows));check(m)
    common.save(RUN/'complete.json',dict(completed=time.time(),changed_rows=13,total_rows=len(rows),controls_unchanged=True,baselines_unchanged=True,models_unchanged=True,policy_unchanged=True))
    common.save(RUN/'progress.json',dict(phase='complete',done=13,total=13,heartbeat=time.time()))
    print(json.dumps({v:d['agent'] for v,d in summary.items()},indent=2),flush=True)


def status():
    p=json.loads((RUN/'progress.json').read_text());print('Stress agent correction | '+p['phase']+' | '+str(p.get('done',0))+'/13',flush=True)
    if 'by_variant' in p:print('Generated:',p['by_variant'],flush=True)
    if (RUN/'matched-results.json').exists():
        for v,values in json.loads((RUN/'matched-results.json').read_text())['by_variant'].items():
            a=values['agent'];print(v,str(a['prevented'])+'/'+str(a['control_liquidations'])+' prevented; '+str(a['worsened'])+' worsened',flush=True)
    if p.get('error'):print('ERROR:',p['error'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run','status']);p.add_argument('--watch',type=float);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='status':
        while True:
            status()
            if not a.watch or (RUN/'complete.json').exists():break
            time.sleep(max(1,a.watch))
    else:
        try:execute()
        except Exception as error:
            common.save(RUN/'progress.json',dict(phase='failed',error=str(error),heartbeat=time.time()));raise
