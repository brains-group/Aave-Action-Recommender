"""Resume-safe paired performSimulations evaluation in a fresh cache namespace."""
import argparse,os,json,hashlib,time,contextlib
from pathlib import Path
from multiprocessing import Pool

def worker(task):
    index,variant=task
    import performSimulations as p
    if variant=='indexed':os.environ.update(AAVE_INDEXED_COVERAGE=CONFIG['coverage'],AAVE_INDEXED_ASSETS=CONFIG['assets'])
    else:os.environ.pop('AAVE_INDEXED_COVERAGE',None)
    arms={};original=p.get_simulation_outcome
    def capture(rec,suffix,**kwargs):
        value=original(rec,suffix,**kwargs)
        arms[suffix]={k:value.get(k) for k in ['user_address','checkpoint_state','final_state','liquidation_stats','execution_failures','indexed_checkpoint']}
        return value
    p.get_simulation_outcome=capture;start=time.time();dest=RUN/'cases'/f'{index:05d}-{variant}.json'
    try:
        with (RUN/'logs'/f'worker-{os.getpid()}.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            p.outputFile=str(RUN/'logs'/f'strategies-{os.getpid()}.log');result=p.process_recommendation(ITEMS[index])
        record={'index':index,'variant':variant,'elapsed':time.time()-start,'result':result,'arms':arms}
        tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(p.convert_to_json_serializable(record)));tmp.replace(dest)
        return {'index':index,'variant':variant,'success':result.get('success'),'elapsed':record['elapsed']}
    finally:p.get_simulation_outcome=original

def main():
    global RUN,CONFIG,ITEMS
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['run-dir','recommendations','coverage','assets']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--limit',type=int);a=ap.parse_args()
    RUN=a.run_dir.resolve();RUN.mkdir(parents=True,exist_ok=True)
    for name in ['cases','logs','cache']:(RUN/name).mkdir(exist_ok=True)
    os.environ.update(AAVE_EVALUATION_CACHE=str(RUN/'cache'),AAVE_RECOMMENDATIONS_FILE=str(a.recommendations.resolve()))
    import performSimulations as p
    import utils.simulations as u
    from simulator import utils as su
    ITEMS=list(p.recommendations.values());CONFIG={'coverage':str(a.coverage.resolve()),'assets':str(a.assets.resolve())};n=min(a.limit or len(ITEMS),len(ITEMS))
    identity={'recommendations_sha256':hashlib.sha256(a.recommendations.read_bytes()).hexdigest(),'code_identity':u.simulation_code_identity(),'coverage_sha256':hashlib.sha256(a.coverage.read_bytes()).hexdigest(),'assets_sha256':hashlib.sha256(a.assets.read_bytes()).hexdigest(),'recommendations':len(ITEMS),'selected':n,'variants':['core','indexed']}
    identity['driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()+Path(p.__file__).read_bytes()).hexdigest()
    profile_stats=[]
    for item in ITEMS[:n]:
        rec,_=p.normalize_recommendation(item);user=rec['user'];paths=[Path(p.PROFILES_DIR)/g/'profiles'/f'user_{user}.json' for g in ['non_liquidated_profiles','liquidated_profiles']]
        path=next((path for path in paths if path.exists()),None)
        profile_stats.append((str(path.resolve()),path.stat().st_size,path.stat().st_mtime_ns) if path else (user,'missing'))
    identity['profile_metadata_sha256']=hashlib.sha256(json.dumps(profile_stats).encode()).hexdigest()
    manifest=RUN/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('Run inputs changed; choose a fresh run directory')
    manifest.write_text(json.dumps(identity,indent=2));su.setup_protocol()
    tasks=[(i,v) for i in range(n) for v in ['core','indexed'] if not (RUN/'cases'/f'{i:05d}-{v}.json').exists()]
    print('Starting',len(tasks),'remaining cases from',n,'recommendations',flush=True);start=time.time()
    with Pool(a.workers) as pool:
        for done,status in enumerate(pool.imap_unordered(worker,tasks,chunksize=1),1):
            if done%20==0 or done==len(tasks):
                (RUN/'progress.json').write_text(json.dumps({'completed_this_run':done,'remaining_this_run':len(tasks)-done,'seconds':time.time()-start,'last':status}));print(done,'/',len(tasks),'elapsed',round(time.time()-start),flush=True)
    (RUN/'complete.json').write_text(json.dumps({'completed':n*2,'seconds':time.time()-start}))
if __name__=='__main__':main()
