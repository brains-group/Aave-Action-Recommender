"""Regenerate a fixed recommendation cohort, then evaluate in isolated caches.

Historical fitted models are held fixed to isolate simulator/funding changes.
No API requests. Missing models fail explicitly instead of fitting on history subsets.
"""
import argparse, os, json, pickle, hashlib, shutil, time, subprocess, sys, traceback
from pathlib import Path
from multiprocessing import Pool

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20), b''): h.update(block)
    return h.hexdigest()

def atomic(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f: pickle.dump(value, f, protocol=4)
    tmp.replace(path)

def prepare(a):
    import pandas as pd
    run=a.run_dir; inputs=run/'inputs';inputs.mkdir(parents=True,exist_ok=True)
    cache=run/'generation-cache';cache.mkdir(exist_ok=True)
    manifest_path=inputs/'manifest.json'
    identity={'cohort_sha256':sha(a.cohort),'old_recommendations_sha256':sha(a.old_recommendations),
              'coverage_sha256':sha(a.coverage),'assets_sha256':sha(a.assets),
              'data':str(a.data.resolve()),'models':str(a.model_cache.resolve()),
              'policy':'fixed fitted survival models; refreshed checkpoint funding and predictions',
              'core_transactions':str(a.core_transactions.resolve())}
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=identity:
        raise ValueError('Inputs changed: choose a fresh run directory')
    manifest_path.write_text(json.dumps(identity,indent=2))
    for source,name in [(a.cohort,'cohort.csv'),(a.old_recommendations,'old-recommendations.pkl'),(a.coverage,'coverage.json'),(a.assets,'assets.json')]:
        dest=inputs/name
        if not dest.exists():shutil.copy2(source,dest)
    for name in ['models','data']:
        dest=cache/name
        if not dest.exists():shutil.copytree(a.model_cache/name,dest)
    if not (cache/'date_ranges.pkl').exists():shutil.copy2(a.model_cache/'date_ranges.pkl',cache/'date_ranges.pkl')
    rows=pd.read_csv(inputs/'cohort.csv'); old=pickle.load(open(inputs/'old-recommendations.pkl','rb'))
    by_user={row.user:row for _,row in rows.iterrows()}
    ordered=[]
    for item in old.values():
        row=by_user[item[0]['user']]
        if int(row.timestamp)+600!=int(item[0]['timestamp']):raise ValueError('Cohort/action timestamp mismatch')
        ordered.append(row)
    atomic(inputs/'ordered-cohort.pkl',ordered)
    cutoff=rows.set_index('user').timestamp.to_dict(); partdir=inputs/'history-parts';partdir.mkdir(exist_ok=True)
    sources=[]
    for source in sorted(a.data.glob('*/*/data.csv')):
        name='-'.join(source.parts[-3:-1]);dest=partdir/(name+'.pkl');stat=source.stat()
        sources.append({'path':str(source.resolve()),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns})
        if dest.exists():continue
        pieces=[]
        for chunk in pd.read_csv(source,chunksize=100000):
            keep=chunk[chunk.timestamp<=chunk.user.map(cutoff)]
            if len(keep):pieces.append(keep)
        value=pd.concat(pieces) if pieces else pd.DataFrame()
        atomic(dest,value);print('Prepared historical features',name,len(value),flush=True)
    (inputs/'historical-feature-sources.json').write_text(json.dumps(sources,indent=2))
    # Training source is explicitly available without overwriting the paper dataset.
    training={'core_transactions':str(a.core_transactions.resolve()),'sha256':sha(a.core_transactions),
      'supplementary_predictors':False,'fitted_models':'historical models copied; no retraining in this controlled comparison',
      'format_command':f'python -m aave_data_pipeline.survival --transactions {a.core_transactions.resolve()} --stablecoins /home/spadef/DMLR_DeFi_Survival_Benchmark/Data/Other_Data/stablecoins.csv --output NEW_OUTPUT_DIRECTORY --tasks journal --semantics corrected --train-cutoff 1722526142 --test-cutoff 1755726959',
      'training_adapter':'AAVE_SURVIVAL_DATA=NEW_OUTPUT_DIRECTORY AAVE_CORE_TRANSACTIONS='+str(a.core_transactions.resolve()),
      'limitation':'Fresh survival formatting requires parity/source review before replacing historical fitted models.'}
    (inputs/'training-data.json').write_text(json.dumps(training,indent=2))
    files={str(f.relative_to(cache)):sha(f) for name in ['models','data'] for f in sorted((cache/name).iterdir()) if f.is_file()}
    (inputs/'model-files.json').write_text(json.dumps(files,indent=2))

def history(user_id,up_to_timestamp):
    import pandas as pd
    frame=HISTORY.get(user_id)
    if frame is None:return pd.DataFrame()
    frame=frame[frame.timestamp<=up_to_timestamp].sort_values('timestamp')
    if len(frame)>1000:frame=frame.tail(10000).reset_index(drop=True)
    return frame

def worker(index):
    import actionAgentTraining as agent
    from utils.logger import set_log_file
    set_log_file(f'generation-{os.getpid()}.log',False,str(RUN/'logs'),file_level='INFO')
    row=ROWS[index];start=time.time();dest=RUN/'generated'/f'{index:05d}.pkl'
    try:
        result=agent.recommend_action(row)
        record={'index':index,'result':result,'elapsed':time.time()-start,'error':None}
    except Exception as exc:
        placeholder=agent.generate_next_transaction(row,'Deposit',amount=0)
        placeholder['generation_error']=str(exc)
        record={'index':index,'result':(placeholder,{}),'elapsed':time.time()-start,'error':traceback.format_exc()}
    atomic(dest,record)
    return {'index':index,'elapsed':record['elapsed'],'error':record['error']}

def generate(a):
    global RUN,ROWS,HISTORY
    import pandas as pd
    RUN=a.run_dir
    for name in ['generated','logs']:(RUN/name).mkdir(exist_ok=True)
    os.environ.update(AAVE_EVALUATION_CACHE=str(RUN/'generation-cache'),AAVE_INDEXED_COVERAGE=str(RUN/'inputs/coverage.json'),AAVE_INDEXED_ASSETS=str(RUN/'inputs/assets.json'),AAVE_FROZEN_MODELS='1')
    # Explicitly keep historical model inputs; refreshed core is recorded separately.
    os.environ.pop('AAVE_SURVIVAL_DATA',None);os.environ.pop('AAVE_CORE_TRANSACTIONS',None)
    import actionAgentTraining as agent
    from simulator import utils as su
    ROWS=pickle.load(open(RUN/'inputs/ordered-cohort.pkl','rb'))
    n=min(a.limit or len(ROWS),len(ROWS))
    parts=[pickle.load(open(f,'rb')) for f in sorted((RUN/'inputs/history-parts').glob('*.pkl'))]
    frame=pd.concat([x for x in parts if len(x)]);del parts
    HISTORY={u:g for u,g in frame.groupby('user',sort=False)};del frame
    agent.get_user_history=history
    su.setup_protocol()
    code=hashlib.sha256(b''.join(Path(f).read_bytes() for f in [agent.__file__,Path(__file__),Path(__file__).parent/'utils/recommendation_funding.py',Path(__file__).parent/'utils/model_training.py'])).hexdigest()
    from utils.simulations import simulation_code_identity
    identity={'generation_code':code,'simulation_code':simulation_code_identity(),'inputs':sha(RUN/'inputs/manifest.json'),'models':sha(RUN/'inputs/model-files.json')}
    path=RUN/'generation-manifest.json'
    if path.exists() and json.loads(path.read_text())!=identity:raise ValueError('Generation inputs/code changed; new run required')
    path.write_text(json.dumps(identity,indent=2))
    tasks=[i for i in range(n) if not (RUN/'generated'/f'{i:05d}.pkl').exists()]
    start=time.time()
    with Pool(a.workers) as pool:
        for done,status in enumerate(pool.imap_unordered(worker,tasks),1):
            (RUN/'generation-progress.json').write_text(json.dumps({'done_this_run':done,'remaining_this_run':len(tasks)-done,'seconds':time.time()-start,'last':status},default=str))
            if done%10==0 or status['error']:print('Generated',done,'/',len(tasks),status,flush=True)
    records=[pickle.load(open(RUN/'generated'/f'{i:05d}.pkl','rb')) for i in range(n)]
    atomic(RUN/'recommendations.pkl',{str(i):r['result'] for i,r in enumerate(records)})
    summary={'selected':n,'errors':sum(bool(r['error']) for r in records),'abstentions':sum(bool(r['result'][1].get('abstention_reason')) for r in records),'seconds':time.time()-start}
    (RUN/'generation-complete.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ['run-dir','cohort','old-recommendations','data','model-cache','coverage','assets','core-transactions']:ap.add_argument('--'+arg,type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--limit',type=int);ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--evaluate',action='store_true');a=ap.parse_args();a.run_dir=a.run_dir.resolve()
    prepare(a)
    if a.prepare_only:return
    generate(a)
    if a.evaluate:
        root=Path(__file__).parent
        cmd=[sys.executable,str(root/'evaluate_recommendations.py'),'--run-dir',str(a.run_dir/'evaluation'),'--recommendations',str(a.run_dir/'recommendations.pkl'),'--coverage',str(a.run_dir/'inputs/coverage.json'),'--assets',str(a.run_dir/'inputs/assets.json'),'--workers',str(a.workers)]
        # Generation-only environment must not leak into the core evaluation arm.
        env=dict(os.environ)
        for k in ['AAVE_INDEXED_COVERAGE','AAVE_INDEXED_ASSETS','AAVE_FROZEN_MODELS']:env.pop(k,None)
        subprocess.run(cmd,check=True,env=env)
        subprocess.run([sys.executable,str(root/'summarize_recommendation_evaluation.py'),str(a.run_dir/'evaluation')],check=True,env=env)
        (a.run_dir/'complete.json').write_text(json.dumps({'generation_and_evaluation_complete':True}))
if __name__=='__main__':main()
