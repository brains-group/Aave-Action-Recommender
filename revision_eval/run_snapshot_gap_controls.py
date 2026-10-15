"""Only previously unavailable controls, using explicit post-history anchoring."""
import argparse
import contextlib
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','gaps','recommendations','coverage','assets','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for name in ('source','gaps','recommendations','coverage','assets','output'):setattr(a,name,getattr(a,name).resolve())
    a.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(a.output/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in ('profiles','data'):
        if not (a.output/name).exists():(a.output/name).symlink_to((a.source/name).resolve())
    os.chdir(a.output)
    os.environ.update(AAVE_EVALUATION_CACHE=str(a.output/'cache'),AAVE_SIMULATION_CACHE=str(a.output/'simulations'),
        AAVE_INDEXED_COVERAGE=str(a.coverage),AAVE_INDEXED_ASSETS=str(a.assets),AAVE_INDEXED_ANCHOR_PHASE='after_history')
    for key in ('AAVE_SURVIVAL_DATA','AAVE_CORE_TRANSACTIONS','AAVE_HISTORICAL_EVIDENCE'):os.environ.pop(key,None)
    sys.path.insert(0,str(a.source))
    import utils.simulations as u
    cases=json.loads(a.gaps.read_text())['cases']
    with a.recommendations.open('rb') as f:recs=list(pickle.load(f).items())
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=dict(indices=[r['index'] for r in cases],code_identity=u.simulation_code_identity(),
        inputs={str(path):sha(path) for path in (a.gaps,a.recommendations,a.coverage,a.assets)},
        runner_sha256=sha(Path(__file__)),variant='post_history_indexed_control',
        note='Previously unavailable only; same saved funding and original variable horizon; no paired agent rerun')
    path=a.output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Changed inputs; use a new output')
    path.write_text(json.dumps(manifest,indent=2));(a.output/'cases').mkdir(exist_ok=True)
    for case in cases:
        index=case['index'];dest=a.output/'cases'/f'{index:05d}.json'
        if dest.exists():continue
        rec=recs[index][1][0];start=time.time()
        record=dict(index=index,case_id=str(recs[index][0]))
        try:
            profile,*_,horizon=u.get_limited_user_profile(rec,return_extras=True)
            anchor=profile['indexed_checkpoint']
            if anchor['timestamp']>=rec['timestamp']:raise ValueError('Anchor reaches decision time')
            if anchor['timestamp']!=case['checkpoint']:raise ValueError('Checkpoint changed')
            profile['counterfactual_start_timestamp']=rec['timestamp']
            with (a.output/'simulation.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                control=u.get_simulation_outcome(rec,'without',profile=profile,lookahead_seconds=horizon)
            record.update(success=True,control=control,horizon_seconds=horizon)
        except Exception as error:record.update(success=False,error=str(error))
        record['seconds']=time.time()-start
        temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(record,default=str));temp.replace(dest)
        print(index,record['success'],record.get('error',''),flush=True)
    rows=[json.loads(path.read_text()) for path in (a.output/'cases').glob('*.json')]
    (a.output/'summary.json').write_text(json.dumps(dict(selected=len(cases),completed=len(rows),
        success=sum(r['success'] for r in rows),errors=[dict(index=r['index'],error=r['error']) for r in rows if not r['success']]),indent=2))

if __name__=='__main__':main()
