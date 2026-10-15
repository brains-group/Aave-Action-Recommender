"""Evaluate the previously frozen tuned rule on existing matched stress cases.

No calibration or new cohort selection. Original stress controls/caches are
read-only. A shared exact-input cache and locks deduplicate interventions.
"""
import argparse
import concurrent.futures as cf
import contextlib
import copy
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from . import audit_baseline_tuning as audit
from . import baseline_tuning_pilot as pilot
from . import run_baseline_tuning as common
from .combine_completed_stress import control_state, same_value
from .followup import fixed_followup
from .policies import propose

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/tuned-rule-stress-20261013'
STUDY = ROOT / 'outputs/high-risk-shortest-20260925/studies/fixed-2592000-seconds-trained-v1'
LEDGER = ROOT / 'outputs/revision-audits-20260929/stress-common-v1/cases.jsonl'
CALIBRATION = ROOT / 'outputs/baseline-tuning-expanded-20261012'
ENGINE = None


def prepare(run, workers):
    if (run / 'manifest.json').exists():
        raise ValueError('Already prepared; resume with run')
    run.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((CALIBRATION / 'frozen-setting.json').read_text())
    assert audit.sha(CALIBRATION / 'calibration-summary.json') == frozen['calibration_summary_sha256']
    assert audit.sha(CALIBRATION / 'calibration-cases.json') == frozen['cases_sha256']
    trigger, target = map(float, frozen['configuration'].split('/'))
    parent = json.loads((STUDY / 'agent-evaluation/manifest.json').read_text())
    assert audit.frozen_identity(ROOT) == parent['code_identity'], 'Stress simulator identity changed'
    historical = json.loads((CALIBRATION / 'manifest.json').read_text())
    coverage = next(p for p in historical['original_inputs'] if p.endswith('/inputs/coverage.json'))
    assets = next(p for p in historical['original_inputs'] if p.endswith('/inputs/assets.json'))
    assert audit.sha(coverage) == parent['coverage_sha256']
    assert audit.sha(assets) == parent['assets_sha256']
    jobs = []
    proposals = Counter()
    for row in map(json.loads, LEDGER.open()):
        if row['status'] != 'common':
            continue
        key = f'{row["index"]:05d}-{row["variant"]}'
        path = STUDY / 'policy-baselines/cases' / key / 'recommendations.pkl'
        pack = pickle.load(path.open('rb'))
        assert str(pack['case_id']) == row['case_id']
        assert pack['variant'] == row['variant'] and pack['index'] == row['index']
        state = pack['state']
        funds = sum(max(0, a['wallet']) * a['price_usd'] for a in state['assets'].values())
        proposal = propose(state, 'static_hf', trigger=trigger, target=target, budget_usd=funds)
        rec = audit.recommendation(pack['recommendations']['static_hf'][0], proposal, state)
        pair = (rec, dict(policy='tuned_rule_bot', is_at_risk=proposal['action'] is not None))
        job = dict(case_id=key, economic_case_id=row['case_id'], variant=row['variant'],
                   pack=str(path), pack_sha256=audit.sha(path), proposal=proposal,
                   recommendation=pair, previous=row)
        jobs.append(job)
        proposals[f'{row["variant"]}:{proposal["action"] or "abstain"}'] += 1
    counts = Counter(j['variant'] for j in jobs)
    assert counts == {'core':1175, 'indexed':520}, counts
    roots = [STUDY/'agent-evaluation/cache/simulation_results', STUDY/'policy-baselines/simulations',
             CALIBRATION/'simulations', *map(Path, historical['cache_roots'])]
    inputs = {str(p):audit.sha(p) for p in [CALIBRATION/'frozen-setting.json', CALIBRATION/'calibration-summary.json',
              CALIBRATION/'calibration-cases.json', LEDGER, STUDY/'agent-evaluation/manifest.json',
              STUDY/'policy-baselines/manifest.json', Path(coverage), Path(assets)]}
    manifest = dict(source=str(ROOT), code_identity=parent['code_identity'], frozen_setting=frozen,
        configuration=frozen['configuration'], coverage=coverage, assets=assets,
        cache_roots=[str(p.resolve()) for p in roots], inputs=inputs,
        runner_sha256=audit.sha(__file__), policy_sha256=audit.sha(ROOT/'revision_eval/policies.py'),
        evaluator_adapter_sha256=audit.sha(ROOT/'revision_eval/run_baseline_tuning.py'),
        workers=workers, cases=dict(counts), proposals=dict(proposals),
        horizon_seconds=30*86400, funding='Unchanged saved stress checkpoint wallets',
        selection='Existing all-method matched stress case IDs; no new selection or tuning',
        created=time.time(), network_requests=0)
    common.save(run/'jobs.json', jobs)
    manifest['jobs_sha256'] = audit.sha(run/'jobs.json')
    common.save(run/'manifest.json', manifest)
    print(json.dumps(dict(cases=counts, proposals=proposals, configuration=frozen['configuration']),indent=2),flush=True)


class StressWorker(common.Worker):
    def __init__(self, run):
        self.run = Path(run)
        self.manifest = json.loads((self.run/'manifest.json').read_text())
        self.source = ROOT
        coverage, assets = self.manifest['coverage'], self.manifest['assets']
        summary = dict(code_identity=self.manifest['code_identity'], inputs={p:audit.sha(p) for p in [coverage, assets]})
        self.identity, self.attach, self.convert, self.adjust = pilot.frozen_helpers(ROOT, summary)
        for name in list(sys.modules):
            if name == 'analysis' or name.startswith('analysis.'):
                del sys.modules[name]
        workspace = self.run/'workers'/str(os.getpid());workspace.mkdir(parents=True,exist_ok=True)
        for name in ('data','profiles'):
            (workspace/name).symlink_to((ROOT/name).resolve(),target_is_directory=True)
        os.chdir(workspace)
        os.environ.update(AAVE_EVALUATION_CACHE=str(workspace/'cache'),
            AAVE_SIMULATION_CACHE=str(self.run/'simulations'), MPLCONFIGDIR=str(workspace/'matplotlib'),
            AAVE_RECOMMENDATIONS_FILE=str(STUDY/'generation-core/recommendations.pkl'))
        for name in ('AAVE_SURVIVAL_DATA','AAVE_CORE_TRANSACTIONS','AAVE_HISTORICAL_EVIDENCE',
                     'AAVE_INDEXED_COVERAGE','AAVE_INDEXED_ASSETS'):
            os.environ.pop(name,None)
        sys.path.insert(0,str(ROOT))
        import performSimulations
        import utils.simulations
        from utils.logger import set_log_file
        self.evaluator,self.simulations = performSimulations,utils.simulations
        self.profile = fixed_followup(self.simulations.get_limited_user_profile,30*86400)
        set_log_file('worker.log',False,str(workspace),file_level='WARNING')
        assert self.simulations.simulation_code_identity() == self.identity
        from simulator.utils import setup_protocol
        with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):setup_protocol()
        self.roots = [self.run/'simulations',*map(Path,self.manifest['cache_roots'])]
        self.status = dict(worker=os.getpid(),phase='idle',new_simulations=0,cache_hits=0,heartbeat=time.time())
        self.lock=threading.Lock();threading.Thread(target=self.heartbeat,daemon=True).start()

    def main_job(self, job):
        dest=self.run/'cases'/(job['case_id']+'.pkl')
        if dest.exists():return dict(kind='main',case_id=job['case_id'],reused=True)
        assert audit.sha(job['pack']) == job['pack_sha256'], 'Checkpoint pack changed'
        if job['variant']=='indexed':
            os.environ.update(AAVE_INDEXED_COVERAGE=self.manifest['coverage'],AAVE_INDEXED_ASSETS=self.manifest['assets'])
        else:
            os.environ.pop('AAVE_INDEXED_COVERAGE',None);os.environ.pop('AAVE_INDEXED_ASSETS',None)
        pair=tuple(job['recommendation']);profile,_,_,_,future,horizon=self.profile(pair[0],return_extras=True)
        control=self.cached(pair[0],'without',dict(profile=profile,lookahead_seconds=horizon))
        old=job['previous']['methods']['agent']
        assert common.outcome(control)==old['control'], 'Stress control flag differs'
        assert same_value(control_state(control),old['control_checkpoint']), 'Stress control state differs'
        value=self.evaluate(dict(profile=profile,horizon=horizon,future=future,control=control),pair)
        value.update(case_id=job['economic_case_id'],index=int(job['case_id'].split('-')[0]),
            variant=job['variant'],proposal=job['proposal'],proposed_capital_usd=job['proposal']['capital_usd'],
            configuration=self.manifest['configuration'])
        common.save(dest,value)
        return dict(kind='main',case_id=job['case_id'],reused=False)


def initialize(run):
    global ENGINE
    ENGINE=StressWorker(run)


def work(job):
    return ENGINE.work(('main',job))


def summarize(run, jobs, manifest):
    rows=[]
    for job in jobs:
        path=run/'cases'/(job['case_id']+'.pkl')
        if not path.exists():raise ValueError('Missing stress case: '+job['case_id'])
        result=pickle.load(path.open('rb'));row=copy.deepcopy(job['previous'])
        assert result['control']==row['methods']['agent']['control']
        row['methods']['tuned_rule_bot']=dict(control=result['control'],intervention=result['intervention'],
             proposed_capital_usd=result['proposed_capital_usd'],execution_failed=result['execution_failed'])
        rows.append(row)
    summary={}
    for variant in ('core','indexed'):
        selected=[r for r in rows if r['variant']==variant]
        summary[variant]={policy:common.count_outcomes([r['methods'][policy] for r in selected])
                          for policy in ('agent',*audit.POLICIES,'tuned_rule_bot')}
    common.save(run/'matched-results.json',dict(configuration=manifest['configuration'],by_variant=summary,
        code_identity=manifest['code_identity'],frozen_setting_sha256=manifest['inputs'][str(CALIBRATION/'frozen-setting.json')]))
    (run/'matched-cases.jsonl').write_text(''.join(json.dumps(r,default=str)+'\n' for r in rows))
    common.save(run/'complete.json',dict(completed=time.time(),cases=len(rows),all_matched=True,calibration_rerun=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','run','status'))
    p.add_argument('--run',type=Path,default=RUN);p.add_argument('--workers',type=int,default=12)
    args=p.parse_args();run=args.run.resolve()
    if args.workers<1:p.error('Workers must be positive')
    if args.mode=='prepare':return prepare(run,args.workers)
    if args.mode=='status':
        manifest=json.loads((run/'manifest.json').read_text());progress=json.loads((run/'progress.json').read_text()) if (run/'progress.json').exists() else {}
        print(json.dumps(dict(configuration=manifest['configuration'],cases=manifest['cases'],progress=progress,
            saved_cases=len(list((run/'cases').glob('*.pkl'))),complete=(run/'complete.json').exists()),indent=2));return
    lock=(run/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((run/'manifest.json').read_text())
    for path,digest in manifest['inputs'].items():assert audit.sha(path)==digest,'Frozen input changed: '+path
    assert audit.sha(__file__)==manifest['runner_sha256']
    assert audit.sha(ROOT/'revision_eval/policies.py')==manifest['policy_sha256']
    assert audit.sha(ROOT/'revision_eval/run_baseline_tuning.py')==manifest['evaluator_adapter_sha256']
    assert audit.frozen_identity(ROOT)==manifest['code_identity']
    assert audit.sha(run/'jobs.json')==manifest['jobs_sha256']
    for folder in ('cases','simulations','locks','simulation-timings','active','failures'):(run/folder).mkdir(exist_ok=True)
    jobs=json.loads((run/'jobs.json').read_text());pending=[j for j in jobs if not (run/'cases'/(j['case_id']+'.pkl')).exists()]
    started=time.time();history=[];failures=0
    with cf.ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(run),)) as pool:
        futures={pool.submit(work,j):j for j in pending}
        while futures:
            done,_=cf.wait(futures,timeout=5,return_when=cf.FIRST_COMPLETED)
            for future in done:
                job=futures.pop(future);receipt=future.result();history.append(receipt);failures+=not receipt['success']
                if receipt['success']:(run/'failures'/(common.job_id(job['case_id'])+'.json')).unlink(missing_ok=True)
                print(f"Stress {len(history)}/{len(pending)} {job['case_id']} success={receipt['success']} {receipt['seconds']:.1f}s",flush=True)
            elapsed=time.time()-started
            common.save(run/'progress.json',dict(phase='stress-testing',heartbeat=time.time(),workers=args.workers,
                total=len(jobs),completed=len(jobs)-len(pending)+sum(r['success'] for r in history),
                processed=len(history),remaining=len(futures),failures=failures,elapsed_seconds=elapsed,
                eta_seconds=elapsed*len(futures)/len(history) if len(history)>=12 else None))
    common.save(run/'task-receipts.json',history)
    if failures:raise SystemExit(f'{failures} failed cases; inspect failures and resume after resolving')
    summarize(run,jobs,manifest)
    print('All stress cases completed and matched.',flush=True)


if __name__=='__main__':main()
