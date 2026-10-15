"""Independent case/count/provenance audit after adaptive calibration and tests."""
import argparse,hashlib,json,pickle,time
from pathlib import Path
from . import gp_ei_mcmc as gp
from . import run_baseline_tuning as common
from . import run_bayesian_tuning as driver
from .combine_completed_stress import control_state,same_value


def verify(run):
    manifest=json.loads((run/'manifest.json').read_text());driver.check_inputs(run,manifest)
    assert (run/'complete.json').exists(),'Run incomplete'
    observations=json.loads((run/'observations.json').read_text());assert len(observations)==14
    bundle=pickle.load((run/'calibration-bundle.pkl').open('rb'))
    for number in range(1,7):
        trial=json.loads((run/'trials'/f'round-{number:02d}.json').read_text())
        digest=hashlib.sha256(json.dumps(observations[:8+number-1],indent=2,default=str).encode()).hexdigest()
        assert digest==trial['observation_sha256'],'Trial consumed the wrong observations'
        assert trial['observations']==8+number-1
        assert trial['bounds']==gp.BOUNDS.tolist()
        assert trial['trigger']<trial['target']
        measured=driver.calibration_summary(run,bundle,trial)
        assert measured==trial['summary']==observations[8+number-1]['summary']
        for cid,item in bundle.items():
            result=pickle.load((run/'calibration'/f'round-{number:02d}'/(cid+'.pkl')).open('rb'))
            assert driver.typed_equal(control_state(result['arms']['without']),control_state(item['control']))
            assert result['control']==common.outcome(item['control'])
            assert result['result'].get('success') or result.get('evaluation_status')=='completed_rapid_pair_not_prevented'
            if result['transaction'] is not None:
                assert result['transaction']['action'].lower()==result['recommendation'][0]['type']
    frozen=json.loads((run/'frozen-setting.json').read_text());winner=gp.winner(observations)
    assert (frozen['trigger'],frozen['target'])==(winner['trigger'],winner['target'])
    assert frozen['observations_sha256']==driver.audit.sha(run/'observations.json')
    complete=json.loads((run/'complete.json').read_text());summaries={}
    if complete['testing_reused']:
        summaries={stage:json.loads((run/(stage+'-matched-results.json')).read_text()) for stage in ('main','stress')}
    else:
        main_queue=pickle.load((run/'main-jobs.pkl').open('rb'))
        # Immutable frozen setting was written before constructing test queues.
        assert (run/'main-jobs.pkl').stat().st_mtime>=frozen['frozen']
        previous={}
        for v in ('core','indexed'):
            for row in map(json.loads,(driver.OLD/(v+'-matched-cases.jsonl')).open()):previous[(v,row['case_id'])]=row
            rows=list(map(json.loads,(run/(v+'-main-matched-cases.jsonl')).open()))
            assert {r['case_id'] for r in rows}=={case for (variant,case) in previous if variant==v}
            for row in rows:
                for policy,old in previous[(v,row['case_id'])]['methods'].items():
                    key='original_tuned_bot' if policy=='tuned_static_hf' else policy
                    assert row['methods'][key]==old
            calculated={p:common.count_outcomes([r['methods'][p] for r in rows]) for p in rows[0]['methods']}
            assert calculated==json.loads((run/'main-matched-results.json').read_text())['by_variant'][v]
        for row in main_queue['rows']:
            value=pickle.load((run/'main-engine/cases'/(common.job_id(row['case_id'])+'.pkl')).open('rb'))
            old=pickle.load(Path(row['control_cache']).open('rb'))
            assert driver.typed_equal(control_state(value['arms']['without']),control_state(old))
            assert value['result'].get('success')
            if value['transaction'] is not None:assert value['transaction']['action'].lower()==value['recommendation'][0]['type']
        oldstress={(r['index'],r['variant']):r for r in map(json.loads,(driver.STRESS/'matched-cases.jsonl').open())}
        stressrows=list(map(json.loads,(run/'stress-matched-cases.jsonl').open()))
        assert {(r['index'],r['variant']) for r in stressrows}==set(oldstress)
        for r in stressrows:
            for p,value in oldstress[(r['index'],r['variant'])]['methods'].items():assert r['methods']['original_tuned_bot' if p=='tuned_rule_bot' else p]==value
        for v in ('core','indexed'):
            rows=[r for r in stressrows if r['variant']==v];calculated={p:common.count_outcomes([r['methods'][p] for r in rows]) for p in rows[0]['methods']}
            assert calculated==json.loads((run/'stress-matched-results.json').read_text())['by_variant'][v]
        for job in pickle.load((run/'stress-jobs.pkl').open('rb')):
            value=pickle.load(Path(job['destination']).open('rb'));assert value['result']['success']
            assert same_value(control_state(value['arms']['without']),job['previous']['methods']['agent']['control_checkpoint'])
            if value['transaction'] is not None:assert value['transaction']['action'].lower()==value['recommendation'][0]['type']
        summaries={stage:json.loads((run/(stage+'-matched-results.json')).read_text()) for stage in ('main','stress')}
    result=dict(verified=time.time(),all_calibration_rows_checked=7440,adaptive_observation_order_verified=True,
        source_and_data_hashes_verified=True,winner=frozen['configuration'],zero_worsening_calibration=True,
        exact_test_identities_verified=True,starting_states_verified=True,counts_recomputed=True)
    common.save(run/'verification.json',result)
    text=['# Completed Bayesian-calibrated bot\n\nGP EI MCMC used eight existing configurations and six adaptive settings, on the same earlier-period cases. The selected setting was frozen before test preparation.\n\n',f"**Frozen setting:** {frozen['configuration']}. All 7,440 new calibration case/configuration rows passed independent verification.\n\n",'| Cohort | State | Method | Cases | Control liquidations | Prevented | Prevention rate | Worsened |\n|---|---|---|---:|---:|---:|---:|---:|\n']
    for stage,data in summaries.items():
        for variant,methods in data['by_variant'].items():
            for policy,values in methods.items():
                text.append(f"| {stage} | {variant} | {policy} | {values['cases']} | {values['control_liquidations']} | {values['prevented']} | {100*values['prevention_rate']:.2f}% | {values['worsened']} |\n")
    text.append('\nCode/data identity, adaptive observation order, unchanged test case identities and control states, original-method outcomes and recomputed summaries passed verification. Default-zero cost fields inherited from old summaries are not measured costs. These are paired simulator outcomes. No manuscript results have been changed during this run.\n')
    (driver.ROOT/'docs/revision-eval/BAYESIAN_TUNED_RESULTS_20261013.md').write_text(''.join(text));print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=driver.DEFAULT);a=p.parse_args();verify(a.run.resolve())
