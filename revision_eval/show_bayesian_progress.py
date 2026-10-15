"""Read-only single progress summary for adaptive calibration and both tests."""
import argparse,json,time
from pathlib import Path
from .run_bayesian_tuning import DEFAULT

def show(run):
    m=json.loads((run/'manifest.json').read_text());p=json.loads((run/'progress.json').read_text());obs=json.loads((run/'observations.json').read_text())
    print('Bayesian tuned bot | GP EI MCMC | %s | %s workers' % (p['phase'],m['workers']))
    print('Calibration: %s/6 adaptive settings finished; original 8 settings reused' % sum('round' in o for o in obs))
    preparation_active=False
    if p['phase']=='preparing_main_test':
        files=list((run/'main-engine'/'cases').glob('*.pkl'))
        latest=max((f.stat().st_mtime for f in files),default=0)
        preparation_active=time.time()-latest<45
        print('Preparing main proposals: %s no-action rows saved; action queue follows.' % len(files))
        if latest:print('Latest preparation output: %.0f seconds ago.' % (time.time()-latest))
    if p.get('round'):print('Current round: %s/6' % p['round'])
    if p.get('total') is not None:
        print('Current stage action tasks: %s/%s completed; %s queued/running; %s errors' % (p['completed'],p['total'],p['remaining'],p['failures']))
        eta=p.get('rate_eta_seconds')
        if eta is not None:print('Current stage rate-based remaining estimate: %.1f minutes (provisional)' % (eta/60))
    active=[]
    for stage in ['cal-engine','main-engine','stress-engine']:
        for file in (run/stage/'active').glob('*.json'):
            try:
                value=json.loads(file.read_text())
                if time.time()-value.get('heartbeat',0)<20 and value.get('phase')!='idle':active.append(value)
            except (ValueError,OSError):pass
    print('Active workers: %s | New simulation files timed: %s' % (len(active),len(list((run/'simulation-timings').glob('*.json')))))
    for stage,label in [('main','Main test'),('stress','Stress test')]:
        path=run/(stage+'-matched-results.json')
        pending='pending evaluation' if (run/'frozen-setting.json').exists() else 'awaiting frozen winner'
        if stage=='main' and p['phase']=='preparing_main_test':pending='preparing saved cases and action proposals'
        print(label+': '+('complete' if path.exists() else pending))
    if (run/'frozen-setting.json').exists():
        s=json.loads((run/'frozen-setting.json').read_text());print('Frozen winner: '+s['configuration'])
    if p['phase']=='failed':print('ERROR: '+p['error'])
    elif p['phase']!='complete' and not preparation_active and time.time()-p.get('heartbeat',0)>45:
        print('Coordinator heartbeat is stale; check runner.log or optimizer selection.')
    print('Independent verification: '+('complete' if (run/'verification.json').exists() else 'pending'))
    if (run/'complete.json').exists():
        for stage in ['main','stress']:
            data=json.loads((run/(stage+'-matched-results.json')).read_text())
            for variant,methods in data['by_variant'].items():
                bot=methods.get('tuned_rule_bot',methods.get('tuned_static_hf'))
                print('%s %s: %s/%s prevented; %s worsened' % (stage,variant,bot['prevented'],bot['control_liquidations'],bot['worsened']))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=DEFAULT);p.add_argument('--watch',type=float);a=p.parse_args()
    while True:
        show(a.run.resolve())
        if not a.watch or (a.run/'verification.json').exists():break
        print(flush=True);time.sleep(max(1,a.watch))
if __name__=='__main__':main()
