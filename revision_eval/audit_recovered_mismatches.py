"""Explain checkpoint guard failures without recomputing a projected control."""
import argparse
import contextlib
import json
import os
import pickle
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','run','recovery','recommendations','coverage','assets','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for name in ('source','run','recovery','recommendations','coverage','assets','output'):setattr(a,name,getattr(a,name).resolve())
    a.output.mkdir(parents=True,exist_ok=True);os.chdir(a.source)
    os.environ.update(AAVE_EVALUATION_CACHE=str(a.output/'cache'),AAVE_SIMULATION_CACHE=str(a.output/'simulations'),AAVE_INDEXED_COVERAGE=str(a.coverage),AAVE_INDEXED_ASSETS=str(a.assets))
    sys.path.insert(0,str(a.source))
    import utils.simulations as u
    from tools import run_single_simulation as runner
    with a.recommendations.open('rb') as f:recs=list(pickle.load(f).items())
    class Stop(BaseException):pass
    results=[]
    for file in sorted((a.run/'cases').glob('*/error.json')):
        if json.loads(file.read_text())['error']!='Checkpoint wallet mismatch':continue
        index=int(file.parent.name.split('-')[0]);dest=a.output/(file.parent.name+'.json')
        if dest.exists():results.append(json.loads(dest.read_text()));continue
        captured={};rec=recs[index][1][0]
        profile,*_=u.get_limited_user_profile(rec,return_extras=True)
        def stop(**kw):
            sim=kw['sim'];user=sim.get_user(kw['user_address'])
            captured.update(wallet=dict(user['wallet']),debt={asset:amount*sim.reserves[asset].variable_borrow_index for asset,amount in user['debt'].items()})
            raise Stop()
        original=runner.run_all_strategies;runner.run_all_strategies=stop
        try:
            with (a.output/'replay.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                try:runner.run_simulation(profile=profile,lookahead_seconds=1)
                except Stop:pass
        finally:runner.run_all_strategies=original
        source=a.recovery/f'control-indexed-{index//3000}'/'cases'/file.parent.name/'control.pkl'
        with source.open('rb') as f:control=pickle.load(f)['control']
        expected=control['checkpoint_state']['wallet_balances']
        differences={asset:dict(cached=expected.get(asset,0),replayed=captured['wallet'].get(asset,0),difference=captured['wallet'].get(asset,0)-expected.get(asset,0)) for asset in set(expected)|set(captured['wallet']) if abs(captured['wallet'].get(asset,0)-expected.get(asset,0))>1e-8*max(1,abs(expected.get(asset,0)))}
        row=dict(index=index,differences=differences,cached_debt=control['checkpoint_state']['debt_balances'],replayed_debt=captured['debt'],source=str(source),code_identity=u.simulation_code_identity())
        dest.write_text(json.dumps(row,indent=2));results.append(row);print(index,differences,flush=True)
    (a.output/'summary.json').write_text(json.dumps(dict(cases=results,scope='Failures present at audit start; historical checkpoint only, guard retained'),indent=2))

if __name__=='__main__':main()
