"""Verify real outer simulator labels against an HF-boundary fixture in staged source."""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulator-root',type=Path,required=True)
    a=parser.parse_args();sys.path.insert(0,str(a.simulator_root.resolve()))
    from tools import run_single_simulation as runner
    from tools.liquidation_detection_strategies import check_liquidation_at_timestamp
    from simulator.protocol import AaveV3Simulator
    def setup():
        s=AaveV3Simulator();s.current_timestamp=10
        s.add_reserve('USDC',decimals=6,ltv=.8,liq_threshold=.85)
        s.set_price('USDC',1);s.reserves['USDC'].last_update_timestamp=10
        s.reserves['USDC'].total_liquidity=1000
        return s
    def strategies(**kwargs):
        flagged,reason,elapsed,details=check_liquidation_at_timestamp(
            kwargs['sim'],kwargs['user_address'],kwargs['base_timestamp'],kwargs['base_timestamp'],
            policy=kwargs['liquidation_policy'],margin_threshold=kwargs['margin_threshold'])
        return dict(consensus=dict(liquidated=flagged,agreement_rate=1.,strategies_detecting_liquidation=int(flagged),strategies_not_detecting=int(not flagged),average_liquidation_time=elapsed),
            results_by_strategy=dict(fixture=dict(liquidated=flagged,liquidation_reason=reason,time_to_liquidation=elapsed)),best_strategy='fixture',total_checks=1)
    profile=dict(user_address='fixture',initial_wallet=dict(USDC=100),transactions=[
        dict(action='Deposit',symbol='USDC',amount=100,timestamp=10),
        dict(action='Borrow',symbol='USDC',amount=80,timestamp=10)])
    results=[]
    for policy,margin,expected in [('protocol',1.1,False),('static_warning',1.05,False),('static_warning',1.1,True),('dynamic_warning',1.1,True)]:
        with patch.object(runner,'setup_protocol',side_effect=setup),patch.object(runner,'run_all_strategies',side_effect=strategies),contextlib.redirect_stdout(io.StringIO()):
            result=runner.run_simulation(profile,liquidation_policy=policy,margin_threshold=margin,lookahead_seconds=1)
        stats=result['liquidation_stats']
        assert stats['policy']==policy
        assert stats['liquidated']==expected
        assert stats['evaluation_target']==('liquidation_eligibility' if policy=='protocol' else 'risk_warning')
        assert stats['margin_threshold_used']==(1.0 if policy=='protocol' else margin)
        assert stats['use_enhanced_hf'] is False
        assert not result['execution_failures']
        results.append(dict(policy=policy,margin=margin,flagged=stats['liquidated'],target=stats['evaluation_target']))
    print(json.dumps(dict(passed=len(results),cases=results),indent=2))

if __name__=='__main__':main()
