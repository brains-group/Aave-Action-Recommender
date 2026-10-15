"""Boundary regression: post-history anchors must not double-apply tied actions."""
import argparse
import contextlib
import copy
import io
import sys
from unittest.mock import patch
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--simulator-root',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.simulator_root.resolve()))
    from tools import run_single_simulation as runner
    from simulator.protocol import AaveV3Simulator
    def setup():
        s=AaveV3Simulator();s.current_timestamp=10;s.add_reserve('USDC',decimals=6,ltv=.8,liq_threshold=.85)
        s.set_price('USDC',1);s.reserves['USDC'].last_update_timestamp=10;s.reserves['USDC'].total_liquidity=1000
        return s
    profile=dict(user_address='fixture',initial_wallet=dict(USDC=100),transactions=[
        dict(action='Deposit',symbol='USDC',amount=10,timestamp=10),
        dict(action='Deposit',symbol='USDC',amount=5,timestamp=10)],
        indexed_checkpoint=dict(timestamp=10,phase='after_history',balances=dict(collateral=dict(USDC=40),debt={}),provenance=[]))
    with patch.object(runner,'setup_protocol',side_effect=setup),contextlib.redirect_stdout(io.StringIO()):
        a=runner.run_simulation(profile,checkpoint_only=True)
        intervention=copy.deepcopy(profile);intervention['counterfactual_start_timestamp']=20
        intervention['transactions'].append(dict(action='Deposit',symbol='USDC',amount=7,timestamp=20))
        b=runner.run_simulation(intervention,checkpoint_only=True)
        assert a['checkpoint_state']['total_collateral_usd']==40
        assert abs(b['checkpoint_state']['total_collateral_usd']-47)<1e-7
        assert a['checkpoint_state']['wallet_balances']['USDC']==85
        assert b['checkpoint_state']['wallet_balances']['USDC']==78
        intervention['counterfactual_start_timestamp']=10
        try:runner.run_simulation(intervention,checkpoint_only=True)
        except ValueError as error:assert 'precede intervention' in str(error)
        else:raise AssertionError('Accepted intervention-time anchor')
    print('Passed: timestamp ties, control/intervention balances, wallet preservation, and intervention guard')

if __name__=='__main__':main()
