"""Small read-only audit probes; no edits/import caches in the simulator checkout."""
import ast,sys,json,typing,hashlib,subprocess
from pathlib import Path
root=Path('/home/spadef/Aave-Action-Recommender/Aave-Simulator');sys.path.insert(0,str(root))
from simulator.protocol import AaveV3Simulator
out=Path(__file__).parent

def extract(path,names,env):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Assign)) and (getattr(n,'name','') in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets))]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)

def sim():
 s=AaveV3Simulator();s.add_reserve('USD',decimals=6,ltv=.8,liq_threshold=.85);s.set_price('USD',1);s.current_timestamp=100
 r=s.reserves['USD'];r.last_update_timestamp=100;r.total_liquidity=10000
 return s
s=sim();u=s.get_user('borrower');u['collateral']['USD']=100;u['debt']['USD']=85/.98;s.reserves['USD'].total_variable_debt=u['debt']['USD'];s.faucet('liquidator','USD',1000)
before=s.get_user_account_data('borrower');s.liquidate('borrower','USD','USD',80,'liquidator',100)
results={'close_factor':{'hf':before['health_factor'],'debt_before':before['total_debt_usd'],'v30_to_v32_max_debt':before['total_debt_usd']*.5,'simulator_repaid':s.liquidation_events[-1]['debt_repaid']}}
env=dict(vars(typing),AaveV3Simulator=AaveV3Simulator)
extract(root/'tools/validate_simulator.py',{'DUST_LIQUIDATION_THRESHOLDS','predict_dust_liquidation','predict_liquidation_by_threshold'},env)
results['zero_debt_dust']=env['predict_dust_liquidation']({'total_debt_usd':0,'total_collateral_usd':100,'health_factor':float('inf')})
s=sim();s.reserves['USD'].liq_threshold=.6;u=s.get_user('healthy');u['collateral']['USD']=100;u['debt']['USD']=10
results['healthy_low_threshold']={'hf':s.get_health_factor('healthy'),'warning':env['predict_liquidation_by_threshold'](s.get_user_account_data('healthy'),s,'healthy',100)}
extract(root/'tools/run_single_simulation.py',{'execute_transaction_silent'},env)
results['unknown_action_success']=env['execute_transaction_silent'](s,'healthy',{'action':'Flashloan','timestamp':100})
# A stale snapshot with non-unit index is copied forward unchanged by the adapter.
from analysis.indexed_history import PositionHistory
asset={'m':{'id':'token','symbol':'USD','decimals':6}}
row={'id':'snapshot','account':{'id':'a'},'position':{'id':'a-m-BORROWER-0'},'balance':'100000000','index':'1100000000000000000000000000','blockNumber':1,'logIndex':0,'timestamp':100}
bal,provenance=PositionHistory([row],asset).before('a',2,100+86400*30)
results['stale_snapshot']={'balance_after_30_days':str(bal['debt']['USD']),'age_seconds':provenance[0]['age_seconds'],'note':'No intervening index/interest update; this is stale last-observed balance, not exact current debt.'}
# Historical HF helper currently selects the next price, not the latest prior price.
s=sim();u=s.get_user('a');u['collateral']['USD']=100;u['debt']['USD']=10
s.set_price_history({'USD':{100:1.,200:2.,300:3.}})
results['future_price_lookup']={'query_timestamp':150,'expected_asof_price':1.,'returned':s.get_health_factor_at_timestamp('a',150)['prices_used']}
# Querying a future HF should not mutate reserve totals.
s=sim();u=s.get_user('a');u['collateral']['USD']=100;u['debt']['USD']=10
r=s.reserves['USD'];r.total_variable_debt=1000
before={'supply':r.total_liquidity,'debt':r.total_variable_debt,'index':r.variable_borrow_index}
s.get_health_factor_at_timestamp('a',100+86400*30)
results['hf_query_mutation']={'before':before,'after':{'supply':r.total_liquidity,'debt':r.total_variable_debt,'index':r.variable_borrow_index}}
env['check_liquidation_at_timestamp']=lambda sim,user,t,base,**kwargs:(True,'fixture',t-base,{})
extract(root/'tools/liquidation_detection_strategies.py',{'strategy_3_binary_search'},env)
results['negative_liquidation_time']=env['strategy_3_binary_search'](None,'a',10000,7200)['time_to_liquidation']
(out/'probe-results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
files=[f for d in ['simulator','tools','analysis','market'] for f in (root/d).rglob('*.py')]
manifest={'revision':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),'files':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
(out/'simulator-source-manifest.json').write_text(json.dumps(manifest,indent=2))
