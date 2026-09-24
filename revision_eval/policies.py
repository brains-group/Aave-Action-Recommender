"""Single-checkpoint policy proposals. Execution requires the common simulator.

No price conversion, wallet top-up, future event access, or protocol feasibility
claim. Input balances are token units; prices USD/token; LT a fraction.
"""
import math
POLICIES = ('control', 'static_hf', 'repay_only', 'deposit_only', 'automation_style')

def propose(state, policy, *, trigger=1.10, target=1.20, budget_usd=1000., gas_usd=None):
    if policy not in POLICIES: raise ValueError('Unknown policy')
    if not (math.isfinite(trigger) and math.isfinite(target) and 1 <= trigger < target):
        raise ValueError('Require 1 <= trigger < target')
    if not math.isfinite(budget_usd) or budget_usd < 0: raise ValueError('Invalid capital budget')
    if gas_usd is not None and (not math.isfinite(gas_usd) or gas_usd < 0): raise ValueError('Invalid gas assumption')
    required = ('case_id', 'timestamp', 'state_timestamp', 'total_debt_usd', 'weighted_collateral_usd', 'assets')
    if any(k not in state for k in required): raise ValueError('Missing checkpoint inputs')
    if state['state_timestamp'] > state['timestamp']: raise ValueError('Future state cannot fund a policy')
    debt, weighted = float(state['total_debt_usd']), float(state['weighted_collateral_usd'])
    if not all(math.isfinite(v) and v >= -1e-8 for v in [debt, weighted]): raise ValueError('Invalid balances')
    # Simulator arithmetic can leave sub-cent negative dust after full repayment.
    # A material negative balance remains an error; no funding is created here.
    debt, weighted = max(0., debt), max(0., weighted)
    out = dict(case_id=state['case_id'], timestamp=state['timestamp'], policy=policy,
               action=None, capital_usd=0., action_count=0, execution_status='not_executed',
               gas_usd_assumption=gas_usd, budget_usd=budget_usd, trigger=trigger, target=target)
    if policy == 'control' or debt == 0 or weighted/debt >= trigger:
        return dict(out, reason='control_or_trigger_not_met')
    candidates = []
    for asset, a in sorted(state['assets'].items()):
        price, wallet, owed, lt = [float(a[k]) for k in ['price_usd','wallet','debt','liquidation_threshold']]
        if not all(math.isfinite(v) for v in [price,wallet,owed,lt]) or price <= 0 or min(wallet*price,owed*price)<-1e-8 or lt<0 or lt>1:
            raise ValueError('Invalid asset inputs')
        wallet, owed = max(0.,wallet), max(0.,owed)
        for action in ['Repay','Deposit']:
            if policy=='repay_only' and action!='Repay' or policy=='deposit_only' and action!='Deposit':continue
            if action=='Deposit' and (lt==0 or not a.get('collateral_enabled',False)):continue
            needed = max(0.,debt-weighted/target) if action=='Repay' else max(0.,(target*debt-weighted)/lt)
            available = min(wallet*price,budget_usd,owed*price if action=='Repay' else budget_usd)
            amount_usd = min(needed,available)
            if amount_usd<=0:continue
            hf = weighted/(debt-amount_usd) if action=='Repay' and debt>amount_usd else (math.inf if action=='Repay' else (weighted+lt*amount_usd)/debt)
            candidates.append(dict(action=action,asset=asset,amount=amount_usd/price,capital_usd=amount_usd,
                                   estimated_post_hf=hf,target_reached=hf>=target-1e-12))
    if not candidates:return dict(out,reason='no_same_asset_funding')
    # Static HF chooses least capital reaching target, else greatest HF improvement.
    # Automation-style is a transparent repay-first heuristic, not a product replica.
    if policy=='automation_style':candidates=[c for c in candidates if c['action']=='Repay'] or candidates
    reached=[c for c in candidates if c['target_reached']]
    c=min(reached,key=lambda c:(c['capital_usd'],c['asset'],c['action'])) if reached else max(candidates,key=lambda c:c['estimated_post_hf'])
    return dict(out,**{k:v for k,v in c.items() if k!='capital_usd'},capital_usd=c['capital_usd'],
                action_count=1,reason='proposal_requires_simulator_validation')
