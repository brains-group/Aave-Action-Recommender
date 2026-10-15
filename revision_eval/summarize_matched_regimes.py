"""Summarize market-volatility slices on the final all-method case intersection."""

import json
from collections import defaultdict
from pathlib import Path
import sys

from .audit_existing_regimes import trailing_volatility

ROOT = Path('outputs/revision-audits-20260929')
POLICIES = ('static_hf', 'repay_only', 'deposit_only', 'automation_style')


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Aave-Simulator'))
    from simulator.utils import get_price_history

    prices = get_price_history()['WETH']
    times = sorted(prices)
    output = ROOT / 'matched-regimes-v1'
    output.mkdir(exist_ok=False)
    results = {}
    with (output / 'cases.jsonl').open('w') as ledger:
        for variant in ('core', 'indexed'):
            full = {r['index']: r for r in rows(ROOT / f'existing-merged-{variant}-v1/cases.jsonl')}
            reviewed = list(rows(ROOT / f'existing-reviewed-{variant}-v2/cases.jsonl'))
            original = json.loads((Path('outputs/revision-audits-20260928') / f'policy-{variant}-regimes/summary.json').read_text())
            low, high = original['volatility_tertiles']
            by_regime = defaultdict(lambda: dict(cases=0, control_flags=0, agent_rescues=0,
                                                 policy_rescues={name: 0 for name in POLICIES}))
            for r in reviewed:
                if r['status'] != 'common':
                    continue
                meta = full[r['index']]
                assert meta['case_id'] == r['case_id']
                vol = trailing_volatility(prices, times, int(meta['checkpoint']))
                regime = ('missing' if vol is None else 'low' if vol <= low else
                          'middle' if vol <= high else 'high')
                group = by_regime[regime]
                group['cases'] += 1
                group['control_flags'] += bool(r['parent']['control'])
                group['agent_rescues'] += bool(r['parent']['control'] and not r['parent']['intervention'])
                for name in POLICIES:
                    method = r['methods'][name]
                    group['policy_rescues'][name] += bool(method['control'] and not method['intervention'])
                ledger.write(json.dumps(dict(index=r['index'], case_id=r['case_id'],
                                             variant=variant, regime=regime, volatility=vol)) + '\n')
            source = json.loads((ROOT / f'existing-reviewed-{variant}-v2/summary.json').read_text())['common']
            assert sum(g['cases'] for g in by_regime.values()) == source['cases']
            assert sum(g['control_flags'] for g in by_regime.values()) == source['control_flags']
            assert sum(g['agent_rescues'] for g in by_regime.values()) == source['methods']['agent']['rescued']
            for name in POLICIES:
                assert sum(g['policy_rescues'][name] for g in by_regime.values()) == source['methods'][name]['rescued']
            results[variant] = dict(volatility_cutoffs=list((low, high)), groups=dict(by_regime))
    (output / 'summary.json').write_text(json.dumps(dict(variants=results,
        definition='Trailing seven-day WETH daily-log-return volatility; tertile cutoffs frozen from 365 pre-selection days. All policies share the final successful case intersection within each state variant.'), indent=2) + '\n')
    print(output / 'summary.json')


if __name__ == '__main__':
    main()
