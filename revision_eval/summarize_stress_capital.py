"""Account overlap and gross proposed amounts for the matched stress cases."""
import argparse
import collections
import hashlib
import json
import math
import pickle
import statistics
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ledger',type=Path,required=True);p.add_argument('--recommendations',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with a.recommendations.open('rb') as source:recommendations=pickle.load(source)
    report={}
    with (a.output/'cases.jsonl').open('w') as target:
        for variant in ('core','indexed'):
            users=set();positive_users=set();rows=[]
            for line in a.ledger.open():
                row=json.loads(line)
                if row['variant']!=variant or row['status']!='common':continue
                rec=recommendations[row['case_id']][0]
                user=str(rec['user']).lower();amount=float(rec.get('amountUSD',0) or 0)
                if not math.isfinite(amount) or amount<0:raise ValueError('Invalid proposal amount: '+row['case_id'])
                users.add(user)
                if row['methods']['agent']['control']:positive_users.add(user)
                item=dict(case_id=row['case_id'],variant=variant,user=user,
                    agent_abstained=bool(rec.get('abstained')),agent_action=str(rec.get('Index Event')),
                    agent_proposed_capital_usd=amount,control_flag=row['methods']['agent']['control'])
                rows.append(item);target.write(json.dumps(item)+'\n')
            values=[r['agent_proposed_capital_usd'] for r in rows]
            ordered=sorted(values)
            report[variant]=dict(cases=len(rows),unique_accounts=len(users),control_positive_accounts=len(positive_users),
                agent_abstentions=sum(r['agent_abstained'] for r in rows),
                agent_proposed_capital_usd=sum(values),median_proposed_usd=statistics.median(values),
                p90_proposed_usd=ordered[int(.9*(len(ordered)-1))] if ordered else None,
                agent_actions=dict(collections.Counter(r['agent_action'] for r in rows)))
    (a.output/'summary.json').write_text(json.dumps(dict(variants=report,ledger_sha256=sha(a.ledger),
        recommendations_sha256=sha(a.recommendations),code_sha256=sha(Path(__file__)),
        interpretation='Saved agent proposal amounts, not executed amounts. Gross token funding may reflect historical wallet inference and does not constitute an equal-capital policy comparison.'),indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
