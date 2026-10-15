"""Verify unresolved prior-snapshot gaps against the shared read-only cache."""
import argparse
import hashlib
import json
import pickle
import sqlite3
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('report','recommendations','coverage','profiles','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    report=json.loads(a.report.read_text());coverage=json.loads(a.coverage.read_text())
    with a.recommendations.open('rb') as f:items=list(pickle.load(f).items())
    database=Path(coverage['entities']['positionSnapshots']['cache']).resolve()
    results=[]
    with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as db:
        for entry in report['unresolved']:
            if entry['reason']!='No strictly prior snapshot coverage for checkpoint':continue
            rec=items[entry['index']][1][0];user=rec['user'];cutoff=rec['timestamp']-600
            paths=[a.profiles/folder/'profiles'/f'user_{user}.json' for folder in ('non_liquidated_profiles','liquidated_profiles')]+[a.profiles/f'user_{user}.json']
            source=next(path for path in paths if path.is_file());profile=json.loads(source.read_text())
            prior=[t for t in profile['transactions'] if t['timestamp']<=cutoff];timestamp=max(t['timestamp'] for t in prior)
            times=[int(json.loads(payload)['timestamp']) for (payload,) in db.execute('SELECT payload FROM rows WHERE entity=? AND id>=? AND id<?',('positionSnapshots',user.lower()+'-',user.lower()+'.'))]
            results.append(dict(index=entry['index'],user=user,checkpoint=timestamp,decision_cutoff=cutoff,
                profile_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),prior_actions=len(prior),
                prior_action_types=sorted({t['action'] for t in prior}),snapshots=len(times),
                strictly_prior=sum(t<timestamp for t in times),at_checkpoint=sum(t==timestamp for t in times),
                first_snapshot=min(times) if times else None,
                resolution='Cannot use same-time/post-event snapshot as pre-event state; requires earlier state or verified complete origin history'))
    (a.output/'summary.json').write_text(json.dumps(dict(cases=results,snapshot_cache=str(database),
        coverage_sha256=hashlib.sha256(a.coverage.read_bytes()).hexdigest(),
        report_sha256=hashlib.sha256(a.report.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
