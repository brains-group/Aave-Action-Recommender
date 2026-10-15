"""Audit recovered existing-case results by source vintage and control state."""
import argparse
import collections
import hashlib
import json
import math
import pickle
from pathlib import Path

POLICIES=('static_hf','repay_only','deposit_only','automation_style')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1048576),b''):h.update(block)
    return h.hexdigest()

def same(a,b):
    if isinstance(a,(int,float)) and isinstance(b,(int,float)) and not isinstance(a,bool) and not isinstance(b,bool):
        return math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-8)
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    return a==b

def summary(rows):
    common=list(rows)
    result=dict(cases=len(common),control_flags=sum(r['parent']['control'] for r in common),methods={})
    for method in ('agent','control',*POLICIES):
        count=collections.Counter()
        for row in common:
            x=row['parent'] if method in ('agent','control') else row['methods'][method]
            if method=='control':x=dict(control=x['control'],intervention=x['control'])
            count['rescued']+=x['control'] and not x['intervention']
            count['worsened']+=not x['control'] and x['intervention']
            if method not in ('agent','control'):
                count['attempted']+=x['attempted'];count['proposed_capital_usd']+=x['proposed_capital_usd']
        result['methods'][method]=dict(count)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ledger',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    ledger_manifest=json.loads((a.ledger/'manifest.json').read_text());parent_manifest=json.loads((a.parent/'manifest.json').read_text())
    if ledger_manifest['parent_manifest_sha256']!=sha(a.parent/'manifest.json'):raise ValueError('Parent manifest changed')
    parent_code=parent_manifest['code_identity']
    code_by_dir={};policy_by_dir={}
    for run in ledger_manifest['additional_runs']:
        meta=json.loads((Path(run['path'])/'manifest.json').read_text())
        code_by_dir[str(Path(run['path']).resolve())]=meta['code_identity']
        policy_by_dir[str(Path(run['path']).resolve())]=meta.get('policies_sha256')
    original_meta=json.loads((Path(ledger_manifest['original'])/'manifest.json').read_text())
    original_code=original_meta['code_identity'];original_policy=original_meta['policies_sha256']
    recovery=Path(ledger_manifest['recovery'])
    for name in ('baseline-errors-fixed','baseline-errors','baseline-errors-indexed'):
        folder=recovery/name
        if (folder/'manifest.json').exists():
            meta=json.loads((folder/'manifest.json').read_text())
            code_by_dir[str(folder.resolve())]=meta['code_identity']
            policy_by_dir[str(folder.resolve())]=meta.get('policies_sha256')
    groups=collections.defaultdict(list)
    with (a.output/'cases.jsonl').open('w') as target:
        for line in (a.ledger/'cases.jsonl').open():
            original=json.loads(line);case=original['case_id'];row=dict(index=original['index'],case_id=case,variant=original['variant'],status='excluded',reason=None)
            if not original['parent_hf_processed']:row['reason']='parent_not_hf_processed'
            elif original['parent'] is None:row['reason']='parent_not_successful_or_paired'
            elif any(k not in original['policies'] for k in POLICIES):row['reason']='missing_policy'
            else:
                parent=json.loads((a.parent/'cases'/(case+'.json')).read_text())
                ref=parent['arms']['without']['checkpoint_state'];row['parent']=original['parent'];row['methods']={}
                versions=[];policy_versions=[]
                for method in POLICIES:
                    path=Path(original['sources'][method]);record=pickle.load(path.open('rb'))
                    if not record.get('result',{}).get('success') or set(record.get('arms',{}))!={'without','with'}:
                        row['reason']='policy_not_successful:'+method;break
                    if str(record['case_id'])!=str(original['index']) or record['index']!=original['index'] or record['variant']!=original['variant']:
                        row['reason']='policy_case_id_mismatch:'+method;break
                    policy=original['policies'][method]
                    if policy['control']!=row['parent']['control']:
                        row['reason']='control_flag_mismatch:'+method;break
                    checkpoint=record['arms']['without']['checkpoint_state']
                    for field in ('timestamp','wallet_balances','debt_balances'):
                        if field not in ref or field not in checkpoint or not same(ref[field],checkpoint[field]):
                            row['reason']='control_checkpoint_mismatch:'+method+':'+field;break
                    if row['reason']:break
                    source_code=original_code;policy_sha=original_policy
                    for folder,identity in code_by_dir.items():
                        if str(path).startswith(folder+'/'):
                            source_code=identity;policy_sha=policy_by_dir[folder];break
                    versions.append(source_code);policy_versions.append(policy_sha)
                    row['methods'][method]=policy
                if row['reason'] is None:
                    if any(code!=parent_code for code in versions):
                        row['status']='separate_source';row['reason']='simulator_code_vintage_mismatch'
                    else:row['status']='common'
                    row['source_codes']=sorted(set(versions));row['policy_versions']=sorted(set(policy_versions))
            groups[row['status']].append(row);target.write(json.dumps(row)+'\n')
    primary=summary(groups['common']);other=summary(groups['separate_source'])
    vintage_counts=dict(collections.Counter('+'.join(r.get('policy_versions',[])) for r in groups['common']))
    report=dict(variant=ledger_manifest['variant'],selected=sum(len(rs) for rs in groups.values()),
        parent_code_identity=parent_code,common=primary,separate_source=other,
        common_policy_version_combinations=vintage_counts,
        exclusions=dict(collections.Counter(r['reason'] for r in groups['excluded'])),
        parent_manifest_sha256=sha(a.parent/'manifest.json'),ledger_manifest_sha256=sha(a.ledger/'manifest.json'),
        source_ledger_sha256=sha(a.ledger/'cases.jsonl'),code_sha256=sha(Path(__file__)),
        interpretation='Primary includes only successful matched cases with identical simulator source identity and matching timestamp/wallet/debt in the no-action arm. Policy implementation hashes may differ and are stratified. Different simulator-code recovered policies remain a separate diagnostic; exclusions are explicit. Full historical state equality is not established beyond these saved fields.')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(variant=report['variant'],common=primary,separate_source=other,exclusions=report['exclusions']),indent=2))

if __name__=='__main__':main()
