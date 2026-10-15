"""Freeze evaluator source and stage output-label correction without changing a live run."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def prepare(root, output):
    output.mkdir(parents=True, exist_ok=False)
    sources = list(root.glob('*.py'))
    for folder in ('utils', 'revision_eval', 'Aave-Simulator'):
        sources.extend(p for p in (root/folder).rglob('*.py')
                       if not any(x in p.parts for x in ('__pycache__', '.venv', 'data', 'cache')))
    hashes = {}
    for source in sources:
        rel = source.relative_to(root)
        target = output/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(rel)] = hashlib.sha256(source.read_bytes()).hexdigest()
    for rel in ('data', 'profiles', 'config.json', 'Aave-Simulator/data',
                'Aave-Simulator/config.json', 'Aave-Simulator/analysis/polygon_asset_registry.json'):
        source = root/rel
        if source.exists():
            target = output/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source.resolve())
    target = output/'Aave-Simulator/tools/run_single_simulation.py'
    before = target.read_text()
    old = '''"evaluation_target": "liquidation_eligibility",
            "policy": "protocol",
            "margin_threshold_used": 1.0,
            "use_enhanced_hf": False,'''
    new = '''"evaluation_target": "liquidation_eligibility" if liquidation_policy == "protocol" else "risk_warning",
            "policy": liquidation_policy,
            "margin_threshold_used": 1.0 if liquidation_policy == "protocol" else margin_threshold,
            "threshold_mode": "dynamic" if liquidation_policy in ("dynamic_warning", "legacy_warning") else "static",
            "use_enhanced_hf": bool(use_enhanced_hf) if liquidation_policy == "legacy_warning" else False,'''
    if before.count(old) != 1:
        raise ValueError('Expected metadata block changed; inspect before applying')
    target.write_text(before.replace(old, new))
    import difflib
    patch = ''.join(difflib.unified_diff(before.splitlines(True), target.read_text().splitlines(True),
                                        fromfile='a/tools/run_single_simulation.py',
                                        tofile='b/tools/run_single_simulation.py'))
    (output/'warning-metadata.patch').write_text(patch)
    (output/'source-manifest.json').write_text(json.dumps(dict(
        source_root=str(root), source_sha256=hashes,
        patched_source_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        changes='Output labels only; active source and historical results untouched',
        data='Symlinks to existing read-only inputs; no independent chain evidence'), indent=2))


def stage_post_history(output):
    """Opt-in end-of-history anchor; never use this to predict its own event."""
    target=output/'utils/indexed_checkpoint.py'
    before=target.read_text()
    s=before.replace("rows=[r for r in rows if int(r['timestamp'])<timestamp]", "after_history=os.environ.get('AAVE_INDEXED_ANCHOR_PHASE')=='after_history'\n    rows=[r for r in rows if int(r['timestamp'])<timestamp or (after_history and int(r['timestamp'])==timestamp)]")
    s=s.replace('history.before(user,2**63-1,timestamp)', 'history.before(user,2**63-1,timestamp+int(after_history))')
    s=s.replace("'timestamp':timestamp,'balances'", "'timestamp':timestamp,'phase':'after_history' if after_history else 'before_timestamp','balances'")
    s += "\n    if after_history:\n        for row in provenance:row['age_seconds']-=1\n        profile['indexed_checkpoint']['source']='Indexed end-of-history snapshot; applied after all historical timestamp ties, before any later intervention; not a pre-event reference'\n"
    if s==before:raise ValueError('Adapter block changed')
    target.write_text(s)
    runner=output/'Aave-Simulator/tools/run_single_simulation.py'
    text=runner.read_text()
    start=text.index('    for transaction in transactions:\n        if anchor and not anchored')
    stop=text.index('        if not execute_transaction_silent',start)
    block=text[start:stop]
    body=block[block.index('            user ='):block.index('            anchored = True')]
    body=''.join(line[4:]+'\n' for line in body.splitlines())
    replacement="""    def apply_checkpoint_anchor():
        nonlocal anchored
"""+body+"""        anchored = True
    after_history = bool(anchor and anchor.get('phase') == 'after_history')
    if after_history and profile.get('counterfactual_start_timestamp') is not None:
        if anchor['timestamp'] >= profile['counterfactual_start_timestamp']:
            raise ValueError('Recorded anchor must precede intervention')
    for transaction in transactions:
        if anchor and not anchored and (transaction['timestamp'] > anchor['timestamp'] if after_history else transaction['timestamp'] >= anchor['timestamp']):
            apply_checkpoint_anchor()
"""
    text=text[:start]+replacement+text[stop:]
    needle="    if anchor and not anchored:\n        raise ValueError('Indexed checkpoint was not reached')"
    text=text.replace(needle,"    if after_history and not anchored and transactions and transactions[-1]['timestamp'] == anchor['timestamp']:\n        apply_checkpoint_anchor()\n"+needle)
    runner.write_text(text)
    import difflib
    parts=[]
    # Preserve both patches with source hashes for review/application after live runs.
    original_runner=(Path(__file__).resolve().parents[1]/'Aave-Simulator/tools/run_single_simulation.py').read_text()
    for name,old,new in [('utils/indexed_checkpoint.py',before,s),('Aave-Simulator/tools/run_single_simulation.py',original_runner,text)]:
        parts.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    (output/'post-history.patch').write_text(''.join(parts))
    manifest=json.loads((output/'source-manifest.json').read_text())
    manifest['staged_post_history']=dict(opt_in='AAVE_INDEXED_ANCHOR_PHASE=after_history',
        adapter_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest())
    (output/'source-manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--post-history',action='store_true')
    a = p.parse_args()
    prepare(Path(__file__).resolve().parents[1], a.output.resolve())
    if a.post_history:stage_post_history(a.output.resolve())
