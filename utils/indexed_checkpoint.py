"""Strictly historical position evidence for the last pre-recommendation action.

Only positions are anchored. Wallet funding and later counterfactual state are preserved.
"""
import json,os,sqlite3
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=1)
def settings():
    path=Path(os.environ['AAVE_INDEXED_COVERAGE'])
    coverage=json.loads(path.read_text());entry=coverage['entities']['positionSnapshots']
    if coverage['market']!='polygon' or entry['status']!='indexed_interval_complete':raise ValueError('Incomplete Polygon snapshot coverage')
    assets=json.loads(Path(os.environ['AAVE_INDEXED_ASSETS']).read_text())
    from analysis.asset_registry import polygon_identity_mappings
    for market, verified in polygon_identity_mappings().items():
        if market not in assets:
            assets[market] = verified
        elif assets[market]['id'].lower() != verified['id'].lower():
            raise ValueError('Configured market mapping conflicts with deployment evidence: '+market)
    return coverage,entry,assets

def attach_checkpoint(profile):
    if not os.environ.get('AAVE_INDEXED_COVERAGE'):return
    from analysis.indexed_history import PositionHistory, position_parts
    txs=profile['transactions']
    if not txs:raise ValueError('No historical checkpoint available')
    timestamp=max(int(t['timestamp']) for t in txs);user=profile['user_address'].lower()
    coverage,entry,assets=settings()
    with sqlite3.connect(Path(entry['cache']).as_uri()+'?mode=ro',uri=True) as db:
        rows=[json.loads(r[0]) for r in db.execute('SELECT payload FROM rows WHERE entity=? AND id>=? AND id<?',('positionSnapshots',user+'-',user+'.'))]
    rows=[r for r in rows if int(r['timestamp'])<timestamp]
    if not rows:raise ValueError('No strictly prior snapshot coverage for checkpoint')
    used_markets={position_parts(row['position']['id'],user)[0] for row in rows}
    # Only markets with a historical position for this user belong to the snapshot.
    # Including every market that shares a display symbol caused false ambiguity
    # even when this account used only one underlying. PositionHistory still
    # rejects true multi-underlying collisions within the selected markets.
    missing=used_markets-set(assets)
    if missing:raise ValueError('Missing indexed asset mapping: '+', '.join(sorted(missing)))
    relevant_assets={market:assets[market] for market in used_markets}
    evidence = None
    if os.environ.get('AAVE_HISTORICAL_EVIDENCE'):
        from analysis.historical_evidence import HistoricalEvidence
        evidence = HistoricalEvidence(json.loads(Path(os.environ['AAVE_HISTORICAL_EVIDENCE']).read_text()))
    history=PositionHistory(rows,relevant_assets, index_evidence=evidence)
    balances,provenance=history.before(user,2**63-1,timestamp)
    profile['indexed_checkpoint']={'timestamp':timestamp,'balances':{side:{symbol:float(amount) for symbol,amount in amounts.items()} for side,amounts in balances.items()},'provenance':provenance,'source':'Polygon indexed positionSnapshots; historical only; ambiguous underlying symbols rejected; stale balances not accrued between indexed observations'}
