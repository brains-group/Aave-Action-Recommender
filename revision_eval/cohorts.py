"""Deterministic sampling from an explicitly supplied eligible checkpoint frame."""
import hashlib

def representative(rows, n, seed='42', excluded_accounts=()):
    if n<0:raise ValueError('n must be nonnegative')
    excluded=set(excluded_accounts);seen=set();eligible=[]
    for r in rows:
        if r['case_id'] in seen:raise ValueError('Duplicate case ID')
        seen.add(r['case_id'])
        if r['account'] in excluded:continue
        # Eligibility/split must be defined upstream without future outcome labels.
        if r['split']!='test' or not r['eligible']:continue
        eligible.append(r)
    key=lambda r:(hashlib.sha256((seed+':'+r['case_id']).encode()).hexdigest(),r['case_id'])
    return sorted(eligible,key=key)[:n]
