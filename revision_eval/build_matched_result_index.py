"""Combine completed cohort comparisons without mixing their case denominators."""

import hashlib
import json
from pathlib import Path


ROOT = Path('outputs/revision-audits-20260929')
METHODS = ('agent', 'static_hf', 'repay_only', 'deposit_only', 'automation_style')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_cases(path, variant):
    with path.open() as source:
        return [row for line in source if (row := json.loads(line))['status'] == 'common'
                and row['variant'] == variant]


def assemble(study, variant, folder):
    path = ROOT / folder
    source = json.loads((path / 'summary.json').read_text())
    group = source['variants'][variant] if study == 'shortest_time_stress' else source['common']
    rows = load_cases(path / 'cases.jsonl', variant)
    assert len(rows) == group['cases' if study == 'existing_cohort' else 'common']
    assert len({row['case_id'] for row in rows}) == len(rows)
    methods = {}
    controls = group['control_flags'] if study == 'existing_cohort' else group['methods']['agent']['control_flags']
    assert sum((row['methods']['agent'] if study == 'shortest_time_stress' else row['parent'])['control']
               for row in rows) == controls
    for method in METHODS:
        records = [row['methods'][method] if study == 'shortest_time_stress' and method != 'agent'
                   else row['methods']['agent'] if study == 'shortest_time_stress'
                   else row['parent'] if method == 'agent' else row['methods'][method]
                   for row in rows]
        rescued = sum(record['control'] and not record['intervention'] for record in records)
        worsened = sum(not record['control'] and record['intervention'] for record in records)
        saved = group['methods'][method]
        assert rescued == saved['rescued'] and worsened == saved['worsened']
        methods[method] = dict(rescued=rescued, worsened=worsened,
                               rescued_fraction_of_control_flags=rescued / controls if controls else None)
    ids = '\n'.join(sorted(row['case_id'] for row in rows)).encode()
    return dict(study=study, variant=variant, common_cases=len(rows), control_flags=controls,
                selected=group.get('selected', source.get('selected')),
                matched_case_ids_sha256=hashlib.sha256(ids).hexdigest(), methods=methods,
                source_summary=str(path / 'summary.json'), source_summary_sha256=sha256(path / 'summary.json'),
                source_cases=str(path / 'cases.jsonl'), source_cases_sha256=sha256(path / 'cases.jsonl'),
                exclusions=group.get('exclusion_reasons', source.get('exclusions', {})))


def main():
    entries = [assemble('existing_cohort', variant, f'existing-reviewed-{variant}-v2')
               for variant in ('core', 'indexed')]
    entries.extend(assemble('shortest_time_stress', variant, 'stress-common-v1')
                   for variant in ('core', 'indexed'))
    result = dict(entries=entries,
                  interpretation='Each row is one matched case intersection across the agent and four baselines. Cohorts and simulator variants remain separate. Rescued means a control liquidation-eligibility flag is absent after intervention within the saved follow-up; it is not observed liquidation prevention. Funding and action thresholds are not equal across methods.')
    output = ROOT / 'matched-results.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
