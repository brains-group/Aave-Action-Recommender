# Shared Aave data pipeline

The real `Aave-Data-Pipeline` Git submodule owns collection, normalization and survival formatting.
The original simulator remains a separate submodule. No datasets or caches moved.

```bash
git -c protocol.file.allow=always submodule update --init --recursive
python -m pip install --no-deps --no-build-isolation -e ./Aave-Data-Pipeline
python collect_aave.py --help
python -m aave_data_pipeline.survival --help
python -m unittest discover -s Aave-Data-Pipeline/tests -v
```

The pipeline submodule currently has a local absolute source URL. Initialization on another
machine requires the sibling source repository at that path, or an authorized remote URL:
`git submodule set-url Aave-Data-Pipeline <authorized-url>` followed by
`git submodule sync`. No new remote was invented or published.

Live collection defaults to zero requests. Only after independently verifying remaining free
capacity and disabled paid overage, use `--free-access-confirmed --max-requests N`.
The budget includes retries and metadata; stopping preserves cached pages. See the shared
README for date semantics, cache audits and survival commands. Never run two writers on a cache.

Select a new journal dataset with `AAVE_SURVIVAL_DATA=/absolute/run/directory`.
Its manifest identifies a separate model/data cache namespace. Without that variable,
existing data and model paths remain the default. The formatter preserves identifiers;
model preprocessing excludes observation_id, outcome_id and split.

Simulation fixes write under `cache/simulation_results/revision_20260908/<input-hash>/`;
old analysis scripts that scan only the directory top level will not discover these new
runs automatically. Use explicit cohort/config manifests with the new checkpoint evaluation
interface before interpreting a new policy result. Old paper results remain untouched.

## Shared output and supplementary downloads

Both consumers now export to `/home/spadef/data/craft-soc/data` and collect into its
hidden `.cache` directory. The legacy craft-soc collection path is a compatibility
symlink; model caches remain separate. New supplementary CSVs and coverage manifests
live under the output folder's `supplementary/aave_v3/polygon/<deployment>/`.

`python collect_aave.py --market polygon --include-supplementary` opts into supported
extra events/history. Add the existing free-access confirmation and bounded request
budget only after verifying the account's remaining free allowance.
Use `--supplementary-only` to avoid refreshing core events/account snapshots.
Unavailable sources are explicit in coverage.json. Schema support does not establish
complete chain history. See Aave-Data-Pipeline/README.md for the complete commands.
