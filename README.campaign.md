# GLM-5.3 optimization campaign H0 tooling

This branch is the benchmark/provenance anchor only. It makes no runtime model,
MLX, or optimization change.

Run Python through `scripts/run_campaign_clean.sh`; the wrapper builds a neutral
environment with the exact runtime worktree on `PYTHONPATH`, user site disabled,
and the content-pinned stock Python 3.12 environment. `campaign/provenance.py`
asserts runtime/import/native-binary/checkpoint identities before load.
`campaign/instrumented_prefill.py` refuses any installed mlx-vlm `ar.py` except
SHA-256 `7076aedecab82c42cdaf520f6e1e02466ccf0f22c99af82d9ce4a414f4bd9fab`.
Its timer starts at the first prompt model submission and stops after one final
logit/cache synchronization and before sampling. Public `generate_step` remains
the TTFT and synchronized-yield decode path.

The reviewed H0 successor adds content-addressed environment/model provenance,
an advisory full-model lock, explicit MLX/RSS/pressure gates, a sequential
paired-order orchestrator, and paired integration-state validation/restoration.
Smoke runs require an explicit repetition count and are never acceptance-eligible.

```text
python -m campaign.provenance capture-model|capture-environment|verify ...
python -m campaign.h0_benchmark --run-class smoke --repetitions 1 ...
python -m campaign.paired_runner --spec paired.json ...
python -m campaign.locking audit|recover ...
python -m campaign.state validate|stage|restore|select ...
python -m campaign.contracts manifests/candidates/chunk-sweep.json
```
