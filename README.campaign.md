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
