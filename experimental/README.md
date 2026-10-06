# Experimental crates

Code from completed GPU experiments, kept buildable but not part of the production engine.

- `pi-snow-gpu-alternatives`: gpu-alternatives-v1 (arms A1+A2, A5 compact kernel, A6, A8 and 3.2 forcing probes).
- `pi-snow-gpu-alternatives-v2`: gpu-alternatives-v2 (3.2b vector forcing textures and the CA-MB driver).

Both moved here from `rust/crates/` on 2026-10-06, after the experiments were merged. Scripts and records under
`experiments/gpu-alternatives-v1/` and `experiments/gpu-alternatives-v2/` were hash-bound at run time and still
name the old `rust/crates/...` and `pipeline/pipeline/gpu_alternatives_v2` paths. To reproduce a recorded run,
check out the commit named in that experiment's results. Binary names did not change.
