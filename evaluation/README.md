# Evaluation

Benchmark-specific entry points and metric code for Savvy.

- `savvy_scannet_runner.py`: Savvy inference and evaluation on ScanNet.
- `savvy_hm3d_runner.py`: Savvy inference and evaluation on HM3D.
- `savvy_vipseg_runner.py`: Savvy inference on VIPSeg.
- `savvy_vipseg_evaluation.py`: VIPSeg metric evaluation for saved predictions.
- `oga_metrics/`: shared Open-world Granularity-Agnostic (OGA) metrics and dataset loaders.

Use the root-level shell scripts (`run_scannet_eval.sh`, `run_hm3d_eval.sh`,
`run_vipseg_eval.sh`) as the main launch commands.
