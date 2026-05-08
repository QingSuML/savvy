from evaluation.oga_metrics import (
    HM3DVOSDataset,
    ScannetVOSDataset,
    VOSEvaluator,
    apply_dynamic_spatial_dilation,
    apply_macro_sever,
    apply_sparse_id_flickering,
    apply_temporal_dropout,
    compute_sequence_stq_vpq,
    evaluate_vos_consistency,
)

__all__ = [
    "HM3DVOSDataset",
    "ScannetVOSDataset",
    "VOSEvaluator",
    "apply_dynamic_spatial_dilation",
    "apply_macro_sever",
    "apply_sparse_id_flickering",
    "apply_temporal_dropout",
    "compute_sequence_stq_vpq",
    "evaluate_vos_consistency",
]


if __name__ == "__main__":
    dataset = ScannetVOSDataset(gt_dir="/path/to/scannet/gt", pred_dir="./eval/")
    evaluator = VOSEvaluator(dataset, max_pattern_size=3)
    evaluator.evaluate(scene_names="scene0019_00")
    evaluator.print_summary()
