from .datasets import HM3DVOSDataset, ScannetVOSDataset
from .evaluator import VOSEvaluator
from .metrics import (
    compute_sequence_stq_vpq,
    evaluate_vos_consistency,
)

_STRESS_EXPORTS = {
    "apply_dynamic_spatial_dilation",
    "apply_macro_sever",
    "apply_sparse_id_flickering",
    "apply_temporal_dropout",
}

def __getattr__(name):
    if name in _STRESS_EXPORTS:
        from . import stress
        value = getattr(stress, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
