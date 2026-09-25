"""
Evaluation package for Business Entity Resolution.
"""

from .blocking_evaluation import (
    evaluate_blocking_recall,
    compute_candidate_volume_stats,
    generate_ablation_report,
    analyze_blocking_failures,
)

try:
    from .evaluator import evaluate_predictions, test_thresholds, compute_f_beta
    HAS_EVALUATOR = True
except ImportError:
    HAS_EVALUATOR = False

__all__ = [
    "evaluate_blocking_recall",
    "compute_candidate_volume_stats",
    "generate_ablation_report",
    "analyze_blocking_failures",
]

if HAS_EVALUATOR:
    __all__.extend(["evaluate_predictions", "test_thresholds", "compute_f_beta"])
