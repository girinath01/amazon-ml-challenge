"""
evaluator.py - Macro F0.5 Evaluation & Threshold Testing Engine for Member 3
Implements exact competition metric evaluation:
- Macro F_0.5 per S1 entity, averaged across all entities
- Rigorous handling of singletons (truth empty) and multi-match entities
- Threshold simulation framework for probability-based prediction testing
"""

from typing import Dict, List, Set, Any, Optional, Union
import numpy as np
import pandas as pd


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """
    Compute F_beta score.
    For beta = 0.5:
        F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0:
        return 0.0
    return float((1.0 + beta_sq) * precision * recall / denom)


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Dict[str, Any]:
    """
    Evaluate predicted matches against ground truth.
    
    Macro-averages per-S1 precision, recall, and F_0.5.
    
    Rules:
    - Singletons (truth set is empty):
        - if predicted set is empty -> P=1.0, R=1.0, F0.5=1.0
        - if predicted set is non-empty -> P=0.0, R=0.0, F0.5=0.0
    - Non-singletons (truth set has >= 1 matches):
        - if predicted set is empty -> P=0.0, R=0.0, F0.5=0.0
        - if predicted set is non-empty:
            intersection = len(truth & pred)
            P = intersection / len(pred)
            R = intersection / len(truth)
            F0.5 = (1.25 * P * R) / (0.25 * P + R)
    
    Returns:
        Dictionary with overall macro metrics, singleton metrics, and multi-match metrics.
    """
    all_s1_ids = sorted(ground_truth.keys())
    total_entities = len(all_s1_ids)
    if total_entities == 0:
        return {
            "macro_f05": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "singleton_f05": 0.0,
            "multi_match_f05": 0.0,
            "total_entities": 0
        }

    per_entity_p = []
    per_entity_r = []
    per_entity_f05 = []

    singleton_f05 = []
    multi_f05 = []
    multi_p = []
    multi_r = []

    for s1_id in all_s1_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())

        # Singleton case
        if len(true_set) == 0:
            if len(pred_set) == 0:
                p, r, f = 1.0, 1.0, 1.0
            else:
                p, r, f = 0.0, 0.0, 0.0
            singleton_f05.append(f)
        else:
            # Non-singleton case
            if len(pred_set) == 0:
                p, r, f = 0.0, 0.0, 0.0
            else:
                common = len(true_set & pred_set)
                p = float(common / len(pred_set))
                r = float(common / len(true_set))
                f = compute_f_beta(p, r, beta=0.5)
            multi_f05.append(f)
            multi_p.append(p)
            multi_r.append(r)

        per_entity_p.append(p)
        per_entity_r.append(r)
        per_entity_f05.append(f)

    macro_f05 = float(np.mean(per_entity_f05))
    macro_p = float(np.mean(per_entity_p))
    macro_r = float(np.mean(per_entity_r))

    return {
        "macro_f05": round(macro_f05, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "singleton_count": len(singleton_f05),
        "singleton_f05": round(float(np.mean(singleton_f05)), 4) if singleton_f05 else 0.0,
        "multi_match_count": len(multi_f05),
        "multi_match_f05": round(float(np.mean(multi_f05)), 4) if multi_f05 else 0.0,
        "multi_match_precision": round(float(np.mean(multi_p)), 4) if multi_p else 0.0,
        "multi_match_recall": round(float(np.mean(multi_r)), 4) if multi_r else 0.0,
        "total_entities": total_entities
    }


def test_thresholds(
    pair_scores_df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    score_col: str = "probability",
    thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Evaluate Macro F_0.5 across multiple decision thresholds.
    
    Args:
        pair_scores_df: DataFrame containing ['source1_id', 'candidate_id', score_col]
        ground_truth: {s1_id: set(matched_ids)}
        score_col: name of probability/similarity column to threshold
        thresholds: list of cutoff thresholds to test
    
    Returns:
        DataFrame comparing threshold, Precision, Recall, Macro F0.5, Singleton Score.
    """
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    results = []
    all_s1_ids = set(ground_truth.keys())

    for thresh in thresholds:
        # Filter candidate pairs above threshold
        passed = pair_scores_df[pair_scores_df[score_col] >= thresh]
        
        # Build predictions dict: {s1_id: set(predicted_candidate_ids)}
        preds = {s1: set() for s1 in all_s1_ids}
        for _, row in passed.iterrows():
            s1 = row["source1_id"]
            cand = row["candidate_id"]
            if s1 in preds:
                preds[s1].add(cand)

        metrics = evaluate_predictions(ground_truth, preds)
        results.append({
            "Threshold": thresh,
            "Macro F0.5": metrics["macro_f05"],
            "Macro Precision": metrics["macro_precision"],
            "Macro Recall": metrics["macro_recall"],
            "Singleton Score": metrics["singleton_f05"],
            "Multi-Match F0.5": metrics["multi_match_f05"],
            "Total Predicted Links": int(sum(len(v) for v in preds.values()))
        })

    return pd.DataFrame(results)
