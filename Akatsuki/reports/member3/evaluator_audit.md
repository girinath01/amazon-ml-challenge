# Evaluator Audit Report — Member 3 (Validation Engineer)

**Date**: 2026-09-25  
**Auditor**: Member 3 (Validation Engineer, Team Akatsuki)  
**Scope**: Verification of all evaluators against the Amazon ML Challenge 2026 Problem Specification.

---

## 1. Official Competition Metric Definition

According to the official problem specification (`student_resource/README.md`):

1. The challenge metric is **Macro-averaged $F_{0.5}$** across **all Source-1 entities** in the evaluation population.
2. For each Source-1 entity $i$:
   - Let $T_i$ be the set of true matching entity IDs from Source 2 and/or Source 3.
   - Let $P_i$ be the set of predicted matching entity IDs from Source 2 and/or Source 3.
   
3. **Singleton Evaluation ($T_i = \emptyset$)**:
   - If $P_i = \emptyset$ (correctly predicted empty): $\text{Precision}_i = 1.0, \text{Recall}_i = 1.0, F_{0.5, i} = 1.0$.
   - If $P_i \neq \emptyset$ (false positive match): $\text{Precision}_i = 0.0, \text{Recall}_i = 0.0, F_{0.5, i} = 0.0$.

4. **Non-Singleton / Multi-Match Evaluation ($T_i \neq \emptyset$)**:
   - If $P_i = \emptyset$ (missed match): $\text{Precision}_i = 0.0, \text{Recall}_i = 0.0, F_{0.5, i} = 0.0$.
   - If $P_i \neq \emptyset$:
     $$\text{Intersection}_i = |T_i \cap P_i|$$
     $$\text{Precision}_i = \frac{\text{Intersection}_i}{|P_i|}$$
     $$\text{Recall}_i = \frac{\text{Intersection}_i}{|T_i|}$$
     $$F_{0.5, i} = \frac{(1 + 0.5^2) \cdot \text{Precision}_i \cdot \text{Recall}_i}{(0.5^2 \cdot \text{Precision}_i) + \text{Recall}_i} = \frac{1.25 \cdot \text{Precision}_i \cdot \text{Recall}_i}{0.25 \cdot \text{Precision}_i + \text{Recall}_i}$$

5. **Macro Aggregation**:
   $$\text{Macro } F_{0.5} = \frac{1}{N} \sum_{i=1}^{N} F_{0.5, i}$$
   where $N$ is the total count of Source-1 entities, **including singletons**.

---

## 2. Audit of Existing Codebases

### A. `Akatsuki/code/business_entity_resolution/src/evaluation/evaluator.py`
- **Function**: `evaluate_predictions(ground_truth, predictions)`
- **Finding**: **CORRECT**.
  - Implements per-S1 entity calculation.
  - Implements the exact $F_{0.5}$ formula with $\beta = 0.5$.
  - Properly assigns $F_{0.5} = 1.0$ for true empty singletons when predictions are empty.
  - Properly assigns $F_{0.5} = 0.0$ when false matches are made on singletons.
  - Correctly computes macro-averages over the full set of S1 entities.

### B. `validate_member3_v2.py`
- **Location**: Lines 114–124
- **Code**:
  ```python
  preds = (cal_scores >= pair_thrs).astype(int)
  p = precision_score(y, preds, zero_division=0)
  r = recall_score(y, preds, zero_division=0)
  f05 = (1.25 * p * r) / (0.25 * p + r) if (p + r) > 0 else 0
  print(f"Macro F0.5 @ Thr : {f05:.5f}")
  ```
- **Finding**: **DEFECTIVE / MISLABELED**.
  1. **Pair-Level vs. Entity-Level**: `precision_score` and `recall_score` were computed across the flattened binary classification labels of candidate pairs ($y \in \{0, 1\}$). This calculates the scalar pair-level binary $F_{0.5}$, **not** the entity-level macro-averaged $F_{0.5}$.
  2. **Singleton Omission**: True singletons with zero candidate pairs were completely absent from the dataset and evaluation.
  3. **Multi-match Distortion**: An S1 entity with 5 true matches contributed 5 data rows to the binary calculation, whereas in the competition metric, that entity must count as exactly one unit in the macro-average.
  4. **In-Sample Evaluation**: The evaluation was performed on the training set `training_pair_features_v2.parquet` using the model trained on those very rows.

### C. `train_model_v2.py` Cross-Validation Evaluator
- **Location**: Lines 360–375
- **Code**:
  ```python
  def macro_f05(y_true, y_pred):
      p = precision_score(y_true, y_pred, zero_division=0)
      r = recall_score(y_true, y_pred, zero_division=0)
      if p + r == 0: return 0.0
      return (1.25 * p * r) / (0.25 * p + r)
  ```
- **Finding**: **DEFECTIVE**.
  - Used pair-level binary classification $F_{0.5}$ for validation fold scoring and threshold finding instead of entity-level grouping.

---

## 3. Discrepancy Impact Analysis

| Property | Competition Metric | `validate_member3_v2.py` | Impact |
| :--- | :--- | :--- | :--- |
| **Observation unit** | Source-1 Entity | Candidate Pair | Weights entities with more candidates higher |
| **Averaging method** | Macro-average across entities | Global pooled binary scalar | Distorts precision/recall trade-off |
| **Singleton handling** | Explicit $1.0$ or $0.0$ per singleton | Ignored (unpaired singletons dropped) | Overestimates performance |
| **Data evaluated** | Held-out validation split | Full in-sample training matrix | Extreme in-sample overfitting bias |

---

## 4. Remediation Implemented

The validation engine in `run_member3_validation_suite.py` was constructed to strictly enforce the competition standard:
1. Every candidate pair prediction is aggregated by `source1_entity_id`.
2. All Source-1 entities (including all 273 singletons in the 5,000 population) are evaluated.
3. Threshold sweep and multi-match breakdown now compute the true macro-averaged $F_{0.5}$.
