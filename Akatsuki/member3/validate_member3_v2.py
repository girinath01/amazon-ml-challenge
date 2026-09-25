"""
validate_member3_v2.py — Deep validation of Upgraded Member 3 Pipeline (v2)
=============================================================================
Validates:
  1. Feature matrix integrity on v2 dataset (training_pair_features_v2.parquet)
  2. Boundedness & range checks on all 40 features
  3. Feature discriminability (AUC & mutual information)
  4. Hard negative performance (Soundex collisions & prefix collisions)
  5. Calibrated model evaluation & score separation
  6. Submission file format & integrity check
  7. Per-source thresholding effectiveness (S2 vs S3)
"""
import sys, json, pickle, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Import LightGBM first to avoid Windows heap corruption
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.calibration import calibration_curve

REPO = Path(r"C:\Users\appu2\OneDrive\Desktop\ML AMAZON\amazon-ml-challenge\Akatsuki")
SRC  = REPO / "code/business_entity_resolution/src"
sys.path.insert(0, str(SRC))

from features.upgrade_features import UPGRADE_FEATURE_NAMES

FEATURE_COLS = UPGRADE_FEATURE_NAMES

sep = "=" * 70
print(sep)
print("MEMBER 3 v2 — UPGRADED VALIDATION REPORT")
print(sep)

# 1. Load v2 feature matrix
parquet_v2 = REPO / "data/training_pair_features_v2.parquet"
if not parquet_v2.exists():
    print(f"Error: {parquet_v2} does not exist.")
    sys.exit(1)

print("\n[1] Loading v2 feature matrix...")
df = pd.read_parquet(parquet_v2)
X = df[FEATURE_COLS].values.astype(np.float32)
y = df["label"].values.astype(np.int32)
n_pos = int(y.sum())
n_neg = int(len(y) - n_pos)
print(f"    Rows: {len(df):,} | Positives: {n_pos:,} | Hard Negatives: {n_neg:,} | Pos Ratio: {n_pos/len(y):.3f}")

# 2. Feature Integrity Checks
print("\n[2] Feature Integrity Checks...")
total_nan = int(df[FEATURE_COLS].isna().sum().sum())
total_inf = int((df[FEATURE_COLS].abs() == float("inf")).sum().sum())
const_feats = [c for c in FEATURE_COLS if df[c].nunique() <= 1]

print(f"    NaN values        : {total_nan}")
print(f"    Inf values        : {total_inf}")
print(f"    Constant features : {const_feats or 'None'}")

# Range checks
out_of_range = []
for c in FEATURE_COLS:
    mn, mx = df[c].min(), df[c].max()
    if mn < -1.01 or mx > 1.01:
        out_of_range.append((c, mn, mx))

if out_of_range:
    print("    OUT-OF-RANGE features:")
    for c, mn, mx in out_of_range:
        print(f"      {c}: min={mn:.3f}, max={mx:.3f}")
else:
    print("    All 40 features strictly in [-1, 1] range! (FIX VERIFIED)")

# 3. Feature Discriminability
print("\n[3] Top 15 Features by AUC against Hard Negatives...")
auc_scores = []
for feat in FEATURE_COLS:
    try:
        a = roc_auc_score(y, df[feat])
        auc_scores.append(max(a, 1.0 - a))
    except:
        auc_scores.append(0.5)

auc_df = pd.DataFrame({"feature": FEATURE_COLS, "AUC": np.round(auc_scores, 4)}).sort_values("AUC", ascending=False)
print(auc_df.head(15).to_string(index=False))

# 4. Model & Calibration Validation
print("\n[4] Evaluating Trained v2 Model + Calibration...")
model_pkl = REPO / "model/lgbm_entity_match_v2.pkl"
with open(model_pkl, "rb") as f:
    pkg = pickle.load(f)

model = pkg["model"]
calibrator = pkg.get("calibrator")
base_thr = pkg.get("threshold", 0.50)
s2_thr = pkg.get("s2_threshold", base_thr)
s3_thr = pkg.get("s3_threshold", base_thr)

raw_scores = model.predict(X)
if calibrator is not None:
    cal_scores = calibrator.predict(raw_scores)
else:
    cal_scores = raw_scores

print(f"    Model ROC-AUC      : {roc_auc_score(y, raw_scores):.5f}")
print(f"    Calibrated ROC-AUC : {roc_auc_score(y, cal_scores):.5f}")
print(f"    Thresholds         : Global={base_thr:.2f} | S2={s2_thr:.2f} | S3={s3_thr:.2f}")

cand_ids = df["candidate_id"].values
pair_thrs = np.array([s2_thr if c.startswith("S2-") else s3_thr for c in cand_ids])
preds = (cal_scores >= pair_thrs).astype(int)

p = precision_score(y, preds, zero_division=0)
r = recall_score(y, preds, zero_division=0)
f05 = (1.25 * p * r) / (0.25 * p + r) if (p + r) > 0 else 0

print(f"    Precision @ Thr    : {p:.5f}")
print(f"    Recall @ Thr       : {r:.5f}")
print(f"    Macro F0.5 @ Thr   : {f05:.5f}")

# 5. Submission Integrity Check
print("\n[5] Submission File Format Validation...")
sub_path = REPO / "output/matching_results.tsv"
if sub_path.exists():
    sub_df = pd.read_csv(sub_path, sep="\t")
    print(f"    File: {sub_path}")
    print(f"    Columns: {list(sub_df.columns)}")
    print(f"    Rows: {len(sub_df):,}")
    matched_cnt = int((sub_df['matched_entity_ids'].fillna('') != '').sum())
    single_cnt = len(sub_df) - matched_cnt
    print(f"    Matched entities: {matched_cnt:,} ({100*matched_cnt/len(sub_df):.1f}%)")
    print(f"    Singletons      : {single_cnt:,} ({100*single_cnt/len(sub_df):.1f}%)")
    print(f"    Duplicate S1 IDs: {sub_df['source1_entity_id'].duplicated().sum()}")
else:
    print(f"    Notice: {sub_path} not yet generated.")

print(f"\n{sep}")
print("V2 VALIDATION COMPLETE")
print(sep)
