"""
train_model.py  — Standalone training script (run as subprocess from run_member3_complete.py)
=============================================================================================
Imports ONLY what is needed for training. No pyarrow, no parquet, no pandas imported first.
This avoids the Windows Store Python + LightGBM C-heap corruption bug.

Usage:
    python train_model.py [--no-optuna] [--optuna-trials N]
"""
import sys
import time
import json
import pickle
import logging
import argparse
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train_model")

parser = argparse.ArgumentParser()
parser.add_argument("--no-optuna",     action="store_true")
parser.add_argument("--optuna-trials", type=int, default=40)
args = parser.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────
_THIS      = Path(__file__).resolve().parent
REPO_ROOT  = _THIS / "amazon-ml-challenge" / "Akatsuki"
SRC_DIR    = REPO_ROOT / "code" / "business_entity_resolution" / "src"
DATA_DIR   = REPO_ROOT / "data"
MODEL_DIR  = REPO_ROOT / "model"
REPORT_DIR = REPO_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_PARQUET = DATA_DIR / "training_pair_features.parquet"
MODEL_PKL       = MODEL_DIR / "lgbm_entity_match.pkl"
MODEL_TXT       = MODEL_DIR / "lgbm_entity_match.txt"
META_JSON       = MODEL_DIR / "training_metadata.json"

sys.path.insert(0, str(SRC_DIR))

# Import LightGBM FIRST before anything else (avoids C-heap corruption on Windows)
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score

# NOW import pandas/pyarrow (after LightGBM is already loaded)
import pandas as pd

FEATURE_COLS = [
    "name_exact", "name_jaccard", "name_levenshtein",
    "name_jaro_winkler", "name_token_sort", "name_token_set",
    "name_char3_cosine", "name_char4_cosine",
    "name_core_similarity", "name_translit_similarity",
    "address_jaccard", "address_levenshtein", "address_char3_cosine",
    "address_digit_overlap", "house_number_match",
    "postal_code_match", "address_exact", "address_missing",
    "country_match", "name_length_ratio", "address_length_ratio",
    "name_token_count_difference", "address_token_count_difference",
    "numeric_conflict",
    "name_address_similarity_product", "name_high_address_low",
    "name_high_address_high", "name_address_conflict",
]


def macro_f05(y_true, y_pred):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    if p + r == 0: return 0.0
    return (1 + 0.25) * p * r / (0.25 * p + r)


def find_optimal_threshold(scores, labels, thresholds=None):
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.30, 0.85, 0.02)]
    best_t, best_f = 0.5, 0.0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f = macro_f05(labels, preds)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


def lgbm_f05_eval(preds, eval_data):
    labels  = eval_data.get_label()
    binary  = (preds >= 0.50).astype(int)
    score   = macro_f05(labels, binary)
    return "f05_macro", score, True


# ── 1. Load data ───────────────────────────────────────────────────────────
log.info(f"Loading parquet: {FEATURE_PARQUET}")
df = pd.read_parquet(FEATURE_PARQUET)
X  = df[FEATURE_COLS].values.astype(np.float32)
y  = df["label"].values.astype(np.int32)
n_pos = int(y.sum()); n_neg = int(len(y) - n_pos)
log.info(f"  Rows={len(df):,}  Pos={n_pos:,}  Neg={n_neg:,}")

BASE_PARAMS = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "boosting_type":    "gbdt",
    "num_leaves":       63,
    "learning_rate":    0.05,
    "feature_fraction": 0.80,
    "bagging_fraction": 0.80,
    "bagging_freq":     5,
    "min_child_samples":20,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "scale_pos_weight": round(n_neg / n_pos, 3) if n_pos > 0 else 1.0,
    "n_jobs":           -1,
    "seed":             42,
    "verbose":          -1,
}

best_params = dict(BASE_PARAMS)

# ── 2. Optuna HP search ────────────────────────────────────────────────────
if not args.no_optuna:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        log.info(f"Optuna search ({args.optuna_trials} trials)...")
        skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

        def objective(trial):
            p = {
                "objective":        "binary",
                "metric":           "binary_logloss",
                "boosting_type":    "gbdt",
                "num_leaves":       trial.suggest_int("num_leaves", 31, 255),
                "learning_rate":    trial.suggest_float("lr", 0.01, 0.15, log=True),
                "feature_fraction": trial.suggest_float("ff", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bf", 0.6, 1.0),
                "bagging_freq":     trial.suggest_int("bfr", 1, 10),
                "min_child_samples":trial.suggest_int("mcs", 10, 80),
                "reg_alpha":        trial.suggest_float("ra", 1e-4, 2.0, log=True),
                "reg_lambda":       trial.suggest_float("rl", 1e-4, 4.0, log=True),
                "scale_pos_weight": best_params["scale_pos_weight"],
                "n_jobs": -1, "seed": trial.number, "verbose": -1,
            }
            scores = []
            for ti, vi in skf3.split(X, y):
                dtr = lgb.Dataset(X[ti], label=y[ti])
                dv  = lgb.Dataset(X[vi], label=y[vi], reference=dtr)
                m   = lgb.train(p, dtr, 300, valid_sets=[dv],
                                callbacks=[lgb.early_stopping(25, verbose=False),
                                           lgb.log_evaluation(-1)])
                preds = m.predict(X[vi], num_iteration=m.best_iteration)
                _, f  = find_optimal_threshold(preds, y[vi])
                scores.append(f)
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=args.optuna_trials)
        log.info(f"Optuna best F0.5 = {study.best_value:.4f}")
        bp = study.best_params
        best_params.update({
            "num_leaves": bp["num_leaves"], "learning_rate": bp["lr"],
            "feature_fraction": bp["ff"],  "bagging_fraction": bp["bf"],
            "bagging_freq": bp["bfr"],     "min_child_samples": bp["mcs"],
            "reg_alpha": bp["ra"],         "reg_lambda": bp["rl"],
        })
    except Exception as e:
        log.warning(f"Optuna failed ({e}) — using baseline params.")

# ── 3. 5-fold CV ───────────────────────────────────────────────────────────
log.info("5-fold Stratified CV...")
skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores, cv_thresholds = [], []

for fold, (ti, vi) in enumerate(skf5.split(X, y)):
    dtr = lgb.Dataset(X[ti], label=y[ti], feature_name=FEATURE_COLS, free_raw_data=False)
    dv  = lgb.Dataset(X[vi], label=y[vi], reference=dtr, feature_name=FEATURE_COLS, free_raw_data=False)
    m   = lgb.train(best_params, dtr, num_boost_round=1000,
                    valid_sets=[dv],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(-1)],
                    feval=lgbm_f05_eval)
    preds = m.predict(X[vi], num_iteration=m.best_iteration)
    opt_t, opt_f = find_optimal_threshold(preds, y[vi])
    cv_scores.append(opt_f)
    cv_thresholds.append(opt_t)
    log.info(f"  Fold {fold+1}  F0.5={opt_f:.4f}  @t={opt_t:.2f}  iter={m.best_iteration}")

best_threshold = float(np.mean(cv_thresholds))
log.info(f"CV Mean F0.5 = {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}  threshold={best_threshold:.2f}")

# ── 4. Final model (full dataset) ─────────────────────────────────────────
log.info("Training final model on full dataset...")
dtr_full = lgb.Dataset(X, label=y, feature_name=FEATURE_COLS, free_raw_data=False)
final_model = lgb.train(best_params, dtr_full, num_boost_round=600,
                        callbacks=[lgb.log_evaluation(100)])

# ── 5. Feature importance ──────────────────────────────────────────────────
fi_vals = final_model.feature_importance(importance_type="gain")
fi_df = pd.DataFrame({"feature": FEATURE_COLS, "gain": fi_vals})
fi_df = fi_df.sort_values("gain", ascending=False).reset_index(drop=True)
fi_csv = REPORT_DIR / "feature_importance.csv"
fi_df.to_csv(fi_csv, index=False)
log.info(f"Feature importance saved → {fi_csv}")

# ── 6. Save model ──────────────────────────────────────────────────────────
with open(MODEL_PKL, "wb") as f:
    pickle.dump(final_model, f)
final_model.save_model(str(MODEL_TXT))
log.info(f"Model saved → {MODEL_PKL}")

# ── 7. Save metadata ───────────────────────────────────────────────────────
meta = {
    "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
    "num_train_pairs":   len(X),
    "num_features":      len(FEATURE_COLS),
    "feature_names":     FEATURE_COLS,
    "cv_folds":          5,
    "cv_f05_scores":     [round(s, 5) for s in cv_scores],
    "cv_mean_f05":       round(float(np.mean(cv_scores)), 5),
    "cv_std_f05":        round(float(np.std(cv_scores)), 5),
    "optimal_threshold": round(best_threshold, 4),
    "best_lgbm_params":  best_params,
    "top10_features":    fi_df.head(10)[["feature","gain"]].to_dict("records"),
}
with open(META_JSON, "w") as f:
    json.dump(meta, f, indent=2)
log.info(f"Metadata saved → {META_JSON}")

# Print summary to stdout for parent process to capture
print(f"CV_MEAN_F05={np.mean(cv_scores):.5f}")
print(f"CV_STD_F05={np.std(cv_scores):.5f}")
print(f"THRESHOLD={best_threshold:.4f}")
print(f"MODEL_PKL={MODEL_PKL}")
