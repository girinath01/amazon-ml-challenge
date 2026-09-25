"""
trainer.py  — Member 3: LightGBM Classifier Training Module
============================================================
Responsibilities
----------------
1. Load training_pair_features.parquet produced by run_member3_pipeline.py
2. 5-fold Stratified Cross-Validation with early stopping
3. Optuna hyper-parameter search (lightweight – 40 trials)
4. Final model fit on the full training set
5. Export:
   - model/lgbm_entity_match.pkl           (trained model)
   - model/lgbm_entity_match.txt           (LightGBM booster text – portable)
   - model/training_metadata.json          (CV scores, optimal threshold, feature importances)

Public API
----------
  from trainer import EntityMatchTrainer
  t = EntityMatchTrainer()
  t.run_full_training()    # train + save everything
"""

import os
import sys
import json
import time
import pickle
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import precision_score, recall_score
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("trainer")

# ── Paths ──────────────────────────────────────────────────────────────────
_THIS = Path(__file__).resolve().parent
_REPO_ROOT = _THIS.parent.parent.parent   # Akatsuki/
DATA_DIR   = _REPO_ROOT / "data"
MODEL_DIR  = _REPO_ROOT / "model"
REPORT_DIR = _REPO_ROOT / "reports"

FEATURE_PARQUET = DATA_DIR / "training_pair_features.parquet"
MODEL_PKL       = MODEL_DIR / "lgbm_entity_match.pkl"
MODEL_TXT       = MODEL_DIR / "lgbm_entity_match.txt"
META_JSON       = MODEL_DIR / "training_metadata.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature columns (28 total) ─────────────────────────────────────────────
NAME_FEATS = [
    "name_exact", "name_jaccard", "name_levenshtein",
    "name_jaro_winkler", "name_token_sort", "name_token_set",
    "name_char3_cosine", "name_char4_cosine",
    "name_core_similarity", "name_translit_similarity",
]
ADDR_FEATS = [
    "address_jaccard", "address_levenshtein", "address_char3_cosine",
    "address_digit_overlap", "house_number_match",
    "postal_code_match", "address_exact", "address_missing",
]
CTX_FEATS = [
    "country_match", "name_length_ratio", "address_length_ratio",
    "name_token_count_difference", "address_token_count_difference",
    "numeric_conflict",
]
INTER_FEATS = [
    "name_address_similarity_product", "name_high_address_low",
    "name_high_address_high", "name_address_conflict",
]
FEATURE_COLS = NAME_FEATS + ADDR_FEATS + CTX_FEATS + INTER_FEATS


# ── F-0.5 Macro Evaluator ──────────────────────────────────────────────────
def macro_f05(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-average F0.5 (precision-weighted 2×) across all entities."""
    beta = 0.5
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    if p + r == 0:
        return 0.0
    return (1 + beta**2) * p * r / (beta**2 * p + r)


def find_optimal_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """Return (best_threshold, best_f05) over a grid search."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.30, 0.85, 0.02)]
    best_t, best_f = 0.5, 0.0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f = macro_f05(labels, preds)
        if f > best_f:
            best_f = f
            best_t = t
    return best_t, best_f


# ── LightGBM custom eval for F0.5 ─────────────────────────────────────────
def lgbm_f05_eval(preds, eval_data):
    """LightGBM custom eval function: returns (name, value, is_higher_better)."""
    labels = eval_data.get_label()
    threshold = 0.50
    binary = (preds >= threshold).astype(int)
    score = macro_f05(labels, binary)
    return "f05_macro", score, True


# ── Main Trainer Class ─────────────────────────────────────────────────────
class EntityMatchTrainer:
    """Full LightGBM training pipeline for pairwise entity matching."""

    # Baseline LightGBM hyper-parameters (used if Optuna skipped)
    BASE_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
        "n_jobs": -1,
        "seed": 42,
        "verbose": -1,
    }
    CV_FOLDS   = 5
    MAX_ROUNDS = 1000
    EARLY_STOP = 50

    def __init__(self, parquet_path: Optional[Path] = None):
        self.parquet_path = parquet_path or FEATURE_PARQUET
        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.best_params: Dict = dict(self.BASE_PARAMS)
        self.best_threshold: float = 0.50
        self.cv_scores: List[float] = []
        self.model: Optional[lgb.Booster] = None
        self.feature_importance: Optional[pd.DataFrame] = None

    # ── Data loading ──────────────────────────────────────────────────────
    def load_data(self):
        log.info(f"Loading feature matrix from {self.parquet_path}")
        self.df = pd.read_parquet(self.parquet_path)
        # Verify all features present
        missing = [c for c in FEATURE_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        self.X = self.df[FEATURE_COLS].values.astype(np.float32)
        self.y = self.df["label"].values.astype(np.int32)
        n_pos = int(self.y.sum())
        n_neg = int(len(self.y) - n_pos)
        log.info(f"  Rows={len(self.df):,}  Positives={n_pos:,}  Negatives={n_neg:,}")
        # Adjust scale_pos_weight for class imbalance
        if n_neg > 0 and n_pos > 0:
            self.best_params["scale_pos_weight"] = round(n_neg / n_pos, 3)
        else:
            self.best_params["scale_pos_weight"] = 1.0


    # ── Cross-Validation ──────────────────────────────────────────────────
    def cross_validate(self) -> float:
        """5-fold GroupKFold CV grouped by source1_id; returns mean Macro F0.5."""
        log.info(f"Running {self.CV_FOLDS}-fold GroupKFold Cross-Validation (Group = source1_id)...")
        gkf = GroupKFold(n_splits=self.CV_FOLDS)
        groups = self.df["source1_id"].values
        fold_scores = []
        fold_thresholds = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(self.X, self.y, groups=groups)):
            # Audit zero S1 entity leakage
            train_s1 = set(self.df.iloc[train_idx]["source1_id"])
            val_s1 = set(self.df.iloc[val_idx]["source1_id"])
            overlap = train_s1 & val_s1
            assert len(overlap) == 0, f"Entity leakage detected in fold {fold+1}: {len(overlap)} S1 entities overlap!"

            X_tr, y_tr = self.X[train_idx], self.y[train_idx]
            X_val, y_val = self.X[val_idx], self.y[val_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr,
                                 feature_name=FEATURE_COLS, free_raw_data=False)
            dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain,
                                 feature_name=FEATURE_COLS, free_raw_data=False)

            model = lgb.train(
                self.best_params,
                dtrain,
                num_boost_round=self.MAX_ROUNDS,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(self.EARLY_STOP, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
                feval=lgbm_f05_eval,
            )

            val_scores = model.predict(X_val, num_iteration=model.best_iteration)
            opt_t, opt_f = find_optimal_threshold(val_scores, y_val)
            fold_scores.append(opt_f)
            fold_thresholds.append(opt_t)
            log.info(f"  Fold {fold+1}/{self.CV_FOLDS}  F0.5={opt_f:.4f}  @t={opt_t:.2f}"
                     f"  best_iter={model.best_iteration}  (0 S1 entity leakage verified)")

        self.cv_scores = fold_scores
        self.best_threshold = float(np.mean(fold_thresholds))
        mean_f05 = float(np.mean(fold_scores))
        std_f05  = float(np.std(fold_scores))
        log.info(f"CV Mean F0.5 = {mean_f05:.4f} ± {std_f05:.4f}  "
                 f"Avg optimal threshold = {self.best_threshold:.2f}")
        return mean_f05

    # ── Optuna hyper-parameter optimisation ───────────────────────────────
    def optuna_search(self, n_trials: int = 40) -> Dict:
        """Lightweight Optuna search over key hyper-parameters.
        Returns best params dict."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            log.warning("Optuna not installed — skipping HP search.")
            return self.best_params

        log.info(f"Starting Optuna HP search ({n_trials} trials)...")
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

        def objective(trial):
            clf = lgb.LGBMClassifier(
                n_estimators      = 400,
                num_leaves        = trial.suggest_int("num_leaves", 31, 255),
                learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                feature_fraction  = trial.suggest_float("feature_fraction", 0.6, 1.0),
                subsample         = trial.suggest_float("bagging_fraction", 0.6, 1.0),
                subsample_freq    = trial.suggest_int("bagging_freq", 1, 10),
                min_child_samples = trial.suggest_int("min_child_samples", 10, 80),
                reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
                reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 4.0, log=True),
                scale_pos_weight  = self.best_params.get("scale_pos_weight", 1.0),
                random_state      = trial.number,
                verbose           = -1,
                n_jobs            = -1,
            )
            scores = []
            for train_idx, val_idx in skf.split(self.X, self.y):
                X_tr  = self.X[train_idx]
                y_tr  = self.y[train_idx]
                X_val = self.X[val_idx]
                y_val = self.y[val_idx]
                clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(30, verbose=False),
                                   lgb.log_evaluation(period=-1)])
                preds = clf.predict_proba(X_val)[:, 1]
                _, f  = find_optimal_threshold(preds, y_val)
                scores.append(f)
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        log.info(f"Optuna best F0.5 = {study.best_value:.4f}")

        best = study.best_params
        self.best_params.update({
            "num_leaves":        best["num_leaves"],
            "learning_rate":     best["learning_rate"],
            "feature_fraction":  best["feature_fraction"],
            "bagging_fraction":  best["bagging_fraction"],
            "bagging_freq":      best["bagging_freq"],
            "min_child_samples": best["min_child_samples"],
            "reg_alpha":         best["reg_alpha"],
            "reg_lambda":        best["reg_lambda"],
        })
        return self.best_params

    # ── Final model training ───────────────────────────────────────────────
    def train_final_model(self, num_boost_round: int = 600):
        """Train a single model on the full dataset with optimised params."""
        log.info(f"Training final model on full dataset ({len(self.X):,} rows)...")
        dtrain = lgb.Dataset(self.X, label=self.y,
                             feature_name=FEATURE_COLS, free_raw_data=False)
        self.model = lgb.train(
            self.best_params,
            dtrain,
            num_boost_round=num_boost_round,
            callbacks=[lgb.log_evaluation(period=100)],
        )

        # Feature importance table
        fi_vals = self.model.feature_importance(importance_type="gain")
        self.feature_importance = pd.DataFrame({
            "feature": FEATURE_COLS,
            "gain":    fi_vals,
        }).sort_values("gain", ascending=False).reset_index(drop=True)
        log.info("Final model trained.")

    # ── Save artefacts ─────────────────────────────────────────────────────
    def save(self):
        """Persist model + metadata."""
        # Pickle (for Python API)
        with open(MODEL_PKL, "wb") as f:
            pickle.dump(self.model, f)
        log.info(f"  Saved model pickle → {MODEL_PKL}")

        # Booster text (portable)
        self.model.save_model(str(MODEL_TXT))
        log.info(f"  Saved booster text  → {MODEL_TXT}")

        # Feature importance CSV
        fi_csv = REPORT_DIR / "feature_importance.csv"
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.feature_importance.to_csv(fi_csv, index=False)
        log.info(f"  Saved feature importance → {fi_csv}")

        # Metadata JSON
        meta = {
            "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_train_pairs":   len(self.X),
            "num_features":      len(FEATURE_COLS),
            "feature_names":     FEATURE_COLS,
            "cv_folds":          self.CV_FOLDS,
            "cv_f05_scores":     [round(s, 5) for s in self.cv_scores],
            "cv_mean_f05":       round(float(np.mean(self.cv_scores)), 5) if self.cv_scores else None,
            "cv_std_f05":        round(float(np.std(self.cv_scores)), 5) if self.cv_scores else None,
            "optimal_threshold": round(self.best_threshold, 4),
            "best_lgbm_params":  self.best_params,
            "top10_features":    self.feature_importance.head(10)[["feature", "gain"]].to_dict("records"),
        }
        with open(META_JSON, "w") as f:
            json.dump(meta, f, indent=2)
        log.info(f"  Saved training metadata → {META_JSON}")

    # ── Orchestrator ───────────────────────────────────────────────────────
    def run_full_training(self, run_optuna: bool = True, optuna_trials: int = 40):
        """Complete pipeline: load → optuna → CV → final model → save."""
        t0 = time.time()
        self.load_data()
        if run_optuna:
            self.optuna_search(n_trials=optuna_trials)
        self.cross_validate()
        self.train_final_model()
        self.save()
        elapsed = round(time.time() - t0, 1)
        log.info(f"\nTraining complete in {elapsed}s")
        log.info(f"  CV Mean F0.5  : {np.mean(self.cv_scores):.4f}")
        log.info(f"  Threshold     : {self.best_threshold:.2f}")
        log.info(f"  Model saved   : {MODEL_PKL}")
        return self
