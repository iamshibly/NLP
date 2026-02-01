#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CKD Benchmarks — Leakage-free Nested CV + Federated TabTransformer + Transfer Learning
====================================================================================

Key features:
- Uses ONLY 2 datasets:
  (1) UCI CKD (ucimlrepo id=336)
  (2) Kaggle: mansoordaku/ckdisease
- Nested CV: Outer=5 folds, Inner=3 folds
  * Inner loop tunes hyperparams (random search + transferred configs)
  * τ is tuned by MAX MCC using OUTER-train OOF predictions (leakage-free)
- No leakage:
  * Train-only feature dropping transformer inside pipeline
  * Imputers/encoders fit ONLY on train folds
  * Threshold τ is tuned using only train OOF predictions
- Federated Learning:
  * Each dataset's TRAIN split is partitioned into 3 clients
  * TabTransformer is trained with FedAvg across clients
  * Global transformer server aggregates client updates (no pretraining)
- Transfer Learning:
  * Top-K configs from dataset 1 are injected into dataset 2's candidate pool
- Metrics:
  * acc, precision, recall, f1, logloss, MCC, ROC-AUC
  * ROC curve data (FPR/TPR) stored for test/holdout outputs
"""

from __future__ import annotations

import os
import glob
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo
import kagglehub

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# -------------------------
# Reproducibility
# -------------------------
SEED = 42


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_all_seeds(SEED)


# -------------------------
# Performance-oriented defaults (fast)
# -------------------------
OUTER_FOLDS = 5
INNER_FOLDS = 3
HOLDOUT_SIZE = 0.20

N_CONFIGS_ET = 8
N_CONFIGS_TT = 6
TRANSFER_TOPK = 2

MISSING_THRESH = 0.60
TAU_STEP = 0.02

# Federated training
CLIENTS_PER_DATASET = 3
FED_ROUNDS = 4
LOCAL_EPOCHS = 3
TT_BATCH_SIZE = 64


# -------------------------
# Dataset sanity prior
# -------------------------
CKD_COL_PRIOR = {
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "rbc",
    "pc",
    "pcc",
    "ba",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wc",
    "rc",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
    "classification",
}
_MISS_TOKENS = {"?", " ?", "\t?", "\t?\t", "nan", "NaN", "None", ""}


def _norm_col(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")


def ckd_overlap_score(cols: List[str]) -> int:
    cols = [_norm_col(c) for c in cols]
    return len(set(cols) & CKD_COL_PRIOR)


# -------------------------
# Cleaning + label normalization
# -------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype(str).str.strip()
            out[c] = out[c].replace({t: np.nan for t in _MISS_TOKENS})
        coerced = pd.to_numeric(out[c], errors="coerce")
        if coerced.notna().mean() >= 0.80:
            out[c] = coerced
    return out


def infer_target_column(df: pd.DataFrame) -> str:
    candidates = [
        "classification",
        "class",
        "target",
        "label",
        "ckd",
        "disease",
        "outcome",
        "diagnosis",
        "result",
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in cols_lower:
            return cols_lower[name]
    return df.columns[-1]


def normalize_ckd_label(y_raw: pd.Series) -> pd.Series:
    y = y_raw.copy()
    if pd.api.types.is_numeric_dtype(y):
        yy = pd.to_numeric(y, errors="coerce")
        uniq = sorted([u for u in yy.dropna().unique()])
        if set(uniq) <= {0, 1}:
            return yy.astype("Int64")
        if set(uniq) <= {1, 2}:
            return yy.map({1: 1, 2: 0}).astype("Int64")
        med = float(np.nanmedian(yy.values))
        return (yy >= med).astype("Int64")

    ys = y.astype(str).str.strip().str.lower()

    def map_one(v: str) -> Any:
        v = v.replace(" ", "").replace("_", "").replace("-", "")
        if v in {"1", "yes", "y", "true", "ckd", "chronickidneydisease", "positive"}:
            return 1
        if v in {"0", "no", "n", "false", "notckd", "notchronickidneydisease", "negative"}:
            return 0
        if ("not" in v or "non" in v) and "ckd" in v:
            return 0
        if "ckd" in v:
            return 1
        if v in {"normal", "healthy"}:
            return 0
        return np.nan

    return ys.map(map_one).astype("Int64")


# -------------------------
# Leakage-free feature selector (train-only)
# -------------------------
class TrainOnlyDropper(BaseEstimator, TransformerMixin):
    def __init__(self, missing_thresh: float = 0.60):
        self.missing_thresh = missing_thresh
        self.drop_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        miss = X.isna().mean()
        drop_missing = miss[miss > self.missing_thresh].index.tolist()
        nunq = X.fillna("__NA__").nunique(dropna=False)
        drop_const = nunq[nunq <= 1].index.tolist()
        self.drop_cols_ = sorted(set(drop_missing + drop_const))
        return self

    def transform(self, X: pd.DataFrame):
        return X.drop(columns=self.drop_cols_, errors="ignore")


# -------------------------
# τ selection (MCC)
# -------------------------
def best_tau_by_mcc(y_true: np.ndarray, proba: np.ndarray, step: float = 0.02) -> float:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    best_tau, best_mcc = 0.5, -1e18
    for tau in np.arange(0.0, 1.0 + 1e-12, step):
        pred = (proba >= tau).astype(int)
        mcc = matthews_corrcoef(y_true, pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_tau = float(tau)
    return float(best_tau)


def compute_metrics(
    y_true: np.ndarray, proba: np.ndarray, tau: float
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= tau).astype(int)
    fpr, tpr, _ = roc_curve(y_true, proba)
    proba_clip = np.clip(proba, 1e-9, 1 - 1e-9)
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "roc_auc_score": float(roc_auc_score(y_true, proba)),
        "roc_auc_curve": float(auc(fpr, tpr)),
        "logloss": float(
            log_loss(y_true, np.vstack([1 - proba_clip, proba_clip]).T, labels=[0, 1])
        ),
        "tau": float(tau),
    }
    curve = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    return metrics, curve


# =========================
# ExtraTrees pipeline
# =========================
def make_extratrees_pipeline(config: Dict[str, Any], X_ref: pd.DataFrame) -> Pipeline:
    num_cols = [c for c in X_ref.columns if pd.api.types.is_numeric_dtype(X_ref[c])]
    cat_cols = [c for c in X_ref.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    clf = ExtraTreesClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        random_state=SEED,
        n_jobs=-1,
    )

    pipe = Pipeline(
        [
            ("dropper", TrainOnlyDropper(missing_thresh=MISSING_THRESH)),
            ("pre", pre),
            ("clf", clf),
        ]
    )
    return pipe


# =========================
# Federated TabTransformer
# =========================
def safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class _TTNet(nn.Module):
    def __init__(
        self,
        cat_cardinalities: List[int],
        n_num: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        mlp_hidden: int,
    ):
        super().__init__()
        self.n_cat = len(cat_cardinalities)
        self.n_num = n_num
        self.d_model = d_model

        self.cat_embeds = nn.ModuleList(
            [nn.Embedding(int(card), d_model) for card in cat_cardinalities]
        )
        self.num_proj = nn.Linear(n_num, d_model) if n_num > 0 else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=max(mlp_hidden, d_model * 2),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x_cat: torch.Tensor, x_num: Optional[torch.Tensor]) -> torch.Tensor:
        tokens = []
        for j in range(self.n_cat):
            tokens.append(self.cat_embeds[j](x_cat[:, j]))
        if self.num_proj is not None and x_num is not None:
            tokens.append(self.num_proj(x_num))
        x = torch.stack(tokens, dim=1) if len(tokens) > 1 else tokens[0].unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        logits = self.head(x).squeeze(-1)
        return logits


class TTPreprocessor:
    def __init__(self):
        self.cat_cols_: List[str] = []
        self.num_cols_: List[str] = []
        self.cat_maps_: Dict[str, Dict[Any, int]] = {}
        self.cat_cardinalities_: List[int] = []
        self.num_median_: Dict[str, float] = {}
        self.num_mean_: Dict[str, float] = {}
        self.num_std_: Dict[str, float] = {}
        self.drop_cols_: List[str] = []

    def fit(self, X: pd.DataFrame) -> "TTPreprocessor":
        dropper = TrainOnlyDropper(missing_thresh=MISSING_THRESH).fit(X)
        self.drop_cols_ = list(dropper.drop_cols_)
        X = dropper.transform(X)

        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in X.columns if c not in num_cols]
        self.num_cols_ = num_cols
        self.cat_cols_ = cat_cols

        self.cat_maps_.clear()
        self.cat_cardinalities_.clear()
        for c in self.cat_cols_:
            s = X[c].astype(str).fillna("__NA__")
            uniq = pd.unique(s)
            mapping = {v: i + 1 for i, v in enumerate(uniq)}
            self.cat_maps_[c] = mapping
            self.cat_cardinalities_.append(len(mapping) + 1)

        for c in self.num_cols_:
            col = pd.to_numeric(X[c], errors="coerce")
            med = float(np.nanmedian(col.values))
            col2 = col.fillna(med)
            mu = float(col2.mean())
            sd = float(col2.std(ddof=0)) if float(col2.std(ddof=0)) > 1e-12 else 1.0
            self.num_median_[c] = med
            self.num_mean_[c] = mu
            self.num_std_[c] = sd

        return self

    def transform(self, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X = X.drop(columns=self.drop_cols_, errors="ignore")
        if self.cat_cols_:
            x_cat = np.zeros((len(X), len(self.cat_cols_)), dtype=np.int64)
            for j, c in enumerate(self.cat_cols_):
                s = X[c].astype(str).fillna("__NA__")
                m = self.cat_maps_[c]
                x_cat[:, j] = [m.get(v, 0) for v in s.values]
        else:
            x_cat = np.zeros((len(X), 0), dtype=np.int64)

        if self.num_cols_:
            x_num = np.zeros((len(X), len(self.num_cols_)), dtype=np.float32)
            for j, c in enumerate(self.num_cols_):
                col = (
                    pd.to_numeric(X[c], errors="coerce")
                    .fillna(self.num_median_[c])
                    .astype(float)
                    .values
                )
                col = (col - self.num_mean_[c]) / self.num_std_[c]
                x_num[:, j] = col.astype(np.float32)
        else:
            x_num = None
        return x_cat, x_num


class FederatedTabTransformerClassifier:
    def __init__(self, config: Dict[str, Any], device: Optional[str] = None):
        self.config = dict(config)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prep_ = TTPreprocessor()
        self.model_: Optional[_TTNet] = None

    def _build_model(self) -> _TTNet:
        return _TTNet(
            cat_cardinalities=self.prep_.cat_cardinalities_,
            n_num=0 if not self.prep_.num_cols_ else len(self.prep_.num_cols_),
            d_model=int(self.config["d_model"]),
            n_heads=int(self.config["n_heads"]),
            n_layers=int(self.config["n_layers"]),
            dropout=float(self.config["dropout"]),
            mlp_hidden=int(self.config["mlp_hidden"]),
        ).to(self.device)

    def _split_clients(self, y: np.ndarray) -> List[np.ndarray]:
        splitter = StratifiedKFold(
            n_splits=CLIENTS_PER_DATASET, shuffle=True, random_state=SEED
        )
        idx = np.arange(len(y))
        return [client_idx for _, client_idx in splitter.split(idx, y)]

    def _make_loader(
        self, x_cat: np.ndarray, x_num: Optional[np.ndarray], y: np.ndarray
    ) -> DataLoader:
        t_cat = torch.tensor(x_cat, dtype=torch.long, device=self.device)
        t_y = torch.tensor(y, dtype=torch.float32, device=self.device)
        if x_num is None:
            ds = TensorDataset(t_cat, t_y)

            def collate(batch):
                xc, yc = zip(*batch)
                return torch.stack(xc, 0), None, torch.stack(yc, 0)
        else:
            t_num = torch.tensor(x_num, dtype=torch.float32, device=self.device)
            ds = TensorDataset(t_cat, t_num, t_y)

            def collate(batch):
                xc, xn, yc = zip(*batch)
                return torch.stack(xc, 0), torch.stack(xn, 0), torch.stack(yc, 0)

        return DataLoader(
            ds, batch_size=int(self.config["batch_size"]), shuffle=True, collate_fn=collate
        )

    def _train_local(
        self, model: _TTNet, loader: DataLoader, epochs: int
    ) -> Dict[str, torch.Tensor]:
        model.train()
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["lr"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            for xb_cat, xb_num, yb in loader:
                opt.zero_grad(set_to_none=True)
                logits = model(xb_cat, xb_num)
                loss = loss_fn(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}

    @staticmethod
    def _fedavg(
        states: List[Dict[str, torch.Tensor]], weights: List[int]
    ) -> Dict[str, torch.Tensor]:
        total = float(sum(weights))
        avg_state = {}
        for k in states[0].keys():
            avg = sum(state[k] * (w / total) for state, w in zip(states, weights))
            avg_state[k] = avg
        return avg_state

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "FederatedTabTransformerClassifier":
        set_all_seeds(SEED)
        y = np.asarray(y).astype(np.float32)

        # Fit preprocess on TRAIN ONLY (per fold)
        self.prep_.fit(X)
        x_cat, x_num = self.prep_.transform(X)

        client_idxs = self._split_clients(y)
        client_loaders = []
        client_sizes = []
        for idx in client_idxs:
            client_loaders.append(
                self._make_loader(
                    x_cat[idx], None if x_num is None else x_num[idx], y[idx]
                )
            )
            client_sizes.append(int(len(idx)))

        self.model_ = self._build_model()
        global_state = {k: v.detach().cpu() for k, v in self.model_.state_dict().items()}

        for _ in range(int(self.config["federated_rounds"])):
            states = []
            for loader in client_loaders:
                local_model = self._build_model()
                local_model.load_state_dict(global_state)
                state = self._train_local(
                    local_model, loader, epochs=int(self.config["local_epochs"])
                )
                states.append(state)
            global_state = self._fedavg(states, client_sizes)

        self.model_.load_state_dict(global_state)
        self.model_.eval()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fit yet.")
        x_cat, x_num = self.prep_.transform(X)
        t_cat = torch.tensor(x_cat, dtype=torch.long, device=self.device)
        t_num = None if x_num is None else torch.tensor(x_num, dtype=torch.float32, device=self.device)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(t_cat, t_num).detach().cpu().numpy()
            proba = 1.0 / (1.0 + np.exp(-logits))
        proba = np.clip(proba, 1e-9, 1 - 1e-9)
        return np.vstack([1 - proba, proba]).T


# =========================
# Config sampling (random + transfer)
# =========================
def sample_et_configs(n: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    grid = {
        "n_estimators": [300, 600],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", 0.7],
    }
    cfgs: List[Dict[str, Any]] = []
    for _ in range(n):
        cfgs.append(
            {
                "n_estimators": int(rng.choice(grid["n_estimators"])),
                "max_depth": grid["max_depth"][int(rng.integers(0, len(grid["max_depth"])))],
                "min_samples_leaf": int(rng.choice(grid["min_samples_leaf"])),
                "max_features": grid["max_features"][int(rng.integers(0, len(grid["max_features"])))],
            }
        )
    return _uniq_configs(cfgs)


def sample_tt_configs(n: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    grid = {
        "d_model": [16, 32],
        "n_heads": [2, 4],
        "n_layers": [2, 3],
        "dropout": [0.1, 0.2],
        "mlp_hidden": [64, 128],
        "lr": [1e-3, 5e-4],
        "weight_decay": [0.0, 1e-4],
    }
    cfgs: List[Dict[str, Any]] = []
    for _ in range(n):
        cfgs.append(
            {
                "d_model": int(rng.choice(grid["d_model"])),
                "n_heads": int(rng.choice(grid["n_heads"])),
                "n_layers": int(rng.choice(grid["n_layers"])),
                "dropout": float(rng.choice(grid["dropout"])),
                "mlp_hidden": int(rng.choice(grid["mlp_hidden"])),
                "lr": float(rng.choice(grid["lr"])),
                "weight_decay": float(rng.choice(grid["weight_decay"])),
                "federated_rounds": int(FED_ROUNDS),
                "local_epochs": int(LOCAL_EPOCHS),
                "batch_size": int(TT_BATCH_SIZE),
            }
        )
    return _uniq_configs(cfgs)


def _uniq_configs(cfgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for c in cfgs:
        key = tuple(sorted((k, str(v)) for k, v in c.items()))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def inject_transfer(
    cfgs: List[Dict[str, Any]], transfer_bank: List[Dict[str, Any]], topk: int
) -> List[Dict[str, Any]]:
    if not transfer_bank:
        return cfgs
    injected = transfer_bank[:topk] + cfgs
    return _uniq_configs(injected)


# =========================
# Nested CV core
# =========================
def fit_predict_proba(
    model_family: str,
    cfg: Dict[str, Any],
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
) -> Tuple[np.ndarray, Any]:
    if model_family == "extratrees":
        pipe = make_extratrees_pipeline(cfg, X_tr)
        pipe.fit(X_tr, y_tr)
        return pipe.predict_proba(X_te)[:, 1], pipe
    if model_family == "tabtransformer":
        tt = FederatedTabTransformerClassifier(cfg)
        tt.fit(X_tr, y_tr)
        return tt.predict_proba(X_te)[:, 1], tt
    raise ValueError(f"Unknown model family: {model_family}")


def inner_cv_score(
    model_family: str, X_tr: pd.DataFrame, y_tr: np.ndarray, cfg: Dict[str, Any], seed: int
) -> Tuple[float, float]:
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    mccs, aucs = [], []
    for a, b in inner.split(X_tr, y_tr):
        X_a, y_a = X_tr.iloc[a], y_tr[a]
        X_b, y_b = X_tr.iloc[b], y_tr[b]
        proba_b, _ = fit_predict_proba(model_family, cfg, X_a, y_a, X_b)
        tau = best_tau_by_mcc(y_b, proba_b, step=TAU_STEP)
        pred_b = (proba_b >= tau).astype(int)
        mccs.append(matthews_corrcoef(y_b, pred_b))
        aucs.append(roc_auc_score(y_b, proba_b))
    return float(np.mean(mccs)), float(np.mean(aucs))


def oof_tau_on_outer_train(
    model_family: str, X_tr: pd.DataFrame, y_tr: np.ndarray, cfg: Dict[str, Any], seed: int
) -> float:
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    oof_proba = np.zeros(len(y_tr), dtype=float)
    for a, b in inner.split(X_tr, y_tr):
        X_a, y_a = X_tr.iloc[a], y_tr[a]
        X_b = X_tr.iloc[b]
        proba_b, _ = fit_predict_proba(model_family, cfg, X_a, y_a, X_b)
        oof_proba[b] = proba_b
    return best_tau_by_mcc(y_tr, oof_proba, step=TAU_STEP)


@dataclass
class NestedCVResult:
    model_family: str
    oof_proba: np.ndarray
    fold_metrics: List[Dict[str, float]]
    fold_taus: List[float]
    fold_best_cfgs: List[Dict[str, Any]]
    fold_inner_mcc: List[float]
    fold_curves: List[Dict[str, List[float]]]


def run_nested_cv(
    model_family: str,
    X: pd.DataFrame,
    y: np.ndarray,
    candidate_cfgs: List[Dict[str, Any]],
    seed: int,
) -> NestedCVResult:
    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    fold_metrics, fold_taus, fold_cfgs, fold_inner, fold_curves = [], [], [], [], []

    for ofold, (tr_idx, te_idx) in enumerate(outer.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_te, y_te = X.iloc[te_idx], y[te_idx]

        best_cfg = None
        best_mcc = -1e18
        best_auc = -1e18
        for cfg in candidate_cfgs:
            mean_mcc, mean_auc = inner_cv_score(model_family, X_tr, y_tr, cfg, seed=seed + 1000 + ofold)
            if (mean_mcc > best_mcc) or (mean_mcc == best_mcc and mean_auc > best_auc):
                best_mcc, best_auc = mean_mcc, mean_auc
                best_cfg = cfg

        tau = oof_tau_on_outer_train(model_family, X_tr, y_tr, best_cfg, seed=seed + 2000 + ofold)
        proba_te, _ = fit_predict_proba(model_family, best_cfg, X_tr, y_tr, X_te)
        oof[te_idx] = proba_te

        met, curve = compute_metrics(y_te, proba_te, tau)
        fold_metrics.append(met)
        fold_taus.append(tau)
        fold_cfgs.append(best_cfg)
        fold_inner.append(best_mcc)
        fold_curves.append(curve)

        print(
            f"  [{model_family} | Outer {ofold}/{OUTER_FOLDS}] inner_mcc={best_mcc:.4f} "
            f"tau={tau:.2f} test_mcc={met['mcc']:.4f} test_auc={met['roc_auc_score']:.4f}"
        )

    return NestedCVResult(
        model_family=model_family,
        oof_proba=oof,
        fold_metrics=fold_metrics,
        fold_taus=fold_taus,
        fold_best_cfgs=fold_cfgs,
        fold_inner_mcc=fold_inner,
        fold_curves=fold_curves,
    )


def summarize_outer_metrics(res: NestedCVResult) -> Dict[str, str]:
    mdf = pd.DataFrame(res.fold_metrics)

    def ms(col: str) -> str:
        return f"{mdf[col].mean():.4f}±{mdf[col].std(ddof=0):.4f}"

    return {
        "acc(mean±std)": ms("accuracy"),
        "f1(mean±std)": ms("f1"),
        "mcc(mean±std)": ms("mcc"),
        "roc_auc(mean±std)": ms("roc_auc_score"),
        "logloss(mean±std)": ms("logloss"),
        "tau(mean±std)": f"{np.mean(res.fold_taus):.3f}±{np.std(res.fold_taus, ddof=0):.3f}",
    }


def pick_topk_cfgs(res: NestedCVResult, k: int) -> List[Dict[str, Any]]:
    mccs = [m["mcc"] for m in res.fold_metrics]
    order = np.argsort(mccs)[::-1][:k]
    top = [res.fold_best_cfgs[i] for i in order]
    return _uniq_configs(top)


# =========================
# Final fit on full train split + holdout eval
# =========================
@dataclass
class FinalFitResult:
    best_cfg: Dict[str, Any]
    tau: float
    proba_holdout: np.ndarray
    metrics_holdout: Dict[str, float]
    curve_holdout: Dict[str, List[float]]
    model_obj: Any


def tune_and_fit_full_train(
    model_family: str,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_holdout: pd.DataFrame,
    y_holdout: np.ndarray,
    candidate_cfgs: List[Dict[str, Any]],
    seed: int,
) -> FinalFitResult:
    best_cfg = None
    best_mcc = -1e18
    best_auc = -1e18
    for cfg in candidate_cfgs:
        mean_mcc, mean_auc = inner_cv_score(model_family, X_tr, y_tr, cfg, seed=seed + 333)
        if (mean_mcc > best_mcc) or (mean_mcc == best_mcc and mean_auc > best_auc):
            best_mcc, best_auc = mean_mcc, mean_auc
            best_cfg = cfg

    tau = oof_tau_on_outer_train(model_family, X_tr, y_tr, best_cfg, seed=seed + 444)
    proba_hold, model_obj = fit_predict_proba(model_family, best_cfg, X_tr, y_tr, X_holdout)
    met_hold, curve_hold = compute_metrics(y_holdout, proba_hold, tau)
    return FinalFitResult(
        best_cfg=best_cfg,
        tau=tau,
        proba_holdout=proba_hold,
        metrics_holdout=met_hold,
        curve_holdout=curve_hold,
        model_obj=model_obj,
    )


# =========================
# Data loading (ONLY 2 datasets)
# =========================
def load_uciml_ckd(dataset_id: int = 336) -> Tuple[pd.DataFrame, pd.Series]:
    ds = fetch_ucirepo(id=dataset_id)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    return X, pd.Series(y)


def pick_best_ckd_csv(path: str) -> str:
    csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csvs:
        raise FileNotFoundError(f"No CSV found under: {path}")

    best, best_score = None, -1e18
    for f in csvs:
        try:
            df = pd.read_csv(f, nrows=3000)
            df = clean_dataframe(df)
            if df.shape[0] < 50 or df.shape[1] < 6:
                continue
            tcol = infer_target_column(df)
            y_norm = normalize_ckd_label(df[tcol])
            n_classes = int(pd.Series(y_norm.dropna().unique()).nunique())
            valid = float(y_norm.notna().mean())
            Xcand = df.drop(columns=[tcol], errors="ignore")
            overlap = ckd_overlap_score(Xcand.columns.tolist())
            uniq_raw = int(pd.Series(df[tcol]).nunique(dropna=True))

            score = 0.0
            score += 20.0 * min(overlap, 12) / 12.0
            score += 10.0 * valid
            score += 30.0 if n_classes == 2 else (-120.0 if n_classes < 2 else -40.0)
            score -= 15.0 * (1.0 if uniq_raw > 30 else 0.0)

            if score > best_score:
                best_score, best = score, f
        except Exception:
            continue

    if best is None:
        best = max(csvs, key=lambda f: os.path.getsize(f))
    return best


def load_kaggle_df(slug: str) -> pd.DataFrame:
    path = kagglehub.dataset_download(slug)
    csv_path = pick_best_ckd_csv(path)
    print(f"[Kaggle] {slug} -> {csv_path}")
    return pd.read_csv(csv_path)


# =========================
# Main runner
# =========================
def _display_df(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    try:
        from IPython.display import display

        display(df)
    except Exception:
        print(df.to_string(index=False))


def prepare_xy(
    name: str, X_raw: pd.DataFrame, y_raw: pd.Series
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    Xc = clean_dataframe(X_raw)
    yn = normalize_ckd_label(pd.Series(y_raw))

    df = Xc.copy()
    df["__y__"] = yn
    df = df.dropna(subset=["__y__"]).reset_index(drop=True)

    X = df.drop(columns=["__y__"])
    y = df["__y__"].astype(int).values

    overlap = ckd_overlap_score(X.columns.tolist())
    if overlap < 4:
        raise ValueError(f"{name}: not patient-level CKD table (ckd_overlap={overlap}).")
    if len(np.unique(y)) < 2:
        raise ValueError(f"{name}: only one class after label normalization.")
    if len(y) < 120:
        raise ValueError(f"{name}: too few labeled rows after cleaning: n={len(y)}")

    meta = {
        "rows": int(len(y)),
        "features": int(X.shape[1]),
        "ckd_overlap": int(overlap),
        "class1_ratio": float(np.mean(y == 1)),
        "missing_mean": float(X.isna().mean().mean()),
    }
    return X, y, meta


def main() -> None:
    rng = np.random.default_rng(SEED)

    datasets: List[Tuple[str, pd.DataFrame, pd.Series]] = []
    X_uci, y_uci = load_uciml_ckd(336)
    datasets.append(("UCI CKD (id=336)", X_uci, y_uci))

    df_k = load_kaggle_df("mansoordaku/ckdisease")
    tcol = infer_target_column(df_k)
    datasets.append(
        ("Kaggle CKD (mansoordaku/ckdisease)", df_k.drop(columns=[tcol], errors="ignore"), df_k[tcol])
    )

    transfer_bank = {"extratrees": [], "tabtransformer": []}

    main_rows = []
    holdout_rows = []
    hp_rows = []
    pr_rows = []
    global_rows = []

    v_intake = []
    v_feat = []
    v_enc = []
    v_cv = []
    v_transfer = []
    v_global = []

    meta_train_X_list = []
    meta_train_y_list = []
    meta_holdout_by_ds = {}
    holdout_curves = {}

    for di, (dname, X_raw, y_raw) in enumerate(datasets, start=1):
        print(f"\n==================== {dname} ====================")
        X, y, meta = prepare_xy(dname, X_raw, y_raw)

        idx = np.arange(len(y))
        tr_idx, ho_idx = train_test_split(
            idx, test_size=HOLDOUT_SIZE, random_state=SEED, stratify=y
        )
        X_tr, y_tr = X.iloc[tr_idx].reset_index(drop=True), y[tr_idx]
        X_ho, y_ho = X.iloc[ho_idx].reset_index(drop=True), y[ho_idx]

        v_intake.append(
            {
                "dataset": dname,
                "n_train": int(len(y_tr)),
                "n_holdout": int(len(y_ho)),
                "features": int(X_tr.shape[1]),
                "class1_ratio": round(float(np.mean(y_tr == 1)), 4),
            }
        )

        dropper_tmp = TrainOnlyDropper(missing_thresh=MISSING_THRESH).fit(X_tr)
        kept = int(X_tr.shape[1] - len(dropper_tmp.drop_cols_))
        v_feat.append(
            {
                "dataset": dname,
                "feat_in": int(X_tr.shape[1]),
                "feat_kept": kept,
                "feat_drop": int(len(dropper_tmp.drop_cols_)),
                "ckd_overlap": int(meta["ckd_overlap"]),
                "missing_thr": MISSING_THRESH,
            }
        )

        num_cols = [c for c in X_tr.columns if pd.api.types.is_numeric_dtype(X_tr[c])]
        cat_cols = [c for c in X_tr.columns if c not in num_cols]
        try:
            tmp_pipe = make_extratrees_pipeline(
                {
                    "n_estimators": 300,
                    "max_depth": None,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
                },
                X_tr,
            )
            tmp_pipe.fit(X_tr, y_tr)
            Xt = tmp_pipe.named_steps["pre"].fit_transform(
                tmp_pipe.named_steps["dropper"].transform(X_tr)
            )
            et_dim = int(Xt.shape[1])
        except Exception:
            et_dim = -1

        v_enc.append(
            {
                "dataset": dname,
                "n_num": int(len(num_cols)),
                "n_cat": int(len(cat_cols)),
                "et_dim": int(et_dim),
                "tt_tokens": int(len(cat_cols) + (1 if len(num_cols) else 0)),
                "tt_d_model": "16-32",
            }
        )

        v_cv.append(
            {
                "dataset": dname,
                "outer": OUTER_FOLDS,
                "inner": INNER_FOLDS,
                "tau_step": TAU_STEP,
                "cfg_et": N_CONFIGS_ET,
                "cfg_tt": N_CONFIGS_TT,
            }
        )

        et_cfgs = inject_transfer(
            sample_et_configs(N_CONFIGS_ET, rng), transfer_bank["extratrees"], TRANSFER_TOPK
        )
        tt_cfgs = inject_transfer(
            sample_tt_configs(N_CONFIGS_TT, rng), transfer_bank["tabtransformer"], TRANSFER_TOPK
        )

        print("Running nested CV (train split only) ...")
        res_et = run_nested_cv("extratrees", X_tr, y_tr, et_cfgs, seed=SEED + 10 * di)
        res_tt = run_nested_cv("tabtransformer", X_tr, y_tr, tt_cfgs, seed=SEED + 20 * di)

        sum_et = summarize_outer_metrics(res_et)
        sum_tt = summarize_outer_metrics(res_tt)

        main_rows.append({"dataset": dname, "model": "ExtraTrees", **sum_et})
        main_rows.append({"dataset": dname, "model": "FedTabTransformer", **sum_tt})

        pr_rows.append(
            {
                "dataset": dname,
                "model": "ExtraTrees",
                "precision(mean±std)": f"{pd.DataFrame(res_et.fold_metrics)['precision'].mean():.4f}±{pd.DataFrame(res_et.fold_metrics)['precision'].std(ddof=0):.4f}",
                "recall(mean±std)": f"{pd.DataFrame(res_et.fold_metrics)['recall'].mean():.4f}±{pd.DataFrame(res_et.fold_metrics)['recall'].std(ddof=0):.4f}",
                "tau(mean±std)": sum_et["tau(mean±std)"],
                "n_train": int(len(y_tr)),
            }
        )
        pr_rows.append(
            {
                "dataset": dname,
                "model": "FedTabTransformer",
                "precision(mean±std)": f"{pd.DataFrame(res_tt.fold_metrics)['precision'].mean():.4f}±{pd.DataFrame(res_tt.fold_metrics)['precision'].std(ddof=0):.4f}",
                "recall(mean±std)": f"{pd.DataFrame(res_tt.fold_metrics)['recall'].mean():.4f}±{pd.DataFrame(res_tt.fold_metrics)['recall'].std(ddof=0):.4f}",
                "tau(mean±std)": sum_tt["tau(mean±std)"],
                "n_train": int(len(y_tr)),
            }
        )

        best_et = pick_topk_cfgs(res_et, 1)[0]
        best_tt = pick_topk_cfgs(res_tt, 1)[0]
        hp_rows.append({"dataset": dname, "model": "ExtraTrees", **best_et})
        hp_rows.append({"dataset": dname, "model": "FedTabTransformer", **best_tt})

        top_et = pick_topk_cfgs(res_et, TRANSFER_TOPK)
        top_tt = pick_topk_cfgs(res_tt, TRANSFER_TOPK)
        if di == 1:
            transfer_bank["extratrees"] = top_et
            transfer_bank["tabtransformer"] = top_tt

        v_transfer.append(
            {
                "dataset": dname,
                "transfer_injected": 0 if di == 1 else TRANSFER_TOPK,
                "topk_saved": TRANSFER_TOPK,
                "et_pool": int(len(et_cfgs)),
                "tt_pool": int(len(tt_cfgs)),
                "note": "UCI→Kaggle" if di == 2 else "source",
            }
        )

        print("Final fit on full train split → holdout ...")
        final_et = tune_and_fit_full_train(
            "extratrees", X_tr, y_tr, X_ho, y_ho, et_cfgs, seed=SEED + 100 + di
        )
        final_tt = tune_and_fit_full_train(
            "tabtransformer", X_tr, y_tr, X_ho, y_ho, tt_cfgs, seed=SEED + 200 + di
        )

        holdout_rows.append({"dataset": dname, "model": "ExtraTrees", **final_et.metrics_holdout})
        holdout_rows.append({"dataset": dname, "model": "FedTabTransformer", **final_tt.metrics_holdout})
        holdout_curves[f"{dname}::ExtraTrees"] = final_et.curve_holdout
        holdout_curves[f"{dname}::FedTabTransformer"] = final_tt.curve_holdout

        meta_X_train = np.column_stack([safe_logit(res_et.oof_proba), safe_logit(res_tt.oof_proba)])
        meta_train_X_list.append(meta_X_train)
        meta_train_y_list.append(y_tr)

        meta_X_holdout = np.column_stack(
            [safe_logit(final_et.proba_holdout), safe_logit(final_tt.proba_holdout)]
        )
        meta_holdout_by_ds[dname] = (meta_X_holdout, y_ho)

        v_global.append(
            {
                "dataset": dname,
                "meta_feats": int(meta_X_train.shape[1]),
                "train_rows": int(meta_X_train.shape[0]),
                "holdout_rows": int(meta_X_holdout.shape[0]),
                "locals": "ET+FedTT",
                "leakage": "no",
            }
        )

    print("\n==================== GLOBAL MODEL (Stacker) ====================")
    X_meta = np.vstack(meta_train_X_list)
    y_meta = np.concatenate(meta_train_y_list)

    stacker = LogisticRegression(solver="liblinear", max_iter=2000, random_state=SEED)
    stacker.fit(X_meta, y_meta)
    p_meta = stacker.predict_proba(X_meta)[:, 1]
    tau_global = best_tau_by_mcc(y_meta, p_meta, step=TAU_STEP)

    for dname, (Xh_meta, yh) in meta_holdout_by_ds.items():
        p_h = stacker.predict_proba(Xh_meta)[:, 1]
        met, curve = compute_metrics(yh, p_h, tau_global)
        global_rows.append({"dataset": dname, "model": "GLOBAL(Stacker)", **met})
        holdout_curves[f"{dname}::GLOBAL(Stacker)"] = curve

    main_df = pd.DataFrame(main_rows)
    hold_df = pd.DataFrame(holdout_rows)
    hp_df = pd.DataFrame(hp_rows)
    pr_df = pd.DataFrame(pr_rows)
    glob_df = pd.DataFrame(global_rows)

    v_intake_df = pd.DataFrame(v_intake)
    v_feat_df = pd.DataFrame(v_feat)
    v_enc_df = pd.DataFrame(v_enc)
    v_cv_df = pd.DataFrame(v_cv)
    v_transfer_df = pd.DataFrame(v_transfer)
    v_global_df = pd.DataFrame(v_global)

    _display_df(main_df, "1) Nested-CV Metrics (Outer mean±std) — per dataset & model")
    _display_df(pr_df, "2) Precision/Recall + τ (Outer aggregates)")
    _display_df(hp_df, "3) Best Hyperparams (best outer-fold config)")
    _display_df(hold_df, "4) Holdout Metrics (train-split tuned → holdout) — LOCAL models")
    _display_df(glob_df, f"5) Holdout Metrics — GLOBAL model (τ_global={tau_global:.2f})")

    print("\n--- Validation Tables (Process-only; 5–6 cols, <=5 rows) ---")
    _display_df(v_intake_df, "V1) Intake / Holdout Split Validation")
    _display_df(v_feat_df, "V2) Train-only Feature Drop Validation")
    _display_df(v_enc_df, "V3) Encoding / Tokenization Validation (train-only)")
    _display_df(v_cv_df, "V4) Nested CV Scheme & Search Budget Validation")
    _display_df(v_transfer_df, "V5) Hyperparam Transfer Validation")
    _display_df(v_global_df, "V6) Global Stacker Dataflow Validation")

    os.makedirs("artifacts", exist_ok=True)
    with open(os.path.join("artifacts", "ckd_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "nested_cv": main_rows,
                "holdout_local": holdout_rows,
                "holdout_global": global_rows,
                "transfer_bank": transfer_bank,
            },
            f,
            indent=2,
        )
    with open(os.path.join("artifacts", "ckd_roc_curves.json"), "w", encoding="utf-8") as f:
        json.dump(holdout_curves, f, indent=2)

    print("\nSaved artifacts:")
    print(" - artifacts/ckd_metrics.json")
    print(" - artifacts/ckd_roc_curves.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
