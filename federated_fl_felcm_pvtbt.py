import os
import time
import math
import random
import sys
import subprocess
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, roc_auc_score,
    classification_report, roc_curve, precision_recall_curve, average_precision_score
)

# -------------------------
# Ensure timm is available
# -------------------------
try:
    import timm
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "timm"])
    import timm

from torchvision import transforms

try:
    from IPython.display import display
except Exception:
    display = print

# -------------------------
# Reproducibility + Device
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

plt.rcParams["figure.dpi"] = 140

print("=" * 92)
print("TRUE FL + GA-FELCM + PVTv2-B2 (MULTICLASS) — 4 Clients, 2 Datasets, NO AUGMENTATION")
print("=" * 92)
print(f"DEVICE: {DEVICE} | torch={torch.__version__}")
print("=" * 92)

# -------------------------
# Configuration
# -------------------------
CFG = {
    # Kaggle dataset base folders (the dataset slugs you added to the notebook)
    "ds1_base": "/kaggle/input/pmram-bangladeshi-brain-cancer-mri-dataset",  # Dataset-1
    "ds2_base": "/kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection",  # Dataset-2

    # image/model
    "img_size": 224 if torch.cuda.is_available() else 160,
    "batch_size": 24 if torch.cuda.is_available() else 12,
    "num_workers": 2,

    # FL
    "clients": 4,
    "rounds": 12,
    "local_epochs": 3,
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "warmup_epochs": 1,
    "label_smoothing": 0.08,
    "grad_clip": 1.0,
    "fedprox_mu": 0.01,

    # split (per dataset)
    "global_val_frac": 0.15,
    "test_frac": 0.15,
    "client_val_frac": 0.12,
    "client_tune_frac": 0.12,
    "min_per_class_per_client": 5,

    # Non-IID Dirichlet (within each dataset)
    "dirichlet_alpha": 0.35,

    # GA / preprocessing
    "use_preprocessing": False,
    "use_ga": False,
    "ga_pop": 10,
    "ga_gens": 5,
    "ga_elites": 3,
    "elite_pool_max": 15,

    # augmentation
    "use_augmentation": False,

    # model adapter/head
    "adapter_dim": 256,
    "adapter_heads": 4,
    "adapter_dropout": 0.1,
    "head_dropout": 0.3,

    # optional late unfreeze
    "unfreeze_after_round": 3,
    "unfreeze_lr_mult": 0.10,
    "unfreeze_tail_frac": 0.17,

    # validation controls
    "quick_hash_subset_per_split": 400,
    "preproc_val_sample_n": 600,
    "before_after_n": 12,
}

OUTDIR = "/kaggle/working"
os.makedirs(OUTDIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTDIR, "FL_FELCM_PVTBT_full_checkpoint.pth")
CSV_PATH = os.path.join(OUTDIR, "ALL_OUTPUTS_AND_METRICS.csv")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# ImageNet normalization (no pooled stats)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

# ============================================================
# Helper: collect all tables into ONE CSV (long format)
# ============================================================
ALL_ROWS = []

def add_table_to_csv(df, table_name):
    df2 = df.copy()
    df2.insert(0, "table_name", table_name)
    for _, row in df2.iterrows():
        ALL_ROWS.append(row.to_dict())

def print_table(df, title):
    print("\n" + "-" * 92)
    print(title)
    print("-" * 92)
    display(df)

# ============================================================
# 1) DATASET DISCOVERY (NO CENTRAL MERGE)
# ============================================================
print("\n" + "=" * 92)
print("STEP 1: LOCATE DATASET ROOTS (NO MERGE)")
print("=" * 92)

REQ1 = {"512Glioma", "512Meningioma", "512Normal", "512Pituitary"}
REQ2 = {"glioma", "meningioma", "notumor", "pituitary"}


def norm_label(name: str):
    s = str(name).strip().lower()
    if "glioma" in s:
        return "glioma"
    if "meningioma" in s:
        return "meningioma"
    if "pituitary" in s:
        return "pituitary"
    if "normal" in s or "no_tumor" in s or "no tumor" in s or "notumor" in s:
        return "notumor"
    return None


def find_root_with_required_class_dirs(base_dir, required_set, prefer_raw=True):
    candidates = []
    for root, dirs, _ in os.walk(base_dir):
        dset = set(dirs)
        if required_set.issubset(dset):
            candidates.append(root)

    if not candidates:
        return None

    def score(p):
        pl = p.lower()
        sc = 0
        if prefer_raw:
            if "raw data" in pl:
                sc += 7
            if os.path.basename(p).lower() == "raw":
                sc += 7
            if "/raw/" in pl or "\\raw\\" in pl:
                sc += 3
            if "augmented" in pl:
                sc -= 20
        sc -= 0.0001 * len(p)
        return sc

    return max(candidates, key=score)


def list_images_under_class_root(class_root, class_dir_name):
    class_dir = os.path.join(class_root, class_dir_name)
    out = []
    for r, _, files in os.walk(class_dir):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                out.append(os.path.join(r, fn))
    return out


# Always detect roots so checkpoint has correct paths
DS1_ROOT = find_root_with_required_class_dirs(CFG["ds1_base"], REQ1, prefer_raw=True)
if DS1_ROOT is None:
    raise RuntimeError(
        f"Could not locate RAW class root for Dataset-1 under: {CFG['ds1_base']}\n"
        f"Expected class dirs: {sorted(list(REQ1))}"
    )
print(f"Dataset-1 RAW root detected:\n  {DS1_ROOT}")

DS2_ROOT = find_root_with_required_class_dirs(CFG["ds2_base"], REQ2, prefer_raw=False)
if DS2_ROOT is None:
    raise RuntimeError(
        f"Could not locate class root for Dataset-2 under: {CFG['ds2_base']}\n"
        f"Expected class dirs: {sorted(list(REQ2))}"
    )
print(f"Dataset-2 root detected:\n  {DS2_ROOT}")


def build_df_from_root(ds_root, class_dirs, source_name):
    rows = []
    for c in class_dirs:
        lab = norm_label(c)
        imgs = list_images_under_class_root(ds_root, c)
        print(f"{source_name}: {c} -> {lab} | {len(imgs)} images")
        for p in imgs:
            rows.append({"path": p, "label": lab, "source": source_name})
    dfm = pd.DataFrame(rows).dropna().reset_index(drop=True)
    dfm["path"] = dfm["path"].astype(str)
    dfm["label"] = dfm["label"].astype(str)
    dfm["source"] = dfm["source"].astype(str)
    dfm = dfm.drop_duplicates(subset=["path"]).reset_index(drop=True)
    dfm["filename"] = dfm["path"].apply(lambda x: os.path.basename(x))
    return dfm


print("\n" + "-" * 92)
print("Building Dataset-1 (RAW only)")
df1 = build_df_from_root(
    DS1_ROOT,
    ["512Glioma", "512Meningioma", "512Normal", "512Pituitary"],
    "ds1_raw",
)
print("Building Dataset-2 (preprocessed)")
df2 = build_df_from_root(
    DS2_ROOT,
    ["glioma", "meningioma", "notumor", "pituitary"],
    "ds2",
)

labels = ["glioma", "meningioma", "notumor", "pituitary"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

for df_ in (df1, df2):
    df_["label"] = df_["label"].astype(str).str.strip().str.lower()
    df_ = df_[df_["label"].isin(set(labels))].reset_index(drop=True)


def enforce_labels(df_):
    df_ = df_.copy()
    df_["label"] = df_["label"].astype(str).str.strip().str.lower()
    df_ = df_[df_["label"].isin(set(labels))].reset_index(drop=True)
    df_["y"] = df_["label"].map(label2id).astype(int)
    return df_


df1 = enforce_labels(df1)
df2 = enforce_labels(df2)

NUM_CLASSES = len(labels)

print("\n" + "-" * 92)
print(f"Dataset-1 images: {len(df1)}")
print(df1["label"].value_counts().reindex(labels, fill_value=0))
print(f"Dataset-2 images: {len(df2)}")
print(df2["label"].value_counts().reindex(labels, fill_value=0))
print("-" * 92)

# ============================================================
# 2) Train/Val/Test Split per Dataset (STRATIFIED)
# ============================================================
print("\n" + "=" * 92)
print("STEP 2: TRAIN/VAL/TEST SPLIT (PER DATASET)")
print("=" * 92)


def split_dataset(df_):
    train_df, temp_df = train_test_split(
        df_,
        test_size=(CFG["global_val_frac"] + CFG["test_frac"]),
        stratify=df_["y"],
        random_state=SEED,
    )
    val_rel = CFG["global_val_frac"] / (CFG["global_val_frac"] + CFG["test_frac"])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_rel),
        stratify=temp_df["y"],
        random_state=SEED,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


train1, val1, test1 = split_dataset(df1)
train2, val2, test2 = split_dataset(df2)

print(f"DS1 TRAIN: {len(train1)} | VAL: {len(val1)} | TEST: {len(test1)}")
print(f"DS2 TRAIN: {len(train2)} | VAL: {len(val2)} | TEST: {len(test2)}")

# ============================================================
# 2.5) Leakage/Sanity Checks (per dataset)
# ============================================================
print("\n" + "=" * 92)
print("STEP 2.5: SANITY / LEAKAGE CHECKS (PER DATASET)")
print("=" * 92)


def split_overlap_checks(train_df, val_df, test_df):
    tr = set(train_df["path"].tolist())
    va = set(val_df["path"].tolist())
    te = set(test_df["path"].tolist())
    checks = {
        "path_overlap_train_val": len(tr.intersection(va)),
        "path_overlap_train_test": len(tr.intersection(te)),
        "path_overlap_val_test": len(va.intersection(te)),
        "unique_paths_train": len(tr),
        "unique_paths_val": len(va),
        "unique_paths_test": len(te),
    }
    trf = set(train_df["filename"].tolist())
    vaf = set(val_df["filename"].tolist())
    tef = set(test_df["filename"].tolist())
    checks.update(
        {
            "filename_overlap_train_val": len(trf.intersection(vaf)),
            "filename_overlap_train_test": len(trf.intersection(tef)),
            "filename_overlap_val_test": len(vaf.intersection(tef)),
        }
    )
    return checks


def md5_file(path, max_bytes=2_000_000):
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            h.update(f.read(max_bytes))
        return h.hexdigest()
    except Exception:
        return None


def quick_hash_subset(frame, n=300):
    n = min(n, len(frame))
    if n <= 0:
        return set()
    idx = np.random.choice(len(frame), size=n, replace=False)
    hashes = []
    for i in idx:
        hv = md5_file(frame.iloc[i]["path"])
        if hv is not None:
            hashes.append(hv)
    return set(hashes)


def leakage_report(name, tr, va, te):
    over = split_overlap_checks(tr, va, te)
    leak_df = pd.DataFrame([over])

    n_hash = int(CFG["quick_hash_subset_per_split"])
    trh = quick_hash_subset(tr, n_hash)
    vah = quick_hash_subset(va, n_hash)
    teh = quick_hash_subset(te, n_hash)

    hash_over = {
        "subset_hash_train_val": len(trh.intersection(vah)),
        "subset_hash_train_test": len(trh.intersection(teh)),
        "subset_hash_val_test": len(vah.intersection(teh)),
        "subset_hash_n_train": len(trh),
        "subset_hash_n_val": len(vah),
        "subset_hash_n_test": len(teh),
    }
    leak_df = pd.concat([leak_df, pd.DataFrame([hash_over])], axis=1)
    print_table(leak_df, f"Leakage / Sanity Summary — {name}")
    add_table_to_csv(leak_df, f"leakage_sanity_{name}")


leakage_report("ds1", train1, val1, test1)
leakage_report("ds2", train2, val2, test2)

# ============================================================
# 3) Non-IID Client Partitioning (within dataset)
# ============================================================
print("\n" + "=" * 92)
print("STEP 3: NON-IID CLIENT PARTITIONING (2 clients per dataset)")
print("=" * 92)


def make_clients_non_iid(train_df, n_clients, num_classes, min_per_class=5, alpha=0.35):
    y = train_df["y"].values
    idx_by_class = {c: np.where(y == c)[0].tolist() for c in range(num_classes)}
    for c in idx_by_class:
        random.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(n_clients)]

    # ensure some per-class coverage
    for c in range(num_classes):
        idxs = idx_by_class[c]
        feasible = min(min_per_class, max(1, len(idxs) // n_clients))
        for k in range(n_clients):
            take = idxs[:feasible]
            idxs = idxs[feasible:]
            client_indices[k].extend(take)
        idx_by_class[c] = idxs

    # Dirichlet for remaining
    for c in range(num_classes):
        idxs = idx_by_class[c]
        if len(idxs) == 0:
            continue
        props = np.random.dirichlet([alpha] * n_clients)
        counts = (props * len(idxs)).astype(int)
        diff = len(idxs) - counts.sum()
        counts[np.argmax(props)] += diff

        start = 0
        for k in range(n_clients):
            client_indices[k].extend(idxs[start : start + counts[k]])
            start += counts[k]

    for k in range(n_clients):
        random.shuffle(client_indices[k])
    return client_indices


def robust_client_splits(train_df, indices, val_frac, tune_frac):
    idxs = np.array(indices, dtype=int)
    if len(idxs) < 3:
        return idxs.tolist(), idxs.tolist(), idxs.tolist()

    yk = train_df.loc[idxs, "y"].values
    # First split off tune
    if len(np.unique(yk)) < 2 or len(idxs) < 20:
        n_tune = max(1, int(round(len(idxs) * tune_frac)))
        n_tune = min(n_tune, max(1, len(idxs) - 2))
        tune_idx = idxs[:n_tune]
        rem_idx = idxs[n_tune:]
    else:
        rem_idx, tune_idx = train_test_split(
            idxs,
            test_size=tune_frac,
            stratify=yk,
            random_state=SEED,
        )
    # Split remaining into train/val
    if len(rem_idx) < 2:
        return rem_idx.tolist(), tune_idx.tolist(), rem_idx.tolist()

    yk2 = train_df.loc[rem_idx, "y"].values
    if len(np.unique(yk2)) < 2 or len(rem_idx) < 12:
        n_val = max(1, int(round(len(rem_idx) * val_frac)))
        n_val = min(n_val, max(1, len(rem_idx) - 1))
        val_idx = rem_idx[:n_val]
        train_idx = rem_idx[n_val:]
    else:
        train_idx, val_idx = train_test_split(
            rem_idx,
            test_size=val_frac,
            stratify=yk2,
            random_state=SEED,
        )
    if len(train_idx) == 0:
        train_idx = val_idx[:]
    if len(val_idx) == 0:
        val_idx = train_idx[:1]
    return train_idx.tolist(), tune_idx.tolist(), val_idx.tolist()


# 2 clients for DS1, 2 clients for DS2
client_indices_ds1 = make_clients_non_iid(
    train1,
    n_clients=2,
    num_classes=NUM_CLASSES,
    min_per_class=CFG["min_per_class_per_client"],
    alpha=CFG["dirichlet_alpha"],
)
client_indices_ds2 = make_clients_non_iid(
    train2,
    n_clients=2,
    num_classes=NUM_CLASSES,
    min_per_class=CFG["min_per_class_per_client"],
    alpha=CFG["dirichlet_alpha"],
)

# Build client splits and keep a consistent client id mapping
client_splits = []
client_meta = []

for k in range(2):
    tr, tune, va = robust_client_splits(
        train1,
        client_indices_ds1[k],
        CFG["client_val_frac"],
        CFG["client_tune_frac"],
    )
    client_splits.append(("ds1", k, tr, tune, va))
    client_meta.append({"client": f"client_{k}", "dataset": "ds1"})
    print(f"DS1 Client {k}: {len(tr)} train, {len(tune)} tune, {len(va)} val")

for k in range(2):
    tr, tune, va = robust_client_splits(
        train2,
        client_indices_ds2[k],
        CFG["client_val_frac"],
        CFG["client_tune_frac"],
    )
    cid = k + 2
    client_splits.append(("ds2", k, tr, tune, va))
    client_meta.append({"client": f"client_{cid}", "dataset": "ds2"})
    print(f"DS2 Client {k} (global id {cid}): {len(tr)} train, {len(tune)} tune, {len(va)} val")

# Partition tests per dataset into 2 clients each (federated test)
client_test_splits = []
for ds_name, test_df in [("ds1", test1), ("ds2", test2)]:
    idxs = list(range(len(test_df)))
    random.shuffle(idxs)
    split = np.array_split(idxs, 2)
    for k in range(2):
        client_test_splits.append((ds_name, k, split[k].tolist()))

# Client class distribution summary

def client_distribution_table():
    dist_rows = []
    for idx, (ds_name, local_id, tr_idx, tune_idx, val_idx) in enumerate(client_splits):
        df_src = train1 if ds_name == "ds1" else train2
        counts = df_src.loc[tr_idx, "label"].value_counts().reindex(labels, fill_value=0)
        row = {
            "client": f"client_{idx}",
            "dataset": ds_name,
            "total_train": len(tr_idx),
            "total_tune": len(tune_idx),
            "total_val": len(val_idx),
        }
        row.update({lab: int(counts[lab]) for lab in labels})
        dist_rows.append(row)
    return pd.DataFrame(dist_rows)


dist_df = client_distribution_table()
print_table(dist_df, "Client class distribution (Non-IID, per dataset)")
add_table_to_csv(dist_df, "client_distribution")

# ============================================================
# 4) Data pipeline (NO AUGMENTATION) + ImageNet Norm
# ============================================================
print("\n" + "=" * 92)
print("STEP 4: DATA LOADERS (NO AUGMENTATION) + IMAGENET NORM")
print("=" * 92)


def load_rgb(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (CFG["img_size"], CFG["img_size"]), (128, 128, 128))


# Evaluation transforms
EVAL_TFMS = transforms.Compose(
    [
        transforms.Resize((CFG["img_size"], CFG["img_size"])),
        transforms.ToTensor(),
    ]
)

# Training transforms (augmentation ON)
if CFG["use_augmentation"]:
    TRAIN_TFMS = transforms.Compose(
        [
            transforms.Resize((CFG["img_size"], CFG["img_size"])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
        ]
    )
else:
    TRAIN_TFMS = EVAL_TFMS


class MRIDataset(Dataset):
    def __init__(self, frame, indices=None, tfms=None):
        self.df = frame
        self.indices = indices if indices is not None else list(range(len(frame)))
        self.tfms = tfms

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        row = self.df.iloc[j]
        img = load_rgb(row["path"])
        x = self.tfms(img) if self.tfms is not None else transforms.ToTensor()(img)
        y = int(row["y"])
        return x, y, row["path"]


def make_weighted_sampler(frame, indices, num_classes):
    if len(indices) == 0:
        return None
    ys = frame.loc[indices, "y"].values
    class_counts = np.bincount(ys, minlength=num_classes)
    class_weights = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = class_weights[ys]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def make_loader(frame, indices, bs, tfms, shuffle=False, sampler=None):
    ds = MRIDataset(frame, indices=indices, tfms=tfms)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=CFG["num_workers"],
        pin_memory=True,
        drop_last=False,
        persistent_workers=(CFG["num_workers"] > 0),
    )


# Build client loaders (train/tune/val)
client_loaders = []
for idx, (ds_name, local_id, tr_idx, tune_idx, val_idx) in enumerate(client_splits):
    df_src = train1 if ds_name == "ds1" else train2
    sampler = make_weighted_sampler(df_src, tr_idx, NUM_CLASSES)
    tr_loader = make_loader(
        df_src,
        tr_idx,
        CFG["batch_size"],
        TRAIN_TFMS,
        shuffle=(sampler is None),
        sampler=sampler,
    )
    tune_loader = make_loader(
        df_src,
        tune_idx if len(tune_idx) > 0 else tr_idx[: max(1, len(tr_idx))],
        CFG["batch_size"],
        EVAL_TFMS,
        shuffle=True,
    )
    val_loader = make_loader(
        df_src,
        val_idx if len(val_idx) > 0 else tr_idx[: max(1, min(len(tr_idx), CFG["batch_size"]))],
        CFG["batch_size"],
        EVAL_TFMS,
        shuffle=False,
    )
    client_loaders.append((tr_loader, tune_loader, val_loader))

# Test loaders per client
client_test_loaders = []
for i, (ds_name, local_id, test_idx) in enumerate(client_test_splits):
    df_src = test1 if ds_name == "ds1" else test2
    t_loader = make_loader(df_src, test_idx, CFG["batch_size"], EVAL_TFMS, shuffle=False)
    client_test_loaders.append((ds_name, local_id, t_loader))

print(f"Augmentation: {'ON ✅' if CFG['use_augmentation'] else 'OFF ✅ (train transforms == eval transforms)'}")
print(f"Preprocessing: {'ON ✅' if CFG['use_preprocessing'] else 'OFF ✅ (identity)'}")

# Optional visualization: before vs after augmentation (train transforms)
if CFG["use_augmentation"]:
    print("\n" + "-" * 92)
    print("AUGMENTATION VISUAL CHECK (Before vs After) — TRAIN TRANSFORMS")
    print("-" * 92)
    sample_frame = train1 if len(train1) > 0 else train2
    sample_n = min(8, len(sample_frame))
    if sample_n > 0:
        idxs = np.random.choice(len(sample_frame), size=sample_n, replace=False)
        raw_ds = MRIDataset(sample_frame, indices=idxs.tolist(), tfms=EVAL_TFMS)
        aug_ds = MRIDataset(sample_frame, indices=idxs.tolist(), tfms=TRAIN_TFMS)

        raws, augs = [], []
        for i in range(sample_n):
            x_raw, _, _ = raw_ds[i]
            x_aug, _, _ = aug_ds[i]
            raws.append(x_raw)
            augs.append(x_aug)

        fig = plt.figure(figsize=(min(16, 2.2 * sample_n), 5))
        for i in range(sample_n):
            ax1 = plt.subplot(2, sample_n, i + 1)
            ax1.imshow(raws[i].permute(1, 2, 0).numpy())
            ax1.set_title("Before Aug", fontsize=8)
            ax1.axis("off")

            ax2 = plt.subplot(2, sample_n, sample_n + i + 1)
            ax2.imshow(augs[i].permute(1, 2, 0).numpy())
            ax2.set_title("After Aug", fontsize=8)
            ax2.axis("off")

        plt.suptitle("Training Augmentation: Before vs After", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()

# ============================================================
# 5) Enhanced FELCM + theta fullforms
# ============================================================
print("\n" + "=" * 92)
print("STEP 5: GA-TUNED ENHANCED FELCM PREPROCESSOR")
print("=" * 92)
if not CFG["use_preprocessing"]:
    print("Preprocessing disabled → GA/FELCM will be skipped (identity).")

THETA_FULLFORMS = {
    "gamma": "Power transform exponent (γ)",
    "alpha": "Local contrast weight (α)",
    "beta": "Contrast sharpness (β)",
    "tau": "Robust clipping threshold (τ)",
    "k": "Blur kernel size (k) for local contrast map",
    "sh": "Sharpen strength (sh)",
    "dn": "Denoise strength (dn)",
}


class EnhancedFELCM(nn.Module):
    def __init__(self, gamma=1.0, alpha=0.35, beta=6.0, tau=2.5, blur_k=7, sharpen=0.0, denoise=0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau = float(tau)
        self.blur_k = int(blur_k)
        self.sharpen = float(sharpen)
        self.denoise = float(denoise)

        lap = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        self.register_buffer("lap", lap.view(1, 1, 3, 3))

        sharp = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32)
        self.register_buffer("sharp_kernel", sharp.view(1, 1, 3, 3))

    def forward(self, x):
        eps = 1e-6
        B, C, H, W = x.shape

        if self.denoise > 0:
            k = 3
            x_blur = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k, 1)
            x = x * (1 - self.denoise) + x_blur * self.denoise

        mu = x.mean(dim=(2, 3), keepdim=True)
        sd = x.std(dim=(2, 3), keepdim=True).clamp_min(eps)
        x0 = (x - mu) / sd
        x0 = x0.clamp(-self.tau, self.tau)

        x1 = torch.sign(x0) * torch.pow(torch.abs(x0).clamp_min(eps), self.gamma)

        gray = x1.mean(dim=1, keepdim=True)
        lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.lap)
        mag = lap.abs()

        k = self.blur_k if self.blur_k % 2 == 1 else self.blur_k + 1
        pad = k // 2
        blur = F.avg_pool2d(F.pad(mag, (pad, pad, pad, pad), mode="reflect"), k, 1)
        C_map = mag / (blur + eps)

        x2 = x1 + self.alpha * torch.tanh(self.beta * C_map)

        if self.sharpen > 0:
            outs = []
            for c in range(C):
                x_c = x2[:, c : c + 1, :, :]
                x_sharp = F.conv2d(F.pad(x_c, (1, 1, 1, 1), mode="reflect"), self.sharp_kernel)
                outs.append(x_c * (1 - self.sharpen) + x_sharp * self.sharpen)
            x2 = torch.cat(outs, dim=1)

        mn = x2.amin(dim=(2, 3), keepdim=True)
        mx = x2.amax(dim=(2, 3), keepdim=True)
        x3 = (x2 - mn) / (mx - mn + eps)
        return x3.clamp(0, 1)


def theta_to_module(theta):
    return EnhancedFELCM(*theta)


def random_theta():
    gamma = random.uniform(0.7, 1.4)
    alpha = random.uniform(0.15, 0.55)
    beta = random.uniform(3.0, 9.0)
    tau = random.uniform(1.8, 3.2)
    blur_k = random.choice([3, 5, 7])
    sharpen = random.uniform(0.0, 0.25)
    denoise = random.uniform(0.0, 0.2)
    return (gamma, alpha, beta, tau, blur_k, sharpen, denoise)


def mutate(theta, p=0.8):
    if random.random() > p:
        return theta
    g, a, b, t, k, sh, dn = theta
    g = float(np.clip(g + np.random.normal(0, 0.06), 0.6, 1.5))
    a = float(np.clip(a + np.random.normal(0, 0.05), 0.08, 0.7))
    b = float(np.clip(b + np.random.normal(0, 0.5), 2.0, 11.0))
    t = float(np.clip(t + np.random.normal(0, 0.2), 1.5, 3.8))
    if random.random() < 0.3:
        k = random.choice([3, 5, 7])
    sh = float(np.clip(sh + np.random.normal(0, 0.04), 0.0, 0.35))
    dn = float(np.clip(dn + np.random.normal(0, 0.03), 0.0, 0.3))
    return (g, a, b, t, int(k), sh, dn)


def crossover(t1, t2):
    return tuple(random.choice([a, b]) for a, b in zip(t1, t2))


def theta_str(th):
    if th is None:
        return "None"
    g, a, b, t, k, sh, dn = th
    return f"(γ={g:.2f}, α={a:.2f}, β={b:.1f}, τ={t:.1f}, k={k}, sh={sh:.2f}, dn={dn:.2f})"


IDENTITY_PRE = nn.Identity().to(DEVICE)

# ============================================================
# 6) Enhanced model (PVTv2-B2 + multi-scale fusion + adapter)
# ============================================================
print("\n" + "=" * 92)
print("STEP 6: MODEL (PVTv2-B2 + MULTI-SCALE FUSION)")
print("=" * 92)

BACKBONE_NAME = "pvt_v2_b2"


class MultiHeadAttentionAdapter(nn.Module):
    def __init__(self, dim, num_heads=4, bottleneck=256, dropout=0.1):
        super().__init__()
        bottleneck = min(bottleneck, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dim),
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        x2 = x.unsqueeze(1)
        x_norm = self.norm1(x2)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x2 = x2 + self.scale * attn_out
        x2 = x2 + self.scale * self.ffn(self.norm2(x2))
        return x2.squeeze(1)


class TokenAttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, 1)

    def forward(self, x):
        # x: [B, N, C]
        attn = torch.softmax(self.query(x).squeeze(-1), dim=1)
        return (x * attn.unsqueeze(-1)).sum(dim=1)


class MultiScaleFusionHead(nn.Module):
    def __init__(self, in_channels: List[int], out_dim: int, num_classes: int, dropout=0.3):
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_dim, kernel_size=1, bias=False),
                nn.GroupNorm(8, out_dim),
                nn.GELU(),
            )
            for c in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
        )
        self.pool = TokenAttentionPooling(out_dim)
        self.adapter = MultiHeadAttentionAdapter(out_dim, num_heads=4, bottleneck=out_dim, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, max(64, out_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(max(64, out_dim // 2), num_classes),
        )

    def forward(self, feats):
        # feats: list of feature maps [B, C, H, W]
        proj_feats = [p(f) for p, f in zip(self.proj, feats)]
        x = proj_feats[-1]
        for f in reversed(proj_feats[:-1]):
            x = F.interpolate(x, size=f.shape[-2:], mode="bilinear", align_corners=False)
            x = x + f
        x = self.fuse(x)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        pooled = self.pool(tokens)
        pooled = self.adapter(pooled)
        return self.classifier(pooled)


class PVTv2B2_MultiScale(nn.Module):
    def __init__(self, num_classes, pretrained=True, head_dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        in_channels = self.backbone.feature_info.channels()
        out_dim = max(256, in_channels[-1] // 2)
        self.head = MultiScaleFusionHead(in_channels, out_dim, num_classes, dropout=head_dropout)
        self._init_weights()

    def _init_weights(self):
        for p in self.head.parameters():
            if p.dim() > 1:
                nn.init.trunc_normal_(p, std=0.02)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def set_trainable_for_round(model, rnd):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if not n.startswith("backbone."):
            p.requires_grad = True
    if rnd >= CFG["unfreeze_after_round"]:
        params = list(model.backbone.parameters())
        if len(params) > 0:
            tail_n = max(1, int(len(params) * CFG["unfreeze_tail_frac"]))
            for p in params[-tail_n:]:
                p.requires_grad = True


def make_optimizer(model):
    head_params, bb_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone."):
            bb_params.append(p)
        else:
            head_params.append(p)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": CFG["lr"]})
    if bb_params:
        groups.append({"params": bb_params, "lr": CFG["lr"] * CFG["unfreeze_lr_mult"]})
    return torch.optim.AdamW(groups, weight_decay=CFG["weight_decay"])

# ============================================================
# 7) GA Fitness (robust)
# ============================================================
print("\n" + "=" * 92)
print("STEP 7: GA FITNESS (ROBUST)")
print("=" * 92)


@torch.no_grad()
def enhanced_separability_score(emb, y):
    eps = 1e-6
    y = y.long()
    classes = torch.unique(y)
    if len(classes) < 2:
        return 0.0

    centroids = []
    within_vars = []
    sizes = []

    for c in classes:
        mask = y == c
        e = emb[mask]
        if e.size(0) < 2:
            continue
        mu = e.mean(dim=0)
        var = (e - mu).pow(2).sum(dim=1).mean().item()
        centroids.append(mu)
        within_vars.append(var)
        sizes.append(e.size(0))

    if len(centroids) < 2:
        return 0.0

    centroids = torch.stack(centroids, dim=0)
    global_mean = centroids.mean(dim=0)
    between = sum(n * (c - global_mean).pow(2).sum().item() for c, n in zip(centroids, sizes))
    within = float(np.mean(within_vars)) if within_vars else eps
    return float(between / (within + eps))


@torch.no_grad()
def ga_fitness(theta, backbone_frozen, batch_x, batch_y, use_separability=True):
    pre = theta_to_module(theta).to(DEVICE)
    x = batch_x.to(DEVICE)
    y = batch_y.to(DEVICE)

    x_p = pre(x)
    gray = x_p.mean(dim=1, keepdim=True)
    lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), pre.lap).abs()
    contrast = float(lap.mean().item())
    dyn_range = float((x_p.max() - x_p.min()).item())

    x_n = (x_p - IMAGENET_MEAN) / IMAGENET_STD

    sep = 0.0
    if use_separability:
        emb = backbone_frozen(x_n)
        if isinstance(emb, (list, tuple)):
            emb = emb[-1]
            emb = emb.mean(dim=(2, 3))
        sep = enhanced_separability_score(emb, y)

    g, a, b, t, k, sh, dn = theta
    cost = (
        0.03 * abs(g - 1.0)
        + 0.05 * a
        + 0.01 * (b / 10.0)
        + 0.02 * abs(t - 2.5)
        + 0.02 * sh
        + 0.02 * dn
    )

    if use_separability:
        return 0.35 * contrast + 0.15 * dyn_range + 1.35 * sep - 0.5 * cost
    return 0.60 * contrast + 0.35 * dyn_range - 0.5 * cost


def _safe_first_batch(dl):
    try:
        it = iter(dl)
        bx, by, _ = next(it)
        return bx, by
    except Exception:
        return None, None


def run_ga_for_client(backbone_frozen, dl_for_eval, elite_pool, use_separability=True):
    bx, by = _safe_first_batch(dl_for_eval)
    if bx is None:
        return None, [], 0.0

    pop = []
    if elite_pool:
        pop.extend(elite_pool[: min(len(elite_pool), CFG["ga_pop"] // 2)])
    while len(pop) < CFG["ga_pop"]:
        pop.append(random_theta())

    bx = bx[: CFG["batch_size"]].contiguous()
    by = by[: CFG["batch_size"]].contiguous()

    for _ in range(CFG["ga_gens"]):
        scored = [(ga_fitness(th, backbone_frozen, bx, by, use_separability), th) for th in pop]
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = [th for _, th in scored[: CFG["ga_elites"]]]

        new_pop = elites[:]
        while len(new_pop) < CFG["ga_pop"]:
            p1, p2 = random.sample(elites + pop[: max(2, CFG["ga_pop"] // 2)], 2)
            child = crossover(p1, p2)
            child = mutate(child, p=0.75)
            new_pop.append(child)
        pop = new_pop

    scored = [(ga_fitness(th, backbone_frozen, bx, by, use_separability), th) for th in pop]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_theta = scored[0][1]
    best_fit = float(scored[0][0])
    top = [th for _, th in scored[: CFG["ga_elites"]]]
    return best_theta, top, best_fit

# ============================================================
# 8) Train / Eval utilities (FULL metrics)
# ============================================================
print("\n" + "=" * 92)
print("STEP 8: TRAIN / EVAL UTILITIES (FULL METRICS)")
print("=" * 92)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def _auc_metrics(y_true, p_pred, num_classes):
    out = {}
    try:
        if num_classes == 2:
            out["auc_roc"] = float(roc_auc_score(y_true, p_pred[:, 1]))
        else:
            out["auc_roc_macro_ovr"] = float(roc_auc_score(y_true, p_pred, multi_class="ovr", average="macro"))
            for c in range(num_classes):
                yc = (y_true == c).astype(int)
                if yc.sum() > 0 and yc.sum() < len(yc):
                    out[f"auc_class_{c}"] = float(roc_auc_score(yc, p_pred[:, c]))
    except Exception:
        pass
    return out


@torch.no_grad()
def evaluate_full(model, loader, preproc_module):
    t0 = time.time()
    model.eval()
    preproc_module.eval()

    all_y, all_p, all_loss = [], [], []
    has_any = False

    for x, y, _ in loader:
        has_any = True
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        x_p = preproc_module(x)
        x_n = (x_p - IMAGENET_MEAN) / IMAGENET_STD
        logits = model(x_n)

        probs = torch.softmax(logits, dim=1)
        loss = F.cross_entropy(logits, y)

        all_loss.append(float(loss.item()))
        all_y.append(y.cpu().numpy())
        all_p.append(probs.cpu().numpy())

    if not has_any:
        met = {
            "loss_ce": np.nan,
            "acc": np.nan,
            "precision_macro": np.nan,
            "recall_macro": np.nan,
            "f1_macro": np.nan,
            "precision_weighted": np.nan,
            "recall_weighted": np.nan,
            "f1_weighted": np.nan,
            "log_loss": np.nan,
            "eval_time_s": float(time.time() - t0),
        }
        return met, np.array([]), np.array([])

    y_true = np.concatenate(all_y)
    p_pred = np.concatenate(all_p)
    y_hat = np.argmax(p_pred, axis=1)

    met = {
        "loss_ce": float(np.mean(all_loss)),
        "acc": float(accuracy_score(y_true, y_hat)),
        "precision_macro": float(precision_score(y_true, y_hat, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_hat, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_hat, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_hat, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_hat, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, p_pred, labels=list(range(NUM_CLASSES)))),
        "eval_time_s": float(time.time() - t0),
    }
    met.update(_auc_metrics(y_true, p_pred, NUM_CLASSES))
    return met, y_true, p_pred


def fedprox_term(local_model, global_model):
    loss = 0.0
    for p_local, p_global in zip(local_model.parameters(), global_model.parameters()):
        loss += ((p_local - p_global.detach()) ** 2).sum()
    return loss


def train_one_epoch(model, loader, optimizer, preproc_module, criterion, global_model=None, scheduler=None, scaler=None, grad_clip=1.0):
    model.train()
    preproc_module.eval()
    losses = []
    correct = 0
    total = 0
    t0 = time.time()

    for x, y, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(device_type=DEVICE.type, enabled=(scaler is not None)):
            x_p = preproc_module(x)
            x_n = (x_p - IMAGENET_MEAN) / IMAGENET_STD
            logits = model(x_n)
            loss = criterion(logits, y)
            if global_model is not None and CFG["fedprox_mu"] > 0:
                prox = fedprox_term(model, global_model)
                loss = loss + 0.5 * CFG["fedprox_mu"] * prox

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.append(float(loss.item()))
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.size(0))

    return float(np.mean(losses)), float(correct / max(1, total)), float(time.time() - t0)


def fedavg_update(global_model, local_models, weights, trainable_names):
    gsd = global_model.state_dict()
    new_sd = {}
    for name in trainable_names:
        acc = None
        for m, w in zip(local_models, weights):
            p = m.state_dict()[name].detach().float().cpu()
            acc = (w * p) if acc is None else (acc + w * p)
        new_sd[name] = acc
    for name, t in new_sd.items():
        gsd[name].copy_(t.to(gsd[name].device).type_as(gsd[name]))
    global_model.load_state_dict(gsd)


@torch.no_grad()
def pick_best_theta_from_pool(model, pool, val_loader, max_candidates=10):
    if not pool:
        return None, None
    cand = pool[:max_candidates]
    best, best_acc = None, -1
    for th in cand:
        pre = theta_to_module(th).to(DEVICE)
        met, _, _ = evaluate_full(model, val_loader, pre)
        if np.isfinite(met["acc"]) and met["acc"] > best_acc:
            best_acc = met["acc"]
            best = th
    return best, best_acc

# ============================================================
# 9) Initialize Global Model + Loss
# ============================================================
print("\n" + "=" * 92)
print("STEP 9: INITIALIZING GLOBAL MODEL")
print("=" * 92)

global_model = PVTv2B2_MultiScale(
    num_classes=NUM_CLASSES,
    pretrained=True,
    head_dropout=CFG["head_dropout"],
).to(DEVICE)

set_trainable_for_round(global_model, rnd=1)


total_params, tuned_params = count_params(global_model)
print("\n" + "-" * 92)
print(f"Backbone: {BACKBONE_NAME} | pretrained_loaded=True")
print(f"Total params: {total_params:,} | Trainable params: {tuned_params:,} ({(tuned_params/total_params)*100:.2f}%)")
print("-" * 92)

# Frozen backbone for GA separability (use last stage tokens)
backbone_frozen = global_model.backbone.eval()
for p in backbone_frozen.parameters():
    p.requires_grad = False

# Class weights per dataset (average)
counts1 = train1["y"].value_counts().sort_index().reindex(range(NUM_CLASSES), fill_value=0).values
counts2 = train2["y"].value_counts().sort_index().reindex(range(NUM_CLASSES), fill_value=0).values
counts = counts1 + counts2
w = (counts.sum() / np.clip(counts, 1, None)).astype(np.float32)
w = w / max(1e-6, w.mean())
class_w = torch.tensor(w, device=DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=CFG["label_smoothing"])

scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

hp_rows = [{"hp_name": k, "hp_value": str(v)} for k, v in CFG.items()]
hp_rows += [
    {
        "hp_name": "GA_theta_ranges",
        "hp_value": "gamma∈[0.7,1.4], alpha∈[0.15,0.55], beta∈[3,9], tau∈[1.8,3.2], k∈{3,5,7}, sh∈[0,0.25], dn∈[0,0.2]",
    },
    {"hp_name": "theta_fullforms", "hp_value": str(THETA_FULLFORMS)},
    {"hp_name": "backbone_name", "hp_value": BACKBONE_NAME},
    {"hp_name": "norm", "hp_value": "ImageNet mean/std"},
]
hp_df = pd.DataFrame(hp_rows)
print_table(hp_df, "Hyperparameters / Search Space")
add_table_to_csv(hp_df, "hyperparameters")

# ============================================================
# 10) Federated Training Loop (true FL simulation)
# ============================================================
print("\n" + "=" * 92)
print("STEP 10: FEDERATED TRAINING (NO CENTRAL VAL/TEST)")
print("=" * 92)

elite_pool_ds1 = []
elite_pool_ds2 = []
history_global = []
history_local = []

print(f"Rounds: {CFG['rounds']} | Clients: {CFG['clients']} | Local epochs: {CFG['local_epochs']}")
print(f"LR={CFG['lr']} | label_smoothing={CFG['label_smoothing']} | grad_clip={CFG['grad_clip']} | FedProx μ={CFG['fedprox_mu']}")
print(f"GA: {'ON' if CFG['use_ga'] else 'OFF'} | GA separability: {'ON'}")
print(f"Unfreeze after round: {CFG['unfreeze_after_round']} | tail frac: {CFG['unfreeze_tail_frac']} | bb lr mult: {CFG['unfreeze_lr_mult']}")

best_global_acc = -1.0
best_model_state = None
best_theta_ds1 = None
best_theta_ds2 = None
best_round_saved = None

t_global_start = time.time()

for rnd in range(1, CFG["rounds"] + 1):
    round_t0 = time.time()
    local_models = []
    local_weights = []
    local_rows = []

    print(f"\n{'='*92}")
    print(f"ROUND {rnd}/{CFG['rounds']}")
    print(f"{'='*92}")

    for k in range(CFG["clients"]):
        tr_loader, tune_loader, val_loader = client_loaders[k]
        ds_name = "ds1" if k < 2 else "ds2"
        elite_pool = elite_pool_ds1 if ds_name == "ds1" else elite_pool_ds2

        # GA (tuned on tune split)
        ga_t0 = time.time()
        if CFG["use_preprocessing"] and CFG["use_ga"]:
            best_theta, top_thetas, best_fit = run_ga_for_client(
                backbone_frozen, tune_loader, elite_pool, use_separability=True
            )
            elite_pool.extend(top_thetas)
            elite_pool[:] = elite_pool[: CFG["elite_pool_max"]]
            pre_k = theta_to_module(best_theta).to(DEVICE) if best_theta is not None else IDENTITY_PRE
        else:
            best_theta, best_fit = None, 0.0
            pre_k = IDENTITY_PRE
        ga_time = float(time.time() - ga_t0)

        if ds_name == "ds1":
            elite_pool_ds1 = elite_pool
        else:
            elite_pool_ds2 = elite_pool

        # Local model
        local_model = PVTv2B2_MultiScale(
            num_classes=NUM_CLASSES,
            pretrained=False,
            head_dropout=CFG["head_dropout"],
        ).to(DEVICE)
        local_model.load_state_dict(global_model.state_dict(), strict=True)

        set_trainable_for_round(local_model, rnd=rnd)
        opt = make_optimizer(local_model)

        total_steps = max(1, len(tr_loader) * CFG["local_epochs"])
        warmup_steps = max(1, len(tr_loader) * CFG["warmup_epochs"])
        scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

        # Train
        tr_losses, tr_accs, tr_time = [], [], 0.0
        for _ in range(CFG["local_epochs"]):
            loss_ep, acc_ep, t_ep = train_one_epoch(
                local_model,
                tr_loader,
                opt,
                pre_k,
                criterion,
                global_model=global_model,
                scheduler=scheduler,
                scaler=scaler,
                grad_clip=CFG["grad_clip"],
            )
            tr_losses.append(loss_ep)
            tr_accs.append(acc_ep)
            tr_time += t_ep

        # Client validation (on val split)
        met_loc, _, _ = evaluate_full(local_model, val_loader, pre_k)

        local_models.append(local_model)
        local_weights.append(len(tr_loader.dataset))

        if best_theta is not None:
            g, a, b, t, kk, sh, dn = best_theta
        else:
            g = a = b = t = kk = sh = dn = None

        row = {
            "round": rnd,
            "client": f"client_{k}",
            "dataset": ds_name,
            "ga_best_fit_score": float(best_fit),
            "ga_time_s": ga_time,
            "theta_str": theta_str(best_theta),
            "gamma_power": g,
            "alpha_contrast_weight": a,
            "beta_contrast_sharpness": b,
            "tau_clip": t,
            "k_blur_kernel_size": kk,
            "sh_sharpen_strength": sh,
            "dn_denoise_strength": dn,
            "train_loss": float(np.mean(tr_losses)),
            "train_acc": float(np.mean(tr_accs)),
            "train_time_s": float(tr_time),
            **{f"val_{k2}": v2 for k2, v2 in met_loc.items()},
        }
        local_rows.append(row)

        auc_val = row.get("val_auc_roc_macro_ovr", row.get("val_auc_roc", np.nan))
        print(
            f"Client {k} ({ds_name}) | train_acc={row['train_acc']:.4f} | "
            f"val_acc={row['val_acc']:.4f} | val_prec={row['val_precision_macro']:.4f} | "
            f"val_rec={row['val_recall_macro']:.4f} | val_f1={row['val_f1_macro']:.4f} | "
            f"val_auc={auc_val:.4f} | val_logloss={row['val_log_loss']:.4f} | "
            f"GA_fit={row['ga_best_fit_score']:.3f} | "
            f"time(train={row['train_time_s']:.1f}s, eval={row['val_eval_time_s']:.1f}s, ga={row['ga_time_s']:.1f}s) | "
            f"theta={row['theta_str']}"
        )

    # FedAvg
    wsum = sum(local_weights)
    weights = [w / wsum for w in local_weights]
    trainable_names = [n for n, p in local_models[0].named_parameters() if p.requires_grad]
    fedavg_update(global_model, local_models, weights, trainable_names)

    # Global validation aggregation (federated): weighted by client val size
    round_time = float(time.time() - round_t0)
    local_val_rows = pd.DataFrame(local_rows)
    local_val_rows["val_size"] = [len(client_loaders[i][2].dataset) for i in range(CFG["clients"])]
    total_val = local_val_rows["val_size"].sum()

    def weighted_avg(key):
        if total_val == 0:
            return np.nan
        return float(np.average(local_val_rows[key], weights=local_val_rows["val_size"]))

    global_metrics = {
        "acc": weighted_avg("val_acc"),
        "f1_macro": weighted_avg("val_f1_macro"),
        "precision_macro": weighted_avg("val_precision_macro"),
        "recall_macro": weighted_avg("val_recall_macro"),
        "log_loss": weighted_avg("val_log_loss"),
        "loss_ce": weighted_avg("val_loss_ce"),
        "eval_time_s": weighted_avg("val_eval_time_s"),
    }

    # Global theta per dataset (pick from each dataset pool using one val client)
    if CFG["use_preprocessing"] and elite_pool_ds1:
        best_theta_ds1, _ = pick_best_theta_from_pool(global_model, elite_pool_ds1, client_loaders[0][2])
    if CFG["use_preprocessing"] and elite_pool_ds2:
        best_theta_ds2, _ = pick_best_theta_from_pool(global_model, elite_pool_ds2, client_loaders[2][2])

    history_local.extend(local_rows)
    history_global.append(
        {
            "round": rnd,
            "round_time_s": round_time,
            "global_theta_ds1": theta_str(best_theta_ds1),
            "global_theta_ds2": theta_str(best_theta_ds2),
            **{f"global_{k2}": v2 for k2, v2 in global_metrics.items()},
        }
    )

    # BEST based on ACC
    if np.isfinite(global_metrics["acc"]) and global_metrics["acc"] > best_global_acc:
        best_global_acc = float(global_metrics["acc"])
        best_model_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        best_round_saved = rnd

    print("\n" + "-" * 92)
    print(
        f"GLOBAL VAL (Round {rnd}) | acc={global_metrics['acc']:.4f} | f1={global_metrics['f1_macro']:.4f} | "
        f"logloss={global_metrics['log_loss']:.4f} | loss_ce={global_metrics['loss_ce']:.4f} | "
        f"round_time={round_time:.1f}s | theta_ds1={theta_str(best_theta_ds1)} | theta_ds2={theta_str(best_theta_ds2)}"
    )
    print(f"BEST SO FAR (by ACC) | best_val_acc={best_global_acc:.4f} at round={best_round_saved}")
    print("-" * 92)

# restore best model
if best_model_state is not None:
    global_model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})

t_total = float(time.time() - t_global_start)
print("\n" + "=" * 92)
print(f"TRAINING COMPLETE ✅ | total_time={t_total:.1f}s | best_val_acc={best_global_acc:.4f} | best_round={best_round_saved}")
print("=" * 92)

glob_df = pd.DataFrame(history_global)
loc_df = pd.DataFrame(history_local)

print_table(glob_df, "GLOBAL per-round metrics (12 rounds)")
print_table(loc_df, "LOCAL per-client per-round metrics (12 rounds × 4 clients = 48 rows)")

add_table_to_csv(glob_df, "global_round_metrics_full")
add_table_to_csv(loc_df, "client_round_metrics_full")

# ============================================================
# 11) Final Evaluation (federated) + Paper-ready tables
# ============================================================
print("\n" + "=" * 92)
print("STEP 11: FINAL EVALUATION (FEDERATED VAL + TEST)")
print("=" * 92)

# Use per-dataset theta
pre_best_ds1 = (
    theta_to_module(best_theta_ds1).to(DEVICE)
    if CFG["use_preprocessing"] and best_theta_ds1 is not None
    else IDENTITY_PRE
)
pre_best_ds2 = (
    theta_to_module(best_theta_ds2).to(DEVICE)
    if CFG["use_preprocessing"] and best_theta_ds2 is not None
    else IDENTITY_PRE
)

# Federated VAL aggregation (already computed by history, but recompute for paper table)
val_metrics_clients = []
for k in range(CFG["clients"]):
    _, _, val_loader = client_loaders[k]
    pre = pre_best_ds1 if k < 2 else pre_best_ds2
    met, _, _ = evaluate_full(global_model, val_loader, pre)
    val_metrics_clients.append((k, met, len(val_loader.dataset)))


def weighted_aggregate(mets):
    total = sum(w for _, _, w in mets)
    if total == 0:
        return {}
    keys = mets[0][1].keys()
    out = {}
    for k in keys:
        vals = [m[1].get(k, np.nan) for m in mets]
        weights = [m[2] for m in mets]
        out[k] = float(np.average(vals, weights=weights))
    return out


val_best = weighted_aggregate(val_metrics_clients)

# Federated TEST (per dataset + global weighted)

def eval_test_per_dataset(ds_name):
    mets = []
    for i, (ds, local_id, t_loader) in enumerate(client_test_loaders):
        if ds != ds_name:
            continue
        pre = pre_best_ds1 if ds == "ds1" else pre_best_ds2
        met, y_true, p_pred = evaluate_full(global_model, t_loader, pre)
        mets.append((met, len(t_loader.dataset), y_true, p_pred))
    agg = weighted_aggregate([(i, m[0], m[1]) for i, m in enumerate(mets)])
    return agg, mets


test_ds1, test_mets_ds1 = eval_test_per_dataset("ds1")
test_ds2, test_mets_ds2 = eval_test_per_dataset("ds2")

# Weighted global test (ds1 + ds2)
all_test_mets = [(0, test_ds1, len(test1)), (1, test_ds2, len(test2))]
global_test = weighted_aggregate(all_test_mets)


def compact_metrics(m):
    keep = [
        "acc",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "log_loss",
    ]
    if "auc_roc_macro_ovr" in m:
        keep.append("auc_roc_macro_ovr")
    if "auc_roc" in m:
        keep.append("auc_roc")
    keep += ["loss_ce", "eval_time_s"]
    return {k: float(m[k]) for k in keep if k in m}


paper_df = pd.DataFrame(
    [
        {"setting": "Enhanced FELCM (Best θ ds1)", "split": "VAL", "dataset": "ds1+ds2 weighted", **compact_metrics(val_best)},
        {"setting": "Enhanced FELCM (Best θ ds1)", "split": "TEST", "dataset": "ds1", **compact_metrics(test_ds1)},
        {"setting": "Enhanced FELCM (Best θ ds2)", "split": "TEST", "dataset": "ds2", **compact_metrics(test_ds2)},
        {"setting": "Enhanced FELCM (Best θ)", "split": "TEST", "dataset": "global weighted", **compact_metrics(global_test)},
    ]
)

print_table(paper_df, "VAL+TEST tables (federated, per-dataset + global)")
add_table_to_csv(paper_df, "paper_ready_metrics")

print("\nPaper selection summary:")
print(f"- Best round (by federated VAL accuracy): round={best_round_saved} | best_val_acc={best_global_acc:.4f}")
print(f"- Best θ ds1: {theta_str(best_theta_ds1)}")
print(f"- Best θ ds2: {theta_str(best_theta_ds2)}")

# ============================================================
# 12) Preprocessing validation (overall, device-safe) + plots
# ============================================================
print("\n" + "=" * 92)
print("STEP 12: PREPROCESSING VALIDATION (VAL SAMPLE)")
print("=" * 92)
if not CFG["use_preprocessing"]:
    print("Preprocessing disabled → skipping validation plots.")


@torch.no_grad()
def entropy_per_image(x01):
    gray = x01.mean(dim=1)
    B = gray.shape[0]
    ent = []
    for i in range(B):
        g = (gray[i].detach().cpu().numpy() * 255).astype(np.uint8)
        hist = np.bincount(g.flatten(), minlength=256).astype(np.float32)
        p = hist / np.clip(hist.sum(), 1, None)
        p = p[p > 0]
        ent.append(float(-(p * np.log2(p)).sum()))
    return np.array(ent)


@torch.no_grad()
def edge_energy(x01, lap_kernel):
    lap_kernel = lap_kernel.to(device=x01.device, dtype=x01.dtype)
    gray = x01.mean(dim=1, keepdim=True)
    lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), lap_kernel).abs()
    return lap.mean(dim=(1, 2, 3)).detach().cpu().numpy()


@torch.no_grad()
def contrast_proxy(x01):
    gray = x01.mean(dim=1)
    return gray.std(dim=(1, 2)).detach().cpu().numpy()


@torch.no_grad()
def run_preproc_validation(frame, preproc, sample_n=600):
    n = min(sample_n, len(frame))
    if n <= 0:
        return pd.DataFrame(), pd.DataFrame(), None, None

    idx = np.random.choice(len(frame), size=n, replace=False)
    ds = MRIDataset(frame, indices=idx.tolist(), tfms=EVAL_TFMS)

    xs = []
    for i in range(len(ds)):
        x, _, _ = ds[i]
        xs.append(x)
    x = torch.stack(xs).to(DEVICE)

    x_after = preproc(x).clamp(0, 1)

    lap_kernel = preproc.lap if hasattr(preproc, "lap") else EnhancedFELCM().to(DEVICE).lap

    ee_before = edge_energy(x, lap_kernel)
    ee_after = edge_energy(x_after, lap_kernel)

    ent_before = entropy_per_image(x)
    ent_after = entropy_per_image(x_after)

    con_before = contrast_proxy(x)
    con_after = contrast_proxy(x_after)

    dfm = pd.DataFrame(
        {
            "edge_energy_before": ee_before,
            "edge_energy_after": ee_after,
            "entropy_before": ent_before,
            "entropy_after": ent_after,
            "contrast_before": con_before,
            "contrast_after": con_after,
            "edge_gain_ratio": (ee_after / np.clip(ee_before, 1e-9, None)),
            "entropy_delta": (ent_after - ent_before),
            "contrast_delta": (con_after - con_before),
        }
    )
    summary = dfm.agg(["mean", "std", "min", "max"]).T.reset_index().rename(columns={"index": "metric"})
    return dfm, summary, x, x_after


if CFG["use_preprocessing"]:
    preproc_df, preproc_summary_df, _, _ = run_preproc_validation(
        val1,
        pre_best_ds1 if best_theta_ds1 is not None else IDENTITY_PRE,
        CFG["preproc_val_sample_n"],
    )
    print_table(preproc_summary_df, "Preprocessing validation summary (DS1 VAL sample)")
    add_table_to_csv(preproc_summary_df, "preprocessing_validation_summary_ds1")
else:
    preproc_df = pd.DataFrame()
    preproc_summary_df = pd.DataFrame()

# ============================================================
# 13) Before vs After preprocessing images (best θ)
# ============================================================
print("\n" + "=" * 92)
print("STEP 13: BEFORE vs AFTER PREPROCESSING IMAGES (BEST θ) — PRINTED")
print("=" * 92)
if not CFG["use_preprocessing"]:
    print("Preprocessing disabled → skipping before/after preprocessing grid.")


@torch.no_grad()
def show_before_after(preproc, frame, n=12):
    per_class = max(1, n // NUM_CLASSES)
    sample = frame.groupby("label").head(per_class).copy()
    if len(sample) < n:
        extra = frame.sample(min(n - len(sample), len(frame)), random_state=SEED)
        sample = pd.concat([sample, extra], axis=0).drop_duplicates(subset=["path"]).head(n)
    sample = sample.sample(min(n, len(sample)), random_state=SEED).reset_index(drop=True)

    ds = MRIDataset(sample, indices=list(range(len(sample))), tfms=EVAL_TFMS)
    xs, ys = [], []
    for i in range(len(ds)):
        x, y, _ = ds[i]
        xs.append(x)
        ys.append(y)
    x = torch.stack(xs).to(DEVICE)
    x_after = preproc(x).clamp(0, 1)

    B = x.size(0)
    fig = plt.figure(figsize=(min(18, 2.2 * B), 6))

    for i in range(B):
        ax1 = plt.subplot(2, B, i + 1)
        ax1.imshow(x[i].cpu().permute(1, 2, 0).numpy())
        ax1.set_title(f"Original\n({id2label[int(ys[i])]})", fontsize=9)
        ax1.axis("off")

        ax2 = plt.subplot(2, B, B + i + 1)
        ax2.imshow(x_after[i].cpu().permute(1, 2, 0).numpy())
        ax2.set_title("FELCM Enhanced", fontsize=9)
        ax2.axis("off")

    plt.suptitle(f"Before vs After FELCM | Best θ (ds1) = {theta_str(best_theta_ds1)}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


if CFG["use_preprocessing"]:
    show_before_after(pre_best_ds1 if best_theta_ds1 is not None else IDENTITY_PRE, test1, n=CFG["before_after_n"])

# ============================================================
# 14) ROC + PR curves (TEST, Best θ) — DS1 example
# ============================================================
print("\n" + "=" * 92)
print("STEP 14: ROC + PR CURVES (TEST, Best θ) — DS1")
print("=" * 92)

# Use first DS1 client test loader for plotting (avoid central access)
_, _, ds1_test_loader = client_test_loaders[0]
_, y_true_test, p_test_best = evaluate_full(global_model, ds1_test_loader, pre_best_ds1)

fig = plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
for c in range(NUM_CLASSES):
    yc = (y_true_test == c).astype(int)
    if yc.sum() == 0 or yc.sum() == len(yc):
        continue
    fpr, tpr, _ = roc_curve(yc, p_test_best[:, c])
    auc_c = roc_auc_score(yc, p_test_best[:, c])
    plt.plot(fpr, tpr, label=f"{labels[c]} (AUC={auc_c:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curves (OvR) — DS1 TEST (Best θ)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.subplot(1, 2, 2)
for c in range(NUM_CLASSES):
    yc = (y_true_test == c).astype(int)
    if yc.sum() == 0:
        continue
    prec, rec, _ = precision_recall_curve(yc, p_test_best[:, c])
    ap = average_precision_score(yc, p_test_best[:, c])
    plt.plot(rec, prec, label=f"{labels[c]} (AP={ap:.3f})")
plt.title("Precision–Recall Curves — DS1 TEST (Best θ)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()

plt.tight_layout()
plt.show()

# ============================================================
# 15) Confusion matrices (counts + row-normalized) — DS1
# ============================================================
print("\n" + "=" * 92)
print("STEP 15: CONFUSION MATRIX (TEST, Best θ) — DS1")
print("=" * 92)


y_hat_test = np.argmax(p_test_best, axis=1)
cm_counts = confusion_matrix(y_true_test, y_hat_test, labels=list(range(NUM_CLASSES)))
cm_norm = cm_counts / np.clip(cm_counts.sum(axis=1, keepdims=True), 1, None)

fig = plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.imshow(cm_counts, interpolation="nearest", aspect="equal")
plt.title("Confusion Matrix (Counts)")
plt.colorbar()
plt.xticks(range(NUM_CLASSES), labels, rotation=25, ha="right")
plt.yticks(range(NUM_CLASSES), labels)
plt.grid(False)
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        v = cm_counts[i, j]
        plt.text(
            j,
            i,
            str(v),
            ha="center",
            va="center",
            fontweight="bold",
            color="white" if v > cm_counts.max() / 2 else "black",
        )
plt.xlabel("Predicted")
plt.ylabel("True")

plt.subplot(1, 2, 2)
plt.imshow(cm_norm, interpolation="nearest", vmin=0, vmax=1, aspect="equal")
plt.title("Confusion Matrix (Row-normalized)")
plt.colorbar()
plt.xticks(range(NUM_CLASSES), labels, rotation=25, ha="right")
plt.yticks(range(NUM_CLASSES), labels)
plt.grid(False)
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        v = cm_norm[i, j]
        plt.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontweight="bold",
            color="white" if v > 0.5 else "black",
        )
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()
plt.show()

# ============================================================
# 16) Calibration (Reliability diagram) — DS1
# ============================================================
print("\n" + "=" * 92)
print("STEP 16: CALIBRATION PLOT (TEST, Best θ) — DS1")
print("=" * 92)


def multiclass_calibration_curve(y_true, p_pred, n_bins=12):
    conf = np.max(p_pred, axis=1)
    pred = np.argmax(p_pred, axis=1)
    acc = (pred == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_conf, bin_acc, bin_count = [], [], []
    for b in range(n_bins):
        m = bin_ids == b
        if m.sum() == 0:
            bin_conf.append(np.nan)
            bin_acc.append(np.nan)
            bin_count.append(0)
        else:
            bin_conf.append(conf[m].mean())
            bin_acc.append(acc[m].mean())
            bin_count.append(int(m.sum()))
    return np.array(bin_conf), np.array(bin_acc), np.array(bin_count)


bin_conf, bin_acc, bin_n = multiclass_calibration_curve(y_true_test, p_test_best, n_bins=12)
fig = plt.figure(figsize=(7, 5))
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
plt.plot(bin_conf, bin_acc, marker="o", label="Model")
plt.title("Reliability Diagram — DS1 TEST (Best θ)")
plt.xlabel("Confidence")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

cal_df = pd.DataFrame({"bin_confidence": bin_conf, "bin_accuracy": bin_acc, "bin_count": bin_n})
print_table(cal_df, "Calibration bins table (DS1)")
add_table_to_csv(cal_df, "calibration_bins_ds1")

# ============================================================
# 17) Unique plots: Radar + Client evolution + Theta evolution
# ============================================================
print("\n" + "=" * 92)
print("STEP 17: UNIQUE PLOTS (RADAR, CLIENT EVOLUTION, THETA EVOLUTION)")
print("=" * 92)


def radar_plot(metrics_a, metrics_b, axes_keys, title):
    vals_a = [metrics_a.get(k, np.nan) for k in axes_keys]
    vals_b = [metrics_b.get(k, np.nan) for k in axes_keys]
    angles = np.linspace(0, 2 * np.pi, len(axes_keys), endpoint=False).tolist()
    vals_a += vals_a[:1]
    vals_b += vals_b[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, vals_a, linewidth=2, label="DS1")
    ax.fill(angles, vals_a, alpha=0.15)
    ax.plot(angles, vals_b, linewidth=2, label="DS2")
    ax.fill(angles, vals_b, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_keys)
    ax.set_title(title, y=1.08, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15))
    plt.show()


rad_keys = ["acc", "f1_macro", "precision_macro", "recall_macro", "log_loss"]
if "auc_roc_macro_ovr" in test_ds1:
    rad_keys.insert(4, "auc_roc_macro_ovr")
elif "auc_roc" in test_ds1:
    rad_keys.insert(4, "auc_roc")

radar_plot(compact_metrics(test_ds1), compact_metrics(test_ds2), rad_keys, "TEST Metrics Radar (DS1 vs DS2)")

fig = plt.figure(figsize=(12, 4))
for k in range(CFG["clients"]):
    sub = loc_df[loc_df["client"] == f"client_{k}"].sort_values("round")
    plt.plot(sub["round"], sub["val_acc"], marker="o", label=f"Client {k} val_acc")
plt.title("Client Validation Accuracy Evolution (per round)")
plt.xlabel("Round")
plt.ylabel("val_acc")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


theta_cols = [
    "gamma_power",
    "alpha_contrast_weight",
    "beta_contrast_sharpness",
    "tau_clip",
    "k_blur_kernel_size",
    "sh_sharpen_strength",
    "dn_denoise_strength",
]
theta_evo = loc_df.groupby("round")[theta_cols].mean(numeric_only=True).reset_index()
print_table(theta_evo, "Mean best-θ parameters over rounds (clients averaged)")
add_table_to_csv(theta_evo, "theta_evolution_mean")

fig = plt.figure(figsize=(14, 6))
for col in theta_cols:
    plt.plot(theta_evo["round"], theta_evo[col], marker="o", label=col)
plt.title("GA Best θ Evolution (Mean across clients)")
plt.xlabel("Round")
plt.grid(True, alpha=0.3)
plt.legend(ncol=2)
plt.show()

# ============================================================
# 18) SAVE ONLY TWO FILES (checkpoint + one CSV)
# ============================================================
print("\n" + "=" * 92)
print("STEP 18: SAVING ONLY TWO FILES (CHECKPOINT + ONE CSV)")
print("=" * 92)

checkpoint = {
    "state_dict": {k: v.detach().cpu() for k, v in global_model.state_dict().items()},
    "config": CFG,
    "seed": SEED,
    "device_used": str(DEVICE),
    "dataset1_raw_root": DS1_ROOT,
    "dataset2_root": DS2_ROOT,
    "labels": labels,
    "label2id": label2id,
    "id2label": id2label,
    "num_classes": NUM_CLASSES,
    "backbone_name": BACKBONE_NAME,
    "best_round_saved": best_round_saved,
    "best_val_acc": best_global_acc,
    "best_theta_ds1": best_theta_ds1,
    "best_theta_ds2": best_theta_ds2,
    "best_theta_str_ds1": theta_str(best_theta_ds1),
    "best_theta_str_ds2": theta_str(best_theta_ds2),
    "theta_fullforms": THETA_FULLFORMS,
    "client_splits": client_splits,
    "client_test_splits": client_test_splits,
    "history_global": glob_df.to_dict(orient="list"),
    "history_local": loc_df.to_dict(orient="list"),
    "final_val_federated": val_best,
    "final_test_ds1": test_ds1,
    "final_test_ds2": test_ds2,
    "final_test_global_weighted": global_test,
    "preprocessing_validation_summary_ds1": preproc_summary_df.to_dict(orient="list") if len(preproc_summary_df) else {},
    "total_training_time_s": t_total,
}

torch.save(checkpoint, MODEL_PATH)
print(f"✅ Saved checkpoint: {MODEL_PATH}")

all_df = pd.DataFrame(ALL_ROWS)
all_df.to_csv(CSV_PATH, index=False)
print(f"✅ Saved CSV (ALL outputs): {CSV_PATH}")

print("\nDONE ✅ (TRUE FL SIMULATION, 4 clients, NO AUGMENTATION, PVTv2-B2 fixed)")
