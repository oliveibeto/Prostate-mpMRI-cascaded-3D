"""
Stage 2 of the cascaded framework: benign (0) vs malignant (1) classification.

Pipeline:  Stage 1 3D U-Net  ->  prostate mask  ->  prostate crop  ->  3D ResNet-10  ->  P(malignant)

Labels: derived from Prostate158 expert tumour annotations (tumour mask present = 1, absent = 0).
These are annotation-based labels, not histopathology (state this in the thesis).
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from torch.utils.data import Dataset, DataLoader, random_split
from monai.networks.nets import UNet, resnet10

# ============================ CONFIG ============================
PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cache"
STAGE1_WEIGHTS = str(PROJECT_ROOT / "prostate_unet_stage1_best.pth")
LABELS_CSV = str(PROJECT_ROOT / "stage2_labels_check.csv")
RAW_ROOT = Path(r"C:\ProstateData\prostate158_train")  # set to your Prostate158 folder

VERIFY = True          # True on your PC (raw data present); False on Kaggle
VERIFY_ONLY = False    # True = check the cache/ID mapping and stop

SEED = 42
N_TRAIN, N_VAL, N_TEST = 97, 20, 22               # identical to the Stage 1 split
OUT_SHAPE = (32, 96, 96)                          # classifier input (depth, height, width)
MARGIN = np.array([2, 8, 8])                      # voxels added around the prostate mask
EPOCHS, BATCH, LR, WEIGHT_DECAY = 40, 8, 1e-4, 1e-2
RUN_BASELINE = True    # also train a model on the whole volume (no Stage 1) for comparison
# ================================================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------- image utilities ----------------------
def keep_largest(mask):
    """Keep the largest connected component of a binary mask."""
    labelled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labelled, range(1, n + 1))
    return labelled == (int(np.argmax(sizes)) + 1)


def crop_resize(img, mask, margin):
    """Crop img [C,D,H,W] around a binary mask [D,H,W] with a margin, then resize to OUT_SHAPE.
    Returns None if the mask is empty."""
    if mask.sum() == 0:
        return None
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + 1 + margin, np.array(mask.shape))
    crop = img[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return F.interpolate(crop[None], size=OUT_SHAPE, mode="trilinear", align_corners=False)[0]


def whole_resize(img):
    return F.interpolate(img[None], size=OUT_SHAPE, mode="trilinear", align_corners=False)[0]


def augment(x):
    """Light augmentation: left-right flip and small intensity changes."""
    if random.random() < 0.5:
        x = torch.flip(x, dims=[3])
    scale = 1 + 0.1 * torch.randn(x.shape[0], 1, 1, 1)
    shift = 0.1 * torch.randn(x.shape[0], 1, 1, 1)
    return x * scale + shift


# ---------------------- dataset and model ----------------------
class TrainDataset(Dataset):
    """mode='crop'  : crop around the reference prostate mask with random margin jitter,
                      so the classifier tolerates imperfect Stage 1 masks.
       mode='whole' : whole volume (non-cascaded baseline only)."""

    def __init__(self, images, masks, labels, mode):
        self.images, self.masks, self.labels, self.mode = images, masks, labels, mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        if self.mode == "crop":
            jitter = np.random.randint(-2, 3, size=3) * np.array([1, 4, 4])
            x = crop_resize(self.images[i], self.masks[i], np.maximum(MARGIN + jitter, 0))
        else:
            x = whole_resize(self.images[i])
        return augment(x), torch.tensor(float(self.labels[i]))


def make_classifier():
    # Binary classifier: one output logit; sigmoid gives P(malignant). Input: T2W + ADC.
    return resnet10(spatial_dims=3, n_input_channels=2, num_classes=1)


@torch.no_grad()
def predict_probs(model, crops, device):
    model.eval()
    probs = []
    for i in range(0, len(crops), BATCH):
        batch = torch.stack(crops[i:i + BATCH]).to(device)
        probs.append(torch.sigmoid(model(batch).squeeze(1)).cpu().numpy())
    return np.concatenate(probs)


# ---------------------- evaluation ----------------------
def classification_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * sens / max(precision + sens, 1e-9)
    return dict(auc=float(roc_auc_score(y, p)), accuracy=float((tp + tn) / len(y)),
                sensitivity=float(sens), specificity=float(spec), precision=float(precision),
                f1=float(f1), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def bootstrap_auc_ci(y, p, n_boot=2000):
    rng, aucs = np.random.RandomState(SEED), []
    for _ in range(n_boot):
        i = rng.randint(0, len(y), len(y))
        if len(set(y[i])) == 2:
            aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(aucs, [2.5, 97.5])


def report(name, y, p, threshold):
    m = classification_metrics(y, p, threshold)
    lo, hi = bootstrap_auc_ci(y, p)
    m["auc_95ci"] = [float(lo), float(hi)]
    print(f"\nTEST | {name} | threshold {threshold:.3f} (chosen on validation)")
    print(f"  AUC {m['auc']:.3f} (95% CI {lo:.3f}-{hi:.3f}) | accuracy {m['accuracy']:.3f} | "
          f"sensitivity {m['sensitivity']:.3f} | specificity {m['specificity']:.3f} | "
          f"precision {m['precision']:.3f} | F1 {m['f1']:.3f}")
    print(f"  confusion matrix: TN {m['tn']}  FP {m['fp']}  FN {m['fn']}  TP {m['tp']}")
    return m


# ---------------------- training ----------------------
def train_classifier(tag, train_ds, val_crops, y_val, y_train, device):
    model = make_classifier().to(device)
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)

    best_auc, log = -1.0, []
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(1), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(x)
        scheduler.step()

        train_loss = running / len(train_ds)
        val_auc = roc_auc_score(y_val, predict_probs(model, val_crops, device))
        log.append([epoch + 1, train_loss, val_auc])
        print(f"[{tag}] epoch {epoch + 1:02d}/{EPOCHS} | train loss {train_loss:.4f} | val AUC {val_auc:.4f}")
        if val_auc > best_auc:                                   # checkpoint chosen on validation AUC
            best_auc = val_auc
            torch.save(model.state_dict(), f"stage2_{tag}_best.pth")

    pd.DataFrame(log, columns=["epoch", "train_loss", "val_auc"]).to_csv(f"stage2_{tag}_log.csv", index=False)
    model.load_state_dict(torch.load(f"stage2_{tag}_best.pth", map_location=device))
    return model


def choose_threshold(y_val, p_val):
    fpr, tpr, thresholds = roc_curve(y_val, p_val)
    return float(thresholds[np.argmax(tpr - fpr)])               # Youden's J on validation data only


# ---------------------- main ----------------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1. Load the Stage 1 cache and the labels
    files = sorted(CACHE_DIR.rglob("case_*.pt"))
    n = len(files)
    assert n == N_TRAIN + N_VAL + N_TEST, \
        f"{n} cached cases, but split sizes add up to {N_TRAIN + N_VAL + N_TEST}"
    cases = [torch.load(f) for f in files]
    images3 = [c["image"].float() for c in cases]                  # 3 channels, as used by Stage 1
    gt_masks = [(c["label"][0].numpy() > 0) for c in cases]        # reference prostate masks

    labels_df = pd.read_csv(LABELS_CSV)
    assert labels_df["ID"].is_unique and len(labels_df) == n, "labels CSV needs one row per cached case"
    ids = sorted(labels_df["ID"].astype(int).tolist())             # cache is assumed to be in ascending ID order
    label_of = dict(zip(labels_df["ID"].astype(int), labels_df["label"].astype(int)))
    y_all = np.array([label_of[i] for i in ids])

    # 2. Verify that cache order matches the case IDs
    if VERIFY:
        meta = pd.concat([pd.read_csv(RAW_ROOT / "train.csv"), pd.read_csv(RAW_ROOT / "valid.csv")])
        meta = meta.set_index(meta["ID"].astype(int))
        raw_volumes, cache_volumes = [], []
        for k, case_id in enumerate(ids):
            path = meta.loc[case_id, "t2_anatomy_reader1"]
            if pd.isna(path) or not (RAW_ROOT / path).exists():
                continue
            img = nib.load(str(RAW_ROOT / path))
            raw_volumes.append((img.get_fdata() > 0).sum() * float(np.prod(img.header.get_zooms()[:3])))
            cache_volumes.append(float(gt_masks[k].sum()))
        rho = spearmanr(raw_volumes, cache_volumes).correlation
        print(f"Cache-order check: Spearman rho (raw vs cached prostate volume) = {rho:.3f} "
              f"over {len(raw_volumes)} cases")
        if rho < 0.9:
            raise SystemExit("Cache order does NOT match ascending IDs. Fix the ID mapping before training.")
        if VERIFY_ONLY:
            print("Mapping verified. Set VERIFY_ONLY = False to train.")
            return

    # 3. Same train/validation/test split as Stage 1
    split = random_split(list(range(n)), [N_TRAIN, N_VAL, N_TEST],
                         generator=torch.Generator().manual_seed(SEED))
    train_idx, val_idx, test_idx = [list(s.indices) for s in split]
    for name, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        print(f"{name}: {len(idx)} cases | malignant {int(y_all[idx].sum())} | "
              f"benign {int(len(idx) - y_all[idx].sum())}")
    y_train, y_val, y_test = y_all[train_idx], y_all[val_idx], y_all[test_idx]
    assert len(set(y_val)) == 2 and len(set(y_test)) == 2, "validation and test sets each need both classes"

    # 4. STAGE 1: predict the prostate on validation and test cases
    stage1 = UNet(spatial_dims=3, in_channels=3, out_channels=3, channels=(16, 32, 64, 128, 256),
                  strides=(2, 2, 2, 2), num_res_units=2).to(device)
    stage1.load_state_dict(torch.load(STAGE1_WEIGHTS, map_location=device))
    stage1.eval()
    pred_masks = {}
    with torch.no_grad():
        for i in val_idx + test_idx:
            out = stage1(images3[i][None].to(device)).argmax(1)[0].cpu().numpy() > 0   # any foreground class
            pred_masks[i] = keep_largest(out)
    gland_dice = [2 * (pred_masks[i] & gt_masks[i]).sum() / max(pred_masks[i].sum() + gt_masks[i].sum(), 1)
                  for i in test_idx]
    print(f"Stage 1 whole-gland Dice on test cases: {np.mean(gland_dice):.3f} +/- {np.std(gland_dice):.3f}")

    images = [img[:2].clone() for img in images3]                  # Stage 2 input: T2W and ADC
    del images3, cases

    def build_crops(indices, source):
        crops, fallbacks = [], 0
        for i in indices:
            mask = gt_masks[i] if source == "reference" else pred_masks[i]
            crop = crop_resize(images[i], mask, MARGIN)
            if crop is None:                                       # Stage 1 produced no prostate; counted and reported
                fallbacks += 1
                crop = whole_resize(images[i])
            crops.append(crop)
        return crops, fallbacks

    val_crops, fb_val = build_crops(val_idx, "stage1")
    test_crops, fb_test = build_crops(test_idx, "stage1")
    test_crops_ref, _ = build_crops(test_idx, "reference")
    print(f"Stage 1 empty-mask fallbacks: validation {fb_val}, test {fb_test}")

    # 5. STAGE 2 (cascade): train on reference crops; validate and test on Stage 1 crops
    train_ds = TrainDataset([images[i] for i in train_idx], [gt_masks[i] for i in train_idx], y_train, "crop")
    model = train_classifier("cascade", train_ds, val_crops, y_val, y_train, device)

    threshold = choose_threshold(y_val, predict_probs(model, val_crops, device))
    p_cascade = predict_probs(model, test_crops, device) 
    p_oracle = predict_probs(model, test_crops_ref, device)
    results = {
        "cascade": report("CASCADE (Stage 1 predicted crops)", y_test, p_cascade, threshold),
        "oracle": report("ORACLE (reference-mask crops, upper bound)", y_test, p_oracle, threshold),
    }
    out = pd.DataFrame({"ID": [ids[i] for i in test_idx], "label": y_test,
                        "p_cascade": p_cascade, "p_oracle": p_oracle})

    # 6. Non-cascaded baseline for comparison
    if RUN_BASELINE:
        val_whole = [whole_resize(images[i]) for i in val_idx]
        test_whole = [whole_resize(images[i]) for i in test_idx]
        base_ds = TrainDataset([images[i] for i in train_idx], [gt_masks[i] for i in train_idx], y_train, "whole")
        baseline = train_classifier("baseline", base_ds, val_whole, y_val, y_train, device)
        thr_base = choose_threshold(y_val, predict_probs(baseline, val_whole, device))
        p_base = predict_probs(baseline, test_whole, device)
        results["baseline"] = report("BASELINE (whole volume, no Stage 1)", y_test, p_base, thr_base)
        out["p_baseline"] = p_base

    print(f"\nMajority-class accuracy on the test set: {max(y_test.mean(), 1 - y_test.mean()):.3f}")
    out.to_csv("stage2_test_predictions.csv", index=False)
    with open("stage2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: stage2_results.json, stage2_test_predictions.csv, stage2_*_log.csv, stage2_*_best.pth")


if __name__ == "__main__":
    main()