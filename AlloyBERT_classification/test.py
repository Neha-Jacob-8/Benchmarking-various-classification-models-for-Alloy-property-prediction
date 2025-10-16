# test_only.py
# Evaluate AlloyBERT on a held-out packet (no training).
# Reads config from configtest.yaml ONLY; training config.yaml is untouched.

import os, json, math, sys
import numpy as np
import pandas as pd
import torch
import yaml
from typing import List
from transformers import RobertaTokenizerFast, RobertaModel
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    log_loss, confusion_matrix, classification_report, roc_curve,ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

from model.network import AlloyBERT

# ------------------------------- utils -------------------------------

def np_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Binary: probs (N,), Multiclass: probs (N,C)."""
    if probs.ndim == 1:
        return float(np.mean((probs - y_true) ** 2))
    n, c = probs.shape
    oh = np.zeros((n, c), dtype=float); oh[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))

def to_py(o):
    import numpy as _np
    if isinstance(o, (_np.floating, _np.integer)): return o.item()
    if isinstance(o, _np.ndarray): return o.tolist()
    return o

def batchify(lst: List, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
def save_cm_images(cm: np.ndarray, labels: np.ndarray, out_dir: str):
    """Save raw and normalized confusion matrix images to out_dir."""
    class_names = [str(c) for c in labels]

    # Raw counts
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, values_format="d", colorbar=False, cmap="Blues")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    # Normalized (row-wise)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.divide(cm.astype(float), row_sums, where=row_sums!=0)
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, values_format=".2f", colorbar=False, cmap="Blues")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix (Normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_normalized.png"), dpi=200)
    plt.close(fig)

# ------------------------------- main -------------------------------

def main():
    # 1) Load test config (mandatory)
    if not os.path.exists("configtest.yaml"):
        print("❌ configtest.yaml not found. Create it for testing mode.")
        sys.exit(1)
    cfg = yaml.safe_load(open("configtest.yaml"))
    print("[INFO] Using configtest.yaml")

    # Required / default config values
    device     = torch.device(str(cfg.get("device", "cpu")))
    num_labels = int(cfg.get("num_labels", 2))
    max_len    = int(cfg.get("max_len", 512))
    batch_size = int(cfg.get("batch_size", 16))
    test_path  = cfg.get("test_path", "data/datapackets/test.csv")
    text_col   = cfg.get("text_col", "text")
    label_cands = cfg.get("label_candidates", ["YS_Class","label","class","target"])
    ckpt_path  = cfg.get("ckpt_path", "outputs/best_model.pt")
    out_dir    = cfg.get("test_out_dir", "data/datapackets/test_results")

    print(f"[TEST] device={device}  ckpt={ckpt_path}")
    print(f"[TEST] test_path={test_path}  out_dir={out_dir}")

    ensure_dir(out_dir)

    # 2) Build model + tokenizer
    base  = RobertaModel.from_pretrained("roberta-base")
    model = AlloyBERT({"task_type":"classification",
                       "num_labels": num_labels,
                       "dropout": cfg.get("dropout", 0.1)}, base).to(device)

    state = torch.load(ckpt_path, map_location=device)
    state = state.get("model_state_dict", state)
    model.load_state_dict(state, strict=True)
    model.eval()

    tok = RobertaTokenizerFast.from_pretrained("roberta-base")

    # 3) Load test data (CSV or PKL)
    if not os.path.exists(test_path):
        print(f"❌ Test file not found: {test_path}")
        sys.exit(1)

    if test_path.endswith(".csv"):
        df = pd.read_csv(test_path)
    elif test_path.endswith(".pkl"):
        df = pd.read_pickle(test_path)
    else:
        print(f"❌ Unsupported test file extension: {test_path}")
        sys.exit(1)

    if text_col not in df.columns:
        raise ValueError(f"'{text_col}' column not found in {test_path}. "
                         f"Columns: {list(df.columns)[:12]}...")

    label_col = None
    for c in label_cands:
        if c in df.columns:
            label_col = c; break
    if label_col is None:
        raise ValueError(f"No label column found in {test_path}. "
                         f"Expected one of: {label_cands}. "
                         f"Columns: {list(df.columns)[:12]}...")

    texts  = df[text_col].astype(str).tolist()
    y_true = df[label_col].astype(int).to_numpy()

    print(f"[TEST] Loaded {len(texts)} rows. label_col={label_col}")

    # 4) Forward pass
    logits_list = []
    with torch.inference_mode():
        for chunk in batchify(texts, batch_size):
            enc = tok(chunk, return_tensors="pt", truncation=True,
                      padding=True, max_length=max_len).to(device)
            out = model(enc["input_ids"], attention_mask=enc["attention_mask"])
            logits = out.logits if hasattr(out, "logits") else out
            logits_list.append(logits.detach().cpu().numpy())

    logits = np.concatenate(logits_list, axis=0)

    # 5) Metrics, save artifacts
    results = {}
    if num_labels == 2:
        probs2 = np_softmax(logits)      # (N,2)
        pos    = probs2[:, 1]
        y_pred = (pos >= 0.5).astype(int)

        results["f1_macro"]    = f1_score(y_true, y_pred, average="macro")
        results["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")
        results["precision"]   = precision_score(y_true, y_pred, average="binary", zero_division=0)
        results["recall"]      = recall_score(y_true, y_pred, average="binary", zero_division=0)
        results["roc_auc"]     = roc_auc_score(y_true, pos)
        results["log_loss"]    = log_loss(y_true, probs2)
        results["brier_score"] = brier_score(y_true, pos)

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=4)

        # Save confusion matrix & report
        pd.DataFrame(cm).to_csv(os.path.join(out_dir, "confusion_matrix.csv"), index=False)
        with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
            f.write(report)

        # ROC points
        fpr, tpr, thr = roc_curve(y_true, pos)
        pd.DataFrame({"fpr":fpr, "tpr":tpr, "threshold":thr}).to_csv(
            os.path.join(out_dir, "roc_points.csv"), index=False
        )

        # Calibration table (10 bins)
        bins = np.linspace(0, 1, 11)
        idx  = np.digitize(pos, bins) - 1
        rows = []
        for b in range(10):
            mask = (idx == b)
            rows.append({
                "bin": b,
                "count": int(mask.sum()),
                "mean_prob": float(pos[mask].mean()) if mask.any() else np.nan,
                "empirical_acc": float(np.mean(y_true[mask] == 1)) if mask.any() else np.nan
            })
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "calibration_table.csv"), index=False)

        # Per-sample predictions
        pd.DataFrame({
            "text": texts,
            "y_true": y_true,
            "y_pred": y_pred,
            "prob_pos": pos
        }).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

        # Metrics JSON
        ser = {k: to_py(v) for k, v in results.items()}
        ser["confusion_matrix"] = cm.tolist()
        with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
            json.dump(ser, f, indent=2)

        # Console print
        print("\n========== Test Metrics ==========")
        for k in ["f1_macro","f1_weighted","precision","recall","roc_auc","log_loss","brier_score"]:
            print(f"{k.replace('_',' ').title():<18} {results[k]:.4f}")
        print("Confusion Matrix:\n", cm)
        print("\nClassification Report:\n", report)
        cm = confusion_matrix(y_true, y_pred)
    # NEW: save images
        save_cm_images(cm, np.unique(y_true), out_dir)

    else:
        # Multiclass (C >= 3)
        probs_mc = np_softmax(logits)
        y_pred   = probs_mc.argmax(axis=1)

        results["f1_macro"]        = f1_score(y_true, y_pred, average="macro")
        results["f1_weighted"]     = f1_score(y_true, y_pred, average="weighted")
        results["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        results["recall_macro"]    = recall_score(y_true, y_pred, average="macro", zero_division=0)
        try:
            results["roc_auc_ovr"] = roc_auc_score(y_true, probs_mc, multi_class="ovr")
        except ValueError:
            results["roc_auc_ovr"] = float("nan")
        results["log_loss"]    = log_loss(y_true, probs_mc)
        results["brier_score"] = brier_score(y_true, probs_mc)

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=4)

        # Save confusion matrix & report
        pd.DataFrame(cm).to_csv(os.path.join(out_dir, "confusion_matrix.csv"), index=False)
        with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
            f.write(report)

        # Per-sample predictions
        pd.DataFrame({
            "text": texts,
            "y_true": y_true,
            "y_pred": y_pred,
            "prob_pred": probs_mc[np.arange(len(y_true)), y_pred]
        }).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

        # Metrics JSON
        ser = {k: to_py(v) for k, v in results.items()}
        ser["confusion_matrix"] = cm.tolist()
        with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
            json.dump(ser, f, indent=2)

        # Console print
        print("\n========== Test Metrics ==========")
        for k in ["f1_macro","f1_weighted","precision_macro","recall_macro","roc_auc_ovr","log_loss","brier_score"]:
            if k in results and isinstance(results[k], float) and not math.isnan(results[k]):
                print(f"{k.replace('_',' ').title():<18} {results[k]:.4f}")
        print("Confusion Matrix:\n", cm)
        print("\nClassification Report:\n", report)


if __name__ == "__main__":
    main()
