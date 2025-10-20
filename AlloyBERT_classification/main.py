# main.py  — clean training loop with proper metrics & no W&B prompts

import os
import yaml
import torch
import random
import numpy as np

from data.dataloader import load_data
from model.network import create_model, cri_opt_sch
from model.utils import train_pt, validate_pt, train_ft, validate_ft, load_pretrained


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_device(cfg):
    """
    Prefer the device specified in YAML (mps/cpu/cuda) if valid; else auto-detect.
    """
    yaml_dev = str(cfg.get("device", "")).lower()
    if yaml_dev in ("cuda", "cpu", "mps"):
        if yaml_dev == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        if yaml_dev == "mps" and not torch.backends.mps.is_available():
            return torch.device("cpu")
        return torch.device(yaml_dev)

    # auto-detect (prefer MPS on Apple, else CPU)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------
# Training runners
# -----------------------------
def run_pretrain(config):
    print(f'{"="*30}{"TRAINING":^20}{"="*30}')
    best_val = float("inf")

    for epoch in range(1, config["epochs"] + 1):
        train_loss, avg_lr = train_pt(model, train_loader, optimizer, scheduler, device)
        val_loss = validate_pt(model, val_loader, device)

        print(
            f"Epoch {epoch}/{config['epochs']} - "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {avg_lr:.8f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs("outputs", exist_ok=True)
            torch.save({"model_state_dict": model.state_dict()}, "outputs/best_model.pt")
            print("\nModel Saved\n")


def run_finetune(config):
    print(f'{"="*30}{"TRAINING":^20}{"="*30}')
    best_val = float("inf")

    for epoch in range(1, config["epochs"] + 1):
        # one epoch of training
        train_loss, avg_lr = train_ft(model, train_loader, optimizer, criterion, scheduler, device)

        # validation
        val_loss, val_metric = validate_ft(model, val_loader, criterion, device)

        # pretty print with correct metric name
        if getattr(model, "task_type", "regression") == "classification":
            # val_metric is accuracy in [0..1]
            print(
                f"Epoch {epoch}/{config['epochs']} - "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Acc: {val_metric*100:.2f}% | "
                f"LR: {avg_lr:.8f}"
            )
        else:
            # val_metric is MAE for regression
            print(
                f"Epoch {epoch}/{config['epochs']} - "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val MAE: {val_metric:.6f} | "
                f"LR: {avg_lr:.8f}"
            )

        # save best by validation loss
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs("outputs", exist_ok=True)
            torch.save({"model_state_dict": model.state_dict()}, "outputs/best_model.pt")
            print("\nModel Saved\n")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # 1) Load config & seed
    config = load_config("config.yaml")
    set_seed(config.get("seed", 42))

    # 2) Resolve device
    device = resolve_device(config)
    config["device"] = device
    print(f"Device: {device}\n")

    # 3) Build data
    train_loader, val_loader = load_data(config)

    # 4) Build model & training state
    model = create_model(config)
    if config.get("stage") == "finetune" and config.get("load_pretrained"):
        load_pretrained(model, config["paths"]["pretrained"])

    criterion, optimizer, scheduler = cri_opt_sch(config, model)

    # 5) Train
    if config.get("stage") == "pretrain":
        run_pretrain(config)
    else:
        run_finetune(config)

    # 6) Done—best model is saved at outputs/best_model.pt
