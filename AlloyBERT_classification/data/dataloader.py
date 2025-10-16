from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast
import pandas as pd
from data.dataset import PretrainDataset, FinetuneDataset
import os


def load_data(config):
    print(f'{"="*30}{"DATA":^20}{"="*30}')

    # -----------------------------
    # Load data (supports CSV or PKL)
    # -----------------------------
    train_path = config["paths"]["train_data"]
    val_path = config["paths"]["val_data"]

    if train_path.endswith(".csv"):
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
    else:
        df_train = pd.read_pickle(train_path)
        df_val = pd.read_pickle(val_path)

    # Ensure text/target columns exist
    text_col = config.get("text_column", "text")
    target_col = config.get("target_column", "target")

    if text_col not in df_train.columns or target_col not in df_train.columns:
        raise KeyError(
            f"Missing text or target columns in dataset. "
            f"Expected: '{text_col}' and '{target_col}'"
        )

    # -----------------------------
    # Tokenizer
    # -----------------------------
    tokenizer = RobertaTokenizerFast.from_pretrained(
        config["paths"]["tokenizer"],
        max_len=512
    )

    # -----------------------------
    # Dataset creation
    # -----------------------------
    if config["stage"] == "pretrain":
        train_dataset = PretrainDataset(
            texts=df_train[text_col].values,
            tokenizer=tokenizer,
            max_length=tokenizer.model_max_length,
        )
        val_dataset = PretrainDataset(
            texts=df_val[text_col].values,
            tokenizer=tokenizer,
            max_length=tokenizer.model_max_length,
        )

    elif config["stage"] == "finetune":
        train_dataset = FinetuneDataset(
            texts=df_train[text_col].values,
            targets=df_train[target_col].values,
            tokenizer=tokenizer,
            max_length=tokenizer.model_max_length,
        )
        val_dataset = FinetuneDataset(
            texts=df_val[text_col].values,
            targets=df_val[target_col].values,
            tokenizer=tokenizer,
            max_length=tokenizer.model_max_length,
        )

    # -----------------------------
    # DataLoaders (macOS-safe)
    # -----------------------------
    num_workers = config.get("dataloader", {}).get("num_workers", 0)
    pin_memory = config.get("dataloader", {}).get("pin_memory", False)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # -----------------------------
    # Logging
    # -----------------------------
    config["train_len"] = len(train_data_loader)

    print("Batch size: ", config["batch_size"])
    print("Train dataset samples: ", len(train_dataset))
    print("Validation dataset samples: ", len(val_dataset))
    print("Train dataset batches: ", len(train_data_loader))
    print("Validation dataset batches: ", len(val_data_loader))
    print()

    return train_data_loader, val_data_loader
