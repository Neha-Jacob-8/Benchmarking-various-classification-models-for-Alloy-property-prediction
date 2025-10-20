import torch
from tqdm import tqdm


def train_pt(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss, lr = 0.0, 0.0

    for batch in tqdm(dataloader):
        ids = batch["ids"].to(device)
        attention_mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        output = model(ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss

        total_loss += loss.item()
        lr += optimizer.param_groups[0]["lr"]

        loss.backward()
        optimizer.step()
        scheduler.step()

    return (total_loss / len(dataloader)), (lr / len(dataloader))


def validate_pt(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(dataloader):
        ids = batch["ids"].to(device)
        attention_mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.inference_mode():
            output = model(ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_ft(model, dataloader, optimizer, criterion, scheduler, device):
    """
    Finetune step that supports both regression and classification.
    Uses model.task_type set in network.py ("classification" or "regression").
    Returns: (avg_loss, avg_lr)
    """
    model.train()
    total_loss, lr = 0.0, 0.0

    # For optional metric: accuracy (cls) or MAE (reg)
    total_correct, total_examples = 0, 0
    total_abs_err = 0.0

    task_type = getattr(model, "task_type", "regression")

    for batch in tqdm(dataloader):
        ids = batch["ids"].to(device)
        attention_mask = batch["mask"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        logits = model(ids, attention_mask=attention_mask)  # (B, C) for cls, (B,1) for reg

        if task_type == "classification":
            # CE expects class indices (Long)
            target = target.long()
            loss = criterion(logits, target)                  # CrossEntropyLoss
            # accuracy (optional)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == target).sum().item()
            total_examples += target.numel()
        else:
            # regression: squeeze to (B,)
            preds = logits.squeeze(-1)
            loss = criterion(preds, target.float())           # MSELoss
            # MAE (optional)
            total_abs_err += torch.mean(torch.abs(target.float() - preds)).item()

        total_loss += loss.item()
        lr += optimizer.param_groups[0]["lr"]

        loss.backward()
        optimizer.step()
        scheduler.step()

    # We keep the return signature unchanged (loss, lr)
    return (total_loss / len(dataloader)), (lr / len(dataloader))


def validate_ft(model, dataloader, criterion, device):
    """
    Validation step that supports both regression and classification.
    Returns: (avg_loss, secondary_metric)
      - For classification: secondary_metric = accuracy (0..1)
      - For regression:     secondary_metric = MAE
    NOTE: main.py may print the second metric as "MAE" even for classification.
    """
    model.eval()
    total_loss = 0.0

    # metrics
    total_correct, total_examples = 0, 0   # for classification
    total_abs_err = 0.0                    # for regression

    task_type = getattr(model, "task_type", "regression")

    for batch in tqdm(dataloader):
        ids = batch["ids"].to(device)
        attention_mask = batch["mask"].to(device)
        target = batch["target"].to(device)

        with torch.inference_mode():
            logits = model(ids, attention_mask=attention_mask)

            if task_type == "classification":
                target = target.long()
                loss = criterion(logits, target)
                preds = logits.argmax(dim=-1)
                total_correct += (preds == target).sum().item()
                total_examples += target.numel()
            else:
                preds = logits.squeeze(-1)
                loss = criterion(preds, target.float())
                total_abs_err += torch.mean(torch.abs(target.float() - preds)).item()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    if task_type == "classification":
        accuracy = (total_correct / total_examples) if total_examples > 0 else 0.0
        return avg_loss, accuracy
    else:
        mae = total_abs_err / len(dataloader)
        return avg_loss, mae


def load_pretrained(model, pretrained_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path)["model_state_dict"]

    matched_keys = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model.load_state_dict(matched_keys, strict=False)
    print("Pretrained Checkpoint Loaded")
