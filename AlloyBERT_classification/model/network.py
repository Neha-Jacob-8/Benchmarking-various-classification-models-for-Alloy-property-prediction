import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from transformers import (
    RobertaForMaskedLM,
    RobertaModel,
    get_scheduler,
    logging,
)
logging.set_verbosity_error()


class AlloyBERT(torch.nn.Module):
    def __init__(self, config, model):
        super(AlloyBERT, self).__init__()

        self.roberta = model

        # classification / regression toggle
        self.task_type = config.get("task_type", "regression")  # default
        num_labels = config.get("num_labels", 1)
        hidden_size = model.embeddings.word_embeddings.embedding_dim

        if self.task_type == "classification":
            assert num_labels >= 2, "num_labels must be >= 2 for classification"
            self.head = torch.nn.Sequential(
                torch.nn.Dropout(config.get("dropout", 0.1)),
                torch.nn.Linear(hidden_size, num_labels)
            )
        else:
            self.head = torch.nn.Linear(hidden_size, 1)

    def forward(self, inputs, attention_mask):
        out = self.roberta(inputs, attention_mask=attention_mask)
        pooled_output = out.pooler_output  # (B, H)
        logits = self.head(pooled_output)
        return logits


def create_model(config):
    if config["stage"] == "pretrain":
        model = RobertaForMaskedLM.from_pretrained("roberta-base").to(config["device"])
    else:  # finetune
        base = RobertaModel.from_pretrained("roberta-base")
        model = AlloyBERT(config, base).to(config["device"])
    return model


def cri_opt_sch(config, model):
    """Return criterion, optimizer, scheduler."""
    if config["stage"] == "pretrain":
        criterion = None
    else:
        if getattr(model, "task_type", "regression") == "classification":
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["optim"]["lr"])

    scheduler = get_scheduler(
        config["sch"]["name"],
        optimizer=optimizer,
        num_warmup_steps=config["sch"]["warmup_steps"],
        num_training_steps=int(config["train_len"] * config["epochs"]),
    )
    return criterion, optimizer, scheduler
