# print_context_text.py
# Print human-readable "context to classifier" for a HF SeqClass model (e.g., AlloyBERT on RoBERTa)

import argparse, sys, math
import numpy as np
import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ----------------------------- helpers -----------------------------

def normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo + eps)

def strip_special(tokens: List[str], *cols: np.ndarray):
    keep_t, keep_cols = [], [[] for _ in cols]
    specials = {"<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]"}
    for i, t in enumerate(tokens):
        if t in specials:
            continue
        keep_t.append(t)
        for k, c in enumerate(cols):
            keep_cols[k].append(float(c[i]))
    return keep_t, [np.array(v, dtype=np.float32) for v in keep_cols]

def ascii_bar(v: float, width: int = 22, char: str = "█") -> str:
    v = max(0.0, min(1.0, float(v)))
    n = int(round(v * width))
    return char * n + " " * (width - n)

def pretty_table(rows: List[Tuple[str, float, float, float]], title: str, k: int = 15):
    print("\n" + title)
    print("-" * len(title))
    head = "{:<18} {:>9}  {:>9}  {:>9}  | {}".format("token", "attn", "saliency", "combined", "bar")
    print(head)
    print("-" * len(head))
    for tok, att, sal, comb in rows[:k]:
        bar = ascii_bar(comb, width=22)
        print("{:<18} {:>9.4f}  {:>9.4f}  {:>9.4f}  | {}".format(tok, att, sal, comb, bar))

def one_line_summary(tokens: List[str], comb: np.ndarray, top_n: int = 5) -> str:
    idx = np.argsort(-comb)[:min(top_n, len(tokens))]
    picks = [tokens[i] for i in idx]
    if not picks:
        return "No salient tokens (input too short or all masked)."
    if len(picks) == 1:
        return f"Model focuses mainly on '{picks[0]}'."
    if len(picks) == 2:
        return f"Model focuses on '{picks[0]}' and '{picks[1]}'."
    return "Model focuses on " + ", ".join(f"'{p}'" for p in picks[:-1]) + f", and '{picks[-1]}'."


# ---------------------------- core logic ---------------------------

@torch.no_grad()
def cls_attention_scores(model, inputs):
    """
    Returns last-layer CLS->token attention averaged over heads. Shape [T]
    """
    outputs = model(**inputs, output_attentions=True, return_dict=True)
    # attentions: list[L] of [B, H, T, T]
    last = outputs.attentions[-1][0]           # [H, T, T] (batch=1)
    cls_to_all = last[:, 0, :]                 # [H, T]
    att = cls_to_all.mean(dim=0).cpu().numpy() # [T]
    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    yhat = int(logits.argmax(dim=-1).item())
    return att, yhat, probs

def gradient_saliency(model, inputs, yhat: int):
    """
    Computes saliency per token as || d logit_yhat / d inputs_embeds ||_2
    using inputs_embeds path to make token-wise grads straightforward.
    """
    # Build inputs_embeds
    emb_weight = model.get_input_embeddings().weight        # [V, H]
    input_ids = inputs["input_ids"]                         # [1, T]
    attn_mask = inputs["attention_mask"]
    inputs_embeds = torch.index_select(emb_weight, 0, input_ids[0]).unsqueeze(0)  # [1, T, H]
    inputs_embeds = inputs_embeds.detach().clone().requires_grad_(True)

    # Forward with inputs_embeds
    out = model(inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                output_attentions=False,
                return_dict=True)
    logit = out.logits[0, yhat]
    model.zero_grad(set_to_none=True)
    if inputs_embeds.grad is not None:
        inputs_embeds.grad.zero_()
    logit.backward()
    grad = inputs_embeds.grad.detach().norm(dim=-1).squeeze(0).cpu().numpy()  # [T]
    return grad

def explain_text(model, tokenizer, text: str, device: torch.device, k: int = 15):
    # Tokenize
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=False, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu().tolist())

    # Attention view
    att_raw, yhat, probs = cls_attention_scores(model, enc)

    # Saliency view
    sal_raw = gradient_saliency(model, enc, yhat)

    # Normalize, strip specials, combine
    att_n = normalize(att_raw)
    sal_n = normalize(sal_raw)
    tokens_ns, (att_ns, sal_ns) = strip_special(tokens, att_n, sal_n)
    comb = 0.5 * att_ns + 0.5 * sal_ns

    # Rank
    order = np.argsort(-comb)
    table = [(tokens_ns[i], float(att_ns[i]), float(sal_ns[i]), float(comb[i])) for i in order]

    # Print: input, prediction, table, summary
    print("\nINPUT")
    print("-----")
    print(text)

    print("\nPREDICTION")
    print("----------")
    prob_str = ", ".join([f"class {i}: {p:.4f}" for i, p in enumerate(probs)])
    print(f"class = {yhat}   |   probs = [{prob_str}]")

    pretty_table(table, "Top tokens (attention, saliency, combined)", k=k)

    print("\nSUMMARY")
    print("-------")
    print(one_line_summary(tokens_ns, comb, top_n=min(5, len(tokens_ns))))

    # Optional: terse full rank
    print("\nFULL RANK (token -> combined)")
    print("-----------------------------")
    for tok, _, _, c in table:
        print(f"{tok:>18}  {c:0.4f}")


# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Print human-readable context from transformer to classifier head")
    ap.add_argument("--model", type=str, required=True,
                    help="Path or hub id for your fine-tuned sequence classification model (e.g., ./outputs/alloybert_cls)")
    ap.add_argument("--text", type=str, default=None,
                    help="Input text. If omitted, a demo alloy description is used.")
    ap.add_argument("--topk", type=int, default=15, help="How many tokens to display")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    args = ap.parse_args()

    # Device pick
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device("cpu")
    print(f"[Device] {device}")

    # Load model & tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model.to(device).eval()

    # Input text
    text = args.text or (
        "Test temp=1000, YS=850, Solidus=3700, W=0.9, Nb=0.1, Hf=0.0, Re=0.0, Zr=0.0, "
        "Ta=0.0, Ti=0.0, C=0.0, Y=0.0, Al=0.0, Si=0.0, V=0.0, Cr=0.0"
    )

    explain_text(model, tokenizer, text, device, k=args.topk)


if __name__ == "__main__":
    main()
