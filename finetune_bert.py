"""
Fine-tune BERT for text classification (binary sentiment analysis on IMDB).

Usage:
    python finetune_bert.py --epochs 3 --batch_size 16 --lr 2e-5

This script:
  1. Loads a pretrained BERT ("bert-base-uncased") from Hugging Face.
  2. Attaches a classification head on top (handled by BertForSequenceClassification).
  3. Tokenizes a text classification dataset (IMDB by default).
  4. Fine-tunes the whole model end-to-end on the labeled data.
  5. Evaluates accuracy / F1 on a held-out split.
  6. Saves the fine-tuned model + tokenizer to disk for later inference.
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for text classification")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                         help="Pretrained BERT checkpoint to start from")
    parser.add_argument("--dataset", type=str, default="imdb",
                         help="Hugging Face dataset name (must have 'text' and 'label' fields)")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--train_subset", type=int, default=None,
                         help="Optional: limit number of training examples (useful for quick tests)")
    parser.add_argument("--eval_subset", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./bert-finetuned")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize_dataset(dataset, tokenizer, max_length):
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    avg_loss = total_loss / len(dataloader)
    return {"loss": avg_loss, "accuracy": acc, "f1": f1}


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    print(f"Loading dataset: {args.dataset}")
    raw_datasets = load_dataset(args.dataset)
    train_data = raw_datasets["train"]
    eval_data = raw_datasets["test"] if "test" in raw_datasets else raw_datasets["validation"]

    if args.train_subset:
        train_data = train_data.shuffle(seed=args.seed).select(range(args.train_subset))
    if args.eval_subset:
        eval_data = eval_data.shuffle(seed=args.seed).select(range(args.eval_subset))

    # 2. Tokenizer + model (pretrained BERT + new classification head)
    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name, num_labels=args.num_labels
    ).to(device)

    # 3. Tokenize
    train_tok = tokenize_dataset(train_data, tokenizer, args.max_length)
    eval_tok = tokenize_dataset(eval_data, tokenizer, args.max_length)

    train_loader = DataLoader(train_tok, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_tok, batch_size=args.batch_size)

    # 4. Optimizer + LR schedule (standard recipe for fine-tuning BERT)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 5. Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if step % 50 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Step {step}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        metrics = evaluate(model, eval_loader, device)
        print(f"\n=== Epoch {epoch+1} done ===")
        print(f"Train loss: {avg_train_loss:.4f}")
        print(f"Eval  loss: {metrics['loss']:.4f} | "
              f"Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}\n")

    # 6. Save the fine-tuned model + tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
