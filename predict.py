"""
Run predictions with a fine-tuned BERT model.

Usage:
    python predict.py --model_dir ./bert-finetuned --text "This movie was surprisingly great!"
"""

import argparse
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./bert-finetuned")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    inputs = tokenizer(
        args.text,
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()
        pred_label = torch.argmax(probs).item()

    print(f"Input text: {args.text}")
    print(f"Predicted label: {pred_label}")
    print(f"Class probabilities: {probs.cpu().numpy()}")


if __name__ == "__main__":
    main()
