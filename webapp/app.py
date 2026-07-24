"""
Small web app for interacting with a fine-tuned BERT classifier.

Usage:
    python app.py --model_dir ../bert-finetuned --labels negative,positive

Then open http://localhost:5000 in a browser.
"""

import argparse
import torch
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizerFast, BertForSequenceClassification

app = Flask(__name__)

# Populated in main() before app.run()
model = None
tokenizer = None
device = None
label_names = None
max_length = 256


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some text."}), 400

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().tolist()

    if isinstance(probs, float):  # single-label edge case safety
        probs = [probs]

    pred_idx = int(torch.tensor(probs).argmax())
    result = {
        "label": label_names[pred_idx] if pred_idx < len(label_names) else str(pred_idx),
        "confidence": probs[pred_idx],
        "probabilities": [
            {"label": label_names[i] if i < len(label_names) else str(i), "value": p}
            for i, p in enumerate(probs)
        ],
    }
    return jsonify(result)


def main():
    global model, tokenizer, device, label_names, max_length

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="../bert-finetuned",
                         help="Path to the fine-tuned model saved by finetune_bert.py")
    parser.add_argument("--labels", type=str, default="negative,positive",
                         help="Comma-separated label names in class-index order")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    label_names = args.labels.split(",")
    max_length = args.max_length

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model_dir} on {device}...")
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()
    print("Model loaded. Starting server...")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
