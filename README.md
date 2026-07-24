# Fine-Tuning BERT for Text Classification

This project fine-tunes `bert-base-uncased` on the IMDB sentiment dataset (positive/negative movie reviews), but works with any Hugging Face dataset that has `text` and `label` fields.

## Setup

```bash
pip install -r requirements.txt
```

Requires internet access to download the pretrained BERT weights and dataset from Hugging Face on first run (they get cached locally afterward).

## Fine-tune

Full run:
```bash
python finetune_bert.py --epochs 3 --batch_size 16 --lr 2e-5
```

Quick smoke test (small subset, to confirm everything works before a full run):
```bash
python finetune_bert.py --train_subset 200 --eval_subset 100 --epochs 1
```

Key arguments:
- `--model_name`: pretrained checkpoint (e.g. `bert-base-uncased`, `bert-large-uncased`, `distilbert-base-uncased` for a smaller/faster model)
- `--dataset`: any Hugging Face text-classification dataset
- `--num_labels`: number of output classes
- `--max_length`: max token length per input (longer = more memory/time)
- `--lr`: 2e-5 to 5e-5 is the standard range for fine-tuning BERT

## Predict with the fine-tuned model

```bash
python predict.py --model_dir ./bert-finetuned --text "This movie was surprisingly great!"
```

## Notes on hardware

- BERT-base fine-tuning is very feasible on a single GPU (even a free Colab T4). CPU-only will work but is slow.
- If you're memory-constrained, lower `--batch_size` and/or `--max_length`, or switch `--model_name` to `distilbert-base-uncased`.
- Typical fine-tuning only needs 2-4 epochs — BERT overfits quickly on small datasets since it's already pretrained.

## What's actually happening

1. `BertForSequenceClassification` loads the pretrained BERT encoder and attaches a fresh linear layer on top of the `[CLS]` token's embedding.
2. All weights (pretrained BERT + new head) are updated together during training, using a small learning rate so the pretrained knowledge isn't destroyed.
3. A linear warmup + decay schedule is used, which is the standard recipe from the original BERT paper.
