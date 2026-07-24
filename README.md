# Fine-Tuning BERT for Text Classification

This project fine-tunes `bert-base-uncased` on the IMDB sentiment dataset (positive/negative movie reviews), but works with any Hugging Face dataset that has `text` and `label` fields.

## Setup

```bash
pip install -r requirements.txt
```

Requires internet access to download the pretrained BERT weights and dataset from Hugging Face on first run (they get cached locally afterward).

### GPU setup on Windows (read this if `torch.cuda.is_available()` prints `False`)

Installing the CUDA toolkit/drivers alone does **not** give you GPU support — `pip install torch` by default installs the CPU-only build. You need the CUDA-specific build, and it only exists for certain Python versions.

1. **Check your Python version.** PyTorch's CUDA wheels currently support **Python 3.9–3.12**. Python 3.13 and 3.14 are too new and will fail with `ERROR: No matching distribution found for torch`.
   ```bash
   python --version
   ```
   If you're on 3.13/3.14, install Python 3.12 from [python.org](https://www.python.org/downloads/) alongside your existing version (no need to remove it), then create a fresh venv with it:
   ```bash
   py -3.12 -m venv .venv312
   .venv312\Scripts\activate
   ```

2. **Check your driver/CUDA version:**
   ```bash
   nvidia-smi
   ```
   The "CUDA Version" shown here is the *maximum* your driver supports — install a PyTorch build at or below that.

3. **Install the CUDA-enabled PyTorch build** (example for CUDA 12.4):
   ```bash
   pip uninstall torch torchvision torchaudio -y
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```
   Use `cu121` instead of `cu124` if `cu124` isn't available for your setup.

4. **Verify:**
   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```
   You want a version ending in `+cu1xx` (not `+cpu`) and `True`.

5. Reinstall the rest of the project's dependencies inside this same venv:
   ```bash
   pip install -r requirements.txt
   ```

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

## Web UI

A small browser interface lives in `webapp/` — type text in, hit Analyze, see the predicted label, confidence, and a meter showing where it falls between negative and positive.

```bash
cd webapp
python app.py --model_dir ../bert-finetuned --labels negative,positive
```

Then open `http://localhost:5000`. Adjust `--labels` to match your `--num_labels` and class order if you fine-tuned on a different dataset.

## Notes on hardware

- BERT-base fine-tuning is very feasible on a single GPU (even a free Colab T4). CPU-only will work but is slow.
- If you're memory-constrained, lower `--batch_size` and/or `--max_length`, or switch `--model_name` to `distilbert-base-uncased`.
- Typical fine-tuning only needs 2-4 epochs — BERT overfits quickly on small datasets since it's already pretrained.

### Low-VRAM GPUs (4GB and under, e.g. GTX 1650)

Full `bert-base-uncased` at the default `--batch_size 16 --max_length 256` will likely throw `CUDA out of memory` on a 4GB card. Start small and confirm the pipeline works before a full run:

```bash
python finetune_bert.py --train_subset 200 --eval_subset 100 --epochs 1 --batch_size 2 --max_length 96
```

If that succeeds with no OOM error, scale up gradually (`--batch_size 4` or `8`, `--max_length 128`) until you find the largest values that fit. Training will be slower with small batches, but still correct — just budget more time (a full 25k-example IMDB run can take an hour or more on a 4GB card).

### Known harmless error on Windows

After training completes and "Model saved to: ..." prints, you may see:
```
Exception ignored in: <function ResourceTracker.__del__...>
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
```
This is a known Python multiprocessing cleanup quirk on Windows that fires *after* your model has already been saved successfully. It's safe to ignore.

## What's actually happening

1. `BertForSequenceClassification` loads the pretrained BERT encoder and attaches a fresh linear layer on top of the `[CLS]` token's embedding.
2. All weights (pretrained BERT + new head) are updated together during training, using a small learning rate so the pretrained knowledge isn't destroyed.
3. A linear warmup + decay schedule is used, which is the standard recipe from the original BERT paper.
