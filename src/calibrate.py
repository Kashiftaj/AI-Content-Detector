import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Temperature scaling calibration")
    p.add_argument("--model_path", required=True, help="Path to model (checkpoint or output dir)")
    p.add_argument("--validation_file", required=False, help="Validation CSV file (if tokenized dataset not provided)")
    p.add_argument("--preprocessed_dir", required=False, help="If you saved a tokenized dataset with save_to_disk, point here")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default=None, help="cuda or cpu (auto-detected if not provided)")
    p.add_argument("--out", type=str, default=None, help="Where to save learned temperature (JSON). Defaults to <model_path>/temperature.json")
    return p.parse_args()


def load_eval_dataset(args, tokenizer, max_length=256):
    if args.preprocessed_dir and Path(args.preprocessed_dir).exists():
        logger.info("Loading tokenized dataset from %s", args.preprocessed_dir)
        ds = load_from_disk(args.preprocessed_dir)
        val = ds["validation"]
        return val

    if args.validation_file:
        logger.info("Loading raw validation CSV from %s and tokenizing on the fly", args.validation_file)
        raw = load_dataset("csv", data_files={"validation": args.validation_file})
        tokenizer_fn = lambda examples: tokenizer(examples["text"], truncation=True, padding=False, max_length=max_length)
        tok = raw["validation"].map(tokenizer_fn, batched=True, batch_size=1000)
        return tok

    raise ValueError("Provide either --preprocessed_dir or --validation_file to load validation data")


def gather_logits(model, dataset, tokenizer, batch_size=64, device=None):
    from torch.utils.data import DataLoader

    # Create a collate function that tokenizes raw text per batch. This is robust
    # whether the dataset is raw (has 'text') or already tokenized.
    def collate_fn(examples):
        # examples: list of dicts
        # Clean examples: drop raw string fields (e.g., 'text') so the HF
        # DataCollator doesn't attempt to convert strings to tensors.
        cleaned_examples = []
        for ex in examples:
            if isinstance(ex, dict):
                cleaned = {k: v for k, v in ex.items() if not isinstance(v, str)}
            else:
                # unexpected item type (e.g., plain string), coerce to dict
                cleaned = {"text": ex}
            cleaned_examples.append(cleaned)

        # If examples already contain input_ids, let the HF collator pad them.
        if "input_ids" in cleaned_examples[0]:
            return DataCollatorWithPadding(tokenizer)(cleaned_examples)

        texts = [ex.get("text", "") for ex in examples]
        enc = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        if "label" in examples[0]:
            enc["label"] = torch.tensor([ex.get("label", -1) for ex in examples], dtype=torch.long)
        return enc

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    model.to(device)
    model.eval()

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("label") if "label" in batch else None
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.detach().cpu()
            all_logits.append(logits)
            if labels is not None:
                all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0) if all_labels else None
    return all_logits, all_labels


def temperature_scale(logits, temperature):
    # logits: Tensor [N, C]
    return logits / temperature


def nll_criterion(logits, labels):
    return nn.CrossEntropyLoss()(logits, labels)


def fit_temperature(logits, labels, device, maxiter=50):
    # Learn log_temp to ensure positivity
    log_temp = torch.zeros(1, requires_grad=True, device=device)
    optimizer = torch.optim.LBFGS([log_temp], max_iter=maxiter, lr=0.1)

    labels = labels.to(device)
    logits = logits.to(device)

    def closure():
        optimizer.zero_grad()
        temp = torch.exp(log_temp)
        scaled = logits / temp
        loss = nn.CrossEntropyLoss()(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    learned_temp = float(torch.exp(log_temp).item())
    return learned_temp


def main():
    args = parse_args()
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize and resolve paths to avoid HF hub treating backslashes as repo ids on Windows
    model_dir = Path(args.model_path).expanduser().resolve()
    if args.preprocessed_dir:
        args.preprocessed_dir = str(Path(args.preprocessed_dir).expanduser().resolve())

    # Use POSIX-style path (forward slashes) to avoid HF hub repo-id validation issues on Windows
    model_dir_posix = model_dir.as_posix()

    try:
        # Force local-only loading to avoid HF hub repo-id parsing on Windows
        tokenizer = AutoTokenizer.from_pretrained(model_dir_posix, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir_posix, local_files_only=True)
    except Exception as e:
        logger.error("Failed to load model/tokenizer from %s: %s", model_dir_posix, e)
        # Show helpful diagnostics: existence + directory listing
        try:
            exists = model_dir.exists()
            logger.error("Path exists: %s", exists)
            if exists:
                files = list(model_dir.iterdir())
                logger.error("Files in %s:\n%s", model_dir_posix, "\n".join(str(p.name) for p in files))
        except Exception as ex2:
            logger.error("Could not list files in %s: %s", model_dir_posix, ex2)
        raise

    val_ds = load_eval_dataset(args, tokenizer)
    logger.info("Gathering logits on validation set (this can take a while)...")
    logits, labels = gather_logits(model, val_ds, tokenizer, batch_size=args.batch_size, device=device)

    if labels is None:
        raise ValueError("Validation dataset must contain labels for calibration")

    logger.info("Fitting temperature on %d examples", logits.shape[0])
    temp = fit_temperature(logits, labels, device)
    logger.info("Learned temperature: %.4f", temp)

    out_path = args.out or (Path(model_dir) / "temperature.json")
    with open(out_path, "w") as f:
        json.dump({"temperature": temp}, f)
    logger.info("Saved temperature to %s", out_path)


if __name__ == "__main__":
    main()
