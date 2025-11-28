#!/usr/bin/env python3
"""Fine-tune a transformer for binary/multi-class classification using Hugging Face Trainer.

Usage example:
  python src/train.py \
    --train_file data/processed/train.csv \
    --validation_file data/processed/val.csv \
    --model_name_or_path distilbert-base-uncased \
    --output_dir outputs/distilbert_finetuned \
    --per_device_train_batch_size 16 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --max_seq_length 256
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, ClassLabel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import transformers
from packaging import version

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

# Runtime sanity check for versions (helps with confusing TypeError on TrainingArguments)
MIN_TRANSFORMERS = version.parse("4.4.0")
try:
    tf_ver = version.parse(transformers.__version__)
except Exception:
    tf_ver = None
if tf_ver is None or tf_ver < MIN_TRANSFORMERS:
    raise RuntimeError(
        f"Detected transformers version {transformers.__version__!s}. "
        f"Trainer requires transformers>={MIN_TRANSFORMERS}. Please upgrade: `pip install -U transformers`"
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a text classifier with Hugging Face Trainer")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--validation_file", required=True)
    parser.add_argument("--model_name_or_path", default="distilbert-base-uncased")
    parser.add_argument("--output_dir", default="outputs/distilbert_finetuned")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary" if len(np.unique(labels)) == 2 else "macro")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    logger.info("Loading datasets")
    data_files = {"train": args.train_file, "validation": args.validation_file}
    raw_datasets = load_dataset("csv", data_files=data_files)

    # Ensure label column exists and is integer
    if "label" not in raw_datasets["train"].column_names:
        raise ValueError("Input CSVs must contain a 'label' column with integer labels (0/1 or 0..k-1)")

    # Determine label info
    labels = list(set(raw_datasets["train"]["label"]))
    labels_sorted = sorted(labels)
    num_labels = len(labels_sorted)
    logger.info(f"Detected labels: {labels_sorted} (num_labels={num_labels})")

    # Load tokenizer and model config
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    # Tokenize
    max_len = args.max_seq_length

    def preprocess_function(examples):
        texts = examples.get("text")
        return tokenizer(texts, truncation=True, max_length=max_len)

    tokenized_train = raw_datasets["train"].map(preprocess_function, batched=True, remove_columns=[c for c in raw_datasets["train"].column_names if c != "text" and c != "label"]) 
    tokenized_eval = raw_datasets["validation"].map(preprocess_function, batched=True, remove_columns=[c for c in raw_datasets["validation"].column_names if c != "text" and c != "label"]) 

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Build TrainingArguments kwargs dynamically to remain compatible with
    # different transformers versions (some older/newer versions may not accept
    # certain keyword args). We inspect the TrainingArguments __init__ signature
    # and only pass supported keys.
    import inspect

    ta_kwargs = dict(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=args.fp16,
        logging_dir=str(output_dir / "logs"),
        logging_strategy="steps",
        logging_steps=200,
        seed=args.seed,
    )

    # Filter to supported args
    sig = inspect.signature(TrainingArguments.__init__)
    supported = {k for k in sig.parameters.keys() if k != "self"}
    filtered_kwargs = {k: v for k, v in ta_kwargs.items() if k in supported}
    if len(filtered_kwargs) != len(ta_kwargs):
        missing = set(ta_kwargs.keys()) - set(filtered_kwargs.keys())
        logger.warning(f"TrainingArguments does not support keys: {missing}; they will be skipped")

    # If evaluation is not supported we must not set load_best_model_at_end or metric_for_best_model
    # because Trainer will raise an error requiring matching save/eval strategies.
    if 'evaluation_strategy' not in supported:
        if filtered_kwargs.pop('load_best_model_at_end', None) is not None:
            logger.warning("Removed 'load_best_model_at_end' because evaluation is not supported")
        if filtered_kwargs.pop('metric_for_best_model', None) is not None:
            logger.warning("Removed 'metric_for_best_model' because evaluation is not supported")

    training_args = TrainingArguments(**filtered_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training")
    trainer.train()
    logger.info("Training finished. Saving model and tokenizer.")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    logger.info("Done")


if __name__ == "__main__":
    main()
