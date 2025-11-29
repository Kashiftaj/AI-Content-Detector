import os
# Suppress noisy TensorFlow / XLA startup messages BEFORE importing packages that trigger them
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Disable WANDB interactive prompts and cloud sync by default
os.environ.setdefault("WANDB_MODE", "offline")

import argparse
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import numpy as np
import evaluate
import inspect
import os
from datasets import load_from_disk

# ---------------- Setup Logging ---------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce noisy TensorFlow / XLA startup messages which commonly appear in Colab
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def parse_args():
    parser = argparse.ArgumentParser(description="Train AI Text Detector")

    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./model_output")

    # Training Parameters
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--fp16", action="store_true")

    # Optimization/Saving Arguments
    # NOTE: Set default to 4 for T4 speedup, but allow CLI override
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate gradients.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for gradient clipping.")
    
    # Save/Checkpointing Arguments (Defaults set for safe checkpointing)
    parser.add_argument("--save_strategy", type=str, default="steps", help="Checkpoint save strategy: 'no'|'epoch'|'steps'")
    parser.add_argument("--save_steps", type=int, default=500, help="Save every X updates when save_strategy='steps' (500 steps is recommended for safety).")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep (older ones deleted).")

    # Utility Arguments
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses for data loading (0 = main process)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch",
                        help="Evaluation strategy: 'no'|'epoch'|'steps'")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Patience (in evaluation steps) for early stopping")
    parser.add_argument("--metric_for_best_model", type=str, default="f1",
                        help="Metric name to use for best model selection (used by early stopping)")
    parser.add_argument("--no_load_best_model_at_end", dest="load_best_model_at_end", action="store_false",
                        help="Don't load the best model at end of training even if available")
    parser.set_defaults(load_best_model_at_end=True)
    parser.add_argument("--dry_run", action="store_true", help="Run a quick tokenization/label sanity check on a small sample and exit")
    parser.add_argument("--dry_run_size", type=int, default=500, help="Number of rows to sample for dry-run")
    parser.add_argument("--preprocess_only", action="store_true", help="Only run preprocessing (tokenize + save) and exit")
    parser.add_argument("--preprocessed_dir", type=str, default=None, help="Directory to save/load preprocessed tokenized dataset")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of processes to use for tokenization mapping")
    
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Data Loading and Tokenization (No Changes Needed) ---
    logger.info("=========== LOADING DATASETS ===========")
    data_files = {"train": args.train_file, "validation": args.validation_file}
    raw_datasets = load_dataset("csv", data_files=data_files)
    logger.info("=========== LOADING TOKENIZER ===========")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Fast dry-run: sample a small subset and run tokenization + label checks, then exit
    if args.dry_run:
        logger.info("Running dry-run with %d rows (quick tokenization + label checks)", args.dry_run_size)
        ds = raw_datasets["train"]
        try:
            total = len(ds)
        except Exception:
            total = None
        n = args.dry_run_size if (total is None or args.dry_run_size <= total) else total
        sample = ds.select(range(n)) if total is None or n <= total else ds.select(range(total))

        def safe_preprocess(batch):
            input_ids = []
            attention_mask = []
            for txt in batch.get("text", []):
                try:
                    enc = tokenizer(txt if isinstance(txt, str) else str(txt), truncation=True, max_length=args.max_seq_length)
                    input_ids.append(enc.get("input_ids", []))
                    attention_mask.append(enc.get("attention_mask", []))
                except Exception as e:
                    input_ids.append([])
                    attention_mask.append([])
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        tokenized_sample = sample.map(safe_preprocess, batched=True, batch_size=64)
        # Safely extract tokenized input_ids from the Dataset
        input_ids = []
        try:
            if "input_ids" in tokenized_sample.column_names:
                input_ids = tokenized_sample["input_ids"]
            else:
                # fallback: iterate examples
                for ex in tokenized_sample:
                    if "input_ids" in ex:
                        input_ids.append(ex["input_ids"])
        except Exception:
            # worst-case: convert to list of dicts
            for ex in list(tokenized_sample):
                if isinstance(ex, dict) and "input_ids" in ex:
                    input_ids.append(ex["input_ids"])

        import statistics
        lens = [len(x) for x in input_ids]
        empty = sum(1 for l in lens if l == 0)
        mean_len = statistics.mean(lens) if lens else 0
        max_len = max(lens) if lens else 0
        logger.info("Dry-run results: sample=%d, empty_tokenized=%d, mean_tokens=%.1f, max_tokens=%d", len(lens), empty, mean_len, max_len)

        # label checks
        if "label" in sample.column_names:
            from collections import Counter
            labels = sample["label"]
            logger.info("Label sample counts: %s", dict(Counter(labels)))
        else:
            logger.warning("No 'label' column found in sample; ensure data has labels for supervised training")

        logger.info("Dry-run complete — exiting without running training. Remove --dry_run to continue full run.")
        return

    # If a preprocessed_dir is provided and exists, load tokenized dataset from disk (fast)
    if args.preprocessed_dir and os.path.exists(args.preprocessed_dir):
        logger.info("Loading preprocessed dataset from %s", args.preprocessed_dir)
        tokenized = load_from_disk(args.preprocessed_dir)
        tokenized_train = tokenized["train"]
        tokenized_val = tokenized["validation"]
    else:
        # If user requested preprocess_only, run mapping and save to preprocessed_dir then exit
        if args.preprocess_only and not args.preprocessed_dir:
            raise ValueError("--preprocess_only requires --preprocessed_dir to be set to save results")
        if args.preprocess_only or (args.preprocessed_dir and not os.path.exists(args.preprocessed_dir)):
            logger.info("Preprocessing full dataset (tokenization). This may take several minutes.")
            def preprocess_batch(batch):
                return tokenizer(batch["text"], truncation=True, padding=False, max_length=args.max_seq_length)

            tokenized = raw_datasets.map(preprocess_batch, batched=True, batch_size=1000, num_proc=args.num_proc)
            # Optionally remove extra columns
            try:
                tokenized = tokenized.remove_columns([c for c in tokenized["train"].column_names if c not in ("input_ids", "attention_mask", "label")])
            except Exception:
                pass
            if args.preprocessed_dir:
                os.makedirs(args.preprocessed_dir, exist_ok=True)
                logger.info("Saving preprocessed dataset to %s", args.preprocessed_dir)
                tokenized.save_to_disk(args.preprocessed_dir)
            if args.preprocess_only:
                logger.info("Preprocess-only run complete. Exiting.")
                return
            tokenized_train = tokenized["train"]
            tokenized_val = tokenized["validation"]

    # If tokenized datasets were not created/loaded above, create them now
    if "tokenized_train" not in locals() or tokenized_train is None:
        def preprocess(example):
            return tokenizer(example["text"], truncation=True, max_length=args.max_seq_length)

        logger.info("=========== TOKENIZING DATASETS ===========")
        tokenized_train = raw_datasets["train"].map(preprocess, batched=True)
        tokenized_val = raw_datasets["validation"].map(preprocess, batched=True)
    labels = sorted(set(tokenized_train["label"]))
    num_labels = len(labels)
    logger.info(f"Detected labels: {labels} (num_labels={num_labels})")

    logger.info("=========== LOADING MODEL ===========")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    data_collator = DataCollatorWithPadding(tokenizer)

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels)["f1"],
        }

    logger.info("=========== SETTING TRAINING ARGS ===========")

    # Use args directly for dynamic values and set evaluation to match save steps
    ta_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        max_grad_norm=args.max_grad_norm,
        
        # --- CORRECTED/REMOVED CONFLICTING DUPLICATES ---
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        
        # Evaluation strategy (user-configurable)
        evaluation_strategy=args.evaluation_strategy,
        # Set eval steps to match save steps when using steps
        eval_steps=args.save_steps if args.evaluation_strategy == 'steps' else None,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        logging_steps=100,
        report_to="none",
    )

    # Introspection Logic (Handles evaluation_strategy vs eval_strategy renaming)
    try:
        sig = inspect.signature(TrainingArguments.__init__)
        supported = set(sig.parameters.keys())

        # Check for the renamed argument
        if "eval_strategy" in supported and "evaluation_strategy" in ta_kwargs and "eval_strategy" not in supported:
            ta_kwargs["eval_strategy"] = ta_kwargs.pop("evaluation_strategy")

        # Check if the save_steps argument should be passed (0 means disabled)
        if ta_kwargs.get("save_steps", 0) == 0:
            ta_kwargs['save_steps'] = None
            
        filtered_kwargs = {k: v for k, v in ta_kwargs.items() if k in supported}
        
        # Final cleanup for evaluation settings (omitted for brevity)

        training_args = TrainingArguments(**filtered_kwargs)
        used_ta_kwargs = filtered_kwargs
    except Exception as e:
        # Fallback remains, but should not be reached with proper dependency management
        logger.warning(f"Could not introspect TrainingArguments signature ({e}). Falling back to minimal args.")
        fallback_kwargs = dict(output_dir=args.output_dir, num_train_epochs=args.num_train_epochs, per_device_train_batch_size=args.per_device_train_batch_size, per_device_eval_batch_size=args.per_device_eval_batch_size,)
        training_args = TrainingArguments(**fallback_kwargs)
        used_ta_kwargs = fallback_kwargs
        supported = set()

    logger.info("Using TrainingArguments keys: %s", ", ".join(sorted(list(used_ta_kwargs.keys()))))

    # Attach EarlyStoppingCallback only if evaluation is enabled and a metric_for_best_model is set
    callbacks = []
    eval_supported = ("evaluation_strategy" in supported) or ("eval_strategy" in supported)
    if eval_supported and used_ta_kwargs.get("metric_for_best_model"):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
        # Warn if user didn't request loading best model at end
        if not used_ta_kwargs.get("load_best_model_at_end", False):
            logger.warning("Using EarlyStoppingCallback without load_best_model_at_end=True. "
                           "Once training is finished, the best model will not be loaded automatically.")
    else:
        if args.early_stopping_patience and not eval_supported:
            logger.warning("Early stopping requested but evaluation is disabled. Early stopping will be skipped.")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    logger.info("=========== STARTING TRAINING ===========")
    # Support resuming from a checkpoint if provided
    resume_ckpt = args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume_ckpt)

    logger.info("=========== SAVING MODEL ===========")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("=========== TRAINING COMPLETE ===========")


if __name__ == "__main__":
    main()