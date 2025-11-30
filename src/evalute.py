import os
import json
import time
from datetime import datetime
import logging
import argparse
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except Exception:
    sns = None
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# -----------------------------------------------------
# Logging
# -----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------
# Load Model + Tokenizer
# -----------------------------------------------------
def load_model(model_dir):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model folder not found: {model_dir}")

    logger.info(f"Loading model from {model_dir}...")

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return model, tokenizer, device


# -----------------------------------------------------
# Plot confusion matrix
# -----------------------------------------------------
def save_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Human", "AI"],
                    yticklabels=["Human", "AI"])
    else:
        # Fallback to matplotlib if seaborn isn't installed
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.xticks([0, 1], ["Human", "AI"])
        plt.yticks([0, 1], ["Human", "AI"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -----------------------------------------------------
# Evaluate Model
# -----------------------------------------------------
def evaluate_model(model_dir, test_file, output_metrics="outputs/logs/metrics.json",
                   output_plot="outputs/plots/confusion_matrix.png",
                   batch_size: int = 64,
                   status_file: str | None = None,
                   verbose: bool = True):

    # Load model
    model, tokenizer, device = load_model(model_dir)

    # Load test data
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test dataset not found: {test_file}")

    logger.info(f"Loading test dataset: {test_file}")
    df = pd.read_csv(test_file)

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    all_preds = []

    logger.info("Running evaluation in batches... total=%d examples, batch_size=%d", len(texts), batch_size)
    processed = 0
    all_preds = []

    iterator = range(0, len(texts), batch_size)
    if tqdm is not None and verbose:
        iterator = tqdm(iterator, desc="Evaluating", unit="batch")

    for i in iterator:
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().tolist()
            all_preds.extend(preds)
        processed += len(batch_texts)

        # Periodically write status to status_file (if provided) so you can inspect progress
        if status_file:
            try:
                partial_acc = None
                if len(all_preds) > 0:
                    partial_acc = float(accuracy_score(labels[:processed], all_preds))
                status = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "processed": processed,
                    "total": len(texts),
                    "partial_accuracy": partial_acc,
                    "last_batch_size": len(batch_texts)
                }
                os.makedirs(os.path.dirname(status_file), exist_ok=True)
                with open(status_file, "w") as sf:
                    json.dump(status, sf)
            except Exception:
                logger.exception("Failed to write status file %s", status_file)

        if verbose:
            logger.info("Processed %d/%d", processed, len(texts))

    # Metrics
    acc = accuracy_score(labels, all_preds)
    precision = precision_score(labels, all_preds)
    recall = recall_score(labels, all_preds)
    f1 = f1_score(labels, all_preds)

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }

    os.makedirs(os.path.dirname(output_metrics), exist_ok=True)
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)

    # Save metrics
    with open(output_metrics, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Saved metrics → {output_metrics}")

    # Save confusion matrix
    save_confusion_matrix(labels, all_preds, output_plot)
    logger.info(f"Saved confusion matrix → {output_plot}")

    return metrics


# -----------------------------------------------------
# Script entry point
# -----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate model on test set")
    parser.add_argument("--model_path", default="model/distilbert_finetuned", help="Path to model dir")
    parser.add_argument("--test_file", default="data/processed/test.csv", help="Path to test CSV")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_metrics", default="outputs/logs/metrics.json")
    parser.add_argument("--out_plot", default="outputs/plots/confusion_matrix.png")
    parser.add_argument("--status_file", default=None, help="Path to JSON status file to write per-batch progress")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging and tqdm")
    args = parser.parse_args()

    # Use provided args
    metrics = evaluate_model(
        args.model_path,
        args.test_file,
        output_metrics=args.out_metrics,
        output_plot=args.out_plot,
        batch_size=args.batch_size,
        status_file=args.status_file,
        verbose=args.verbose,
    )
    print("\nEvaluation Complete!")
    print(json.dumps(metrics, indent=4))
