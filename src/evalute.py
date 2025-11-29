import os
import json
import logging
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Human", "AI"],
                yticklabels=["Human", "AI"])
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
                   output_plot="outputs/plots/confusion_matrix.png"):

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

    logger.info("Running evaluation...")

    for text in texts:
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            all_preds.append(pred_label)

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
    MODEL_DIR = "models/distilbert_finetuned"
    TEST_FILE = "data/processed/test.csv"

    metrics = evaluate_model(MODEL_DIR, TEST_FILE)
    print("\nEvaluation Complete!")
    print(json.dumps(metrics, indent=4))
