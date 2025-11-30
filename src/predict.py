import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import logging
import sys
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)


# -----------------------------
#  Load Model + Tokenizer
# -----------------------------
def load_model(model_path):
    """
    Loads a fine-tuned DistilBERT model for prediction.
    """

    if not os.path.exists(model_path):
        logging.error(f"❌ Model path not found: {model_path}")
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except Exception as e:
            # Common case: model saved with safetensors but package not installed
            logging.error("Failed to load model from %s: %s", model_path, e)
            if os.path.exists(os.path.join(model_path, "model.safetensors")):
                logging.error("It looks like the model was saved as 'safetensors'. Please install the 'safetensors' package: `pip install safetensors` and retry.")
            return None, None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        # Try to load a learned temperature if present (temperature.json)
        temp_path = Path(model_path) / "temperature.json"
        temperature = None
        if temp_path.exists():
            try:
                with open(temp_path, "r") as f:
                    data = json.load(f)
                    temperature = float(data.get("temperature", None)) if data.get("temperature", None) is not None else None
                    logging.info("Loaded temperature scaling: %s", temperature)
            except Exception as e:
                logging.warning("Could not read temperature file %s: %s", temp_path, e)

        logging.info("✅ Model & tokenizer loaded successfully.")

        return tokenizer, model, temperature

    except Exception as e:
        logging.error(f"❌ Failed to load model: {e}")
        return None, None


# -----------------------------
#  Run Prediction
# -----------------------------
def predict_text(text, tokenizer, model, temperature=None):
    """
    Predicts whether given text is AI-generated or human-written.
    Returns probabilities and predicted class.
    """

    if tokenizer is None or model is None:
        return {
            "error": "Model not loaded. Make sure model files exist."
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenize input text
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )

    # Move tensors to the right device
    try:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception:
        # If something unexpected, fall back to keeping CPU tensors
        logging.warning("Could not move inputs to device; proceeding with CPU tensors.")

    # Run model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Apply temperature scaling if provided
    if temperature is not None and temperature > 0:
        try:
            logits = logits / float(temperature)
        except Exception:
            logging.warning("Failed to apply temperature scaling; continuing without it")

    # Softmax for probabilities (use scaled logits)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    label = int(torch.argmax(logits, dim=1)[0].cpu().numpy())

    result = {
        "predicted_label": label,               # 0 = Human, 1 = AI
        "human_prob": round(float(probs[0]), 4),
        "ai_prob": round(float(probs[1]), 4),
    }

    return result


# -----------------------------
#  CLI Mode (optional)
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("python predict.py <model_path> <text>")
        sys.exit()

    model_path = sys.argv[1]
    input_text = " ".join(sys.argv[2:])

    tokenizer, model, temperature = load_model(model_path)
    result = predict_text(input_text, tokenizer, model, temperature=temperature)

    print("\nPrediction Result:")
    print(result)
