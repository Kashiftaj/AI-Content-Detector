import streamlit as st
import torch
import os
import re
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from docx import Document
import io

# -----------------------------
# CONFIG
# -----------------------------
MODEL_HF = "kashiftaj/ai-content-detector"  # HF repo id to load latest uploaded model
DEMO_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Local candidates to try before falling back to HF hub (helps offline/local testing)
MODEL_CANDIDATES = [
    "model/distilbert_finetuned/checkpoint-193000",
    "model/distilbert_finetuned",
    "models/distilbert_finetuned",
]

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    # Prefer local checkpoints if available
    for p in MODEL_CANDIDATES:
        if os.path.exists(p):
            try:
                tokenizer = AutoTokenizer.from_pretrained(p)
                model = AutoModelForSequenceClassification.from_pretrained(p)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                model.eval()
                return tokenizer, model, p
            except Exception as e:
                # If safetensors present but package missing, show hint
                if os.path.exists(os.path.join(p, "model.safetensors")):
                    st.warning(
                        "Local model at %s looks like it was saved with safetensors. If loading fails, run: `pip install safetensors`." % p
                    )
                # otherwise try next candidate

    # Try loading from HF hub (latest uploaded model)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_HF)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_HF)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return tokenizer, model, MODEL_HF
    except Exception as e:
        st.warning(f"Could not load HF Hub model, falling back to demo: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(DEMO_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(DEMO_MODEL)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            return tokenizer, model, DEMO_MODEL
        except Exception:
            return None, None, None

tokenizer, model, model_used = load_model()

# Try to load a learned temperature for calibration if present
temperature = None
try:
    if model_used and os.path.isdir(model_used):
        tpath = Path(model_used) / "temperature.json"
        if tpath.exists():
            with open(tpath, "r") as tf:
                temperature = json.load(tf).get("temperature")
except Exception:
    temperature = None

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="AI Text Detector", layout="wide")
st.title("📝 AI vs Human Text Detector")
st.write("Paste your text below or upload a document to check if it's AI-generated.")

# Status
if tokenizer is None or model is None:
    st.warning("⚠ Model not loaded yet. Detection disabled.")
else:
    st.success(f"✅ Model Loaded: {model_used}")

# -----------------------------
# TEXT INPUT AREA + FILE UPLOAD
# -----------------------------
st.subheader("Enter Text or Upload File (.txt or .docx)")
user_text = st.text_area(
    "Paste your text here:",
    height=200,
    placeholder="Write or paste text to analyze…",
)
uploaded_file = st.file_uploader(
    "Or upload a .txt or .docx file", type=["txt", "docx"], accept_multiple_files=False
)
detect_button = st.button("Detect AI Content")

# -----------------------------
# HELPERS
# -----------------------------
def read_uploaded_file(uploaded):
    if not uploaded:
        return ""
    name = uploaded.name.lower()
    if name.endswith(".txt"):
        try:
            return uploaded.getvalue().decode("utf-8", errors="replace")
        except Exception:
            return ""
    elif name.endswith(".docx"):
        try:
            doc = Document(uploaded)
            paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
            return "\n\n".join(paragraphs)
        except Exception as e:
            st.error(
                "Could not read .docx file. Please install 'python-docx' or upload a .txt file. "
                f"Error: {e}"
            )
            return ""
    return ""


def predict_text(text):
    if tokenizer is None or model is None or not text.strip():
        return None
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception:
        st.warning("Could not move inputs to GPU; running on CPU.")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        if temperature is not None and isinstance(temperature, (int, float)):
            try:
                logits = logits / float(temperature)
            except Exception:
                pass
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    human_prob = float(probs[0])
    ai_prob = float(probs[1]) if probs.shape[0] > 1 else 1.0 - human_prob
    return {"human_prob": human_prob, "ai_prob": ai_prob}


def predict_sentences(sentences, batch_size=32):
    """Predict AI probability for a list of sentences (batched). Returns list of ai_probs."""
    if tokenizer is None or model is None:
        return [None] * len(sentences)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ai_probs = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, return_tensors="pt")
        try:
            enc = {k: v.to(device) for k, v in enc.items()}
        except Exception:
            pass
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits
            if temperature is not None and isinstance(temperature, (int, float)):
                try:
                    logits = logits / float(temperature)
                except Exception:
                    pass
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for p in probs:
                ai = float(p[1]) if p.shape[0] > 1 else float(1.0 - p[0])
                ai_probs.append(ai)
    return ai_probs

# -----------------------------
# HANDLE DETECTION
# -----------------------------
if detect_button:
    text_to_check = read_uploaded_file(uploaded_file) if uploaded_file else user_text

    if tokenizer is None or model is None:
        st.error("❌ Model is not loaded. Please try again later.")
    elif not text_to_check.strip():
        st.warning("⚠ Please enter some text or upload a supported file.")
    else:
        with st.spinner("Analyzing…"):
            # Split on dot+space to get sentences; keep it simple to avoid heavy NLP deps
            sentences = [s.strip() for s in re.split(r"\.[\s\n]*", text_to_check) if s.strip()]
            if not sentences:
                st.error("Could not parse any sentences from the input.")
            else:
                # Predict per-sentence (batched)
                ai_probs = predict_sentences(sentences, batch_size=32)

                # Aggregate: simple average of per-sentence AI probabilities
                valid_probs = [p for p in ai_probs if p is not None]
                if not valid_probs:
                    st.error("Model did not return predictions.")
                else:
                    avg_ai = float(sum(valid_probs) / len(valid_probs))
                    ai_percent = avg_ai * 100.0
                    human_percent = 100.0 - ai_percent

                    st.subheader("🔎 Detection Result")
                    st.write(f"**AI-generated Probability:** `{ai_percent:.2f}%`  — Human: `{human_percent:.2f}%`")

                    # Verdict
                    if ai_percent > 70.0:
                        st.error("⚠ This text is likely AI-generated.")
                    elif ai_percent > 40.0:
                        st.warning("⚠ This text may contain AI-generated patterns.")
                    else:
                        st.success("✅ This text is likely human-written.")

                    
