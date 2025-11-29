import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# CONFIG
# -----------------------------
# Try several likely model paths (local checkpoint first)
MODEL_CANDIDATES = [
    "model/distilbert_finetuned/checkpoint-41500",
    "model/distilbert_finetuned",
    "models/distilbert_finetuned",
]

# Public demo fallback if local model not available
DEMO_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


@st.cache_resource
def load_model():
    # Try local candidates first
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
                # If safetensors present but package missing, surface helpful hint
                if os.path.exists(os.path.join(p, "model.safetensors")):
                    st.warning(
                        "Local model at %s looks like it was saved with safetensors. If loading fails, run: `pip install safetensors` in your environment." % p
                    )
                # otherwise try next candidate
    # Fallback to public demo model (fast to download and suitable for UI testing)
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


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="AI Text Detector", layout="wide")

st.title("📝 AI vs Human Text Detector")
st.write("Paste your text below or upload a document to check if it's AI-generated.")

# Status Box
if tokenizer is None or model is None:
    st.warning("⚠ Model not loaded yet. The UI is ready, but detection is disabled.")
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

uploaded_file = st.file_uploader("Or upload a .txt or .docx file", type=["txt", "docx"], accept_multiple_files=False)

detect_button = st.button("Detect AI Content")


# -----------------------------
# HELPERS
# -----------------------------
def read_uploaded_file(uploaded):
    # uploaded is a Streamlit UploadedFile (BytesIO-compatible)
    if not uploaded:
        return ""
    name = uploaded.name.lower()
    if name.endswith(".txt"):
        try:
            raw = uploaded.getvalue()
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return ""
    elif name.endswith(".docx"):
        try:
            # python-docx supports file-like objects
            from docx import Document

            doc = Document(uploaded)
            paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
            return "\n\n".join(paragraphs)
        except Exception as e:
            st.error("Could not read .docx file. Please install 'python-docx' or upload a .txt file. Error: %s" % e)
            return ""
    else:
        return ""


def predict_text(text):
    if tokenizer is None or model is None:
        return None

    # Short-circuit empty text
    if not text or not text.strip():
        return None

    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception:
        st.warning("Could not move inputs to GPU; running on CPU.")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()

    # Map numeric label to human-friendly name
    # Common convention: 0=human,1=ai — adapt if your model uses different mapping
    human_prob = float(probs[0])
    ai_prob = float(probs[1]) if probs.shape[0] > 1 else 1.0 - human_prob
    return {"human_prob": human_prob, "ai_prob": ai_prob}


# -----------------------------
# HANDLE DETECTION
# -----------------------------
if detect_button:

    # If file uploaded, prefer it over textarea
    if uploaded_file is not None:
        text_to_check = read_uploaded_file(uploaded_file)
    else:
        text_to_check = user_text

    if tokenizer is None or model is None:
        st.error("❌ Model is not loaded. Please try again later.")
    elif not text_to_check or not text_to_check.strip():
        st.warning("⚠ Please enter some text or upload a supported file.")
    else:
        with st.spinner("Analyzing…"):
            result = predict_text(text_to_check)

        if result is None:
            st.error("Could not run prediction.")
        else:
            ai_score = result["ai_prob"]
            st.subheader("🔎 Detection Result")
            st.write(f"**AI-generated Probability:** `{ai_score*100:.2f}%`")

            if ai_score > 0.7:
                st.error("⚠ This text is likely AI-generated.")
            elif ai_score > 0.4:
                st.warning("⚠ This text may contain AI-generated patterns.")
            else:
                st.success("✅ This text is likely human-written.")

            # Show raw probabilities
            st.markdown("**Details**")
            st.write({"human_prob": f"{result['human_prob']:.4f}", "ai_prob": f"{result['ai_prob']:.4f}"})

