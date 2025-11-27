# AI Text Detector (Human vs AI Text Classification)

This project fine-tunes a DistilBERT model to classify text as Human-written or AI-generated.
It includes training scripts, evaluation methods, a paraphrased dataset generator, and a Streamlit app for deployment.

## Project Structure
- data/raw → original datasets
- data/processed → cleaned & split datasets
- src → preprocessing, training, evaluation code
- models → saved DistilBERT model
- app → Streamlit UI
- notebooks → experiments & visualizations

## Setup Instructions
1. Clone repo
2. Create virtual environment
3. Install dependencies
4. Prepare dataset
5. Run training script
6. Launch Streamlit app

## Goals
- High accuracy (>90%)
- Low false-positive rate (<10%)
- Robust detection of paraphrased AI text
