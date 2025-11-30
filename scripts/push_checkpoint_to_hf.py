"""
Improved script to upload a local checkpoint to a Hugging Face model repo
WITHOUT overwriting old checkpoints.

Each upload is stored in a unique subfolder:
repo_id/checkpoint-123000/
repo_id/checkpoint-200000/

Optional: also update a 'latest/' folder.
"""

import os
from pathlib import Path
import argparse
from huggingface_hub import HfApi, HfFolder, upload_folder
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default=os.environ.get("MODEL_DIR", "model/distilbert_finetuned/checkpoint-193000"),
        help="Local checkpoint directory"
    )
    parser.add_argument(
        "--repo_id",
        default=os.environ.get("HF_REPO", None),
        help="Hugging Face repo id, e.g. username/model"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", None),
        help="HF token (optional)"
    )
    parser.add_argument(
        "--update_latest",
        action="store_true",
        help="Also update a 'latest/' folder in the repo"
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"❌ Model directory not found: {model_dir}")

    if not args.repo_id:
        raise SystemExit("❌ Please provide --repo_id or set HF_REPO env variable")

    repo_id = args.repo_id
    token = args.token

    # Save token if provided
    if token:
        HfFolder.save_token(token)

    checkpoint_name = model_dir.name  # e.g. "checkpoint-193000"

    print(f"📦 Loading model from: {model_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    except Exception as e:
        raise SystemExit(f"❌ Failed to load model: {e}")

    # --- Push model & tokenizer into a subfolder: repo/checkpoint-123000 ---
    remote_subdir = checkpoint_name  # hf_repo/checkpoint-193000/

    print(f"⬆ Uploading checkpoint to: {repo_id}/{remote_subdir}")

    upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=remote_subdir,
        token=token,
        commit_message=f"Upload checkpoint: {checkpoint_name}",
    )

    print(f"✅ Uploaded as folder: {remote_subdir}")

    # --- Optional: Update the /latest folder ---
    if args.update_latest:
        print("🔄 Updating 'latest/' folder with this checkpoint...")
        upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="latest",
            token=token,
            commit_message=f"Update latest to {checkpoint_name}",
        )
        print("✨ 'latest/' is now updated.")

    print("\n🎉 Done — checkpoint uploaded without overwriting previous versions.")


if __name__ == "__main__":
    main()
