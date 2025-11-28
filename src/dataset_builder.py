import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import DistilBertTokenizerFast


class TextDataset(Dataset):
    """
    Custom PyTorch Dataset for AI vs Human text classification.
    Loads text and label, and applies DistilBERT tokenization.
    """

    def __init__(self, csv_path, tokenizer, max_length=256):
        """
        Args:
            csv_path (str): Path to the CSV file (train/val/test).
            tokenizer: DistilBERT tokenizer.
            max_length (int): Max token length for DistilBERT input.
        """
        assert os.path.exists(csv_path), f"File not found: {csv_path}"
        
        self.data = pd.read_csv(csv_path)
        self.texts = self.data["text"].astype(str).tolist()
        self.labels = self.data["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Return:
            dictionary containing:
                - input_ids
                - attention_mask
                - label
        """
        text = self.texts[idx]
        label = int(self.labels[idx])

        # Tokenize using DistilBERT tokenizer
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),     # tensor
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def get_dataloaders(
    train_path,
    val_path,
    tokenizer_name="distilbert-base-uncased",
    batch_size=16,
    max_length=256,
    num_workers=2
):
    """
    Loads the tokenizer, creates Datasets and DataLoaders for training.

    Args:
        train_path (str)
        val_path (str)
        tokenizer_name (str)
        batch_size (int)
        max_length (int)
        num_workers (int)

    Returns:
        train_loader, val_loader, tokenizer
    """

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)

    print("Creating training dataset...")
    train_dataset = TextDataset(train_path, tokenizer, max_length=max_length)

    print("Creating validation dataset...")
    val_dataset = TextDataset(val_path, tokenizer, max_length=max_length)

    # DataLoaders create batches and handle shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print("DataLoaders ready:")
    print(f"- Train batches: {len(train_loader)}")
    print(f"- Val batches: {len(val_loader)}")

    return train_loader, val_loader, tokenizer
