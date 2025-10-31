"""
Dataset utilities for text generation and sentiment analysis

Handles:
- Text dataset loading and preprocessing
- Character-level tokenization
- IMDB sentiment dataset
- Vocabulary building
"""

import os
import requests
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np


# ============================================================================
# TEXT GENERATION DATASET
# ============================================================================

class CharDataset(Dataset):
    """
    Character-level dataset for text generation.

    Args:
        text (str): Input text corpus
        seq_length (int): Sequence length for training
    """

    def __init__(self, text, seq_length=100):
        self.text = text
        self.seq_length = seq_length

        # Build vocabulary
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # Create char to index mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        # Encode entire text
        self.encoded_text = [self.char_to_idx[ch] for ch in text]

        print(f"Dataset loaded: {len(text):,} characters")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Unique characters: {''.join(self.chars[:50])}...")

    def __len__(self):
        return len(self.encoded_text) - self.seq_length

    def __getitem__(self, idx):
        # Input sequence
        x = self.encoded_text[idx:idx + self.seq_length]
        # Target sequence (shifted by 1)
        y = self.encoded_text[idx + 1:idx + self.seq_length + 1]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def load_shakespeare_data(data_dir='./data'):
    """
    Download and load Shakespeare dataset.

    Returns:
        str: Complete Shakespeare text
    """
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, 'shakespeare.txt')

    if not os.path.exists(file_path):
        print("Downloading Shakespeare dataset...")
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        response = requests.get(url)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("Download complete!")

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    return text


def get_text_generation_dataloader(text=None, seq_length=100, batch_size=64,
                                   split_ratio=0.9, data_dir='./data'):
    """
    Create data loaders for text generation.

    Args:
        text (str): Input text (if None, loads Shakespeare)
        seq_length (int): Sequence length
        batch_size (int): Batch size
        split_ratio (float): Train/validation split ratio
        data_dir (str): Directory to store data

    Returns:
        tuple: (train_loader, val_loader, dataset)
    """
    # Load text
    if text is None:
        text = load_shakespeare_data(data_dir)

    # Create dataset
    dataset = CharDataset(text, seq_length)

    # Split into train and validation
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    return train_loader, val_loader, dataset


# ============================================================================
# SENTIMENT ANALYSIS DATASET
# ============================================================================

class IMDBDataset(Dataset):
    """
    Simple IMDB sentiment dataset.

    Args:
        texts (list): List of review texts
        labels (list): List of labels (0=negative, 1=positive)
        vocab (dict): Word to index mapping
        max_length (int): Maximum sequence length
    """

    def __init__(self, texts, labels, vocab, max_length=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize and encode
        tokens = text.lower().split()
        encoded = [self.vocab.get(word, self.vocab['<UNK>']) for word in tokens]

        # Truncate or pad
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        else:
            encoded = encoded + [0] * (self.max_length - len(encoded))

        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def build_vocab(texts, max_vocab_size=20000):
    """
    Build vocabulary from texts.

    Args:
        texts (list): List of texts
        max_vocab_size (int): Maximum vocabulary size

    Returns:
        dict: Word to index mapping
    """
    counter = Counter()
    for text in texts:
        tokens = text.lower().split()
        counter.update(tokens)

    # Get most common words
    most_common = counter.most_common(max_vocab_size - 2)  # Reserve for PAD and UNK

    # Build vocabulary
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)

    return vocab


def load_imdb_data(data_dir='./data', max_samples=None):
    """
    Load IMDB dataset from torchtext or create dummy data for demo.

    Args:
        data_dir (str): Directory to store data
        max_samples (int): Maximum samples to load (None for all)

    Returns:
        tuple: (train_texts, train_labels, test_texts, test_labels)
    """
    try:
        from torchtext.datasets import IMDB

        print("Loading IMDB dataset...")

        # Load datasets
        train_iter = IMDB(root=data_dir, split='train')
        test_iter = IMDB(root=data_dir, split='test')

        train_texts, train_labels = [], []
        for label, text in train_iter:
            train_texts.append(text)
            train_labels.append(1 if label == 'pos' else 0)
            if max_samples and len(train_texts) >= max_samples:
                break

        test_texts, test_labels = [], []
        for label, text in test_iter:
            test_texts.append(text)
            test_labels.append(1 if label == 'pos' else 0)
            if max_samples and len(test_texts) >= max_samples // 5:
                break

        print(f"Loaded {len(train_texts)} training samples")
        print(f"Loaded {len(test_texts)} test samples")

        return train_texts, train_labels, test_texts, test_labels

    except Exception as e:
        print(f"Could not load IMDB dataset: {e}")
        print("Creating dummy dataset for demonstration...")

        # Create dummy data
        positive_templates = [
            "This movie was absolutely amazing and wonderful",
            "I loved every minute of this fantastic film",
            "Brilliant acting and great storyline",
            "One of the best movies I've ever seen",
            "Highly recommended, truly exceptional work"
        ]

        negative_templates = [
            "This movie was terrible and boring",
            "Worst film I have ever watched",
            "Awful acting and poor storyline",
            "Complete waste of time and money",
            "Do not watch this horrible movie"
        ]

        train_texts = positive_templates * 200 + negative_templates * 200
        train_labels = [1] * 1000 + [0] * 1000

        test_texts = positive_templates * 40 + negative_templates * 40
        test_labels = [1] * 200 + [0] * 200

        return train_texts, train_labels, test_texts, test_labels


def get_sentiment_dataloaders(batch_size=64, max_length=256,
                              max_vocab_size=20000, data_dir='./data'):
    """
    Create data loaders for sentiment analysis.

    Args:
        batch_size (int): Batch size
        max_length (int): Maximum sequence length
        max_vocab_size (int): Maximum vocabulary size
        data_dir (str): Directory to store data

    Returns:
        tuple: (train_loader, test_loader, vocab)
    """
    # Load data
    train_texts, train_labels, test_texts, test_labels = load_imdb_data(data_dir)

    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(train_texts, max_vocab_size)
    print(f"Vocabulary size: {len(vocab)}")

    # Create datasets
    train_dataset = IMDBDataset(train_texts, train_labels, vocab, max_length)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab, max_length)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader, vocab


if __name__ == "__main__":
    print("Testing text generation dataset...")
    train_loader, val_loader, dataset = get_text_generation_dataloader(seq_length=50, batch_size=2)
    x, y = next(iter(train_loader))
    print(f"Input batch shape: {x.shape}")
    print(f"Target batch shape: {y.shape}")

    print("\nTesting sentiment dataset...")
    train_loader, test_loader, vocab = get_sentiment_dataloaders(batch_size=2)
    x, y = next(iter(train_loader))
    print(f"Input batch shape: {x.shape}")
    print(f"Label batch shape: {y.shape}")
