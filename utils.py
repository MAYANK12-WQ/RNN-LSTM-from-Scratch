"""
Utility functions for RNN/LSTM training and visualization

Features:
- Training history plotting
- Text generation with different sampling strategies
- Confusion matrix visualization
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_history(history_path, save_path=None):
    """
    Plot training history curves.

    Args:
        history_path (str): Path to history JSON file
        save_path (str): Optional path to save figure
    """
    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    # Check if this is text generation or sentiment analysis
    is_text_gen = 'train_perplexity' in history

    if is_text_gen:
        # Text generation plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Perplexity
        axes[1].plot(epochs, history['train_perplexity'], 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, history['val_perplexity'], 'r-', label='Val', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Perplexity', fontsize=12)
        axes[1].set_title('Perplexity', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Learning rate
        axes[2].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Learning Rate', fontsize=12)
        axes[2].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)

    else:
        # Sentiment analysis plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Learning rate
        axes[2].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Learning Rate', fontsize=12)
        axes[2].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes=['Negative', 'Positive'], save_path=None):
    """
    Plot confusion matrix for sentiment analysis.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        save_path: Optional path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def compare_sampling_temperatures(model, char_to_idx, idx_to_char,
                                  seed_text="The ", length=200,
                                  temperatures=[0.5, 0.8, 1.0, 1.2],
                                  device='cpu'):
    """
    Generate text with different temperature values for comparison.

    Args:
        model: Trained model
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        seed_text: Starting text
        length: Generation length
        temperatures: List of temperature values to try
        device: Device to use

    Returns:
        dict: Generated texts for each temperature
    """
    results = {}

    for temp in temperatures:
        from generate_text import generate_text
        text = generate_text(model, char_to_idx, idx_to_char,
                           seed_text=seed_text, length=length,
                           temperature=temp, device=device)
        results[temp] = text

    return results


def analyze_vocabulary(vocab, top_k=50):
    """
    Analyze vocabulary statistics.

    Args:
        vocab (dict): Vocabulary mapping
        top_k (int): Number of top words to display
    """
    print(f"Vocabulary Statistics:")
    print(f"  Total words: {len(vocab)}")

    # Get word frequencies (if available)
    if isinstance(vocab, dict):
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])[:top_k]
        print(f"\nFirst {top_k} words in vocabulary:")
        for word, idx in sorted_vocab:
            print(f"  {word}: {idx}")


if __name__ == "__main__":
    print("Utility functions loaded!")
    print("Available functions:")
    print("  - plot_training_history()")
    print("  - plot_confusion_matrix()")
    print("  - compare_sampling_temperatures()")
    print("  - analyze_vocabulary()")
