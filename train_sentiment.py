"""
Training script for sentiment analysis using BiLSTM

Features:
- Bidirectional LSTM for sentiment classification
- Training with validation
- Model checkpointing
- Confusion matrix and classification report
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from models.sentiment_lstm import SentimentLSTM
from dataset import get_sentiment_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device, clip_grad=5.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc='Training')

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        # Statistics
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    avg_loss = total_loss / total
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate(model, test_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Validation', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = 100. * correct / total

    return avg_loss, accuracy, all_preds, all_labels


def train_model(args):
    """Main training function."""
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    print("Loading IMDB dataset...")
    train_loader, test_loader, vocab = get_sentiment_dataloaders(
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_vocab_size=args.max_vocab_size,
        data_dir=args.data_dir
    )

    # Initialize model
    print("\nInitializing model...")
    model = SentimentLSTM(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=2,
        dropout=args.dropout,
        bidirectional=args.bidirectional
    )
    model = model.to(device)
    print(f"Total parameters: {model.get_num_parameters():,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    # Create checkpoint directory
    os.makedirs(args.save_path, exist_ok=True)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }

    best_val_acc = 0.0
    print("\nStarting training...")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, args.clip_grad
        )

        # Validate
        val_loss, val_acc, all_preds, all_labels = validate(
            model, test_loader, criterion, device
        )

        # Learning rate scheduling
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)

        # Print results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'vocab': vocab,
                'vocab_size': len(vocab),
            }
            save_file = os.path.join(args.save_path, 'sentiment_best.pth')
            torch.save(checkpoint, save_file)
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")

    print("\n" + "=" * 70)
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Print final classification report
    print("\nFinal Classification Report:")
    print("=" * 70)
    print(classification_report(all_labels, all_preds,
                               target_names=['Negative', 'Positive']))

    # Save training history
    history_file = os.path.join(args.save_path, 'sentiment_history.json')
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_file}")


def main():
    parser = argparse.ArgumentParser(description='Train BiLSTM for sentiment analysis')

    # Model hyperparameters
    parser.add_argument('--embedding-dim', type=int, default=300,
                       help='Embedding dimension (default: 300)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='LSTM hidden dimension (default: 256)')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of LSTM layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate (default: 0.5)')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                       help='Use bidirectional LSTM (default: True)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--max-length', type=int, default=256,
                       help='Maximum sequence length (default: 256)')
    parser.add_argument('--max-vocab-size', type=int, default=20000,
                       help='Maximum vocabulary size (default: 20000)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--clip-grad', type=float, default=5.0,
                       help='Gradient clipping threshold (default: 5.0)')

    # System settings
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory (default: ./data)')
    parser.add_argument('--save-path', type=str, default='./checkpoints',
                       help='Checkpoint save path (default: ./checkpoints)')

    args = parser.parse_args()

    # Print configuration
    print("Training Configuration:")
    print("-" * 70)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-" * 70)

    # Train model
    train_model(args)


if __name__ == "__main__":
    main()
