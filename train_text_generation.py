"""
Training script for character-level text generation using LSTM

Features:
- Character-level LSTM training
- Learning rate scheduling
- Model checkpointing
- Sample text generation during training
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.lstm import CharLSTM
from dataset import get_text_generation_dataloader


def train_epoch(model, train_loader, criterion, optimizer, device, clip_grad=5.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc='Training')

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)

        # Initialize hidden state
        hidden = model.init_hidden(batch_size, device)

        # Forward pass
        optimizer.zero_grad()
        outputs, hidden = model(inputs, hidden)

        # Reshape for loss calculation
        outputs = outputs.view(-1, model.vocab_size)
        targets = targets.view(-1)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        # Statistics
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(avg_loss))

    return avg_loss, perplexity.item()


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Validation', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            # Initialize hidden state
            hidden = model.init_hidden(batch_size, device)

            # Forward pass
            outputs, hidden = model(inputs, hidden)

            # Reshape for loss calculation
            outputs = outputs.view(-1, model.vocab_size)
            targets = targets.view(-1)

            # Calculate loss
            loss = criterion(outputs, targets)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(avg_loss))

    return avg_loss, perplexity.item()


def generate_sample(model, dataset, seed_text="The ", length=200, device='cpu'):
    """Generate sample text during training."""
    model.eval()

    # Encode seed text
    chars = [dataset.char_to_idx.get(ch, 0) for ch in seed_text]
    input_seq = torch.tensor(chars, dtype=torch.long).unsqueeze(0).to(device)

    # Initialize hidden state
    hidden = model.init_hidden(1, device)

    # Generate
    generated = seed_text
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            output = output[:, -1, :]  # Get last time step
            probs = torch.softmax(output, dim=-1)
            next_char_idx = torch.multinomial(probs, 1).item()
            next_char = dataset.idx_to_char[next_char_idx]
            generated += next_char

            # Update input
            input_seq = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)

    return generated


def train_model(args):
    """Main training function."""
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    print("Loading dataset...")
    train_loader, val_loader, dataset = get_text_generation_dataloader(
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )

    # Initialize model
    print("\nInitializing model...")
    model = CharLSTM(
        vocab_size=dataset.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    model = model.to(device)
    print(f"Total parameters: {model.get_num_parameters():,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # Create checkpoint directory
    os.makedirs(args.save_path, exist_ok=True)

    # Training history
    history = {
        'train_loss': [],
        'train_perplexity': [],
        'val_loss': [],
        'val_perplexity': [],
        'learning_rates': []
    }

    best_val_loss = float('inf')
    print("\nStarting training...")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)

        # Train
        train_loss, train_ppl = train_epoch(
            model, train_loader, criterion, optimizer, device, args.clip_grad
        )

        # Validate
        val_loss, val_ppl = validate(model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['train_perplexity'].append(train_ppl)
        history['val_loss'].append(val_loss)
        history['val_perplexity'].append(val_ppl)
        history['learning_rates'].append(current_lr)

        # Print results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Perplexity: {train_ppl:.2f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Perplexity:   {val_ppl:.2f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Generate sample text
        if epoch % args.sample_every == 0:
            print(f"\nGenerated sample:")
            print("-" * 70)
            sample = generate_sample(model, dataset, seed_text="ROMEO:\n", length=200, device=device)
            print(sample)
            print("-" * 70)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'vocab_size': dataset.vocab_size,
                'char_to_idx': dataset.char_to_idx,
                'idx_to_char': dataset.idx_to_char,
            }
            save_file = os.path.join(args.save_path, 'text_gen_best.pth')
            torch.save(checkpoint, save_file)
            print(f"  ✓ Saved best model (Val Loss: {val_loss:.4f})")

    print("\n" + "=" * 70)
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save training history
    history_file = os.path.join(args.save_path, 'text_gen_history.json')
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_file}")


def main():
    parser = argparse.ArgumentParser(description='Train LSTM for text generation')

    # Model hyperparameters
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimension (default: 128)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='LSTM hidden dimension (default: 256)')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of LSTM layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--seq-length', type=int, default=100,
                       help='Sequence length (default: 100)')
    parser.add_argument('--lr', type=float, default=0.002,
                       help='Learning rate (default: 0.002)')
    parser.add_argument('--clip-grad', type=float, default=5.0,
                       help='Gradient clipping threshold (default: 5.0)')

    # System settings
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory (default: ./data)')
    parser.add_argument('--save-path', type=str, default='./checkpoints',
                       help='Checkpoint save path (default: ./checkpoints)')
    parser.add_argument('--sample-every', type=int, default=5,
                       help='Generate sample every N epochs (default: 5)')

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
