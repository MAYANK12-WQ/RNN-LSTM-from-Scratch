"""
Text generation inference script

Generate text using trained character-level LSTM model.
"""

import argparse
import torch
import torch.nn.functional as F

from models.lstm import CharLSTM


def generate_text(model, char_to_idx, idx_to_char, seed_text="The ",
                 length=500, temperature=0.8, device='cpu'):
    """
    Generate text using trained model.

    Args:
        model: Trained LSTM model
        char_to_idx (dict): Character to index mapping
        idx_to_char (dict): Index to character mapping
        seed_text (str): Starting text
        length (int): Number of characters to generate
        temperature (float): Sampling temperature (higher = more random)
        device (str): Device to use

    Returns:
        str: Generated text
    """
    model.eval()

    # Encode seed text
    chars = [char_to_idx.get(ch, 0) for ch in seed_text]
    input_seq = torch.tensor(chars, dtype=torch.long).unsqueeze(0).to(device)

    # Initialize hidden state
    hidden = model.init_hidden(1, device)

    # Feed seed text through model
    with torch.no_grad():
        for i in range(len(chars) - 1):
            output, hidden = model(input_seq[:, i:i+1], hidden)

    # Generate new text
    generated = seed_text
    last_char_idx = chars[-1]

    with torch.no_grad():
        for _ in range(length):
            # Input is last generated character
            input_seq = torch.tensor([[last_char_idx]], dtype=torch.long).to(device)

            # Forward pass
            output, hidden = model(input_seq, hidden)
            output = output[:, -1, :]  # Get last time step

            # Apply temperature
            output = output / temperature

            # Sample from probability distribution
            probs = F.softmax(output, dim=-1)
            next_char_idx = torch.multinomial(probs, 1).item()

            # Get character
            next_char = idx_to_char.get(next_char_idx, '')
            generated += next_char

            # Update for next iteration
            last_char_idx = next_char_idx

    return generated


def main():
    parser = argparse.ArgumentParser(description='Generate text using trained LSTM')

    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--seed', type=str, default='ROMEO:\n',
                       help='Seed text to start generation (default: "ROMEO:\\n")')
    parser.add_argument('--length', type=int, default=500,
                       help='Number of characters to generate (default: 500)')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (0.5-1.5, default: 0.8)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--num-samples', type=int, default=1,
                       help='Number of samples to generate (default: 1)')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load checkpoint
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    vocab_size = checkpoint['vocab_size']
    char_to_idx = checkpoint['char_to_idx']
    idx_to_char = checkpoint['idx_to_char']

    # Initialize model
    model = CharLSTM(vocab_size=vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"\nGeneration settings:")
    print(f"  Seed: \"{args.seed}\"")
    print(f"  Length: {args.length} characters")
    print(f"  Temperature: {args.temperature}")
    print(f"  Number of samples: {args.num_samples}")
    print("\n" + "=" * 80)

    # Generate text samples
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\nSample {i+1}:")
            print("-" * 80)

        generated_text = generate_text(
            model, char_to_idx, idx_to_char,
            seed_text=args.seed,
            length=args.length,
            temperature=args.temperature,
            device=device
        )

        print(generated_text)
        print("=" * 80)


if __name__ == "__main__":
    main()
