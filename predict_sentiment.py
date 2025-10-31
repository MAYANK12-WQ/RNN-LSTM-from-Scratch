"""
Sentiment prediction inference script

Predict sentiment of text using trained BiLSTM model.
"""

import argparse
import torch
import torch.nn.functional as F

from models.sentiment_lstm import SentimentLSTM


def preprocess_text(text, vocab, max_length=256):
    """
    Preprocess text for model input.

    Args:
        text (str): Input text
        vocab (dict): Vocabulary mapping
        max_length (int): Maximum sequence length

    Returns:
        torch.Tensor: Encoded tensor
    """
    # Tokenize and encode
    tokens = text.lower().split()
    encoded = [vocab.get(word, vocab.get('<UNK>', 1)) for word in tokens]

    # Truncate or pad
    if len(encoded) > max_length:
        encoded = encoded[:max_length]
    else:
        encoded = encoded + [0] * (max_length - len(encoded))

    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def predict_sentiment(model, text, vocab, device='cpu', max_length=256):
    """
    Predict sentiment of text.

    Args:
        model: Trained model
        text (str): Input text
        vocab (dict): Vocabulary mapping
        device (str): Device to use
        max_length (int): Maximum sequence length

    Returns:
        tuple: (predicted_class, confidence, probabilities)
    """
    model.eval()

    # Preprocess
    input_tensor = preprocess_text(text, vocab, max_length).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

    predicted_class = 'Positive' if predicted.item() == 1 else 'Negative'
    confidence_score = confidence.item() * 100

    return predicted_class, confidence_score, probabilities[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Predict sentiment using trained BiLSTM')

    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to analyze (optional, will prompt if not provided)')
    parser.add_argument('--max-length', type=int, default=256,
                       help='Maximum sequence length (default: 256)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load checkpoint
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    vocab = checkpoint['vocab']
    vocab_size = checkpoint['vocab_size']

    # Initialize model
    model = SentimentLSTM(vocab_size=vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%\n")
    print("=" * 80)

    # Interactive mode
    if args.interactive or args.text is None:
        print("Interactive Sentiment Analysis")
        print("Type 'quit' or 'exit' to stop\n")

        while True:
            try:
                text = input("\nEnter text to analyze: ").strip()

                if text.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not text:
                    continue

                # Predict
                predicted_class, confidence, probs = predict_sentiment(
                    model, text, vocab, device, args.max_length
                )

                # Display results
                print("\n" + "-" * 80)
                print(f"Text: \"{text}\"")
                print("-" * 80)
                print(f"Prediction: {predicted_class}")
                print(f"Confidence: {confidence:.2f}%")
                print(f"\nProbabilities:")
                print(f"  Negative: {probs[0]*100:.2f}%")
                print(f"  Positive: {probs[1]*100:.2f}%")

                # Visual bar
                bar_length = int(confidence / 2)
                bar = "█" * bar_length
                print(f"\n{predicted_class}: {bar} {confidence:.1f}%")
                print("-" * 80)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

    # Single prediction mode
    else:
        text = args.text

        # Predict
        predicted_class, confidence, probs = predict_sentiment(
            model, text, vocab, device, args.max_length
        )

        # Display results
        print(f"\nText: \"{text}\"")
        print("=" * 80)
        print(f"Prediction: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"\nProbabilities:")
        print(f"  Negative: {probs[0]*100:.2f}%")
        print(f"  Positive: {probs[1]*100:.2f}%")

        # Visual bar
        bar_length = int(confidence / 2)
        bar = "█" * bar_length
        print(f"\n{predicted_class}: {bar} {confidence:.1f}%")
        print("=" * 80)


if __name__ == "__main__":
    main()
