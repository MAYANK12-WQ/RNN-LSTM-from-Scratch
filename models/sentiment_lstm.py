"""
Bidirectional LSTM for Sentiment Analysis

This module implements a BiLSTM classifier for binary sentiment classification.
Architecture: Embedding → BiLSTM → Pooling → FC → Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SentimentLSTM(nn.Module):
    """
    Bidirectional LSTM for sentiment classification.

    Architecture:
        Embedding(vocab_size → embedding_dim)
        Bidirectional LSTM(embedding_dim → hidden_dim)
        Max/Average Pooling over sequence
        FC(hidden_dim*2 → intermediate_dim) → ReLU → Dropout
        FC(intermediate_dim → num_classes)

    Args:
        vocab_size (int): Size of vocabulary
        embedding_dim (int): Dimension of word embeddings
        hidden_dim (int): LSTM hidden dimension
        num_layers (int): Number of LSTM layers
        num_classes (int): Number of output classes (2 for binary)
        dropout (float): Dropout probability
        bidirectional (bool): Use bidirectional LSTM
        pretrained_embeddings (torch.Tensor): Optional pretrained embeddings
    """

    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256,
                 num_layers=2, num_classes=2, dropout=0.5,
                 bidirectional=True, pretrained_embeddings=None):
        super(SentimentLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Load pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True  # Fine-tune embeddings

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * self.num_directions, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout2 = nn.Dropout(dropout * 0.5)

        # Initialize weights
        self._init_weights()

    def forward(self, x, lengths=None):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length)
            lengths (torch.Tensor): Optional sequence lengths for packing

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # Embedding: (batch, seq_len) → (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)
        embedded = self.dropout2(embedded)

        # Pack sequences if lengths provided
        if lengths is not None:
            lengths = lengths.cpu()
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )

        # LSTM: (batch, seq_len, embedding_dim) → (batch, seq_len, hidden_dim*2)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Unpack if packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )

        # Pooling strategies
        # Option 1: Use final hidden states from both directions
        if self.bidirectional:
            # Concatenate final hidden states from forward and backward
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        # Option 2: Max pooling over sequence (commented out)
        # hidden, _ = torch.max(lstm_out, dim=1)

        # Option 3: Average pooling over sequence (commented out)
        # hidden = torch.mean(lstm_out, dim=1)

        # Fully connected layers
        out = self.fc1(hidden)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)

        return out

    def _init_weights(self):
        """Initialize network weights."""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        # Initialize linear layers
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0)

    def get_num_parameters(self):
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """Test function to verify model architecture."""
    vocab_size = 10000
    batch_size = 4
    seq_length = 200

    model = SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=300,
        hidden_dim=256,
        num_layers=2,
        num_classes=2,
        bidirectional=True
    )

    # Create dummy input
    x = torch.randint(1, vocab_size, (batch_size, seq_length))
    lengths = torch.randint(50, seq_length, (batch_size,))

    # Forward pass
    output = model(x, lengths)

    print("Model Architecture Test:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {model.get_num_parameters():,}")
    print("\nModel Summary:")
    print(model)

    # Test prediction
    probs = F.softmax(output, dim=1)
    print(f"\nSample predictions (probabilities):")
    print(probs)


if __name__ == "__main__":
    test_model()
