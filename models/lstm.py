"""
Custom LSTM Implementation for Text Generation

This module implements a character-level LSTM for sequence generation.
Architecture: Embedding → LSTM → Dropout → Linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharLSTM(nn.Module):
    """
    Character-level LSTM for text generation.

    Architecture:
        Embedding(vocab_size → embedding_dim)
        LSTM(embedding_dim → hidden_dim, num_layers)
        Dropout(dropout_rate)
        Linear(hidden_dim → vocab_size)

    Args:
        vocab_size (int): Size of vocabulary
        embedding_dim (int): Dimension of character embeddings
        hidden_dim (int): LSTM hidden dimension
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout probability
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 num_layers=2, dropout=0.3):
        super(CharLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights
        self._init_weights()

    def forward(self, x, hidden=None):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length)
            hidden (tuple): Optional initial hidden state (h_0, c_0)

        Returns:
            tuple: (output, hidden_state)
                - output: (batch_size, seq_length, vocab_size)
                - hidden_state: (h_n, c_n) tuple
        """
        # Embedding: (batch, seq_len) → (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # LSTM: (batch, seq_len, embedding_dim) → (batch, seq_len, hidden_dim)
        if hidden is not None:
            lstm_out, hidden = self.lstm(embedded, hidden)
        else:
            lstm_out, hidden = self.lstm(embedded)

        # Dropout
        lstm_out = self.dropout(lstm_out)

        # Output: (batch, seq_len, hidden_dim) → (batch, seq_len, vocab_size)
        output = self.fc(lstm_out)

        return output, hidden

    def init_hidden(self, batch_size, device='cpu'):
        """
        Initialize hidden state with zeros.

        Args:
            batch_size (int): Batch size
            device (str): Device to create tensors on

        Returns:
            tuple: (h_0, c_0) initial hidden and cell states
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

    def _init_weights(self):
        """Initialize network weights."""
        # Initialize embedding
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        # Initialize linear layer
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)

    def get_num_parameters(self):
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """Test function to verify model architecture."""
    vocab_size = 100
    batch_size = 4
    seq_length = 50

    model = CharLSTM(vocab_size=vocab_size, embedding_dim=128,
                     hidden_dim=256, num_layers=2)

    # Create dummy input
    x = torch.randint(0, vocab_size, (batch_size, seq_length))

    # Forward pass
    output, hidden = model(x)

    print("Model Architecture Test:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden state shape: {hidden[0].shape}, {hidden[1].shape}")
    print(f"Total parameters: {model.get_num_parameters():,}")
    print("\nModel Summary:")
    print(model)


if __name__ == "__main__":
    test_model()
