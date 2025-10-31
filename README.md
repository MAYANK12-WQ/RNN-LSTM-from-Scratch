# RNN & LSTM from Scratch: Understanding Sequential Models

A comprehensive implementation of Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks built from scratch using PyTorch, demonstrating sequence modeling for text generation and sentiment analysis.

## Overview

This project implements custom RNN and LSTM architectures for two tasks:
1. **Character-level Text Generation** - Generate Shakespeare-like text
2. **Sentiment Analysis** - IMDB movie review classification (90%+ accuracy)

Built to understand the mathematical foundations and practical implementation of sequential models.

## Features

- **Custom RNN/LSTM Implementation**: Built from ground-up with detailed comments
- **Two Complete Applications**:
  - Text generation with temperature sampling
  - Sentiment classification on IMDB dataset
- **Training Pipeline**: Complete training loops with gradient clipping
- **Visualization Suite**:
  - Training curves
  - Attention weights visualization
  - Generated text samples
  - Confusion matrix for sentiment analysis
- **Model Checkpointing**: Save and resume training
- **Google Colab Ready**: Notebooks for instant experimentation

## Architecture

### LSTM Architecture for Text Generation
```
Input (sequence_length, vocab_size)
  → Embedding(vocab_size → 128)
  → LSTM(128 → 256) × 2 layers
  → Dropout(0.3)
  → FC(256 → vocab_size)
Output (sequence_length, vocab_size)
```

### LSTM Architecture for Sentiment Analysis
```
Input (sequence_length, vocab_size)
  → Embedding(vocab_size → 300, pretrained GloVe)
  → Bidirectional LSTM(300 → 256)
  → Dropout(0.5)
  → FC(512 → 128) → ReLU → Dropout
  → FC(128 → 2)
Output (2 classes: positive/negative)
```

## Theory Background

### Recurrent Neural Networks (RNN)

RNNs process sequences by maintaining hidden states:
```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
```

**Problem**: Vanishing/exploding gradients in long sequences.

### Long Short-Term Memory (LSTM)

LSTMs solve the vanishing gradient problem using gating mechanisms:

**Gates**:
- **Forget Gate**: `f_t = σ(W_f · [h_{t-1}, x_t] + b_f)` - What to forget from cell state
- **Input Gate**: `i_t = σ(W_i · [h_{t-1}, x_t] + b_i)` - What new info to store
- **Output Gate**: `o_t = σ(W_o · [h_{t-1}, x_t] + b_o)` - What to output

**Cell State Update**:
```
c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)  # Candidate values
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t         # New cell state
h_t = o_t ⊙ tanh(c_t)                    # New hidden state
```

## Installation

```bash
git clone https://github.com/MAYANK12-WQ/RNN-LSTM-from-Scratch.git
cd RNN-LSTM-from-Scratch
pip install -r requirements.txt
```

## Quick Start

### 1. Text Generation

**Training:**
```bash
python train_text_generation.py --epochs 50 --seq-length 100
```

**Generate Text:**
```bash
python generate_text.py --model-path checkpoints/text_gen_best.pth --seed "To be or not to be" --length 500 --temperature 0.8
```

**Arguments:**
- `--seed`: Starting text prompt
- `--length`: Number of characters to generate
- `--temperature`: Sampling temperature (0.5-1.5, higher = more creative)

### 2. Sentiment Analysis

**Training:**
```bash
python train_sentiment.py --epochs 20 --batch-size 64
```

**Inference:**
```bash
python predict_sentiment.py --model-path checkpoints/sentiment_best.pth --text "This movie was absolutely amazing!"
```

## Project Structure

```
RNN-LSTM-from-Scratch/
├── models/
│   ├── rnn.py                    # Custom RNN implementation
│   ├── lstm.py                   # Custom LSTM implementation
│   └── sentiment_lstm.py         # Bidirectional LSTM for sentiment
├── train_text_generation.py      # Text generation training
├── train_sentiment.py             # Sentiment analysis training
├── generate_text.py               # Text generation inference
├── predict_sentiment.py           # Sentiment prediction
├── dataset.py                     # Data loading utilities
├── utils.py                       # Visualization and helpers
├── requirements.txt               # Dependencies
├── demo_text_gen.ipynb           # Text generation demo
├── demo_sentiment.ipynb          # Sentiment analysis demo
└── README.md
```

## Results

### Text Generation

**Shakespeare Text Sample** (after 50 epochs):
```
ROMEO:
What light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief...
```

**Training**: ~1 hour on GPU, ~6-7 hours on CPU

### Sentiment Analysis

| Metric | Score |
|--------|-------|
| Test Accuracy | 90.4% |
| Precision | 0.901 |
| Recall | 0.908 |
| F1 Score | 0.904 |

**Examples:**
- "This movie was fantastic!" → **Positive (98.3%)**
- "Worst film I've ever seen." → **Negative (96.7%)**

## Key Implementation Details

### 1. Gradient Clipping
Prevents exploding gradients in RNN training:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

### 2. Temperature Sampling
Controls randomness in text generation:
```python
probs = F.softmax(logits / temperature, dim=-1)
next_char = torch.multinomial(probs, 1)
```

### 3. Bidirectional LSTM
Processes sequences in both directions for better context:
```python
lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
```

### 4. Packed Sequences
Efficient processing of variable-length sequences:
```python
packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
output, hidden = lstm(packed)
```

## Datasets

### Text Generation
- **Shakespeare Dataset**: Complete works of Shakespeare
- **Size**: ~1.1M characters
- **Vocabulary**: 65 unique characters

### Sentiment Analysis
- **IMDB Movie Reviews**: 50K reviews (25K train, 25K test)
- **Classes**: Binary (positive/negative)
- **Vocabulary**: 20K most frequent words

## Learning Outcomes

This project demonstrates:
1. **Sequential Modeling**: Understanding time-series and sequence data
2. **RNN/LSTM Architecture**: From theory to implementation
3. **NLP Applications**: Text generation and classification
4. **Advanced Techniques**: Gradient clipping, temperature sampling, bidirectional processing
5. **Production Skills**: Model checkpointing, inference pipelines

## Extensions & Ideas

- [ ] Implement GRU (Gated Recurrent Unit)
- [ ] Add attention mechanism
- [ ] Multi-layer stacked LSTMs
- [ ] Beam search for text generation
- [ ] Fine-tune on different text corpora
- [ ] Named Entity Recognition (NER) task
- [ ] Machine translation with seq2seq

## Common Issues & Solutions

**Q: Training is slow**
A: Use GPU acceleration (`--device cuda`) and reduce sequence length

**Q: Generated text is repetitive**
A: Increase temperature (0.8-1.2) or train longer

**Q: Model not learning**
A: Check learning rate, increase model capacity, or train longer

## References

- [Understanding LSTM Networks - colah's blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of RNNs - Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Deep Learning Book - Goodfellow et al.](https://www.deeplearningbook.org/)
- [PyTorch RNN Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)

## License

MIT License - feel free to use for learning and research!

## Author

**Mayank** - Aspiring AI/ML Engineer focused on Computer Vision and Robotics
[GitHub](https://github.com/MAYANK12-WQ) | [LinkedIn](#)

---

**Understanding sequences from first principles** 🚀
