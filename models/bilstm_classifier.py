import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=300,
        hidden_size=256,
        num_layers=2,
        num_labels=13,
        dropout=0.3,
    ):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # BiLSTM
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

        # Linear classifier
        self.fc = nn.Linear(hidden_size * 2, num_labels)

        # Loss function for compatibility with HuggingFace models
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Mimics HuggingFace output format:
        return (loss, logits)
        """
        # input_ids = [batch, seq_len]
        embedded = self.embedding(input_ids)              # -> [batch, seq, embed_dim]

        # Pack padded for efficiency (optional)
        output, (h_n, c_n) = self.lstm(embedded)          # -> output [batch, seq, hidden*2]

        # Pooling strategy (mean pooling + mask)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)           # [batch, seq, 1]
            output = output * mask
            pooled = output.sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = output.mean(dim=1)

        pooled = self.dropout(pooled)

        logits = self.fc(pooled)                          # -> [batch, num_labels]

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return loss, logits
