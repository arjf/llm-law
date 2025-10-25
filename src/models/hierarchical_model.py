"""
Hierarchical InLegalBERT model architecture.
Combines BERT with BiGRU and attention mechanism for document classification.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class HierarchicalInLegalBERT(nn.Module):
    """Hierarchical BERT: InLegalBERT + BiGRU + Attention"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.bert = AutoModel.from_pretrained(
            config.base_model, cache_dir=config.cache_dir
        )

        self.bigru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.gru_hidden_size,
            num_layers=config.gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.gru_num_layers > 1 else 0,
        )

        gru_output_size = config.gru_hidden_size * 2
        self.attention = nn.Sequential(
            nn.Linear(gru_output_size, 128), nn.Tanh(), nn.Linear(128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(gru_output_size, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.num_classes),
        )

    def forward(self, input_ids, attention_mask, num_chunks):
        batch_size, max_chunks, seq_len = input_ids.size()

        input_ids_flat = input_ids.view(-1, seq_len)
        attention_mask_flat = attention_mask.view(-1, seq_len)

        bert_output = self.bert(
            input_ids=input_ids_flat, attention_mask=attention_mask_flat
        )

        cls_embeddings = bert_output.last_hidden_state[:, 0, :]
        chunk_embeddings = cls_embeddings.view(batch_size, max_chunks, -1)

        gru_output, _ = self.bigru(chunk_embeddings)

        attention_scores = self.attention(gru_output).squeeze(-1)
        chunk_mask = torch.arange(max_chunks).unsqueeze(0).to(num_chunks.device)
        chunk_mask = (chunk_mask < num_chunks.unsqueeze(1)).float()
        attention_scores = attention_scores.masked_fill(chunk_mask == 0, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1)

        document_embedding = torch.bmm(
            attention_weights.unsqueeze(1), gru_output
        ).squeeze(1)

        logits = self.classifier(document_embedding)

        return logits, attention_weights
