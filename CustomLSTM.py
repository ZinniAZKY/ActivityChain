import torch
import torch.nn as nn


class CustomLSTMModel(nn.Module):
    def __init__(self, attribute_vocab_size=608, traffic_vocab_size=1024, spatial_vocab_size=3125, vocab_size=33, hidden_size=512, num_layers=4, dropout=0.2):
        super(CustomLSTMModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Embeddings
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(48, hidden_size)
        self.attribute_embeddings = nn.Embedding(attribute_vocab_size, hidden_size)
        self.traffic_embeddings = nn.Embedding(traffic_vocab_size, hidden_size)
        self.spatial_embeddings = nn.Embedding(spatial_vocab_size, hidden_size)

        # LSTM layers
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, combined_attribute_indices, combined_traffic_indices, combined_spatial_indices, attention_mask=None, use_embeds=True):
        # Embeddings
        token_embeddings = self.embeddings(input_ids)
        position_ids = torch.arange(0, input_ids.size(1)).unsqueeze(0).repeat(input_ids.size(0), 1).to(input_ids.device)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings

        if use_embeds:
            attribute_embeds = self.attribute_embeddings(combined_attribute_indices)
            traffic_embeds = self.traffic_embeddings(combined_traffic_indices)
            spatial_embeds = self.spatial_embeddings(combined_spatial_indices)
            embeddings += attribute_embeds.unsqueeze(1) + traffic_embeds.unsqueeze(1) + spatial_embeds.unsqueeze(1)

        # LSTM layers
        lstm_output, _ = self.lstm(embeddings)

        # Apply mask to nullify the impact of padding tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(lstm_output.size())
            lstm_output = lstm_output * mask

        logits = self.output_layer(lstm_output)

        return logits
