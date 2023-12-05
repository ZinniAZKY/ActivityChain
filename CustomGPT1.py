import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math


# Usually, the dimension of hidden size equals to position embeddings.
class CustomGPT1Model(nn.Module):
    def __init__(self, attribute_vocab_size=608, traffic_vocab_size=1024, spatial_vocab_size=3125, vocab_size=33,
                 hidden_size=576, num_layers=2, num_heads=2, max_sequence_len=48, dropout=0.2):
        super(CustomGPT1Model, self).__init__()
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_sequence_len, hidden_size)
        self.attribute_embeddings = nn.Embedding(attribute_vocab_size, hidden_size)
        self.traffic_embeddings = nn.Embedding(traffic_vocab_size, hidden_size)
        self.spatial_embeddings = nn.Embedding(spatial_vocab_size, hidden_size)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = hidden_size

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                init.normal_(module.weight, mean=0.0, std=0.02)
                if hasattr(module, "bias") and module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                init.ones_(module.weight)
                init.zeros_(module.bias)

    def forward(self, input_ids, combined_attribute_indices, combined_traffic_indices, combined_spatial_indices,
                attention_mask=None, use_embeds=True, **kwargs):
        token_embeddings = self.embeddings(input_ids)
        position_ids = torch.arange(0, input_ids.size(1)).unsqueeze(0).repeat(input_ids.size(0), 1).to(input_ids.device)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings
        # comment this if using embeddings
        embeddings = embeddings.transpose(0, 1)

        if use_embeds:
            attribute_embeds = self.attribute_embeddings(combined_attribute_indices)
            traffic_embeds = self.traffic_embeddings(combined_traffic_indices)
            spatial_embeds = self.spatial_embeddings(combined_spatial_indices)
            embeddings = embeddings + attribute_embeds.unsqueeze(1) + traffic_embeds.unsqueeze(
                1) + spatial_embeds.unsqueeze(1)
            embeddings = embeddings.transpose(0, 1)

        for layer in self.transformer_layers:
            embeddings = layer(embeddings, src_key_padding_mask=attention_mask)
            embeddings = self.dropout(embeddings)

        logits = self.output_layer(embeddings)

        return logits
