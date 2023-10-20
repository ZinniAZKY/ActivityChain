import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math


class TransformerLayerWithNorm(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerLayerWithNorm, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, *args, **kwargs):
        return self.transformer_layer(x, *args, **kwargs), self.layer_norm(x)


# Usually, the dimension of hidden size equals to position embeddings.
class CustomGPT1Model(nn.Module):
    def __init__(self, vocab_size=33, hidden_size=256, occupation_vocab_size=16, gender_vocab_size=2,
                 occupation_embed_dim=64, gender_embed_dim=8, num_layers=2, num_heads=2, max_sequence_len=128, dropout=0.2):
        super(CustomGPT1Model, self).__init__()
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_sequence_len, hidden_size)
        self.occupation_embeddings = nn.Embedding(occupation_vocab_size, occupation_embed_dim)
        self.gender_embeddings = nn.Embedding(gender_vocab_size, gender_embed_dim)
        self.transformer_layers = nn.ModuleList([
            TransformerLayerWithNorm(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.occupation_embed_dim = occupation_embed_dim
        self.gender_embed_dim = gender_embed_dim
        self.embedding_projection = nn.Linear(occupation_embed_dim + gender_embed_dim, hidden_size)
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

    def forward(self, input_ids, occupation_ids=None, gender_ids=None, attention_mask=None, use_embeds=True, **kwargs):
        token_embeddings = self.embeddings(input_ids)

        position_ids = torch.arange(0, input_ids.size(1)).unsqueeze(0).repeat(input_ids.size(0), 1).to(input_ids.device)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = token_embeddings + position_embeddings

        if use_embeds:
            # Lookup embeddings using indices
            occupation_embeds = self.occupation_embeddings(occupation_ids)
            gender_embeds = self.gender_embeddings(gender_ids)

            aggregated_embeds = torch.cat([occupation_embeds, gender_embeds], dim=-1)
            aggregated_embeds_projected = self.embedding_projection(aggregated_embeds).unsqueeze(1)
            embeddings += aggregated_embeds_projected

        for layer_with_norm in self.transformer_layers:
            transformer_layer, layer_norm = layer_with_norm.transformer_layer, layer_with_norm.layer_norm

            # LayerNorm before Self Attention
            normalized_embeddings = layer_norm(embeddings)

            attention_scores = torch.matmul(normalized_embeddings, normalized_embeddings.transpose(-1, -2)) / math.sqrt(self.d_model)
            if attention_mask is not None:
                extended_attention_mask = (1.0 - attention_mask.unsqueeze(1)) * -1e9
                attention_scores = attention_scores + extended_attention_mask

            attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
            attention_output = torch.matmul(attention_probs, normalized_embeddings)

            # Residual connection after self-attention
            attention_output += embeddings

            # LayerNorm before Feed Forward
            normalized_attention_output = layer_norm(attention_output)

            # Feed Forward
            intermediate = transformer_layer.linear1(normalized_attention_output)
            intermediate = F.relu(intermediate)
            ff_output = transformer_layer.linear2(intermediate)

            # Residual connection after feed-forward
            ff_output += attention_output

            # Assigning the output to embeddings for the next loop iteration
            embeddings = ff_output
            embeddings = self.dropout(embeddings)

        logits = self.output_layer(embeddings)

        return logits
