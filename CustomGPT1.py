import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math


# Original model before embedding
# class CustomGPT1Model(nn.Module):
#     def __init__(self, vocab_size=33, hidden_size=256, num_layers=4, num_heads=4, max_sequence_len=48, dropout=0.2):
#         super(CustomGPT1Model, self).__init__()
#         self.vocab_size = vocab_size
#         self.embeddings = nn.Embedding(vocab_size, hidden_size)
#         self.position_embeddings = nn.Embedding(max_sequence_len, hidden_size)
#         self.transformer_layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
#             for _ in range(num_layers)
#         ])
#         self.layer_norm = nn.LayerNorm(hidden_size)
#         self.output_layer = nn.Linear(hidden_size, vocab_size)
#         self.dropout = nn.Dropout(dropout)
# 
#     def initialize_weights(self):
#         for module in self.modules():
#             if isinstance(module, (nn.Linear, nn.Embedding)):
#                 init.normal_(module.weight, mean=0.0, std=0.02)
#                 if hasattr(module, "bias") and module.bias is not None:
#                     init.zeros_(module.bias)
#             elif isinstance(module, nn.LayerNorm):
#                 init.ones_(module.weight)
#                 init.zeros_(module.bias)
# 
#     def forward(self, input_ids, attention_mask=None, **kwargs):
#         token_embeddings = self.embeddings(input_ids)
#         position_ids = torch.arange(0, input_ids.size(1)).unsqueeze(0).to(input_ids.device)
#         position_embeddings = self.position_embeddings(position_ids)
#         embeddings = token_embeddings + position_embeddings
# 
#         # Convert attention mask to expected format (True where padding, False otherwise)
#         if attention_mask is not None:
#             attention_mask = attention_mask == 0
# 
#         for layer in self.transformer_layers:
#             embeddings_norm = self.layer_norm(embeddings)
#             embeddings = layer(embeddings_norm, src_key_padding_mask=attention_mask)
#             embeddings += embeddings_norm
#             embeddings = self.dropout(embeddings)
# 
#         embeddings = self.layer_norm(embeddings)
#         logits = self.output_layer(embeddings)
# 
#         return logits


class TransformerLayerWithNorm(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerLayerWithNorm, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, *args, **kwargs):
        return self.transformer_layer(x, *args, **kwargs), self.layer_norm(x)


# Usually, the dimension of hidden size equals to position embeddings.
class CustomGPT1Model(nn.Module):
    def __init__(self, attribute_vocab_size=608, vocab_size=33, hidden_size=512,
                 num_layers=4, num_heads=4, max_sequence_len=48, dropout=0.2):
        super(CustomGPT1Model, self).__init__()
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_sequence_len, hidden_size)
        self.attribute_embeddings = nn.Embedding(attribute_vocab_size, hidden_size)
        self.transformer_layers = nn.ModuleList([
            TransformerLayerWithNorm(hidden_size, num_heads)
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

    def forward(self, input_ids, combined_indices, attention_mask=None, use_embeds=True, **kwargs):
        token_embeddings = self.embeddings(input_ids)
        position_ids = torch.arange(0, input_ids.size(1)).unsqueeze(0).repeat(input_ids.size(0), 1).to(input_ids.device)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings

        if use_embeds:
            attribute_embeds = self.attribute_embeddings(combined_indices)
            embeddings += attribute_embeds.unsqueeze(1)

        for layer_block in self.transformer_layers:
            transformer_layer, layer_norm = layer_block.transformer_layer, layer_block.layer_norm

            # LayerNorm before Self Attention
            normalized_embeddings = layer_norm(embeddings)

            attention_scores = torch.matmul(normalized_embeddings, normalized_embeddings.transpose(-1, -2)) / math.sqrt(
                self.d_model)
            if attention_mask is not None:
                extended_attention_mask = (1.0 - attention_mask.unsqueeze(1)) * -1e9
                attention_scores = attention_scores + extended_attention_mask

            attention_probs = self.dropout(torch.nn.functional.softmax(attention_scores, dim=-1))
            attention_output = torch.matmul(attention_probs, normalized_embeddings)

            # Residual connection after self-attention
            attention_output += embeddings
            attention_output = self.dropout(attention_output)

            # LayerNorm before Feed Forward
            normalized_attention_output = layer_norm(attention_output)

            # Feed Forward
            intermediate = transformer_layer.linear1(normalized_attention_output)
            intermediate = F.gelu(intermediate)
            ff_output = transformer_layer.linear2(intermediate)

            # Residual connection after feed-forward
            ff_output += attention_output
            ff_output = self.dropout(ff_output)

            # Assigning the output to embeddings for the next loop iteration
            embeddings = ff_output

        logits = self.output_layer(embeddings)

        return logits
