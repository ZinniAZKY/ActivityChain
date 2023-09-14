import torch
import torch.nn as nn
import torch.nn.init as init


# class CustomGPT1Model(nn.Module):
#     def __init__(self, vocab_size=71, hidden_size=8, num_layers=2, num_heads=2, max_sequence_len=128, dropout=0.2):
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

#     def initialize_weights(self):
#         for module in self.modules():
#             if isinstance(module, (nn.Linear, nn.Embedding)):
#                 init.normal_(module.weight, mean=0.0, std=0.02)
#                 if hasattr(module, "bias") and module.bias is not None:
#                     init.zeros_(module.bias)
#             elif isinstance(module, nn.LayerNorm):
#                 init.ones_(module.weight)
#                 init.zeros_(module.bias)

#     def forward(self, input_ids, labels=None, weights=None, **kwargs):
#         token_embeddings = self.embeddings(input_ids)
#         position_ids = torch.arange(0, input_ids.size(1)).unsqueeze(0).to(input_ids.device)
#         position_embeddings = self.position_embeddings(position_ids)
#         embeddings = token_embeddings + position_embeddings

#         for layer in self.transformer_layers:
#             embeddings_norm = self.layer_norm(embeddings)
#             embeddings = layer(embeddings_norm)
#             embeddings += embeddings_norm
#             embeddings = self.dropout(embeddings)

#         embeddings = self.layer_norm(embeddings)
#         logits = self.output_layer(embeddings)

#         if labels is not None:
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             if weights is not None:
#                 loss_func = nn.CrossEntropyLoss(weight=weights)
#             else:
#                 loss_func = nn.CrossEntropyLoss()
#             loss = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#             return loss, logits

#         return logits


# V2.0
class CustomGPT1Model(nn.Module):
    def __init__(self, vocab_size=33, hidden_size=8, num_layers=4, num_heads=4, max_sequence_len=128, dropout=0.4):
        super(CustomGPT1Model, self).__init__()
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_sequence_len, hidden_size)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                init.normal_(module.weight, mean=0.0, std=0.02)
                if hasattr(module, "bias") and module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                init.ones_(module.weight)
                init.zeros_(module.bias)

    def forward(self, input_ids, **kwargs):
        token_embeddings = self.embeddings(input_ids)
        position_ids = torch.arange(0, input_ids.size(1)).unsqueeze(0).to(input_ids.device)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings

        for layer in self.transformer_layers:
            embeddings_norm = self.layer_norm(embeddings)
            embeddings = layer(embeddings_norm)
            embeddings += embeddings_norm
            embeddings = self.dropout(embeddings)

        embeddings = self.layer_norm(embeddings)
        logits = self.output_layer(embeddings)

        return logits
