# import torch
# import torch.nn as nn
# import torch.nn.init as init


# # Original model before embedding
# class CustomGPT1Model(nn.Module):
#     def __init__(self, vocab_size=33, hidden_size=256, num_layers=4, num_heads=4, max_sequence_len=128, dropout=0.2):
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

#     def forward(self, input_ids, attention_mask=None, **kwargs):
#         token_embeddings = self.embeddings(input_ids)
#         position_ids = torch.arange(0, input_ids.size(1)).unsqueeze(0).to(input_ids.device)
#         position_embeddings = self.position_embeddings(position_ids)
#         embeddings = token_embeddings + position_embeddings

#         # Convert attention mask to expected format (True where padding, False otherwise)
#         if attention_mask is not None:
#             attention_mask = attention_mask == 0

#         for layer in self.transformer_layers:
#             embeddings_norm = self.layer_norm(embeddings)
#             embeddings = layer(embeddings_norm, src_key_padding_mask=attention_mask)
#             embeddings += embeddings_norm
#             embeddings = self.dropout(embeddings)

#         embeddings = self.layer_norm(embeddings)
#         logits = self.output_layer(embeddings)

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
    def __init__(self, vocab_size=33, hidden_size=128, occupation_embed_dim=64, gender_embed_dim=8, num_layers=2,
                 num_heads=2, max_sequence_len=128, dropout=0.2):
        super(CustomGPT1Model, self).__init__()
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_sequence_len, hidden_size)
        # self.transformer_layers = nn.ModuleList([
        #     nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        #     for _ in range(num_layers)
        # ])
        self.transformer_layers = nn.ModuleList([
            TransformerLayerWithNorm(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        # self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.occupation_embed_dim = occupation_embed_dim
        self.gender_embed_dim = gender_embed_dim
        self.embedding_projection = nn.Linear(occupation_embed_dim + gender_embed_dim, hidden_size)

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                init.normal_(module.weight, mean=0.0, std=0.02)
                if hasattr(module, "bias") and module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                init.ones_(module.weight)
                init.zeros_(module.bias)

    def forward(self, input_ids, occupation_embeds, gender_embeds, attention_mask=None, **kwargs):
        token_embeddings = self.embeddings(input_ids)

        position_ids = torch.arange(0, input_ids.size(1)).unsqueeze(0).repeat(input_ids.size(0), 1).to(input_ids.device)
        position_embeddings = self.position_embeddings(position_ids)
        aggregated_embeds = torch.cat([occupation_embeds, gender_embeds], dim=-1)
        aggregated_embeds_projected = self.embedding_projection(aggregated_embeds).unsqueeze(1)
        embeddings = token_embeddings + position_embeddings + aggregated_embeds_projected

        if attention_mask is not None:
            attention_mask = attention_mask == 0

        # for layer in self.transformer_layers:
        #
        #     print("After attention mask:", torch.isnan(attention_mask).any())
        #
        #     normalized_embeddings = self.layer_norm(embeddings)
        #
        #     print("LayerNorm weight:", self.layer_norm.weight)
        #     print("LayerNorm bias:", self.layer_norm.bias)
        #     print("After normalized embeddings:", torch.isnan(normalized_embeddings).any())
        #
        #     transformer_out = layer(normalized_embeddings, src_key_padding_mask=attention_mask)
        #     embeddings = transformer_out + embeddings
        #     embeddings = self.dropout(embeddings)

        for layer_with_norm in self.transformer_layers:
            transformer_layer, layer_norm = layer_with_norm.transformer_layer, layer_with_norm.layer_norm
            normalized_embeddings = layer_norm(embeddings)

            # Self Attention/Added
            attention_output, _ = layer_with_norm.transformer_layer.self_attn(normalized_embeddings,
                                                                              normalized_embeddings,
                                                                              normalized_embeddings)

            attention_residual = attention_output + normalized_embeddings

            # Layer Normalization after Self Attention
            attention_normalized = layer_norm(attention_residual)

            # Feed Forward
            intermediate = layer_with_norm.transformer_layer.linear1(attention_normalized)

            # Activation Function (ReLU in this case)
            intermediate = F.relu(intermediate)

            ff_output = layer_with_norm.transformer_layer.linear2(intermediate)
            ff_residual = ff_output + attention_normalized

            # Layer Normalization after Feed Forward
            ff_normalized = layer_norm(ff_residual)

            # Output of this transformer layer/end of Added
            transformer_out = ff_normalized

            # # Original code
            # transformer_out = transformer_layer(normalized_embeddings, src_key_padding_mask=attention_mask)

            embeddings = transformer_out
            embeddings = self.dropout(embeddings)

        logits = self.output_layer(embeddings)

        return logits

