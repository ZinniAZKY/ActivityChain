import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup
from transformers import PreTrainedTokenizerFast
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

train_file = "/home/ubuntu/Documents/TokyoPT/PTChain/train.txt"
val_file = "/home/ubuntu/Documents/TokyoPT/PTChain/eval.txt"

with open(train_file, "r", encoding="utf-8") as f:
    train_texts = [line.strip() for line in f.readlines()]

with open(val_file, "r", encoding="utf-8") as f:
    val_texts = [line.strip() for line in f.readlines()]

tokenizer = PreTrainedTokenizerFast(tokenizer_file="/home/ubuntu/Documents/Tokenizer/trip_chain_tokenizer.json")
tokenizer.pad_token = "[PAD]"
tokenizer.eos_token = "[EOS]"

train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=96)
val_encodings = tokenizer(val_texts, padding=True, truncation=True, max_length=96)

# Convert tokenized encodings to torch tensors
train_input_ids = torch.tensor(train_encodings.input_ids)
train_labels = torch.tensor(train_encodings.input_ids)  # Labels are the same as inputs for language modeling tasks

val_input_ids = torch.tensor(val_encodings.input_ids)
val_labels = torch.tensor(val_encodings.input_ids)

batch_size = 4
train_dataset = TensorDataset(train_input_ids, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_input_ids, val_labels)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


class CustomGPT1Model(nn.Module):
    def __init__(self, vocab_size=33, hidden_size=4, num_layers=1, num_heads=1, max_sequence_len=128, dropout=0.25):
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
                # Apply weight initialization for linear layers and embeddings
                init.normal_(module.weight, mean=0.0, std=0.02)
                if hasattr(module, "bias") and module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # Apply weight initialization for layer normalization layers
                init.ones_(module.weight)
                init.zeros_(module.bias)

    def forward(self, input_ids):
        token_embeddings = self.embeddings(input_ids)
        position_ids = torch.arange(0, input_ids.size(1)).unsqueeze(0).to(input_ids.device)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        for layer in self.transformer_layers:
            embeddings = layer(embeddings)
        embeddings = self.layer_norm(embeddings)
        logit_result = self.output_layer(embeddings)

        return logit_result


def calculate_metrics(preds, labels):
    preds = np.argmax(preds, axis=-1).flatten()
    labels = labels.flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    return precision, recall, f1


def calculate_perplexity(loss):
    return np.exp(loss)


def evaluate_model(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            logits = model(inputs)
            loss = loss_fn(logits.view(-1, model.vocab_size), labels.view(-1))
            total_loss += loss.item()
            num_samples += inputs.size(0)

    average_loss = total_loss / num_samples
    return average_loss


model = CustomGPT1Model()
model.initialize_weights()

loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 0.000003
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
num_epochs = 10
num_train_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0.1 * num_train_steps,
    num_training_steps=num_train_steps
)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    all_preds, all_labels = [], []

    for step, batch in enumerate(train_dataloader):
        inputs, labels = batch
        logits = model(inputs)
        loss = loss_fn(logits.view(-1, model.vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        total_loss += loss.item()

        preds = np.argmax(logits.detach().cpu().numpy(), axis=-1).flatten()
        labels = labels.detach().cpu().numpy().flatten()

        all_preds.extend(preds)
        all_labels.extend(labels)

        if step % 10000 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} | Step {step}/{len(train_dataloader)} | Loss: {loss.item():.8f}")

    average_loss = total_loss / len(train_dataloader)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    print(f"Epoch {epoch + 1}/{num_epochs} | Average Loss: {average_loss:.8f} | "
          f"Precision: {precision:.8f} | Recall: {recall:.8f} | F1: {f1:.8f} | Perplexity: {calculate_perplexity(average_loss):.8f}")

    val_loss = evaluate_model(model, val_dataloader, loss_fn)
    print(
        f"Epoch {epoch + 1}/{num_epochs} | Validation Loss: {val_loss:.8f} | Validation Perplexity: {calculate_perplexity(val_loss):.8f}")

    torch.save(model.state_dict(), f'/home/ubuntu/Documents/model_{epoch+1}.pth')

