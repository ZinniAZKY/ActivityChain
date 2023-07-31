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

batch_size = 32
train_dataset = TensorDataset(train_input_ids, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_input_ids, val_labels)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


class CustomGPT1Model(nn.Module):
    def __init__(self, vocab_size=33, hidden_size=256, num_layers=4, num_heads=4, max_sequence_len=512):
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
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 5
num_train_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0.1 * num_train_steps,
    num_training_steps=num_train_steps
)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_precision, total_recall, total_f1 = 0.0, 0.0, 0.0

    for step, batch in enumerate(train_dataloader):
        inputs, labels = batch
        logits = model(inputs)
        loss = loss_fn(logits.view(-1, model.vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        print(f"Current learning rate: {scheduler.get_last_lr()[0]}")
        total_loss += loss.item()

        preds = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        precision, recall, f1 = calculate_metrics(preds, labels)
        total_precision += precision
        total_recall += recall
        total_f1 += f1

        if step % 100 == 0:
            avg_precision = total_precision / (step + 1)
            avg_recall = total_recall / (step + 1)
            avg_f1 = total_f1 / (step + 1)
            print(f"Epoch {epoch + 1}/{num_epochs} | Step {step}/{len(train_dataloader)} | "
                  f"Loss: {loss.item():.8f} | Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f} | F1: {avg_f1:.4f}")

    average_loss = total_loss / len(train_dataloader)
    avg_precision = total_precision / len(train_dataloader)
    avg_recall = total_recall / len(train_dataloader)
    avg_f1 = total_f1 / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs} | Average Loss: {average_loss:.8f} | "
          f"Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f} | F1: {avg_f1:.4f} | Perplexity: {calculate_perplexity(average_loss):.4f}")

    val_loss = evaluate_model(model, val_dataloader, loss_fn)
    print(f"Epoch {epoch + 1}/{num_epochs} | Validation Loss: {val_loss:.8f} | Validation Perplexity: {calculate_perplexity(val_loss):.4f}")

torch.save(model.state_dict(), f'/Users/zhangkunyi/Downloads/PTFolder/PTChain/model_{epoch}.pth')

