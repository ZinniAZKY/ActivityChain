import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json

torch.manual_seed(42)


class JSONTokenizer:
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
        self.vocab = data['model']['vocab']
        self.token_to_id = {token: int(id) for token, id in self.vocab.items()}
        self.id_to_token = {int(id): token for token, id in self.vocab.items()}

    def encode(self, text):
        return [self.token_to_id.get(token, self.token_to_id['[UNK]']) for token in text.split()]

    def decode(self, token_ids):
        return ' '.join(self.id_to_token.get(id, '[UNK]') for id in token_ids)


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, sequence_length=39, predict_length=180, n_rows=None):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.predict_length = predict_length
        self.pairs = []

        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if n_rows is not None and i >= n_rows:
                    break
                encoded_line = self.tokenizer.encode(line.strip())
                if len(encoded_line) >= sequence_length + predict_length:
                    input_seq = encoded_line[:sequence_length]
                    target_seq = encoded_line[sequence_length:sequence_length + predict_length]
                    self.pairs.append((input_seq, target_seq))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_seq, target_seq = self.pairs[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded)
        last_output = lstm_out[:, -1, :]
        predictions = [self.fc(last_output.unsqueeze(1))]
        current_token_ids = last_output.argmax(-1)
        current_input = self.embedding(current_token_ids).unsqueeze(1)

        for _ in range(179):  # Generate 179 additional tokens to get a total of 180
            output, hidden = self.lstm(current_input, hidden)
            output = self.fc(output)
            predictions.append(output)

            current_token_ids = output.argmax(-1)
            if current_token_ids.ndim == 1:
                current_input = self.embedding(current_token_ids).unsqueeze(1)
            else:
                current_input = self.embedding(current_token_ids.squeeze()).unsqueeze(1)

        predictions = torch.cat(predictions, dim=1)
        return predictions

    def init_hidden(self, batch_size):
        # Initialize hidden and cell states with zeros
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())
        return hidden


def train_and_validate(model, train_dataset, val_dataset, batch_size, num_epochs, lr, device):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.transpose(1, 2), targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.transpose(1, 2), targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_dataloader)

        print(f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            output, _ = model(inputs)
            loss = criterion(output.transpose(1, 2), targets)
            total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss


def generate_text(model, tokenizer, start_seq, length, device='cuda'):
    model.eval()
    start_tokens = tokenizer.encode(start_seq)
    input = torch.tensor(start_tokens, dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)

    generated_text = start_seq
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input, hidden)
            probabilities = torch.softmax(output[:, -1, :], dim=1)
            next_token_id = torch.multinomial(probabilities, 1).item()
            generated_text += tokenizer.decode([next_token_id])

            input = torch.tensor([[next_token_id]], dtype=torch.long).to(device)

    return generated_text


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    json_tokenizer_path = '/home/ubuntu/Documents/Tokenizer/trip_chain_tokenizer_Tokyo_attr_loc_mode.json'
    tokenizer = JSONTokenizer(json_tokenizer_path)
    vocab_size = len(tokenizer.token_to_id)

    train_file_path = '/home/ubuntu/Documents/TokyoPT/PTChain/PTAttrActLocMode/Tokyo2008PTChain_Mode_6_24_Train.txt'
    val_file_path = '/home/ubuntu/Documents/TokyoPT/PTChain/PTAttrActLocMode/Tokyo2008PTChain_Mode_6_24_Eval.txt'
    train_dataset = TextDataset(train_file_path, tokenizer, sequence_length=39, predict_length=180, n_rows=25000)
    val_dataset = TextDataset(val_file_path, tokenizer, sequence_length=39, predict_length=180, n_rows=2500)

    device = get_device()
    model = LSTMModel(vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2).to(device)

    train_and_validate(model, train_dataset, val_dataset, batch_size=64, num_epochs=50, lr=5e-6, device=device)
