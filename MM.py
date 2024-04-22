import json
import numpy as np
from torch.utils.data import Dataset


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


class SimpleDataset(Dataset):
    def __init__(self, file_path, tokenizer, n_rows=None):
        self.tokenizer = tokenizer
        self.sequences = []

        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if n_rows is not None and i >= n_rows:
                    break
                encoded_line = self.tokenizer.encode(line.strip())
                if len(encoded_line) > 1:
                    self.sequences.append(encoded_line)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class MarkovModel:
    def __init__(self):
        self.transitions = {}

    def train(self, dataset):
        # Count transitions
        for sequence in dataset:
            for i in range(len(sequence) - 1):
                current, next_token = sequence[i], sequence[i + 1]
                if current not in self.transitions:
                    self.transitions[current] = {}
                if next_token not in self.transitions[current]:
                    self.transitions[current][next_token] = 0
                self.transitions[current][next_token] += 1

        # Convert counts to probabilities
        for current in self.transitions:
            total_transitions = sum(self.transitions[current].values())
            for next_token in self.transitions[current]:
                self.transitions[current][next_token] /= total_transitions

    def generate_text(self, tokenizer, start_seq, length):
        generated_text = start_seq
        current_token = tokenizer.encode(start_seq.split()[-1])[0]
        for _ in range(length):
            next_token = self.sample_next(current_token)
            generated_text += ' ' + tokenizer.decode([next_token])
            current_token = next_token
        return generated_text

    def sample_next(self, current_token):
        if current_token not in self.transitions:
            return current_token  # Handle case with no known transitions
        next_tokens, probs = zip(*self.transitions[current_token].items())
        next_token = np.random.choice(next_tokens, p=probs)
        return next_token


if __name__ == "__main__":
    json_tokenizer_path = '/home/zhangky/Documents/ZhangKY/Tokenizer/trip_chain_tokenizer_Tokyo_attr_loc_mode.json'
    train_file_path = '/home/zhangky/Documents/ZhangKY/TokyoPT/Tokyo2008PTChain_Mode_6_24_Train.txt'
    val_file_path = '/home/zhangky/Documents/ZhangKY/TokyoPT/Tokyo2008PTChain_Mode_6_24_Eval.txt'
    tokenizer = JSONTokenizer(json_tokenizer_path)
    train_dataset = SimpleDataset(train_file_path, tokenizer)
    val_dataset = SimpleDataset(val_file_path, tokenizer)

    model = MarkovModel()
    model.train(train_dataset)

    for i in range(min(2500, len(val_dataset))):
        initial_sequence = val_dataset[i]
        if len(initial_sequence) >= 219:
            start_tokens = initial_sequence[:39]
            initial_text = tokenizer.decode(start_tokens)
            generated_text = model.generate_text(tokenizer, initial_text, 180)
            print(f"Generated text for validation sample {i + 1}: {generated_text}")
        else:
            print(f"Validation sample {i + 1} does not meet the expected number of tokens.")
