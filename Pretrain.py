from CustomGPT1 import CustomGPT1Model
import torch
from transformers import PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from Embedding import id_to_attributes
import numpy as np
import torch.nn.functional as F


def load_data(train_file, val_file):
    with open(train_file, "r", encoding="utf-8") as f:
        train_texts = [line.strip() for line in f.readlines()]

    with open(val_file, "r", encoding="utf-8") as f:
        val_texts = [line.strip() for line in f.readlines()]

    train_ids, train_texts_only = zip(*[(line.split()[0], ' '.join(line.split()[1:])) for line in train_texts])
    val_ids, val_texts_only = zip(*[(line.split()[0], ' '.join(line.split()[1:])) for line in val_texts])

    return train_texts, val_texts, train_texts_only, val_texts_only


def tokenize_data(tokenizer, train_texts_only, val_texts_only):
    train_encodings = tokenizer(train_texts_only, padding=True, truncation=True, max_length=48)
    val_encodings = tokenizer(val_texts_only, padding=True, truncation=True, max_length=48)
    return train_encodings, val_encodings


def shift_input_target(text_encodings):
    seq_tensor = torch.tensor(text_encodings.input_ids)
    batch_size, max_len = seq_tensor.shape

    input_seqs = []
    target_seqs = []

    for i in range(36, max_len):
        input_seq = seq_tensor[:, :i]
        padding_size = max_len - i
        input_seq_padded = F.pad(input_seq, (0, padding_size), value=PAD_TOKEN_ID)
        input_seqs.append(input_seq_padded)

        target_seq = seq_tensor[:, i]
        target_seqs.append(target_seq)

    # Stack all the tensors
    input_sequences_padded = torch.stack(input_seqs, dim=1).view(-1, max_len)
    target_sequences = torch.cat(target_seqs, dim=0)

    return input_sequences_padded, target_sequences


def create_dataset_with_indices(shifted_input_ids, shifted_labels, original_texts):
    occupation_indices = []
    gender_indices = []

    # For each original_text, generate indices and replicate them for each of the shifted sequences
    for text in original_texts:
        person_id, *tokens = text.split()
        person_id = int(person_id)
        attributes = id_to_attributes[person_id]

        # Append the occupation and gender indices instead of embeddings
        occupation_index = attributes['occupation']
        gender_index = attributes['gender']

        occupation_indices.extend([occupation_index] * 12)
        gender_indices.extend([gender_index] * 12)

    # Convert the lists of indices to tensors
    occupation_indices = torch.tensor(occupation_indices, dtype=torch.long)
    gender_indices = torch.tensor(gender_indices, dtype=torch.long)

    # Ensure the lengths match
    assert len(shifted_input_ids) == len(occupation_indices) == len(gender_indices)

    return TensorDataset(shifted_input_ids, shifted_labels, occupation_indices, gender_indices)


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
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels, occupation_ids, gender_ids = batch
            inputs, labels, occupation_ids, gender_ids = inputs.to(device), labels.to(
                device), occupation_ids.to(device), gender_ids.to(device)
            attention_mask = (inputs != PAD_TOKEN_ID).int()

            logits = model(inputs, occupation_ids=occupation_ids, gender_ids=gender_ids, attention_mask=attention_mask, use_embeds=False)

            loss = loss_fn(logits[:, -1, :], labels)
            total_loss += loss.item()
            preds = np.argmax(logits[:, -1, :].detach().cpu().numpy(), axis=-1).flatten()
            predictions.extend(preds)

    average_loss = total_loss / len(dataloader)
    return average_loss, predictions


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_file = "/home/ubuntu/Documents/TokyoPT/PTChain/IDcombinedhalfTrain (copy 1).txt"
    val_file = "/home/ubuntu/Documents/TokyoPT/PTChain/IDcombinedhalfEval (copy 1).txt"
    train_texts, val_texts, train_texts_only, val_texts_only = load_data(train_file, val_file)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file="/home/ubuntu/Documents/Tokenizer/trip_chain_tokenizer.json")
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[EOS]"
    PAD_TOKEN_ID = tokenizer.convert_tokens_to_ids("[PAD]")
    EOS_TOKEN_ID = tokenizer.convert_tokens_to_ids("[EOS]")
    UNK_TOKEN_ID = tokenizer.convert_tokens_to_ids("[UNK]")

    train_encodings, val_encodings = tokenize_data(tokenizer, train_texts_only, val_texts_only)
    train_input_ids, train_labels = shift_input_target(train_encodings)
    val_input_ids, val_labels = shift_input_target(val_encodings)

    batch_size = 1024
    train_dataset = create_dataset_with_indices(train_input_ids, train_labels, train_texts)
    val_dataset = create_dataset_with_indices(val_input_ids, val_labels, val_texts)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CustomGPT1Model()
    model.initialize_weights()
    model.to(device)
    tokenizer_vocab_size = len(tokenizer.get_vocab())
    loss_fn = torch.nn.CrossEntropyLoss()

    learning_rate = 0.000002
    num_epochs = 50
    num_train_steps = len(train_dataloader) * num_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.01 * num_train_steps,
        num_training_steps=num_train_steps
    )

    torch.cuda.empty_cache()

    for epoch in range(num_epochs):
        model.train()
        total_preds, total_labels = [], []
        total_loss = 0.00
        total_correct_preds = 0

        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Splitting the batch to its individual components
            inputs, labels, occupation_indices, gender_indices = batch
            inputs, labels, occupation_indices, gender_indices = inputs.to(device), labels.to(
                device), occupation_indices.to(device), gender_indices.to(device)
            attention_mask = (inputs != PAD_TOKEN_ID).int()
            logits = model(inputs, occupation_ids=occupation_indices, gender_ids=gender_indices,
                           attention_mask=attention_mask, use_embeds=False)

            # prevent the model from predicting non-activity tokens.
            logits[:, -1, EOS_TOKEN_ID] = -1e9
            logits[:, -1, UNK_TOKEN_ID] = -1e9
            logits[:, -1, PAD_TOKEN_ID] = -1e9

            loss = loss_fn(logits[:, -1, :], labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            preds = np.argmax(logits[:, -1, :].detach().cpu().numpy(), axis=-1).flatten()
            labels = labels.detach().cpu().numpy().flatten()
            correct_preds = (preds == labels).sum()
            total_correct_preds += correct_preds
            total_preds.extend(preds)
            total_labels.extend(labels)

            if step % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} | Step {step}/{len(train_dataloader)} | Loss: {loss.item():.8f}")

        average_train_loss = total_loss / len(train_dataloader)
        accuracy = total_correct_preds / len(total_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(total_labels, total_preds, average='macro',
                                                                   zero_division=0)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Average Loss: {average_train_loss:.8f} | "
              f"Accuracy: {accuracy:.8f} | "
              f"Precision: {precision:.8f} | "
              f"Recall: {recall:.8f} | "
              f"F1: {f1:.8f} | "
              f"Perplexity: {calculate_perplexity(average_train_loss):.8f}")

        val_loss, val_predictions = evaluate_model(model, val_dataloader, loss_fn)
        actual_labels = val_labels.tolist()
        predicted_tokens = [tokenizer.decode([pid]) for pid in val_predictions]
        actual_tokens = [tokenizer.decode([aid]) for aid in actual_labels]
        predicted_token_counts = Counter(predicted_tokens)
        actual_token_counts = Counter(actual_tokens)

        print("Token Occurrences in Ground Truth:")
        for token, count in actual_token_counts.most_common():
            print(f"{token}: {count}")

        print("\nToken Occurrences in Predictions:")
        for token, count in predicted_token_counts.most_common():
            print(f"{token}: {count}")

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Validation Loss: {val_loss:.8f} | "
            f"Validation Perplexity: {calculate_perplexity(val_loss):.8f}")

        torch.save(model, f'/home/ubuntu/Documents/model_{epoch + 1}.pth')
