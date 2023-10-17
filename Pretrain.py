# from CustomGPT1 import CustomGPT1Model
# import torch
# from transformers import PreTrainedTokenizerFast
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np
# from sklearn.metrics import precision_recall_fscore_support
# from transformers import get_linear_schedule_with_warmup
# from collections import Counter

# train_file = "/home/ubuntu/Documents/TokyoPT/PTChain/combinedhalfTrain.txt"
# val_file = "/home/ubuntu/Documents/TokyoPT/PTChain/combinedhalfEval.txt"

# with open(train_file, "r", encoding="utf-8") as f:
#     train_texts = [line.strip() for line in f.readlines()]

# with open(val_file, "r", encoding="utf-8") as f:
#     val_texts = [line.strip() for line in f.readlines()]

# tokenizer = PreTrainedTokenizerFast(tokenizer_file="/home/ubuntu/Documents/Tokenizer/trip_chain_tokenizer.json")
# tokenizer.pad_token = "[PAD]"
# tokenizer.eos_token = "[EOS]"
# PAD_TOKEN_ID = tokenizer.convert_tokens_to_ids("[PAD]")

# train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=48)
# val_encodings = tokenizer(val_texts, padding=True, truncation=True, max_length=48)


# def shift_input_target(text_encodings):
#     input_sequences = []
#     target_sequences = []

#     for seq in text_encodings.input_ids:
#         input_sequences.extend([seq[:i] for i in range(36, len(seq))])
#         target_sequences.extend(seq[i] for i in range(36, len(seq)))

#     input_sequences = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in input_sequences], batch_first=True, padding_value=PAD_TOKEN_ID)

#     return input_sequences, torch.tensor(target_sequences)


# train_input_ids, train_labels = shift_input_target(train_encodings)
# val_input_ids, val_labels = shift_input_target(val_encodings)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # # Now calculate class weights
# # label_freqs = Counter(train_labels.tolist())
# # max_freq = max(label_freqs.values())
# # class_weights = {label: max_freq / count for label, count in label_freqs.items()}
# #
# # num_classes = 33
# # weights = [class_weights.get(i, 1.0) for i in range(num_classes)]
# # weights_tensor = torch.tensor(weights).to(device)


# batch_size = 1024
# train_dataset = TensorDataset(train_input_ids, train_labels)
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataset = TensorDataset(val_input_ids, val_labels)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# def calculate_metrics(preds, labels):
#     preds = np.argmax(preds, axis=-1).flatten()
#     labels = labels.flatten()
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
#     return precision, recall, f1


# def calculate_perplexity(loss):
#     return np.exp(loss)


# def evaluate_model(model, dataloader, loss_fn):
#     model.eval()
#     total_loss = 0.0
#     predictions = []

#     with torch.no_grad():
#         for batch in dataloader:
#             inputs, labels = batch
#             inputs, labels = inputs.to(device), labels.to(device)
#             attention_mask = (inputs != PAD_TOKEN_ID).int().transpose(0, 1)
#             logits = model(inputs, attention_mask=attention_mask)
#             loss = loss_fn(logits[:, -1, :], labels)
#             total_loss += loss.item()
#             preds = np.argmax(logits[:, -1, :].detach().cpu().numpy(), axis=-1).flatten()
#             predictions.extend(preds)

#     average_loss = total_loss / len(dataloader)
#     return average_loss, predictions


# model = CustomGPT1Model()
# model.to(device)
# model.initialize_weights()
# tokenizer_vocab_size = len(tokenizer.get_vocab())

# loss_fn = torch.nn.CrossEntropyLoss()
# # loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)

# learning_rate = 0.000005
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
# num_epochs = 15
# num_train_steps = len(train_dataloader) * num_epochs
# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=0.01 * num_train_steps,
#     num_training_steps=num_train_steps
# )

# for epoch in range(num_epochs):
#     model.train()
#     total_preds, total_labels = [], []
#     total_loss = 0.00
#     total_correct_preds = 0

#     for step, batch in enumerate(train_dataloader):
#         inputs, labels = batch
#         inputs, labels = inputs.to(device), labels.to(device)
#         attention_mask = (inputs != PAD_TOKEN_ID).int().transpose(0, 1)
#         logits = model(inputs, attention_mask=attention_mask)
#         loss = loss_fn(logits[:, -1, :], labels)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         scheduler.step()

#         total_loss += loss.item()

#         preds = np.argmax(logits[:, -1, :].detach().cpu().numpy(), axis=-1).flatten()
#         labels = labels.detach().cpu().numpy().flatten()
#         correct_preds = (preds == labels).sum()
#         total_correct_preds += correct_preds
#         total_preds.extend(preds)
#         total_labels.extend(labels)

#         if step % 250 == 0:
#             print(f"Epoch {epoch + 1}/{num_epochs} | Step {step}/{len(train_dataloader)} | Loss: {loss.item():.8f}")

#     average_train_loss = total_loss / len(train_dataloader)
#     accuracy = total_correct_preds / len(total_labels)
#     precision, recall, f1, _ = precision_recall_fscore_support(total_labels, total_preds, average='macro',
#                                                                zero_division=0)

#     print(f"Epoch {epoch + 1}/{num_epochs} | "
#           f"Average Loss: {average_train_loss:.8f} | "
#           f"Accuracy: {accuracy:.8f} | "
#           f"Precision: {precision:.8f} | "
#           f"Recall: {recall:.8f} | "
#           f"F1: {f1:.8f} | "
#           f"Perplexity: {calculate_perplexity(average_train_loss):.8f}")

#     val_loss, val_predictions = evaluate_model(model, val_dataloader, loss_fn)
#     actual_labels = val_labels.tolist()
#     predicted_tokens = [tokenizer.decode([pid]) for pid in val_predictions]
#     actual_tokens = [tokenizer.decode([aid]) for aid in actual_labels]
#     predicted_token_counts = Counter(predicted_tokens)
#     actual_token_counts = Counter(actual_tokens)

#     print("Token Occurrences in Ground Truth:")
#     for token, count in actual_token_counts.most_common():
#         print(f"{token}: {count}")

#     print("\nToken Occurrences in Predictions:")
#     for token, count in predicted_token_counts.most_common():
#         print(f"{token}: {count}")

#     print(
#         f"Epoch {epoch + 1}/{num_epochs} | Validation Loss: {val_loss:.8f} | "
#         f"Validation Perplexity: {calculate_perplexity(val_loss):.8f}")

#     torch.save(model, f'/home/ubuntu/Documents/model_{epoch + 1}.pth')



from CustomGPT1 import CustomGPT1Model
import torch
from transformers import PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from Embedding import occupation_embedding, gender_embedding, id_to_attributes
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


def create_dataset_with_embeddings(shifted_input_ids, shifted_labels, original_texts):
    occupation_embeds = []
    gender_embeds = []

    # For each original_text, generate embeddings and replicate them for each of the shifted sequences
    for text in original_texts:
        person_id, *tokens = text.split()
        person_id = int(person_id)
        attributes = id_to_attributes[person_id]
        occupation_embed = occupation_embedding(torch.tensor([attributes['occupation']]))
        gender_embed = gender_embedding(torch.tensor([attributes['gender']]))

        occupation_embeds.extend([occupation_embed] * 12)
        gender_embeds.extend([gender_embed] * 12)

    # Convert the lists of embeddings to tensors
    occupation_embeds = torch.cat(occupation_embeds).squeeze(1)
    gender_embeds = torch.cat(gender_embeds).squeeze(1)

    # Ensure the lengths match
    assert len(shifted_input_ids) == len(occupation_embeds) == len(gender_embeds)

    return TensorDataset(shifted_input_ids, shifted_labels, occupation_embeds, gender_embeds)


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
            inputs, labels, occupation_embeds, gender_embeds = batch
            inputs, labels, occupation_embeds, gender_embeds = inputs.to(device), labels.to(
                device), occupation_embeds.to(device), gender_embeds.to(device)
            attention_mask = (inputs != PAD_TOKEN_ID).int().transpose(0, 1)

            logits = model(inputs, occupation_embeds=occupation_embeds, gender_embeds=gender_embeds,
                           attention_mask=attention_mask)

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

    train_encodings, val_encodings = tokenize_data(tokenizer, train_texts_only, val_texts_only)
    train_input_ids, train_labels = shift_input_target(train_encodings)
    val_input_ids, val_labels = shift_input_target(val_encodings)

    batch_size = 128
    train_dataset = create_dataset_with_embeddings(train_input_ids, train_labels, train_texts)
    val_dataset = create_dataset_with_embeddings(val_input_ids, val_labels, val_texts)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CustomGPT1Model()
    model.initialize_weights()
    model.to(device)
    tokenizer_vocab_size = len(tokenizer.get_vocab())
    loss_fn = torch.nn.CrossEntropyLoss()

    learning_rate = 0.000002
    num_epochs = 15
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
            inputs, labels, occupation_embeds, gender_embeds = batch
            inputs, labels, occupation_embeds, gender_embeds = inputs.to(device), labels.to(device), occupation_embeds.to(
                device), gender_embeds.to(device)

            occupation_embeds = occupation_embeds.detach().to(device)
            gender_embeds = gender_embeds.detach().to(device)

            attention_mask = (inputs != PAD_TOKEN_ID).int().transpose(0, 1)
            logits = model(inputs, occupation_embeds=occupation_embeds, gender_embeds=gender_embeds,
                           attention_mask=attention_mask)

            loss = loss_fn(logits[:, -1, :], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
