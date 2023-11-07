from CustomGPT1 import CustomGPT1Model
import torch
from transformers import PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from Embedding import id_to_attributes
import numpy as np
import torch.nn.functional as F
import math
# 
# 
# def load_texts_from_file(filename):
#     with open(filename, "r", encoding="utf-8") as f:
#         return [line.strip() for line in f.readlines()]
# 
# 
# def shift_input_target(text_encodings):
#     input_sequences = []
#     target_sequences = []
# 
#     for seq in text_encodings.input_ids:
#         input_sequences.extend([seq[:i] for i in range(36, len(seq))])
#         target_sequences.extend(seq[i] for i in range(36, len(seq)))
# 
#     input_sequences = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in input_sequences], batch_first=True,
#                                                       padding_value=PAD_TOKEN_ID)
# 
#     return input_sequences, torch.tensor(target_sequences)
# 
# 
# def calculate_metrics(preds, labels):
#     preds = np.argmax(preds, axis=-1).flatten()
#     labels = labels.flatten()
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
#     return precision, recall, f1
# 
# 
# def calculate_perplexity(loss):
#     return np.exp(loss)
# 
# 
# def evaluate_model(model, dataloader, loss_fn, tokenizer):
#     model.eval()
# 
#     total_val_loss = 0.0
#     total_predictions = []
# 
#     with torch.no_grad():
#         for batch in dataloader:
#             inputs, labels = batch
#             inputs, labels = inputs.to(device), labels.to(device)
#             attention_mask = (inputs != PAD_TOKEN_ID).int().transpose(0, 1)
#             logits = model(inputs, attention_mask=attention_mask)
#             logits[:, -1, EOS_TOKEN_ID] = -1e9
#             logits[:, -1, UNK_TOKEN_ID] = -1e9
#             logits[:, -1, PAD_TOKEN_ID] = -1e9
#             loss = loss_fn(logits[:, -1, :], labels)
# 
#             total_val_loss += loss.item()
# 
#             preds = np.argmax(logits[:, -1, :].cpu().numpy(), axis=-1).flatten()
#             total_predictions.extend(preds)
# 
#     average_val_loss = total_val_loss / len(dataloader)
# 
#     actual_labels = [label for batch in dataloader for label in batch[1].tolist()]
#     predicted_tokens = [tokenizer.decode([pid]) for pid in total_predictions]
#     actual_tokens = [tokenizer.decode([aid]) for aid in actual_labels]
#     predicted_token_counts = Counter(predicted_tokens)
#     actual_token_counts = Counter(actual_tokens)
# 
#     return average_val_loss, total_predictions, actual_token_counts, predicted_token_counts
# 
# 
# train_file = "/home/ubuntu/Documents/TokyoPT/PTChain/combinedhalfTrainTest.txt"
# val_file = "/home/ubuntu/Documents/TokyoPT/PTChain/combinedhalfEvalTest.txt"
# train_texts = load_texts_from_file(train_file)
# val_texts = load_texts_from_file(val_file)
# tokenizer = PreTrainedTokenizerFast(tokenizer_file="/home/ubuntu/Documents/Tokenizer/trip_chain_tokenizer.json")
# batch_size = 256
# learning_rate = 0.000005
# num_epochs = 15
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 
# tokenizer.pad_token = "[PAD]"
# tokenizer.eos_token = "[EOS]"
# PAD_TOKEN_ID = tokenizer.convert_tokens_to_ids("[PAD]")
# EOS_TOKEN_ID = tokenizer.convert_tokens_to_ids("[EOS]")
# UNK_TOKEN_ID = tokenizer.convert_tokens_to_ids("[UNK]")
# 
# train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=48)
# val_encodings = tokenizer(val_texts, padding=True, truncation=True, max_length=48)
# 
# train_input_ids, train_labels = shift_input_target(train_encodings)
# val_input_ids, val_labels = shift_input_target(val_encodings)
# 
# # calculate class weights
# weight_list = train_labels.tolist()
# weight_counts = Counter(weight_list)
# all_token_ids = list(tokenizer.get_vocab().values())
# token_to_id = tokenizer.get_vocab()
# id_to_token = {v: k for k, v in token_to_id.items()}
# token_weights = {token_id: 1 for token_id in all_token_ids}
# max_weight = max(weight_counts.values())
# smoothing_factor = 10
# smoothed_weights = {token_id: max_weight / (count + smoothing_factor) for token_id, count in weight_counts.items()}
# max_allowed_weight = 10
# capped_weights = {k: min(v, max_allowed_weight) for k, v in smoothed_weights.items()}
# # token_weights[token_to_id["House"]] = token_weights[token_to_id["House"]] * 0.3
# class_weights_tensor = torch.zeros(len(all_token_ids), dtype=torch.float).to(device)
# 
# for token_id, weight in capped_weights.items():
#     class_weights_tensor[token_id] = weight
# 
# train_dataset = TensorDataset(train_input_ids, train_labels)
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataset = TensorDataset(val_input_ids, val_labels)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
# 
# model = CustomGPT1Model()
# model.to(device)
# model.initialize_weights()
# tokenizer_vocab_size = len(tokenizer.get_vocab())
# 
# # loss_fn = torch.nn.CrossEntropyLoss()
# # loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
# loss_fn = WeightedFocalLoss(alpha=0.25, gamma=2.5)
# # loss_fn = WeightedFocalLoss(alpha=0.25, gamma=4, class_weights=class_weights_tensor)
# 
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
# num_train_steps = len(train_dataloader) * num_epochs
# # scheduler = get_linear_schedule_with_warmup(
# #     optimizer,
# #     num_warmup_steps=0.01 * num_train_steps,
# #     num_training_steps=num_train_steps
# # )
# scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
# 
# for epoch in range(num_epochs):
#     model.train()
#     total_preds, total_labels = [], []
#     total_loss = 0.00
#     total_correct_preds = 0
# 
#     for step, batch in enumerate(train_dataloader):
#         optimizer.zero_grad()
#         inputs, labels = batch
#         inputs, labels = inputs.to(device), labels.to(device)
#         attention_mask = (inputs != PAD_TOKEN_ID).int().transpose(0, 1)
#         logits = model(inputs, attention_mask=attention_mask)
#         logits[:, -1, EOS_TOKEN_ID] = -1e9
#         logits[:, -1, UNK_TOKEN_ID] = -1e9
#         logits[:, -1, PAD_TOKEN_ID] = -1e9
#         loss = loss_fn(logits[:, -1, :], labels)
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
# 
#         # calculate gradient of each step
#         average_grad_norm = 0.0
#         num_grads = 0
#         for name, param in model.named_parameters():
#             if param.requires_grad and param.grad is not None:
#                 average_grad_norm += param.grad.data.norm(2).item()
#                 num_grads += 1
# 
#         print_frequency = 100
#         if step % print_frequency == 0:
#             print(f"Epoch: {epoch}, Step: {step}, Average Gradient Norm: {average_grad_norm:.4f}")
# 
#         if num_grads > 0:
#             average_grad_norm /= num_grads
# 
#         total_loss += loss.item()
# 
#         preds = np.argmax(logits[:, -1, :].detach().cpu().numpy(), axis=-1).flatten()
#         labels = labels.detach().cpu().numpy().flatten()
#         correct_preds = (preds == labels).sum()
#         total_correct_preds += correct_preds
#         total_preds.extend(preds)
#         total_labels.extend(labels)
# 
#         if step % 100 == 0:
#             print(f"Epoch {epoch + 1}/{num_epochs} | Step {step}/{len(train_dataloader)} | Loss: {loss.item():.8f}")
# 
#     average_train_loss = total_loss / len(train_dataloader)
#     accuracy = total_correct_preds / len(total_labels)
#     precision, recall, f1, _ = precision_recall_fscore_support(total_labels, total_preds, average='macro',
#                                                                zero_division=0)
# 
#     val_loss, val_predictions, actual_token_counts, predicted_token_counts = evaluate_model(model, val_dataloader, loss_fn, tokenizer)
# 
#     print(f"Epoch {epoch + 1}/{num_epochs} | "
#           f"Average Loss: {average_train_loss:.8f} | "
#           f"Accuracy: {accuracy:.8f} | "
#           f"Precision: {precision:.8f} | "
#           f"Recall: {recall:.8f} | "
#           f"F1: {f1:.8f} | "
#           f"Perplexity: {calculate_perplexity(average_train_loss):.8f}")
# 
#     print("Token Occurrences in Ground Truth:")
#     for token, count in actual_token_counts.most_common():
#         print(f"{token}: {count}")
# 
#     print("\nToken Occurrences in Predictions:")
#     for token, count in predicted_token_counts.most_common():
#         print(f"{token}: {count}")
# 
#     print(
#         f"Epoch {epoch + 1}/{num_epochs} | Validation Loss: {val_loss:.8f} | "
#         f"Validation Perplexity: {calculate_perplexity(val_loss):.8f}")
# 
#     torch.save(model, f'/home/ubuntu/Documents/model_{epoch + 1}.pth')


from CustomGPT1 import CustomGPT1Model
import torch
from transformers import PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from Embedding import id_to_attributes
import numpy as np
import torch.nn.functional as F
from FocalLoss import WeightedFocalLoss


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
    combined_indices = []
    # Assume you have these maximums calculated or set beforehand
    occu_max = max(id_to_attributes.values(), key=lambda x: x['occupation'])['occupation'] + 1
    gender_max = max(id_to_attributes.values(), key=lambda x: x['gender'])['gender'] + 1
    age_max = max(id_to_attributes.values(), key=lambda x: x['age'])['age'] + 1
    base = occu_max * gender_max * age_max  # This is the base for your mixed-base calculation

    for text in original_texts:
        person_id, *tokens = text.split()
        person_id = int(person_id)
        attributes = id_to_attributes.get(person_id, {'occupation': 0, 'gender': 0, 'age': 0})
        occupation_index = attributes['occupation']
        gender_index = attributes['gender']
        age_index = attributes['age']
        combined_id = (age_index * gender_max * occu_max) + (gender_index * occu_max) + occupation_index
        combined_indices.extend([combined_id] * 12)  # 12 is the length of the shifted sequences

    combined_indices = torch.tensor(combined_indices, dtype=torch.long)
    assert len(shifted_input_ids) == len(combined_indices)  # Ensuring the tensor sizes match

    return TensorDataset(shifted_input_ids, shifted_labels, combined_indices)


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
            inputs, labels, combined_indices = batch
            inputs, labels, combined_indices = inputs.to(device), labels.to(device), combined_indices.to(device)
            attention_mask = (inputs != PAD_TOKEN_ID).int()
            logits = model(inputs, combined_indices=combined_indices, attention_mask=attention_mask)

            logits[:, -1, EOS_TOKEN_ID] = -1e9
            logits[:, -1, UNK_TOKEN_ID] = -1e9
            logits[:, -1, PAD_TOKEN_ID] = -1e9

            loss = loss_fn(logits[:, -1, :], labels)
            total_loss += loss.item()
            preds = np.argmax(logits[:, -1, :].detach().cpu().numpy(), axis=-1).flatten()
            predictions.extend(preds)

    average_loss = total_loss / len(dataloader)
    return average_loss, predictions


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_file = "/home/ubuntu/Documents/TokyoPT/PTChain/IDNoAllHome618Train.txt"
    val_file = "/home/ubuntu/Documents/TokyoPT/PTChain/IDNoAllHome618Eval.txt"
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

    # Initialize class weights with default value of 1 for each token ID
    all_token_ids = list(tokenizer.get_vocab().values())  # Retrieve all token IDs from the tokenizer
    class_weights_tensor = torch.ones(len(all_token_ids), dtype=torch.float).to(device)  # Create a tensor of ones

    # Modify the weight manually
    token_to_id = tokenizer.get_vocab()  # Get the mapping from tokens to their respective IDs
    house_token_id = token_to_id["House"]
    backhome_token_id = token_to_id['Back_Home']
    office_token_id = token_to_id["Office"]
    class_weights_tensor[house_token_id] = 0.925
    class_weights_tensor[backhome_token_id] = 0.975
    class_weights_tensor[office_token_id] = 0.925

    batch_size = 1024
    learning_rate = 0.00000005
    num_epochs = 50
    train_dataset = create_dataset_with_indices(train_input_ids, train_labels, train_texts)
    val_dataset = create_dataset_with_indices(val_input_ids, val_labels, val_texts)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CustomGPT1Model()
    model.initialize_weights()
    model.to(device)
    tokenizer_vocab_size = len(tokenizer.get_vocab())
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = WeightedFocalLoss(alpha=0.25, gamma=2, class_weights=class_weights_tensor)
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
            inputs, labels, combined_indices = batch
            inputs, labels, combined_indices = inputs.to(device), labels.to(device), combined_indices.to(device)
            attention_mask = (inputs != PAD_TOKEN_ID).int()
            logits = model(inputs, combined_indices=combined_indices, attention_mask=attention_mask)

            logits[:, -1, EOS_TOKEN_ID] = -1e9
            logits[:, -1, UNK_TOKEN_ID] = -1e9
            logits[:, -1, PAD_TOKEN_ID] = -1e9

            loss = loss_fn(logits[:, -1, :], labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            probs = F.softmax(logits[:, -1, :], dim=-1).detach().cpu().numpy()
            preds = [np.random.choice(len(p), p=p) for p in probs]
            preds = np.array(preds).flatten()
            labels = labels.detach().cpu().numpy().flatten()
            correct_preds = (preds == labels).sum()
            total_correct_preds += correct_preds
            total_preds.extend(preds)
            total_labels.extend(labels)

            if step % 250 == 0:
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



        
