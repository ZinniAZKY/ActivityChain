from CustomGPT1 import CustomGPT1Model
from CustomGRU import CustomGRUModel
import torch
from transformers import PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import Counter
from Embedding import id_to_attributes
import numpy as np
import torch.nn.functional as F
from FocalLoss import WeightedFocalLoss
import time


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
    combined_attribute_indices = []
    combined_traffic_indices = []
    combined_spatial_indices = []

    # Calculate the class of each semantics and use the unique ID of them to create embeddings.
    occu_max = max(id_to_attributes.values(), key=lambda x: x['occupation'])['occupation'] + 1
    gender_max = max(id_to_attributes.values(), key=lambda x: x['gender'])['gender'] + 1
    walk_max = max(id_to_attributes.values(), key=lambda x: x['class_walk_count'])['class_walk_count'] + 1
    car_max = max(id_to_attributes.values(), key=lambda x: x['class_car_count'])['class_car_count'] + 1
    bus_max = max(id_to_attributes.values(), key=lambda x: x['class_bus_count'])['class_bus_count'] + 1
    rail_max = max(id_to_attributes.values(), key=lambda x: x['class_rail_count'])['class_rail_count'] + 1
    res_area_max = max(id_to_attributes.values(), key=lambda x: x['class_landarea'])['class_landarea'] + 1
    pop_den_max = max(id_to_attributes.values(), key=lambda x: x['class_pop_den'])['class_pop_den'] + 1
    rni_max = max(id_to_attributes.values(), key=lambda x: x['class_RNI'])['class_RNI'] + 1
    pop_dnratio_max = max(id_to_attributes.values(), key=lambda x: x['class_pop_day_night'])['class_pop_day_night'] + 1

    for text in original_texts:
        person_id, *tokens = text.split()
        person_id = int(person_id)
        attributes = id_to_attributes.get(person_id, {'occupation': 0, 'gender': 0, 'age': 0})
        occupation_index = attributes['occupation']
        gender_index = attributes['gender']
        age_index = attributes['age']
        walk_index = attributes['class_walk_count']
        car_index = attributes['class_car_count']
        bus_index = attributes['class_bus_count']
        rail_index = attributes['class_rail_count']
        logistics_index = attributes['class_logistic_count']
        res_area_index = attributes['class_landarea']
        pop_den_index = attributes['class_pop_den']
        rni_index = attributes['class_RNI']
        pop_dnratio_index = attributes['class_pop_day_night']
        avg_age_index = attributes['class_avg_age']

        combined_attribute_id = (age_index * gender_max * occu_max) + (gender_index * occu_max) + occupation_index
        combined_traffic_id = (logistics_index * (walk_max * car_max * bus_max * rail_max)) + \
                              (rail_index * (walk_max * car_max * bus_max)) + \
                              (bus_index * (walk_max * car_max)) + \
                              (car_index * walk_max) + \
                              walk_index
        combined_spatial_id = (avg_age_index * (pop_dnratio_max * rni_max * pop_den_max * res_area_max)) + \
                              (pop_dnratio_index * (rni_max * pop_den_max * res_area_max)) + \
                              (rni_index * (pop_den_max * res_area_max)) + \
                              (pop_den_index * res_area_max) + \
                              res_area_index

        combined_attribute_indices.extend([combined_attribute_id] * 12)  # 12 is the length of the shifted sequences
        combined_traffic_indices.extend([combined_traffic_id] * 12)
        combined_spatial_indices.extend([combined_spatial_id] * 12)

    combined_attribute_indices = torch.tensor(combined_attribute_indices, dtype=torch.long)
    combined_traffic_indices = torch.tensor(combined_traffic_indices, dtype=torch.long)
    combined_spatial_indices = torch.tensor(combined_spatial_indices, dtype=torch.long)

    assert len(shifted_input_ids) == len(combined_attribute_indices) == len(combined_traffic_indices) == len(combined_spatial_indices)  # Ensuring the tensor sizes match

    return TensorDataset(shifted_input_ids, shifted_labels, combined_attribute_indices, combined_traffic_indices,
                         combined_spatial_indices)


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
            inputs, labels, combined_attribute_indices, combined_traffic_indices, combined_spatial_indices = batch
            inputs, labels, combined_attribute_indices, combined_traffic_indices, combined_spatial_indices = inputs.to(
                device), labels.to(device), combined_attribute_indices.to(device), combined_traffic_indices.to(
                device), combined_spatial_indices.to(device)
            attention_mask = (inputs != PAD_TOKEN_ID).bool()
            logits = model(inputs, combined_attribute_indices=combined_attribute_indices,
                           combined_traffic_indices=combined_traffic_indices,
                           combined_spatial_indices=combined_spatial_indices, attention_mask=attention_mask, use_embeds=False)

            # Penalize the logits for [EOS], [UNK], and [PAD] tokens
            logits[:, :, tokenizer.convert_tokens_to_ids('[EOS]')] = -1e9
            logits[:, :, tokenizer.convert_tokens_to_ids('[UNK]')] = -1e9
            logits[:, :, tokenizer.convert_tokens_to_ids('[PAD]')] = -1e9

            # Find the index of the last non-padding token in each sequence
            last_non_padding_idx = torch.sum(attention_mask, dim=1) - 1
            last_logits = logits[last_non_padding_idx, torch.arange(logits.size(1)), :]
            loss = loss_fn(last_logits, labels)

            # loss = loss_fn(logits[-1, :, :], labels)
            total_loss += loss.item()
            # preds = np.argmax(logits[-1, :, :].detach().cpu().numpy(), axis=-1).flatten()
            last_logits = last_logits.detach().cpu().numpy()
            preds = np.argmax(last_logits, axis=-1).flatten()
            predictions.extend(preds)

    # Added
    predictions_reshaped = [predictions[i:i + 12] for i in range(0, len(predictions), 12)]

    average_loss = total_loss / len(dataloader)
    return average_loss, predictions_reshaped


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
    class_weights_tensor = torch.ones(len(all_token_ids), dtype=torch.float).to(device)
    token_to_id = tokenizer.get_vocab()  # Get the mapping from tokens to their respective IDs
    house_token_id = token_to_id["House"]
    backhome_token_id = token_to_id['Back_Home']
    office_token_id = token_to_id["Office"]
    class_weights_tensor[house_token_id] = 0.5
    class_weights_tensor[backhome_token_id] = 0.5
    class_weights_tensor[office_token_id] = 0.5

    batch_size = 1024
    learning_rate = 0.0000001
    num_epochs = 50
    train_dataset = create_dataset_with_indices(train_input_ids, train_labels, train_texts)
    val_dataset = create_dataset_with_indices(val_input_ids, val_labels, val_texts)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CustomGPT1Model()
    # model = CustomGRUModel()
    model.initialize_weights()
    model.to(device)
    tokenizer_vocab_size = len(tokenizer.get_vocab())
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = WeightedFocalLoss(alpha=0.25, gamma=2, class_weights=class_weights_tensor)
    num_train_steps = len(train_dataloader) * num_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.01 * num_train_steps,
                                                num_training_steps=num_train_steps)

    torch.cuda.empty_cache()
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_preds, total_labels = [], []
        total_loss = 0.00
        total_correct_preds = 0

        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Splitting the batch to its individual components
            inputs, labels, combined_attribute_indices, combined_traffic_indices, combined_spatial_indices = batch
            inputs, labels, combined_attribute_indices, combined_traffic_indices, combined_spatial_indices = inputs.to(
                device), labels.to(device), combined_attribute_indices.to(device), combined_traffic_indices.to(
                device), combined_spatial_indices.to(device)
            attention_mask = (inputs != PAD_TOKEN_ID).bool()
            logits = model(inputs, combined_attribute_indices=combined_attribute_indices,
                           combined_traffic_indices=combined_traffic_indices,
                           combined_spatial_indices=combined_spatial_indices, attention_mask=attention_mask, use_embeds=False)

            # Penalize the logits for [EOS], [UNK], and [PAD] tokens
            logits[:, :, tokenizer.convert_tokens_to_ids('[EOS]')] = -1e9
            logits[:, :, tokenizer.convert_tokens_to_ids('[UNK]')] = -1e9
            logits[:, :, tokenizer.convert_tokens_to_ids('[PAD]')] = -1e9

            # Find the index of the last non-padding token in each sequence
            last_non_padding_idx = torch.sum(attention_mask, dim=1) - 1
            last_logits = logits[last_non_padding_idx, torch.arange(logits.size(1)), :]
            loss = loss_fn(last_logits, labels)

            # loss = loss_fn(logits[-1, :, :], labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            # probs = F.softmax(logits[-1, :, :], dim=-1).detach().cpu().numpy()
            probs = F.softmax(last_logits, dim=-1).detach().cpu().numpy()
            preds = [np.random.choice(len(p), p=p) for p in probs]
            preds = np.array(preds).flatten()
            labels = labels.detach().cpu().numpy().flatten()
            total_preds.extend(preds)
            total_labels.extend(labels)

            if step % 250 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} | Step {step}/{len(train_dataloader)} | Loss: {loss.item():.8f}")

        average_train_loss = total_loss / len(train_dataloader)
        accuracy = accuracy_score(total_labels, total_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(total_labels, total_preds, average='macro', zero_division=0)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Average Loss: {average_train_loss:.8f} | "
              f"Accuracy: {accuracy:.8f} | "
              f"Precision: {precision:.8f} | "
              f"Recall: {recall:.8f} | "
              f"F1: {f1:.8f} | "
              f"Perplexity: {calculate_perplexity(average_train_loss):.8f}")

        # Perform evaluation lopp and write the predictions to a file
        val_loss, val_predictions_grouped = evaluate_model(model, val_dataloader, loss_fn)
        validation_predictions_file = f'/home/ubuntu/Documents/validation_predictions_epoch_{epoch + 1}.txt'
        with open(validation_predictions_file, 'w') as file:
            for sentence_id, sentence_group in enumerate(val_predictions_grouped):
                predicted_sentence = tokenizer.decode(sentence_group, skip_special_tokens=True)
                original_sentence = val_texts[
                    sentence_id]
                file.write(f'Original sentence: {original_sentence}\n')
                file.write(f'Predicted sentence: {predicted_sentence}\n\n')

        # Decode each group of predictions into sentences and then split them into tokens for statistical analysis
        val_predicted_tokens = []
        for group in val_predictions_grouped:
            for token_id in group:
                if token_id == tokenizer.pad_token_id:
                    decoded_token = "[PAD]"
                elif token_id == tokenizer.eos_token_id:
                    decoded_token = "[EOS]"
                elif token_id == tokenizer.unk_token_id:
                    decoded_token = "[UNK]"
                else:
                    decoded_token = tokenizer.decode([token_id])
                val_predicted_tokens.append(decoded_token)
        groundtruth_tokens = [tokenizer.decode([label_id], skip_special_tokens=True) for label_id in
                              val_labels.tolist()]

        # Convert tokens to their respective IDs for metric calculation
        val_predicted_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in val_predicted_tokens]
        groundtruth_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in groundtruth_tokens]
        val_predicted_token_counts = Counter(val_predicted_tokens)
        groundtruth_token_counts = Counter(groundtruth_tokens)

        # Calculate accuracy, precision, recall, and F1 score in evaluation loop
        accuracy = accuracy_score(groundtruth_token_ids, val_predicted_token_ids)
        precision, recall, f1, _ = precision_recall_fscore_support(
            groundtruth_token_ids, val_predicted_token_ids, average='macro', zero_division=0)

        print(f"Evaluation Loops : "
              f"Accuracy: {accuracy:.8f} | "
              f"Precision: {precision:.8f} | "
              f"Recall: {recall:.8f} | "
              f"F1 Score: {f1:.8f} | ")

        print("Token Occurrences in Ground Truth:")
        for token, count in groundtruth_token_counts.most_common():
            print(f"{token}: {count}")

        print("\nToken Occurrences in Predictions:")
        for token, count in val_predicted_token_counts.most_common():
            print(f"{token}: {count}")

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Validation Loss: {val_loss:.8f} | "
            f"Validation Perplexity: {calculate_perplexity(val_loss):.8f}"
        )

        torch.save(model, f'/home/ubuntu/Documents/model_{epoch + 1}.pth')

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time} seconds")
