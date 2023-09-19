from CustomGPT1 import CustomGPT1Model
import torch
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from transformers import get_linear_schedule_with_warmup

# train_file = "/home/ubuntu/Documents/TokyoPT/PTChain/combinedAttrTrain.txt"
# val_file = "/home/ubuntu/Documents/TokyoPT/PTChain/combinedAttrEval.txt"

# with open(train_file, "r", encoding="utf-8") as f:
#     train_texts = [line.strip() for line in f.readlines()]

# with open(val_file, "r", encoding="utf-8") as f:
#     val_texts = [line.strip() for line in f.readlines()]

# tokenizer = PreTrainedTokenizerFast(tokenizer_file="/home/ubuntu/Documents/Tokenizer/trip_chain_tokenizer_attr.json")
# tokenizer.pad_token = "[PAD]"
# tokenizer.eos_token = "[EOS]"

# train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=51)
# val_encodings = tokenizer(val_texts, padding=True, truncation=True, max_length=51)

# # Convert tokenized encodings to torch tensors
# train_input_ids = torch.tensor(train_encodings.input_ids)
# train_labels = torch.tensor(train_encodings.input_ids)  # Labels are the same as IDs for language modeling tasks
# val_input_ids = torch.tensor(val_encodings.input_ids)
# val_labels = torch.tensor(val_encodings.input_ids)

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
#     num_samples = 0

#     with torch.no_grad():
#         for batch in dataloader:
#             inputs, labels = batch
#             logits = model(inputs)
#             loss = loss_fn(logits.view(-1, model.vocab_size), labels.view(-1))
#             total_loss += loss.item()
#             num_samples += inputs.size(0)

#     average_loss = total_loss / len(dataloader)
#     return average_loss


# model = CustomGPT1Model()
# model.initialize_weights()
# tokenizer_vocab_size = len(tokenizer.get_vocab())
# assert model.vocab_size == tokenizer_vocab_size, f"Model's vocab size {model.vocab_size} doesn't match tokenizer's vocab size {tokenizer_vocab_size}"

# loss_fn = torch.nn.CrossEntropyLoss()
# learning_rate = 0.0001
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
# num_epochs = 10
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
#         logits = model(inputs)
#         loss = loss_fn(logits.view(-1, model.vocab_size), labels.view(-1))
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         scheduler.step()

#         total_loss += loss.item()

#         preds = np.argmax(logits.detach().cpu().numpy(), axis=-1).flatten()
#         labels = labels.detach().cpu().numpy().flatten()

#         correct_preds = (preds == labels).sum()
#         total_correct_preds += correct_preds
#         total_preds.extend(preds)
#         total_labels.extend(labels)

#         if step % 100 == 0:
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

#     val_loss = evaluate_model(model, val_dataloader, loss_fn)
#     print(
#         f"Epoch {epoch + 1}/{num_epochs} | Validation Loss: {val_loss:.8f} | "
#         f"Validation Perplexity: {calculate_perplexity(val_loss):.8f}")

#     torch.save(model, f'/home/ubuntu/Documents/model_{epoch + 1}.pth')


# V2.0 contains methods of 'Slide Windows' and 'Autoregressive'
train_file = "/home/ubuntu/Documents/TokyoPT/PTChain/combinedhalfTrain.txt"
val_file = "/home/ubuntu/Documents/TokyoPT/PTChain/combinedhalfEval.txt"

with open(train_file, "r", encoding="utf-8") as f:
    train_texts = [line.strip() for line in f.readlines()]

with open(val_file, "r", encoding="utf-8") as f:
    val_texts = [line.strip() for line in f.readlines()]

tokenizer = PreTrainedTokenizerFast(tokenizer_file="/home/ubuntu/Documents/Tokenizer/trip_chain_tokenizer.json")
tokenizer.pad_token = "[PAD]"
tokenizer.eos_token = "[EOS]"
PAD_TOKEN_ID = tokenizer.convert_tokens_to_ids("[PAD]")

train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=48)
val_encodings = tokenizer(val_texts, padding=True, truncation=True, max_length=48)


def shift_input_target(text_encodings):
    input_sequences = []
    target_sequences = []

    for seq in text_encodings.input_ids:
        for i in range(36, len(seq)):
            input_sequence = seq[:i]
            target_sequence = seq[i]
            input_sequences.append(torch.tensor(input_sequence))
            target_sequences.append(target_sequence)

    input_sequences = torch.nn.utils.rnn.pad_sequence(input_sequences, batch_first=True, padding_value=PAD_TOKEN_ID)

    return input_sequences, torch.tensor(target_sequences)


train_input_ids, train_labels = shift_input_target(train_encodings)
val_input_ids, val_labels = shift_input_target(val_encodings)

batch_size = 1024
train_dataset = TensorDataset(train_input_ids, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(val_input_ids, val_labels)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


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
    # num_samples = 0
    predictions = []

    # with torch.no_grad():
    #     for batch in dataloader:
    #         inputs, labels = batch
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         attention_mask = (inputs != PAD_TOKEN_ID).int().transpose(0, 1)
    #         logits = model(inputs, attention_mask=attention_mask)
    #         loss = loss_fn(logits[:, -1, :], labels)
    #         total_loss += loss.item()
    #         num_samples += inputs.size(0)
    #
    # average_loss = total_loss / len(dataloader)
    # return average_loss

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            attention_mask = (inputs != PAD_TOKEN_ID).int().transpose(0, 1)
            logits = model(inputs, attention_mask=attention_mask)
            loss = loss_fn(logits[:, -1, :], labels)
            total_loss += loss.item()
            preds = np.argmax(logits[:, -1, :].detach().cpu().numpy(), axis=-1).flatten()
            predictions.extend(preds)

    average_loss = total_loss / len(dataloader)
    return average_loss, predictions


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomGPT1Model()
model.to(device)
model.initialize_weights()
tokenizer_vocab_size = len(tokenizer.get_vocab())

loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 0.000005
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
num_epochs = 15
num_train_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0.01 * num_train_steps,
    num_training_steps=num_train_steps
)

for epoch in range(num_epochs):
    model.train()
    total_preds, total_labels = [], []
    total_loss = 0.00
    total_correct_preds = 0

    for step, batch in enumerate(train_dataloader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        attention_mask = (inputs != PAD_TOKEN_ID).int().transpose(0, 1)
        logits = model(inputs, attention_mask=attention_mask)
        loss = loss_fn(logits[:, -1, :], labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        total_loss += loss.item()

        # preds = np.argmax(logits.detach().cpu().numpy(), axis=-1).flatten()
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
