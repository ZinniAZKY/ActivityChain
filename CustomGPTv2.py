from transformers import GPT2LMHeadModel, GPT2Config
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.utils.data import Dataset
import nltk
from nltk.translate.bleu_score import sentence_bleu
import difflib


# Define a new configuration
config = GPT2Config(
  # vocab_size=50257, # keep the original vocab size
  # n_positions=1024, # keep the original position embeddings
  # n_ctx=1024,       # context size
  n_embd=256,       # new hidden size
  n_layer=4,        # new number of layers
  n_head=4          # new number of attention heads
)


nltk.download('punkt')


def calculate_bleu(reference_texts, candidate_text):
    """
    Calculate the BLEU score for a candidate text given reference texts.
    :param reference_texts: List of reference texts
    :param candidate_text: The candidate text
    :return: BLEU score
    """
    reference_tokens = [nltk.word_tokenize(ref.lower()) for ref in reference_texts]
    candidate_tokens = nltk.word_tokenize(candidate_text.lower())
    return sentence_bleu(reference_tokens, candidate_tokens)


def calculate_lcss(seq1, seq2):
    """
    Calculate the Longest Common Subsequence Score.
    :param seq1: First sequence
    :param seq2: Second sequence
    :return: LCSS score
    """
    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    match = matcher.find_longest_match(0, len(seq1), 0, len(seq2))
    lcss_length = match.size
    max_length = max(len(seq1), len(seq2))
    return lcss_length / max_length if max_length > 0 else 0


def decode_tokens(token_ids, tokenizer):
    return tokenizer.decode(token_ids, skip_special_tokens=True)


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=51):
        self.tokenizer = tokenizer
        self.examples = []

        with open(file_path, encoding='utf-8') as f:
            for line in f:
                tokenized = tokenizer.encode(line, add_special_tokens=True, truncation=True, max_length=block_size)
                self.examples.append(tokenized)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        input_tokens = self.examples[item][:51]
        target_tokens = self.examples[item][:51]
        # target_tokens += [self.tokenizer.pad_token_id] * (50 - len(target_tokens))  # Pad if necessary
        return torch.tensor(input_tokens), torch.tensor(target_tokens)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = Tokenizer.from_file("/home/ubuntu/Documents/Tokenizer/trip_chain_tokenizer_noadd.json")
# model = GPT2LMHeadModel.from_pretrained('gpt2')
model = GPT2LMHeadModel(config)
model.to(device)

# Wrap with PreTrainedTokenizerFast
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
fast_tokenizer.add_special_tokens({'pad_token': '[PAD]', 'eos_token': '[EOS]', 'unk_token': '[UNK]'})
model.resize_token_embeddings(len(fast_tokenizer))

train_file = "/home/ubuntu/Documents/TokyoPT/PTChain/SumPTAttrChainHalfNoAddress/SumPTAttrChainHalfNoAddressTrain.txt"
val_file = "/home/ubuntu/Documents/TokyoPT/PTChain/SumPTAttrChainHalfNoAddress/SumPTAttrChainHalfNoAddressEval.txt"
train_dataset = TextDataset(train_file, fast_tokenizer)
val_dataset = TextDataset(val_file, fast_tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False)

num_epochs = 50
optimizer = AdamW(model.parameters(), lr=1e-8)
num_train_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.01 * num_train_steps, num_training_steps=num_train_steps)

for epoch in range(num_epochs):
    # Training loop
    model.train()
    total_train_loss = 0
    for step, (input_batch, label_batch) in enumerate(train_dataloader):
        optimizer.zero_grad()
        input_batch = torch.nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=fast_tokenizer.pad_token_id)
        label_batch = torch.nn.utils.rnn.pad_sequence(label_batch, batch_first=True, padding_value=fast_tokenizer.pad_token_id)
        inputs = input_batch.to(device)
        labels = label_batch.to(device)
        attention_mask = (inputs != fast_tokenizer.pad_token_id).type(torch.long).to(device)

        outputs = model(inputs, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()

        # Print training loss every 50 steps
        if (step + 1) % 250 == 0:
            print(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item()}")

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Average Training Loss for Epoch {epoch+1}: {avg_train_loss}")

    # Validation loop
    model.eval()
    total_val_loss = 0
    total_bleu_score = 0
    total_lcss_score = 0
    num_batches = 0
    total_examples = 0

    with torch.no_grad():
        for input_batch, label_batch in val_dataloader:
            input_batch = torch.nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=fast_tokenizer.pad_token_id)
            label_batch = torch.nn.utils.rnn.pad_sequence(label_batch, batch_first=True, padding_value=fast_tokenizer.pad_token_id)

            inputs = input_batch.to(device)
            labels = label_batch.to(device)
            attention_mask = (inputs != fast_tokenizer.pad_token_id).type(torch.long).to(device)

            outputs = model(inputs, attention_mask=attention_mask)
            # top k, temperzture
            logits = outputs.logits
            # Manually calculate the loss
            loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, ignore_index=fast_tokenizer.pad_token_id)
            total_val_loss += loss.item()
            predictions = logits.argmax(dim=-1)

            # Calculate BLEU and LCSS for each batch
            for pred, true in zip(predictions, labels):
                pred_text = decode_tokens(pred.cpu().numpy(), fast_tokenizer)
                true_text = decode_tokens(true.cpu().numpy(), fast_tokenizer)
                pred_tokens = pred_text.split()
                true_tokens = true_text.split()
                bleu_score = calculate_bleu([true_text], pred_text)
                lcss_score = calculate_lcss(pred_tokens, true_tokens)

                total_bleu_score += bleu_score
                total_lcss_score += lcss_score
                total_examples += 1

            num_batches += 1

    avg_val_loss = total_val_loss / num_batches
    avg_bleu_score = total_bleu_score / total_examples if total_examples > 0 else 0
    avg_lcss_score = total_lcss_score / total_examples if total_examples > 0 else 0

    print(f"Validation Loss: {avg_val_loss}")
    print(f"Average BLEU Score: {avg_bleu_score}")
    print(f"Average LCSS Score: {avg_lcss_score}")
    torch.save(model, f'/home/ubuntu/Documents/model_{epoch + 1}.pth')

# input_text = "House House House House House House House House House House House House House House House House House " \
#              "House House House House House House House House Go_Eat Socializing Socializing Socializing " \
#              "Private_Space Private_Space Private_Space Back_Home House House House "
#
# # Encode the input using the fast_tokenizer
# input_ids = fast_tokenizer.encode(input_text, return_tensors="pt")
#
# # Generate text
# output = model.generate(input_ids, max_length=48, pad_token_id=fast_tokenizer.pad_token_id)
# generated_ids = output[0]
# generated_text = fast_tokenizer.decode(generated_ids, skip_special_tokens=True)
#
# print(generated_text)
