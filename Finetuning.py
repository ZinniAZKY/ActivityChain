from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, LineByLineTextDataset, PreTrainedTokenizerFast
import logging
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# frequency使用每个微调的数据集的frequency
token_frequencies = {
    'House': 2527564,
    'Commute': 111670,
    'Office': 798887,
    'Store_Daily': 1940,
    'Go_School': 43850,
    'School': 411453,
    'Back_Home': 1041,
    'Shopping_Daily': 663,
    'Shopping_Nondaily': 399,
    'Store_Nondaily': 866,
    'Go_Eat': 2260,
    'Socializing': 6708,
    'Go_Recreational_Facility': 0,
    'Pickup_Drop_Off': 1386,
    'Go_Sightseeing': 1113,
    'Tourist_Spot.py': 3467,
    'Private_Movement': 2017,
    'Private_Space': 7994,
    'Delivering': 1181,
    'Business_Place': 36439,
    'Attend_Meeting': 2331,
    'Go_Occupation': 2466,
    'Go_Agricultural_Work': 273,
    'Natural_Area': 2939,
    'Go_Other_Business': 4147,
    'Go_Exercise': 0,
    'Pitch': 0,
    'Volunteering': 374,
    'Public_Space': 1976,
    'Welcoming': 2766,
    '[UNK]': 0,
    '[PAD]': 0,
    '[EOS]': 0,
}

total_samples = sum(token_frequencies.values())
class_weights = {}

for token, freq in token_frequencies.items():
    if freq == 0:
        token_frequencies[token] = 0.001

# 平衡过多的House
max_weight = max(total_samples / (1 * freq) for freq in token_frequencies.values())
for token, freq in token_frequencies.items():
    if token == "House":
        class_weights[token] = max_weight * 100
    else:
        class_weights[token] = total_samples / (100 * freq)

max_weight_value = max(class_weights.values())
for token in class_weights:
    class_weights[token] /= max_weight_value

# Convert to tensor format for PyTorch
weights_tensor = [class_weights[token] for token in sorted(token_frequencies.keys())]
weights_tensor = torch.tensor(weights_tensor).to('cuda')


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_logs = []

    def log(self, logs):
        logs = {k: f"{v:.8f}" if isinstance(v, float) else v for k, v in logs.items()}
        self.train_logs.append(logs)
        super().log(logs)

    def get_train_logs(self):
        return self.train_logs

    def evaluation_loop(self, dataloader, description, prediction_loss_only=False, ignore_keys=None,
                        metric_key_prefix="eval"):
        true_labels = []
        predicted_labels = []

        for step, inputs in enumerate(dataloader):
            inputs = {name: tensor.to(self.args.device) for name, tensor in inputs.items()}
            outputs = self.model(**inputs)
            predictions = outputs[1]
            predicted_indices = predictions.argmax(-1)
            true_labels.extend(inputs["labels"].cpu().numpy().flatten())
            predicted_labels.extend(predicted_indices.cpu().numpy().flatten())
            predicted_tokens_batch = [tokenizer.convert_ids_to_tokens(indices) for indices in predicted_indices]
            softmax_predictions = F.softmax(predictions, dim=-1)
            predicted_probs_batch = softmax_predictions.gather(2, predicted_indices.unsqueeze(-1)).squeeze(-1)

            if step == 0:
                for b in range(len(predicted_indices)):
                    print(f"Batch item {b}:")
                    for idx, token in enumerate(predicted_tokens_batch[b][:10]):
                        print(f"Token: {token}, Probability: {predicted_probs_batch[b, idx]:.5f}")

        accuracy = (np.array(true_labels) == np.array(predicted_labels)).mean()
        metrics = {
            "accuracy": accuracy,
        }
        original_results = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys,
                                                   metric_key_prefix)
        original_metrics = original_results.metrics if hasattr(original_results, "metrics") else {}
        combined_metrics = {**original_metrics, **metrics}

        return original_results._replace(metrics=combined_metrics)
        # return super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels, weights=weights_tensor)
        # Save past state if it exists (default None if not present)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        # return loss, and optionally decoded logits for interpretability
        return (outputs[0], outputs) if return_outputs else outputs[0]


logging.basicConfig(
    filename='/home/ubuntu/Documents/TokyoPT/PTChain/training_logs.txt',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

val_logger = logging.getLogger('validation')
val_logger.setLevel(logging.INFO)
val_handler = logging.FileHandler('/home/ubuntu/Documents/TokyoPT/PTChain/validation_logs.txt')
val_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
val_handler.setFormatter(val_formatter)
val_logger.addHandler(val_handler)

# 注意tokenizer路径
tokenizer = PreTrainedTokenizerFast(tokenizer_file="/home/ubuntu/Documents/Tokenizer/trip_chain_tokenizer.json")
tokenizer.pad_token = "[PAD]"

# model = GPT2LMHeadModel.from_pretrained('distilgpt2')
model = torch.load('/home/ubuntu/Documents/model_6.pth')
# model.resize_token_embeddings(len(tokenizer))

# 微调数据集需要使用不同PTtrip chain
train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/home/ubuntu/Documents/TokyoPT/PTChain/Chukyo2011PTChainRefinedhalfUpsampleTrain.txt",
    block_size=48,
)

eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/home/ubuntu/Documents/TokyoPT/PTChain/Chukyo2011PTChainRefinedhalfEval.txt",
    block_size=48,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

training_args = TrainingArguments(
    output_dir="/home/ubuntu/Documents/TokyoPT",
    logging_dir='./logs',
    overwrite_output_dir=True,
    num_train_epochs=15,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=1000,
    logging_steps=1000,
    save_total_limit=1,
    learning_rate=0.01,
    # gradient_accumulation_steps=4,
    evaluation_strategy="epoch",
    # lr_scheduler_type='cosine_with_restarts',
    lr_scheduler_type='cosine',
    warmup_steps=5000,
    weight_decay=0.05
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
eval_results = trainer.evaluate()

train_logs = trainer.get_train_logs()
with open('/home/ubuntu/Documents/TokyoPT/PTChain/training_logs.txt', 'a') as train_log_file:
    for step, logs in enumerate(train_logs):
        train_log_file.write(f"Step {step} - Training Loss: {logs.get('loss', 'N/A')}\n")

for key in sorted(eval_results.keys()):
    val_logger.info(f"{key}: {eval_results[key]}")
