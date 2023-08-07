from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, LineByLineTextDataset, PreTrainedTokenizerFast
import logging
import torch
import torch.nn.functional as F
from collections import defaultdict


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
        for step, inputs in enumerate(dataloader):
            inputs = {name: tensor.to(self.args.device) for name, tensor in inputs.items()}
            outputs = self.model(**inputs)
            predictions = outputs[1]
            predicted_indices = predictions.argmax(-1)
            predicted_tokens_batch = [tokenizer.convert_ids_to_tokens(indices) for indices in predicted_indices]

            softmax_predictions = F.softmax(predictions, dim=-1)
            predicted_probs_batch = softmax_predictions.gather(2, predicted_indices.unsqueeze(-1)).squeeze(-1)

            if step == 0:
                for b in range(len(predicted_indices)):
                    print(f"Batch item {b}:")
                    for idx, token in enumerate(predicted_tokens_batch[b][:25]):
                        print(f"Token: {token}, Probability: {predicted_probs_batch[b, idx]:.5f}")

        return super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)


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

tokenizer = PreTrainedTokenizerFast(tokenizer_file="/home/ubuntu/Documents/Tokenizer/trip_chain_tokenizer.json")
tokenizer.pad_token = "[PAD]"

# model = GPT2LMHeadModel.from_pretrained('distilgpt2')
model = torch.load('/home/ubuntu/Documents/model_3.pth')
# model.resize_token_embeddings(len(tokenizer))

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/home/ubuntu/Documents/TokyoPT/PTChain/Chukyo2011PTChainRefinedHalfTrain.txt",
    block_size=48,
)

eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/home/ubuntu/Documents/TokyoPT/PTChain/Chukyo2011PTChainRefinedHalfEval.txt",
    block_size=48,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

training_args = TrainingArguments(
    output_dir="/home/ubuntu/Documents/TokyoPT",
    logging_dir='./logs',
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_steps=500,
    logging_steps=500,
    save_total_limit=1,
    learning_rate=0.00005,
    gradient_accumulation_steps=8,
    evaluation_strategy="epoch",
    # lr_scheduler_type='cosine_with_restarts',
    lr_scheduler_type='cosine',
    warmup_steps=1000,
    weight_decay=0.01
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
