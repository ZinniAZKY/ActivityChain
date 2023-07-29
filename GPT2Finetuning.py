from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, LineByLineTextDataset, PreTrainedTokenizerFast
import logging


class CustomTrainer(Trainer):
    # def log(self, logs):
    #     logs = {k: round(v, 2) if isinstance(v, float) else round(v.item(), 2) for k, v in logs.items()}
    #
    #     super().log(logs)
    #     if self.state.global_step % self.args.logging_steps == 0 and self.state.global_step > 0:
    #         inputs = next(iter(self.get_train_dataloader()))
    #         inputs = {name: tensor.to(self.args.device) for name, tensor in inputs.items()}
    #         outputs = self.model(**inputs)
    #         predicted_tokens = tokenizer.convert_ids_to_tokens(outputs.logits.argmax(-1)[0].tolist())
    #         input_text = tokenizer.decode(inputs['input_ids'][0])
    #
    #         print(f"At step {self.state.global_step}, original_input_train: {input_text}")
    #         print(f"At step {self.state.global_step}, decoded_input_token: {predicted_tokens}")

    def evaluation_loop(self, dataloader, description, prediction_loss_only=False, ignore_keys=None,
                        metric_key_prefix="eval"):
        for step, inputs in enumerate(dataloader):
            inputs = {name: tensor.to(self.args.device) for name, tensor in inputs.items()}
            outputs = self.model(**inputs)
            predictions = outputs[0]
            predicted_tokens = tokenizer.convert_ids_to_tokens(predictions.argmax(-1).tolist())
            if step <= 5:
                print("validation_predicted_tokens:", predicted_tokens)
                # input_text = tokenizer.decode(inputs['input_ids'][0])
                # print("original_input_validation:", input_text)

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

model = GPT2LMHeadModel.from_pretrained('distilgpt2')
model.resize_token_embeddings(len(tokenizer))

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/home/ubuntu/Documents/TokyoPT/PTChain/train.txt",
    block_size=96,
)

eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/home/ubuntu/Documents/TokyoPT/PTChain/eval.txt",
    block_size=96,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

training_args = TrainingArguments(
    output_dir="/home/ubuntu/Documents/TokyoPT",
    logging_dir='./logs',
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_steps=250,
    logging_steps=250,
    save_total_limit=1,
    learning_rate=0.001,
    gradient_accumulation_steps=8,
    evaluation_strategy="epoch",
    lr_scheduler_type='cosine_with_restarts',
    warmup_steps=500,
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

for key in sorted(eval_results.keys()):
    val_logger.info(f"{key}: {eval_results[key]}")
