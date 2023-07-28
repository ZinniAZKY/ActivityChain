from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, LineByLineTextDataset
import logging


class CustomTrainer(Trainer):
    def evaluation_loop(self, dataloader, description, prediction_loss_only=False, ignore_keys=None,
                        metric_key_prefix="eval"):
        for step, inputs in enumerate(dataloader):
            # print("inputs:", inputs)
            inputs = {name: tensor.to(self.args.device) for name, tensor in inputs.items()}
            outputs = self.model(**inputs)
            predictions = outputs[0]
            # print("predictions:", predictions)
            predicted_tokens = tokenizer.convert_ids_to_tokens(predictions.argmax(-1).tolist())
            print("predicted_tokens:", predicted_tokens)
            # predicted_sequences = [' '.join(tokens) for tokens in predicted_tokens]
            # print("predicted_sequences:", predicted_sequences)
            input_text = tokenizer.decode(inputs['input_ids'][0])
            print("input_text is", input_text)

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

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token
special_tokens = ["Commute", "House", "Office", "Store_Daily", "Go_School", "School", "Back_Home", "Shopping_Daily", "Shopping_Nondaily", "Store_Nondaily", "Go_Eat", "Socializing", "Go_Recreational_Facility", "Pickup&Drop_Off", "Go_Sightseeing", "Tourist_Spot", "Private_Movement", "Private_Space", "Delivering", "Business_Place", "Attend_Meeting", "Go_Occupation", "Go_Agricultural_Work", "Natural_Area", "Go_Other_Business", "Go_Exercise", "Pitch", "Volunteering", "Public_Space", "Welcoming"]
tokenizer.add_tokens(special_tokens)

# encoded_input = tokenizer.encode("House Office Go_Shopping", return_tensors='pt')
# decoded_output = tokenizer.decode(encoded_input[0])
# print("decoded output is", decoded_output)

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
