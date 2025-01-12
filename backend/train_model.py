from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

# Set the padding token to the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Load and preprocess the dataset
dataset = load_dataset("text", data_files={"train": "data/dataset.txt"})
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Split dataset into train and evaluation sets
train_size = int(0.9 * len(tokenized_datasets["train"]))
train_dataset = tokenized_datasets["train"].select(range(train_size))
eval_dataset = tokenized_datasets["train"].select(range(train_size, len(tokenized_datasets["train"])))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./models/fine_tuned_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    evaluation_strategy="steps",  # Enable evaluation
    eval_steps=500,              # Frequency of evaluation
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Include evaluation dataset
)

# Start training
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("models/fine_tuned_model")
tokenizer.save_pretrained("models/fine_tuned_model")
