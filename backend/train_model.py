from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

# Set the padding token to the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
def tokenize_function(examples):
    # Tokenize the input text
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True)
    # Copy input_ids to labels
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# Load and preprocess the dataset
dataset = load_dataset("text", data_files={"train": "data/dataset.txt"})
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./models/fine_tuned_model",  # Save directory
    num_train_epochs=3,                      # Number of training epochs
    per_device_train_batch_size=4,           # Batch size per device
    save_steps=500,                          # Save checkpoint every 500 steps
    logging_dir="./logs",                    # Logging directory
    logging_steps=100                        # Log every 100 steps
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"]
)

# Start training
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("models/fine_tuned_model")
tokenizer.save_pretrained("models/fine_tuned_model")
