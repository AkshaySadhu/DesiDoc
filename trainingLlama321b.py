import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset

if __name__ == '__main__':

    print("Starting training")

    # Load the Alpaca model and tokenizer
    model_name = "RohitSuresh15/llama3.2_1b-medical-v1"
    print(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.gradient_checkpointing_enable()  # Enable gradient checkpointing
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Model loaded")

    # Load training data from JSON file
    with open("./training_data_3.json", "r") as file:
        train_data = json.load(file)

    print("Train data loaded")


    # Tokenization function
    def tokenize(sample):
        input_text = f"Instruction: {sample['instruction']}\nInput: {sample['input']}\nOutput: {sample['output']}"
        tokenized = tokenizer(input_text, padding='max_length', truncation=True, max_length=256)  # Reduce max_length
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized


    # Split data into smaller chunks
    chunk_size = 50  # Reduce chunk size
    data_chunks = [train_data[i:i + chunk_size] for i in range(0, len(train_data), chunk_size)]

    print(f"Data split into {len(data_chunks)} chunks of size up to {chunk_size}")


    # Convert to a PyTorch Dataset
    class TrainDataset(torch.utils.data.Dataset):
        def __init__(self, tokenized_data):
            self.tokenized_data = tokenized_data

        def __len__(self):
            return len(self.tokenized_data)

        def __getitem__(self, idx):
            item = self.tokenized_data[idx]
            return {key: torch.tensor(val) for key, val in item.items()}


    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=1,  # Reduce batch size
        gradient_accumulation_steps=4,  # Simulate larger batch size
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        save_steps=500,
        save_total_limit=2,
        fp16=True,  # Enable mixed precision
    )

    print("Starting incremental training")

    # Train on each chunk
    for idx, chunk in enumerate(data_chunks):
        print(f"Processing chunk {idx + 1}/{len(data_chunks)}")

        # Tokenize the current chunk
        tokenized_train_data = [tokenize(sample) for sample in chunk]
        train_dataset = TrainDataset(tokenized_train_data)

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        print(f"Training on chunk {idx + 1}/{len(data_chunks)}")
        trainer.train()

        # Save model after training on each chunk
        model.save_pretrained(f"./results/chunk_{idx + 1}")
        tokenizer.save_pretrained(f"./results/chunk_{idx + 1}")

        print(f"Chunk {idx + 1} training completed and model saved.")

    print("All chunks processed. Training complete.")
