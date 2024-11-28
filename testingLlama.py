import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, accuracy_score, f1_score
from rouge_score import rouge_scorer
import numpy as np


# Function to compute evaluation metrics
def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    # Replace -100 (ignored index) in labels to the padding token ID
    predictions = np.argmax(predictions, axis=-1)  # Get the token with max probability
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode predictions and labels into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute accuracy, precision, and F1 score at the token level
    flattened_preds = [token for seq in predictions for token in seq]
    flattened_labels = [token for seq in labels for token in seq]

    precision = precision_score(flattened_labels, flattened_preds, average="weighted", zero_division=1)
    accuracy = accuracy_score(flattened_labels, flattened_preds)
    f1 = f1_score(flattened_labels, flattened_preds, average="weighted")

    # Compute ROUGE score at the text level
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(pred, label) for pred, label in zip(decoded_preds, decoded_labels)]

    rouge1 = np.mean([score["rouge1"].fmeasure for score in rouge_scores])
    rouge2 = np.mean([score["rouge2"].fmeasure for score in rouge_scores])
    rougeL = np.mean([score["rougeL"].fmeasure for score in rouge_scores])

    return {
        "accuracy": accuracy,
        "precision": precision,
        "f1": f1,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
    }


# Dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        item = self.tokenized_data[idx]
        return {key: torch.tensor(val) for key, val in item.items()}


# Define the tokenize function (this was missing)
def tokenize(sample):
    input_text = f"Instruction: {sample['instruction']}\nInput: {sample['input']}\nOutput: {sample['output']}"
    tokenized = tokenizer(input_text, padding='max_length', truncation=True, max_length=1024)
    tokenized["labels"] = tokenized["input_ids"].copy()  # Labels are the same as input_ids for causal models
    return tokenized


# Load the saved model and tokenizer
model_name = "./HindiMedLLM/models/final_model_100k"  # Path to saved model
print(f"Loading model from {model_name}...")

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Load validation data
with open("./22687585/release_validate_patients/sample_validate_data_500.json", "r") as val_file:  # Replace with your validation file path
    validation_data = json.load(val_file)

# Tokenize validation data
subset_size = len(validation_data)  # Get 1/4 of the data for validation
tokenized_val_data = [tokenize(sample) for sample in validation_data[:subset_size]]
validation_dataset = CustomDataset(tokenized_val_data)

# Prepare evaluation arguments
eval_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=1,  # Adjust batch size as necessary
    eval_accumulation_steps=10,  # Accumulate eval results to save memory
    fp16=True,  # Enable mixed precision
)

# Initialize Trainer for evaluation only (no training)
trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,  # Add metrics function
)

# Run evaluation
print("Evaluating the model...")
eval_results = trainer.evaluate()

# Print the evaluation results
print("Evaluation Results:")
for key, value in eval_results.items():
    print(f"{key}: {value}")
