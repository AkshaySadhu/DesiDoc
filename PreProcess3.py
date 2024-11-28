import pandas as pd
import json
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset

# Load the dataset
train_data = pd.read_csv('/HindiMedLLM/22687585/release_train_patients/release_train_patients')
evidence_dict = json.load(open('/HindiMedLLM/22687585/release_evidences.json'))
conditions = json.load(open('/HindiMedLLM/22687585/release_conditions.json'))

# Data Cleaning
def clean_text(text):
    symptom_texts = []
    antecedents = []
    description = ""
    for evidence_code in text:
        # Separate multi-choice evidence by value
        if "_@_" in evidence_code:
            evidence, value = evidence_code.split('_@_')
            evidence_text = evidence_dict[evidence]['question_en']
            value_text = evidence_dict[evidence]['value_meaning'].get(value)
            value_text = value_text['en'] if value_text is not None else value
            if evidence_dict[evidence]['is_antecedent']:
                antecedents.append(f"{evidence_text}: {value_text}")
            else:
                symptom_texts.append(f"{evidence_text}: {value_text}")
        else:
            if evidence_dict[evidence_code]['is_antecedent']:
                antecedents.append(evidence_dict[evidence_code]['question_en']+' Y ')
            else:
                symptom_texts.append(evidence_dict[evidence_code]['question_en']+' Y ')

    description += "Antecedents:" + "; ".join(antecedents) + ". Symptoms: " + "; ".join(symptom_texts) + ". "
    print(description)
    return description.lower().strip()

train_data['PATHOLOGY'] = train_data['PATHOLOGY'].apply(clean_text)
train_data['EVIDENCES'] = train_data['EVIDENCES'].apply(lambda x: [clean_text(e) for e in eval(x)])

# Text Normalization
def normalize_evidence(evidence):
    if '@' in evidence:
        name, value = evidence.split('@')
        return f"{name.strip()} is {value.strip()}"
    return evidence

train_data['EVIDENCES'] = train_data['EVIDENCES'].apply(lambda x: [normalize_evidence(e) for e in x])

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

# Formatting for Fine-tuning
def format_input(row):
    evidences = ", ".join(row['EVIDENCES'])
    return f"Patient: Age {row['AGE']}, Sex {row['SEX']}. Symptoms and antecedents: {evidences}"

def format_output(row):
    return f"Diagnosis: {row['PATHOLOGY']}"

train_data['input'] = train_data.apply(format_input, axis=1)
train_data['output'] = train_data.apply(format_output, axis=1)

print(train_data)

# Create a custom dataset
class MedicalDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        input_encoding = self.tokenizer.encode_plus(
            item['input'],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        output_encoding = self.tokenizer.encode_plus(
            item['output'],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': output_encoding['input_ids'].flatten()
        }

# Split the data
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# Create datasets
train_dataset = MedicalDataset(train_data, tokenizer)
val_dataset = MedicalDataset(val_data, tokenizer)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Example of accessing a sample
sample = train_dataset[0]
print("Input:", tokenizer.decode(sample['input_ids']))
print("Output:", tokenizer.decode(sample['labels']))