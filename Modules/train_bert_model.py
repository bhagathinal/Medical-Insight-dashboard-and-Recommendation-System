# train_bert_model.py (Final Version with Corrected Consolidation)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
import json
import os

# --- 1. Configuration ---
CONFIG = {
    "model_name": "emilyalsentzer/Bio_ClinicalBERT",
    "max_length": 256,
    "batch_size": 16,
    "epochs": 5,  # Increased to 5 epochs to allow for more learning time
    "learning_rate": 3e-5,
    "save_path": "./clinicalbert_model_v3" # Save to a new folder for the final version
}

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 2. Load and Prepare the Data ---
print("Loading the balanced dataset...")
df = pd.read_csv('/kaggle/input/balanced-medical-data/balanced_medical_dataset.csv')
df.dropna(subset=['transcription', 'medical_specialty'], inplace=True)

# --- CORRECTED & AGGRESSIVE CLASS CONSOLIDATION ---
print("Applying corrected and aggressive class consolidation...")
# Define lists of classes to merge
general_note_classes = [
    'Consult - History and Phy.', 'SOAP / Chart / Progress Notes', 'Discharge Summary',
    'Office Notes', 'Letters', 'Emergency Room Reports', 'Pediatrics - Neonatal'
]
surgical_specialties = [
    'Neurosurgery', 'Cosmetic / Plastic Surgery', 'Orthopedic'
]
specialized_medicine = [
    # Merging classes that previously had very low F1-scores or high overlap
    'Dermatology', 'Hematology - Oncology', 'Nephrology', 'Ophthalmology'
]

# Apply the merges sequentially
df['medical_specialty'] = df['medical_specialty'].replace(general_note_classes, 'General Note')
df['medical_specialty'] = df['medical_specialty'].replace(surgical_specialties, 'Surgical Specialties')
df['medical_specialty'] = df['medical_specialty'].replace(specialized_medicine, 'Specialized Medicine')

# --- VERIFY CONSOLIDATION ---
unique_labels = sorted(df['medical_specialty'].unique())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for i, label in enumerate(unique_labels)}

# This print statement will now show the CORRECT number of classes
print(f"\nNumber of unique classes after consolidation: {len(unique_labels)}\n")
print("Final Classes:", unique_labels) # Print the list of final classes to be sure

os.makedirs(CONFIG['save_path'], exist_ok=True)
with open(f"{CONFIG['save_path']}/label_map.json", 'w') as f:
    json.dump({'label2id': label2id, 'id2label': id2label}, f)

df['label'] = df['medical_specialty'].map(label2id)


# --- 3. Create a PyTorch Dataset (No changes needed) ---
class MedicalNotesDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Split the data
df_train, df_val = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])
tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
train_dataset = MedicalNotesDataset(df_train.transcription.to_numpy(), df_train.label.to_numpy(), tokenizer, CONFIG['max_length'])
val_dataset = MedicalNotesDataset(df_val.transcription.to_numpy(), df_val.label.to_numpy(), tokenizer, CONFIG['max_length'])
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])


# --- 4. Initialize Model, Optimizer, and Scheduler (No changes needed) ---
model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG['model_name'], num_labels=len(unique_labels)
).to(device)
optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
total_steps = len(train_loader) * CONFIG['epochs']
num_warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
)


# --- 5. Training and Validation Loop (No changes needed) ---
print("Starting final training run...")
for epoch in range(CONFIG['epochs']):
    print(f"\n======== Epoch {epoch + 1}/{CONFIG['epochs']} ========")
    model.train()
    total_train_loss, total_train_correct = 0, 0
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        model.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs.loss, outputs.logits
        preds = torch.argmax(logits, dim=1)
        total_train_correct += torch.sum(preds == labels)
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = total_train_correct / len(train_dataset)
    print(f"\n  - Average Training Loss: {avg_train_loss:.4f}")
    print(f"  - Training Accuracy: {train_accuracy:.4f}")

    model.eval()
    total_val_correct = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            total_val_correct += torch.sum(preds == labels)
            
    val_accuracy = total_val_correct / len(val_dataset)
    print(f"  - Validation Accuracy: {val_accuracy:.4f}")


# --- 6. Final Evaluation (No changes needed) ---
print("\n--- Final Model Evaluation ---")
model.eval()
final_predictions, final_true_labels = [], []
with torch.no_grad():
    for batch in tqdm(val_loader, desc="Final Evaluation"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).flatten()
        final_predictions.extend(preds.cpu().numpy())
        final_true_labels.extend(labels.cpu().numpy())

print("\n--- Final Classification Report ---")
target_names = [id2label[i] for i in range(len(id2label))]
print(classification_report(final_true_labels, final_predictions, target_names=target_names, zero_division=0))


# --- 7. Save the Fine-Tuned Model and Tokenizer (No changes needed) ---
print(f"Saving model and tokenizer to {CONFIG['save_path']}...")
model.save_pretrained(CONFIG['save_path'])
tokenizer.save_pretrained(CONFIG['save_path'])

print("\nTraining and saving complete!")