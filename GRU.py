import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Assuming your class is in a file named GRU_model.py
from GRU_model import BiGRU

# --- 1. Parameters ---
HIDDEN_SIZE = 8
EMB_DIM = 200
OUTPUT_SIZE = 2
LAYER_NUM = 1
DROPOUT = 0.5
LEARN_RATE = 0.005
EPOCHS = 15
SPATIAL_DROPOUT = True
BATCH_SIZE = 1024

# --- Experiment Folder ---
EXP_NAME = f"BiGRU_layers{LAYER_NUM}_emb{EMB_DIM}"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join("experiments", f"{EXP_NAME}_{TIMESTAMP}")
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Saving everything to: {SAVE_DIR}")

# --- 2. Data Loading & Vocabulary Building ---
train_df = pd.read_csv('dataset/datasets_feat_clean/train_feat_clean.csv', usecols=['clean_review', 'label'])
val_df = pd.read_csv('dataset/datasets_feat_clean/val_feat_clean.csv', usecols=['clean_review', 'label'])
test_df = pd.read_csv('dataset/datasets_feat_clean/test_feat_clean.csv',usecols=['clean_review', 'label'])
print(f'Training samples: {len(train_df)}, Validation samples: {len(val_df)}, Testing samples: {len(test_df)}')

# Simple tokenizer and vocab builder
def tokenize(text):
    return str(text).split()

# Build Vocab from training data
all_tokens = [token for text in train_df['clean_review'] for token in tokenize(text)]
vocab = {word: i + 2 for i, word in enumerate(set(all_tokens))}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1
VOCAB_SIZE = len(vocab)

def text_to_sequence(text):
    return [vocab.get(word, 1) for word in tokenize(text)]

# --- 3. Dataset and Collate Function ---
class ReviewDataset(Dataset):
    def __init__(self, df):
        # Filter out rows with empty clean_reviews if any
        df = df.dropna(subset=['clean_review'])
        self.sequences = [text_to_sequence(t) for t in df['clean_review']]
        self.labels = df['label'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.LongTensor(self.sequences[idx]), torch.tensor(self.labels[idx])

def collate_fn(batch):
    """
    Handles padding and length extraction for BiGRU.
    Pytorch's pack_padded_sequence requires sequences sorted by length.
    """
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, labels = zip(*batch)
    lengths = torch.LongTensor([len(x) for x in sequences])
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.LongTensor(labels)
    
    return {
        'input_seq': sequences_padded,
        'target': labels,
        'x_lengths': lengths
    }

train_loader = DataLoader(ReviewDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(ReviewDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(ReviewDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print("Data loaders created.")

# --- 4. Model Initialization ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BiGRU(
    hidden_size=HIDDEN_SIZE, 
    vocab_size=VOCAB_SIZE, 
    embedding_dim=EMB_DIM, 
    output_size=OUTPUT_SIZE, 
    n_layers=LAYER_NUM, 
    dropout=DROPOUT,
    spatial_dropout=SPATIAL_DROPOUT
)

model.to(device)
model.add_device(device)
model.add_loss_fn(nn.NLLLoss())
model.add_optimizer(optim.Adam(model.parameters(), lr=LEARN_RATE))

# --- 5. Training Loop ---
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

print('Starting training...')

for epoch in range(EPOCHS):
    print(f'\n--- Epoch {epoch+1}/{EPOCHS} ---')
    
    # Run training epoch
    _, avg_train_loss, train_acc = model.train_model(train_loader)
    
    # Run evaluation epoch
    _, avg_val_loss, val_acc, conf_matrix = model.evaluate_model(val_loader, conf_mtx=True)
    
    # Save metrics
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    print(f'Summary -> Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')


model.eval()

all_preds = []
all_labels = []
test_loss = 0.0
criterion = nn.NLLLoss()

with torch.no_grad():
    for batch in test_loader:
        inputs = batch['input_seq'].to(device)
        lengths = batch['x_lengths'].to(device)
        labels = batch['target'].to(device)

        outputs = model(inputs, lengths)   # log-probs
        loss = criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)

        test_loss += loss.item() * labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

test_loss /= len(all_labels)
test_acc = (all_preds == all_labels).mean()

# Confusion matrix
test_conf_mtx = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)
for t, p in zip(all_labels, all_preds):
    test_conf_mtx[t, p] += 1

print(f"Test Accuracy: {test_acc:.4f}")

test_results_df = pd.DataFrame({
    "true_label": all_labels,
    "predicted_label": all_preds
})

CSV_PATH = os.path.join(SAVE_DIR, "test_predictions.csv")
test_results_df.to_csv(CSV_PATH, index=False)

# --- Save Model and Statistics ---
MODEL_PATH = os.path.join(SAVE_DIR, "bigru_model.pt")
torch.save(model.state_dict(), MODEL_PATH)

METRICS_PATH = os.path.join(SAVE_DIR, "results.txt")

with open(METRICS_PATH, "w") as f:
    f.write("===== MODEL CONFIGURATION =====\n")
    f.write(f"HIDDEN_SIZE: {HIDDEN_SIZE}\n")
    f.write(f"EMB_DIM: {EMB_DIM}\n")
    f.write(f"LAYER_NUM: {LAYER_NUM}\n")
    f.write(f"DROPOUT: {DROPOUT}\n")
    f.write(f"SPATIAL_DROPOUT: {SPATIAL_DROPOUT}\n")
    f.write(f"LEARNING_RATE: {LEARN_RATE}\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
    f.write(f"EPOCHS: {EPOCHS}\n")

    f.write("\n===== TRAINING HISTORY =====\n")
    for i in range(EPOCHS):
        f.write(
            f"Epoch {i+1}: "
            f"Train Loss={history['train_loss'][i]:.4f}, "
            f"Val Loss={history['val_loss'][i]:.4f}, "
            f"Train Acc={history['train_acc'][i]:.4f}, "
            f"Val Acc={history['val_acc'][i]:.4f}\n"
        )

    f.write("\n" + "=" * 30 + "\n")
    f.write("TEST RESULTS\n")
    f.write("=" * 30 + "\n")

    f.write(f"Test Loss     : {test_loss:.6f}\n")
    f.write(f"Test Accuracy : {test_acc:.6f}\n\n")

    TN, FP = test_conf_mtx[0]
    FN, TP = test_conf_mtx[1]

    f.write("Confusion Matrix (Binary Classification):\n")
    f.write(f"  TP: {TP}\n")
    f.write(f"  TN: {TN}\n")
    f.write(f"  FP: {FP}\n")
    f.write(f"  FN: {FN}\n")



# --- 6. Plotting Results ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.title('Loss History')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Val')
plt.title('Accuracy History')
plt.legend()

PLOT_PATH = os.path.join(SAVE_DIR, "training_curves.png")
plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
plt.close()
