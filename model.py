import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
from tqdm import tqdm
import re

# --- Configuration ---
DATA_FILE = 'SID.csv'  # Uses your uploaded file
MODEL_FILE = 'sqli_model_hybrid.pth' 
TOKENIZER_FILE = 'tokenizer_hybrid.pkl'

# --- HYPERPARAMETERS ---
VOCAB_SIZE = 25000
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
LSTM_LAYERS = 1
MAX_LENGTH = 200
EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Tokenizer ---
class SimpleTokenizer:
    def __init__(self, max_words, max_len):
        self.max_words = max_words
        self.max_len = max_len
        self.word_index = {}
        self.pad_token = 0
        self.unk_token = 1
        
    def fit(self, texts):
        counts = Counter()
        for text in tqdm(texts, desc="Building Vocabulary"):
            counts.update(self._clean(str(text)).split()) # Ensure text is string
        vocab = ["<PAD>", "<UNK>"] + [w for w, c in counts.most_common(self.max_words - 2)]
        self.word_index = {word: i for i, word in enumerate(vocab)}
        
    def _clean(self, text):
        text = str(text).lower()
        # Fix for "mysql.user" and "UN/**/ION"
        # We Keep dots and stars attached to see patterns like ".*" or "/**/"
        text = re.sub(r"([!?'=()%;:\-\[\]\"+#|&^@])", r" \1 ", text) 
        # However, we add spaces around commas to separate arguments
        text = text.replace(",", " , ")
        return text

    def encode_single(self, text):
        words = self._clean(text).split()
        seq = [self.word_index.get(w, self.unk_token) for w in words]
        if len(seq) < self.max_len:
            seq = seq + [self.pad_token] * (self.max_len - len(seq))
        else:
            seq = seq[:self.max_len]
        return np.array(seq, dtype=np.int64)

    def save(self, filepath):
        with open(filepath, 'wb') as f: pickle.dump(self.word_index, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f: self.word_index = pickle.load(f)

# --- 2. HYBRID ARCHITECTURE (CNN + LSTM) ---
class SQLiHybrid(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers):
        super(SQLiHybrid, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # CNN Layer
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2) 
        
        # LSTM Layer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=n_layers, bidirectional=True, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        x = self.dropout(hidden)
        x = self.fc(x)
        return x

# --- 3. Dataset Class ---
class SQLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text_vector = self.tokenizer.encode_single(self.texts[idx])
        label = self.labels[idx]
        return torch.tensor(text_vector), torch.tensor(label, dtype=torch.float32)

# --- 4. Helper Functions ---
def calculate_accuracy(outputs, labels):
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float()
    return (preds == labels).sum().item()

def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item() * inputs.size(0)
            total_correct += calculate_accuracy(outputs, labels)
            total_samples += inputs.size(0)
    return total_loss / total_samples, total_correct / total_samples

# --- 5. Training Logic ---
def train_model():
    print("\n" + "="*40)
    print("   STARTING TRAINING (SID.CSV)   ")
    print("="*40)
    
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"❌ {DATA_FILE} not found!")

    print("Loading & Cleaning Data...")
    # --- AUTO-CLEANING SID.CSV ---
    try:
        # Read CSV, ignoring errors
        df = pd.read_csv(DATA_FILE, on_bad_lines='skip')
        
        # 1. Remove empty 'Unnamed' columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # 2. Normalize column names
        df.columns = df.columns.str.strip().str.capitalize() # 'Query', 'Label'
        
        # 3. Ensure Label is numeric (converts errors to NaN, then 0)
        df['Label'] = pd.to_numeric(df['Label'], errors='coerce').fillna(0).astype(float)
        
        # 4. Ensure Query is string
        df['Query'] = df['Query'].astype(str)
        
        print(f"✅ Data Cleaned. Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Error cleaning CSV: {e}")
        return None, None

    # 70/15/15 Split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        df['Query'].values, df['Label'].values, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=42)
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    tokenizer = SimpleTokenizer(VOCAB_SIZE, MAX_LENGTH)
    tokenizer.fit(X_train)
    tokenizer.save(TOKENIZER_FILE)

    train_loader = DataLoader(SQLDataset(X_train, y_train, tokenizer), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(SQLDataset(X_val, y_val, tokenizer), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(SQLDataset(X_test, y_test, tokenizer), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SQLiHybrid(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, LSTM_LAYERS).to(device)
    
    # Dynamic Class Weighting (Fixes False Negatives)
    n_benign = (y_train == 0).sum()
    n_malicious = (y_train == 1).sum()
    # If attacks are rare, boost their weight. If balanced, boost slightly (1.2) to favor security.
    weight_val = (n_benign / n_malicious) * 1.5 if n_malicious > 0 else 1.0
    pos_weight = torch.tensor([weight_val]).to(device)
    print(f"Attack Weight Multiplier: {weight_val:.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"Model initialized on {device}. Starting epochs...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss, running_correct, total = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_correct += calculate_accuracy(outputs, labels)
            total += inputs.size(0)
            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{running_correct/total:.4f}")

        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        print(f"  └── Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    print("\n--- Final Testing ---")
    test_loss, test_acc = evaluate_model(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"✅ Model saved to {MODEL_FILE}")
    return model, tokenizer

# --- 6. Interactive Mode ---
def predict(model, tokenizer, query):
    model.eval()
    with torch.no_grad():
        vec = tokenizer.encode_single(query)
        tensor_vec = torch.tensor(np.array([vec]), dtype=torch.long).to(device)
        return torch.sigmoid(model(tensor_vec)).item()

def batch_test(model, tokenizer):
    print("\n--- Paste queries (type END to finish) ---")
    queries = []
    while True:
        line = input()
        if line.strip() == 'END': break
        if line.strip(): queries.append(line.strip())

    print(f"\n{'Query':<50} | {'Verdict':<10} | {'Conf':<6}")
    print("-" * 75)
    for q in queries:
        prob = predict(model, tokenizer, q)
        verdict = "MALICIOUS" if prob > 0.5 else "SAFE"
        color = "\033[91m" if prob > 0.5 else "\033[92m"
        print(f"{(q[:47] + '..') if len(q)>47 else q:<50} | {color}{verdict:<10}\033[0m | {prob*100:.1f}%")

# --- 7. Main ---
if __name__ == "__main__":
    should_train = True
    if os.path.exists(MODEL_FILE) and os.path.exists(TOKENIZER_FILE):
        if os.path.exists(DATA_FILE) and os.path.getmtime(DATA_FILE) > os.path.getmtime(MODEL_FILE):
            print("🔄 New dataset detected! Retraining Hybrid Model...")
        else:
            print("✅ Loading existing Hybrid Model...")
            should_train = False
    
    if should_train:
        model, tokenizer = train_model()
    else:
        tokenizer = SimpleTokenizer(VOCAB_SIZE, MAX_LENGTH)
        tokenizer.load(TOKENIZER_FILE)
        model = SQLiHybrid(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, LSTM_LAYERS).to(device)
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        model.eval()

    if model:
        while True:
            sel = input("\n(1) Single Test, (2) Batch Test, (q) Quit: ").lower()
            if sel == 'q': break
            elif sel == '2': batch_test(model, tokenizer)
            elif sel == '1':
                q = input("Enter Query: ")
                if q.strip():
                    prob = predict(model, tokenizer, q)
                    verdict = "MALICIOUS" if prob > 0.5 else "SAFE"
                    color = "\033[91m" if prob > 0.5 else "\033[92m"
                    print(f"Verdict: {color}{verdict}\033[0m ({prob*100:.2f}%)")