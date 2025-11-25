import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import numpy as np
import pandas as pd
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="SQL INJECTION DETECTION",
    page_icon="🛡️",
    layout="centered"
)

# --- CONFIGURATION ---
MODEL_PATH = 'sqli_model_hybrid.pth'
TOKENIZER_PATH = 'tokenizer_hybrid.pkl'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. CLASSES (Must match training exactly) ---
class SimpleTokenizer:
    def __init__(self, max_words=25000, max_len=200):
        self.word_index = {}
        self.pad_token = 0
        self.unk_token = 1
        self.max_len = max_len
        
    def _clean(self, text):
        text = str(text).lower()
        # Exact regex from your training script
        text = re.sub(r"([!?'=()%;:\-\[\]\"+#|&^@*/])", r" \1 ", text) 
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

class SQLiHybrid(nn.Module):
    def __init__(self, vocab_size=25000, embed_dim=128, hidden_dim=128, n_layers=1):
        super(SQLiHybrid, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2) 
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

# --- 2. LOAD RESOURCES (Cached for speed) ---
@st.cache_resource
def load_resources():
    try:
        # Load Tokenizer
        with open(TOKENIZER_PATH, 'rb') as f:
            vocab = pickle.load(f)
        
        tokenizer = SimpleTokenizer()
        tokenizer.word_index = vocab

        # Load Model
        model = SQLiHybrid().to(DEVICE)
        # map_location ensures it loads on CPU if user doesn't have CUDA
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        
        return model, tokenizer, True
    except Exception as e:
        return None, None, str(e)

# Initialize
model, tokenizer, status = load_resources()

# --- 3. UI LOGIC ---

st.title("🛡️ SQL INJECTION DETECTOR")
st.markdown("### Hybrid Deep Learning Security Scanner")

if status is not True:
    st.error(f"Failed to load model files: {status}")
    st.warning("Please ensure `sqli_model_hybrid.pth` and `tokenizer_hybrid.pkl` are in the same directory.")
    st.stop()

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/security-checked.png", width=80)
st.sidebar.header("Scan Mode")
mode = st.sidebar.radio("Choose an option:", ["Single Query Scanner", "Batch File Upload"])
st.sidebar.markdown("---")
st.sidebar.info(f"**Model:** Hybrid CNN-LSTM")

# --- SINGLE SCANNER ---
if mode == "Single Query Scanner":
    st.subheader("🔍 Inspect Single Query")
    query = st.text_area("Enter SQL Query:", height=150, placeholder="SELECT * FROM users WHERE...")
    
    if st.button("Analyze Query", type="primary"):
        if not query.strip():
            st.warning("Please enter a query first.")
        else:
            with st.spinner("Analyzing..."):
                # Inference
                with torch.no_grad():
                    vec = tokenizer.encode_single(query)
                    tensor_vec = torch.tensor(np.array([vec]), dtype=torch.long).to(DEVICE)
                    logit = model(tensor_vec)
                    prob = torch.sigmoid(logit).item()

                # Lower threshold for stricter security as discussed
                is_malicious = prob > 0.3  

                st.markdown("---")
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if is_malicious:
                        st.image("https://img.icons8.com/fluency/96/high-priority.png")
                    else:
                        st.image("https://img.icons8.com/fluency/96/verified-account.png")

                with col2:
                    if is_malicious:
                        st.error(f"### 🚨 MALICIOUS DETECTED")
                        st.markdown("**Action:** Block Request")
                    else:
                        st.success(f"### ✅ SAFE QUERY")
                        st.markdown("**Action:** Allow Request")

# --- BATCH SCANNER ---
elif mode == "Batch File Upload":
    st.subheader("📂 Batch File Scanner")
    st.markdown("Upload a `.txt` (one query per line) or `.csv` (column name 'Query') file.")
    
    uploaded_file = st.file_uploader("Upload file", type=["txt", "csv"])
    
    if uploaded_file is not None:
        queries = []
        
        # Read File
        if uploaded_file.name.endswith('.txt'):
            stringio = uploaded_file.getvalue().decode("utf-8")
            queries = [line.strip() for line in stringio.split('\n') if line.strip()]
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            if 'Query' in df.columns:
                queries = df['Query'].astype(str).tolist()
            else:
                st.error("CSV must have a column named 'Query'")
        
        if queries:
            st.info(f"Loaded {len(queries)} queries. Scanning now...")
            
            results = []
            progress_bar = st.progress(0)
            
            # Batch Inference
            with torch.no_grad():
                for i, q in enumerate(queries):
                    vec = tokenizer.encode_single(q)
                    tensor_vec = torch.tensor(np.array([vec]), dtype=torch.long).to(DEVICE)
                    logit = model(tensor_vec)
                    prob = torch.sigmoid(logit).item()
                    
                    # Lower threshold for stricter security
                    is_malicious = prob > 0.3 
                    verdict = "MALICIOUS" if is_malicious else "SAFE"
                    
                    results.append({
                        "Query": q,
                        "Verdict": verdict,
                        "Is_Malicious": is_malicious
                    })
                    progress_bar.progress((i + 1) / len(queries))
            
            # Display Results
            res_df = pd.DataFrame(results)
            
            st.markdown("### 📊 Scan Results")
            
            # Filter Metrics
            total = len(res_df)
            malicious_count = len(res_df[res_df['Is_Malicious'] == True])
            
            m1, m2 = st.columns(2)
            m1.metric("Total Queries", total)
            m2.metric("Threats Found", malicious_count, delta_color="inverse")
            
            # Highlight malicious rows
            def highlight_rows(row):
                return ['background-color: #ff4b4b; color: white' if row.Verdict == 'MALICIOUS' else 'background-color: #90ee90; color: black'] * len(row)

            # Drop the Is_Malicious column for cleaner display
            display_df = res_df.drop(columns=['Is_Malicious'])
            st.dataframe(display_df.style.apply(highlight_rows, axis=1))
            
            # Download Button
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Report",
                csv,
                "scan_report.csv",
                "text/csv",
                key='download-csv'
            )