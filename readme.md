&nbsp;\*AI-Powered SQL Injection Detection System\*

\- Overview

The SQLi Detection System is a state-of-the-art Deep Learning security system designed to detect and block SQL Injection (SQLi) attacks in real-time. Unlike traditional Web Application Firewalls (WAFs) that rely on static Regular Expressions (Regex), this system uses a Hybrid Neural Network (CNN + Bi-LSTM) to understand the semantic context and logic of SQL queries.

This project demonstrates a move from signature-based detection to context-aware AI defense, effectively blocking advanced evasion techniques, obfuscation, and logical injection attacks while maintaining a zero false-positive rate for legitimate traffic.

\- Key Features

Context-Aware Detection: Distinguishes between malicious SQL logic (e.g., OR 1=1) and safe benign data (e.g., id = 10-2).

Hybrid Architecture: Combines CNN (for feature extraction of attack signatures) and Bi-LSTM (for sequence logic analysis).

Adversarial Robustness: Trained on a dataset augmented with obfuscation techniques (/\*\*/ comments, Hex encoding, URL encoding).

System-Level Protection: Specifically trained to detect reconnaissance against system tables (information\_schema, mysql.user).

Interactive UI: Includes a user-friendly Streamlit web interface for single-query checking and batch file scanning.

\- Architecture

The model utilizes a custom tokenization pipeline feeding into a hybrid deep learning structure:

Custom Tokenizer: Preprocesses SQL queries, separating special symbols (\*, =, (, )) to preserve structural integrity.

Embedding Layer: 128-dimensional dense vector representation.

1D Convolutional Neural Network (CNN): Extracts local n-gram features (suspicious keyword combinations).

Bidirectional LSTM: Analyzes the sequence flow to understand the intent of the query.

Dense Classifier: Outputs a probability score (0-1) indicating maliciousness.

Hyperparameters:

Vocabulary: 25,000 tokens

Batch Size: 256

Optimizer: Adam (LR=0.001) with Weight Decay

Loss Function: Binary Cross Entropy with Logic-Based Class Weighting

\- Performance \& Results

The model was evaluated on a rigorous unseen test set of 38,717 samples and a specialized "Stress Test" of 80 complex adversarial queries.

Metric

Score

Accuracy

98.79%

Precision

0.99

Recall

0.99

F1-Score

0.99

* Visualization

The Confusion Matrix below demonstrates zero False Positives and zero False Negatives on the test set.

&nbsp;Installation \& Usage

1. Prerequisites

Ensure you have Python installed. Then, install the dependencies:

pip install -r requirements.txt





2. Running the Application (One-Click)

Simply double-click the included batch file:
 - launch.bat

3. Running via Command Line

To launch the Streamlit interface manually:

streamlit run app.py





&nbsp;- Project Structure

app.py: Main Streamlit web application.

model.py: Complete training and validation script.

sqli\_model\_hybrid.pth: The trained model weights.

tokenizer\_hybrid.pkl: The saved tokenizer vocabulary.

SID.csv: The augmented training dataset.

\- Stress Testing

The model has been verified against highly complex vectors, including:

Obfuscation: UN/**/ION SE/**/LECT (Detected ✅)

Benign Ambiguity: SELECT \* FROM orders WHERE id = 10-2 (Allowed ✅)

Schema Recon: SELECT \* FROM information\_schema.tables (Detected ✅)

JSON Injection: SELECT \* FROM json\_data WHERE info->>'$.role' = 'admin' (Allowed ✅)

