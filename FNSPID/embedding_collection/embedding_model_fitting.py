import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from joblib import dump

# === Settings ===
CSV_FILES      = [
    "/data/user_data/spushpit/FNSPID/stock_news_author_integrated/GILD.csv",
    "/data/user_data/spushpit/FNSPID/stock_news_author_integrated/USO.csv",
    "/data/user_data/spushpit/FNSPID/stock_news_author_integrated/ORCL.csv",
    "/data/user_data/spushpit/FNSPID/stock_news_author_integrated/GSK.csv",
    "/data/user_data/spushpit/FNSPID/stock_news_author_integrated/BIDU.csv",
"/data/user_data/spushpit/FNSPID/stock_news_author_integrated/EBAY.csv"
]
TEXT_COL       = "New_text"
CLOSE_COL      = "Close"
INPUT_LENGTH   = 50
OUTPUT_LENGTH  = 1
SPLIT_RATIO    = 0.85
RANDOM_SEED    = 42

# === Model for embeddings ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "meta-llama/Llama-2-7b-hf"
access_token = #your token here
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    token=access_token
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)

def compute_embeddings(texts):
    all_embs = []
    for text in texts:
        encoded = tokenizer.encode(text, return_tensors='pt').to(device)
        with torch.no_grad():
            output = model(encoded, output_hidden_states=True)
        embedding = torch.mean(output.hidden_states[-1][0], dim=0).to(dtype=torch.float32).cpu().numpy()
        all_embs.append(embedding)
        del encoded
        del output
    return np.vstack(all_embs)

def create_sequences(embeddings, prices, input_length, output_length):
    X, y = [], []
    T = len(prices)
    for i in range(T - input_length - output_length + 1):
        X.append(embeddings[i : i + input_length])
        y.append(prices[i + input_length])
    return np.array(X), np.array(y)

def process_csv(csv_path):
    df = pd.read_csv(csv_path)
    texts = df[TEXT_COL].astype(str).tolist()
    closes = df[CLOSE_COL].values.reshape(-1, 1)

    embeddings = compute_embeddings(texts)
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    E_scaled = x_scaler.fit_transform(embeddings)
    C_scaled = y_scaler.fit_transform(closes)

    X, y = create_sequences(E_scaled, C_scaled, INPUT_LENGTH, OUTPUT_LENGTH)
    return X, y, x_scaler, y_scaler

def main():
    X_all, y_all = [], []

    for csv_file in CSV_FILES:
        print(f"Processing {csv_file}...")
        X, y, _, _ = process_csv(csv_file)
        X_all.append(X)
        y_all.append(y)

    X_all = np.vstack(X_all)
    y_all = np.vstack(y_all)

    N_train = int(SPLIT_RATIO * len(X_all))
    X_train, X_test = X_all[:N_train], X_all[N_train:]
    y_train, y_test = y_all[:N_train], y_all[N_train:]

    B, L, D = X_train.shape
    X_train_flat = X_train.reshape(B, L * D)
    X_test_flat = X_test.reshape(len(X_test), L * D)

    reg = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    reg.fit(X_train_flat, y_train)
    y_pred = reg.predict(X_test_flat)

    # Fit a global y_scaler for inverse transform (retrain on all y_all)
    y_scaler = MinMaxScaler()
    y_scaler.fit(y_all)
    y_test_inv = y_scaler.inverse_transform(y_test)
    y_pred_inv = y_scaler.inverse_transform(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    print(f"Test RMSE over next {OUTPUT_LENGTH} day(s): {rmse:.4f}")

    for h in range(OUTPUT_LENGTH):
        r, p = pearsonr(y_test_inv[:, h], y_pred_inv[:, h])
        print(f"Pearson r for horizon t+{h+1}: {r:.4f} (p={p:.3g})")

    dump(reg, "ridge_model.joblib")
    print("Saved Ridge model → ridge_model.joblib")

if __name__ == "__main__":
    main()
