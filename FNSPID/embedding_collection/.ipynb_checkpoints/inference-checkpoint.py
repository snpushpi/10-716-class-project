import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from joblib import load
import torch, gc, numpy as np

TEXT_COL      = "New_text"                    # same text‐column name
CLOSE_COL     = "Close"                       # same close‐price column (needed to build windows)
INPUT_LENGTH  = 50
MODEL_PATH    = "ridge_model.joblib"
MODEL_NAME    = "meta-llama/Llama-2-7b-hf"
ACCESS_TOKEN  = "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"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Load model + tokenizer
reg      = load(MODEL_PATH)
tokenizer= AutoTokenizer.from_pretrained(MODEL_NAME, token=ACCESS_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
model    = AutoModel.from_pretrained(
               MODEL_NAME,
               torch_dtype=torch.bfloat16,
               device_map="auto",
               attn_implementation="flash_attention_2",
               token=ACCESS_TOKEN
           ).to(DEVICE)

model.config.use_cache = False     #  ⬅︎ turn the cache off
model.eval()

@torch.inference_mode()            # same as no_grad + a few extras

def compute_embeddings(texts, max_len=256, batch_size=8):
    """Return an (N, D) NumPy array of mean‑pooled last‑layer embeddings."""
    embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(model.device)

        out = model(**enc, output_hidden_states=True, use_cache=False)
        last = out.hidden_states[-1].mean(dim=1)           # (B, D)

        embs.append(last.float().cpu().numpy())
        # aggressively release blocks & defragment
        del enc, out, last
        torch.cuda.empty_cache()
        gc.collect()

    return np.vstack(embs)


# 4) Build 1‑step sequences of X only
def create_X_sequences(embeds, input_length):
    X = []
    T = len(embeds)
    for i in range(T - input_length):
        X.append(embeds[i : i + input_length])
    return np.array(X)  # (N, input_length, D)

def main(CSV_IN,CSV_OUT):
    # 5) Load your new CSV
    df = pd.read_csv(CSV_IN)
    texts  = df[TEXT_COL].astype(str).tolist()
    closes = df[CLOSE_COL].values.reshape(-1,1)  # needed only to fit scaler
    
    # 6) Compute fresh scalers on the new data
    E_new = compute_embeddings(texts)           # (T, D)
    x_scaler = MinMaxScaler().fit(E_new)
    E_s     = x_scaler.transform(E_new)
    
    y_scaler = MinMaxScaler().fit(closes)       # only if you want to invert later
    
    # 7) Create X windows and flatten
    X_seq = create_X_sequences(E_s, INPUT_LENGTH)      # (N, L, D)
    N, L, D = X_seq.shape
    X_flat = X_seq.reshape(N, L*D)
    
    # 8) Predict (these are in *training* y‐scale)
    y_pred_scaled = reg.predict(X_flat)               # (N,)
    
    # 9) (Optionally) Invert to original‐price scale using the new y‐scaler
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    pred_col = np.concatenate([ [np.nan]*INPUT_LENGTH, y_pred ])
    df["Potency"] = pred_col
    # 10) Append to DataFrame (pad first INPUT_LENGTH rows with NaN)
    close_price = df[CLOSE_COL].tolist()[INPUT_LENGTH:]
    gpt = df['Scaled_sentiment'].tolist()[INPUT_LENGTH:]
    r1, p1 = pearsonr(gpt, close_price)
    r2, p2 = pearsonr(y_pred, close_price)
    print(r1, r2)
    df.to_csv(CSV_OUT, index=False)
    print(f"Wrote predictions → {CSV_OUT}")

if __name__ == "__main__":

    CSV_FILES_trained = [
        "/data/user_data/spushpit/FNSPID/stock_news_author_integrated/GILD.csv",
        "/data/user_data/spushpit/FNSPID/stock_news_author_integrated/USO.csv",
        "/data/user_data/spushpit/FNSPID/stock_news_author_integrated/ORCL.csv",
        "/data/user_data/spushpit/FNSPID/stock_news_author_integrated/GSK.csv",
        "/data/user_data/spushpit/FNSPID/stock_news_author_integrated/BIDU.csv",
        "/data/user_data/spushpit/FNSPID/stock_news_author_integrated/EBAY.csv"
    ]
    
    # 1) determine parent directory
    parent_dir_in = os.path.dirname(CSV_FILES_trained[0])
    parent_dir_out = "/data/user_data/spushpit/FNSPID/stock_news_author_int_final"
    # 2) set of basenames you've already trained on
    trained_set = {os.path.basename(p) for p in CSV_FILES_trained}
    
    # 3) list all files in that dir
    all_in_dir = [
        f for f in os.listdir(parent_dir_in)
        if os.path.isfile(os.path.join(parent_dir_in, f))
    ]
    
    # 4) filter out the trained ones
    to_train = [f for f in all_in_dir if f not in trained_set]
    
    for fname in sorted(to_train):
        CSV_IN = os.path.join(parent_dir_in, fname)
        CSV_OUT = os.path.join(parent_dir_out, fname)
        main(CSV_IN,CSV_OUT)

