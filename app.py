import streamlit as st
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from pathlib import Path

# --- Model definition ---
class TransformerScalarClassifier(torch.nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.scalar_vector = torch.nn.Parameter(torch.randn(d_model))

    def forward(self, x, padding_mask=None):
        x = self.linear(x)        # (B, L, d_model)
        x = x.transpose(0, 1)     # (L, B, d_model)
        x = self.encoder(x)       # (L, B, d_model)
        x = x.transpose(0, 1)     # (B, L, d_model)
        logits = torch.matmul(x, self.scalar_vector)  # (B, L)
        if padding_mask is not None:
            logits = logits.masked_fill(padding_mask, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return logits, probs


# --- Streamlit UI ---
st.title("Gene Probability Predictor")
st.write("Upload a locus feature file and a model checkpoint to get gene probabilities.")

# --- File inputs ---
ckpt_file = st.file_uploader("Upload model checkpoint (.pt or .pth)", type=["pt", "pth"])
uploaded_file = st.file_uploader("Upload locus features (.csv or .npy)", type=["csv", "npy"])

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- When checkpoint is uploaded ---
if ckpt_file is not None:
    ckpt_bytes = ckpt_file.read()
    ckpt_path = Path("model_0.pt")
    with open(ckpt_path, "wb") as f:
        f.write(ckpt_bytes)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Try to get config from checkpoint
    if "config" in checkpoint:
        config = checkpoint["config"]
        st.success("Loaded model configuration from checkpoint.")
    else:
        # try to infer something minimal
        st.warning("No config found in checkpoint; using fallback defaults.")
        config = dict(input_dim=13, d_model=64, n_heads=4, n_layers=2)

    # Initialize model
    model = TransformerScalarClassifier(**config).to(device)

    # Load weights
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    st.success("Model loaded successfully.")
    st.json(config)

    # --- When features are uploaded ---
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            x = torch.tensor(df.values, dtype=torch.float32)
            gene_names = list(df.columns)
        else:
            arr = np.load(uploaded_file)
            x = torch.tensor(arr, dtype=torch.float32)
            gene_names = [f"Gene_{i+1}" for i in range(x.shape[0])]

        if x.ndim == 2:
            x = x.unsqueeze(0)  # (1, seq_len, input_dim)

        with torch.no_grad():
            logits, probs = model(x.to(device))

        probs = probs.cpu().numpy().flatten()
        result_df = pd.DataFrame({"Gene": gene_names, "Probability": probs})
        st.subheader("Predicted Probabilities")
        st.dataframe(result_df.style.format({"Probability": "{:.4f}"}))

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions", csv, "predictions.csv", "text/csv")
else:
    st.info("Upload a model checkpoint to begin.")
