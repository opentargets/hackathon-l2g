import streamlit as st
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import skops.io as sio

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_results_from_l2g(df):
    
    model_path = "checkpoints/classifier.skops"
    loaded_model = sio.load(
        model_path, trusted=sio.get_untrusted_types(file=model_path)
    )
    probs = loaded_model.predict_proba(df.iloc[:, 3:].values)[:,1]
    results = df.iloc[:, :3].assign(probs=probs) 
    return results


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


# dict of { "ENSG_ID": "gene symbol"}
name_mapping = pd.read_csv("data/ensg_to_symbol_mapping.csv").set_index('id').approvedSymbol.to_dict()

# --- Streamlit UI ---
st.sidebar.title("L2G: predicting the causal gene in a locus")
st.sidebar.write("Upload a locus feature file and a model checkpoint to get gene probabilities.")

# --- File inputs ---
ckpt_file = st.sidebar.file_uploader("Upload model checkpoint (.pt or .pth)", type=["pt", "pth"])

feature_matrix_test = pd.read_parquet("data/test.parquet")

# Results from current production L2G model (not calculated on the fly)
current_l2g_results = load_results_from_l2g(feature_matrix_test)

study_locus_id = st.sidebar.selectbox(label="Study locus ID", options=feature_matrix_test.studyLocusId.unique())
# uploaded_file = st.file_uploader("Upload locus features (.csv or .npy)", type=["csv", "npy"])


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
        config.pop("fold")
        # st.success("Loaded model configuration from checkpoint.")
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

    current_l2g_results_for_locus = current_l2g_results.query("studyLocusId == @study_locus_id")

    # Data as required by our transformer model (13 features)
    feature_matrix_test_for_locus = feature_matrix_test.query("studyLocusId == @study_locus_id")
    feature_matrix_test_for_locus = feature_matrix_test_for_locus.loc[:,~feature_matrix_test_for_locus.columns.str.contains('Neighbourhood', case=False)]
    feature_matrix_test_for_locus = feature_matrix_test_for_locus.loc[:,~feature_matrix_test_for_locus.columns.str.contains('GeneCount', case=False)]
    gene_names = feature_matrix_test_for_locus.geneId.tolist()
    label = feature_matrix_test_for_locus.goldStandardSet
    x = torch.tensor(feature_matrix_test_for_locus.iloc[:, 3:].values)

    # --- When features are uploaded ---
    if x.ndim == 2:
        x = x.unsqueeze(0)  # (1, seq_len, input_dim) where seq_len is number of genes in the neighbourhood
    
    with torch.no_grad():
        logits, probs = model(x.to(device))

    probs = probs.cpu().numpy().flatten()
    transformer_result_df = pd.DataFrame({"Gene": gene_names, "Probability": probs})
    transformer_result_df['label'] = label.reset_index(drop=True)
    
    transformer_result_df = transformer_result_df.\
        sort_values("Probability", ascending=False).\
        reset_index(drop=True)
    
    # Mapping ENSG to approved gene symbol
    transformer_result_df = transformer_result_df.assign(gene_name=lambda df: df.Gene.apply(lambda ensg: name_mapping[ensg]))

    all_results_df = pd.merge(transformer_result_df, current_l2g_results_for_locus, left_on="Gene", right_on="geneId").\
        drop(["goldStandardSet", "geneId", "studyLocusId"], axis=1)

    st.subheader("Predicted Probabilities")
    all_results_df = all_results_df.rename(
        {"Probability": "Transformer", "probs": "XGBoost", "gene_name": "Gene symbol", "Gene": "Gene ID"}, axis=1
    )
    all_results_df = all_results_df[["Gene ID", "Gene symbol", "label", "Transformer", "XGBoost"]]
    st.dataframe(all_results_df.style.format({"Probability": "{:.4f}"}))

else:
    st.info("Upload a model checkpoint to begin.")
