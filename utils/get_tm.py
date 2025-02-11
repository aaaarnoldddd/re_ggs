from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from tqdm import tqdm

def get_ogt_model():
    model_path = "AI4Protein/Prime_690M"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model = model.to(device)
    return model, tokenizer, device

def get_ogt_entrance(seqs):
    model, tokenizer, device = get_ogt_model()
    if isinstance(seqs, str):
        with torch.no_grad():
            tokenied_results = tokenizer(seqs, return_tensors="pt")
            input_ids = tokenied_results.input_ids.to(device)
            attention_mask = tokenied_results.attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).predicted_values
            return logits.item()

    ogt = []
    with torch.no_grad():
        for sequence in tqdm(seqs, total=len(seqs), desc="calc pred_OGT"):
            tokenied_results = tokenizer(sequence, return_tensors="pt")
            input_ids = tokenied_results.input_ids.to(device)
            attention_mask = tokenied_results.attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).predicted_values
            ogt.append(logits.item())
    return ogt
# seqs = pd.read_csv("data/GFP/mutant_7_percentile_0.0_0.3/filtered_dataset.csv")["sequence"]
# print(len(tm))