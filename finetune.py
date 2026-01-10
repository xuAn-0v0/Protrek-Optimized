import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize, cross_entropy
from peft import LoraConfig, get_peft_model

# Add the current directory to sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel

class HardNegativeMutationDataset(Dataset):
    def __init__(self, df, tokenizer, text_tokenizer, fluorescent_texts, max_len=512):
        self.df = df
        self.tokenizer = tokenizer
        self.text_tokenizer = text_tokenizer
        self.fluorescent_texts = fluorescent_texts
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row["protein_sequence"]
        text = row["description"]
        
        # Hard Negative Mining Strategy:
        # Increase probability to 80% to force color discrimination
        if "fluorescent" in text.lower() and random.random() < 0.8:
            neg_text = random.choice([t for t in self.fluorescent_texts if t != text])
        else:
            # Otherwise pick a random description from the batch
            neg_idx = random.randint(0, len(self.df) - 1)
            while self.df.iloc[neg_idx]["description"] == text:
                neg_idx = random.randint(0, len(self.df) - 1)
            neg_text = self.df.iloc[neg_idx]["description"]
        
        return seq, text, neg_text

def collate_fn(batch, tokenizer, text_tokenizer):
    seqs, texts, neg_texts = zip(*batch)
    p_inputs = tokenizer(list(seqs), return_tensors="pt", padding=True, truncation=True, max_length=256)
    t_inputs = text_tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=256)
    n_inputs = text_tokenizer(list(neg_texts), return_tensors="pt", padding=True, truncation=True, max_length=256)
    return p_inputs, t_inputs, n_inputs

def main():
    model_dir = "weights/ProTrek_650M"
    config = {
        "protein_config": f"{model_dir}/esm2_t33_650M_UR50D",
        "text_config": f"{model_dir}/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "structure_config": f"{model_dir}/foldseek_t30_150M",
        "from_checkpoint": f"{model_dir}/ProTrek_650M.pt",
        "use_reranker": True,
        "gradient_checkpointing": False
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model with Reranker
    print("Loading ProTrek Model (this may take 1-2 minutes)...")
    model = ProTrekTrimodalModel(**config).to(device)
    print("Model loaded successfully.")

    # Apply LoRA and Freeze Layers for maximum speed
    print("Applying LoRA and Freezing early layers...")
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["query", "value"], 
        lora_dropout=0.05, 
        bias="none"
    )
    # Apply to encoders
    model.protein_encoder.model.esm = get_peft_model(model.protein_encoder.model.esm, peft_config)
    
    # FREEZE first 24 layers of ESM2 to save computation
    # ESM2-650M has 33 layers
    for i in range(24):
        for param in model.protein_encoder.model.esm.base_model.model.encoder.layer[i].parameters():
            param.requires_grad = False
            
    model.text_encoder.model = get_peft_model(model.text_encoder.model, peft_config)

    # Note: Reranker is fully trained
    for param in model.reranker.parameters():
        param.requires_grad = True

    model.train()

    # Load data
    data_path = "protrek_data.tsv"
    print(f"Loading data from {data_path} (450MB, please wait)...")
    
    # Expanded Fluorescent Family for training
    family_data = [
        {"protein_sequence": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK", 
         "description": "Green fluorescent protein. Transduce blue chemiluminescence into green fluorescent light.", "type": "fluorescent"},
        {"protein_sequence": "MRSSKNVIKEFMRFKVRMEGTVNGHEFEIEGEGEGRPYEGHNTVKLKVTKGGPLPFAWDILSPQFQYGSKVYVKHPADIPDYKKLSFPEGFKWERVMNFEDGGVVTVTQDSSLQDGCFIYKVKFIGVNFPSDGPVMQKKTMGWEASTERLYPRDGVLKGEIHKALKLKDGGHYLVEFKSIYMAKKPVQLPGYYYVDSKLDITSHNEDYTIVEQYERTEGRHHLFL", 
         "description": "Red fluorescent protein. Play a role in photoprotection and convert blue light into longer wavelengths.", "type": "fluorescent"},
        {"protein_sequence": "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFGYGLQCFARYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK", 
         "description": "Yellow fluorescent protein. A genetic mutant of GFP with shifted emission spectrum.", "type": "fluorescent"},
        {"protein_sequence": "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSWGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK", 
         "description": "Cyan fluorescent protein. Emission shifted towards blue/cyan.", "type": "fluorescent"}
    ]

    if not os.path.exists(data_path):
        df = pd.DataFrame(family_data * 100)
        df["stage"] = "train"
    else:
        # Load a large subset (100k) for fast startup but sufficient diversity
        print(f"Loading 100,000 samples from {data_path} for fast startup...")
        df = pd.read_csv(data_path, sep="\t", usecols=["protein_sequence", "description", "stage"], nrows=100000)
        target_df = pd.DataFrame(family_data * 50)
        target_df["stage"] = "train"
        df = pd.concat([df, target_df], ignore_index=True)
        print(f"Successfully loaded {len(df)} samples.")
    
    # Extract only fluorescent descriptions for hard negative mining
    fluorescent_texts = [d["description"] for d in family_data]
    
    #  Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = torch.cuda.amp.GradScaler() # Add GradScaler for AMP
    
    # Losses
    bce_loss = nn.BCEWithLogitsLoss()
    margin_loss_fn = nn.MarginRankingLoss(margin=0.2)

    # Training Loop
    print("Starting Optimized 3-Hour Training")
    epochs = 24  
    batch_size = 24  
    accumulation_steps = 1
    samples_per_epoch = 15000 
    
    for epoch in range(epochs):
        train_df = df[df["stage"] == "train"]
        target_cases = train_df[train_df["description"].str.contains("fluorescent", na=False)]
        other_cases = train_df[~train_df["description"].str.contains("fluorescent", na=False)]
        
        if len(other_cases) > samples_per_epoch:
            sampled_df = pd.concat([target_cases, other_cases.sample(n=samples_per_epoch)])
        else:
            sampled_df = train_df
            
        dataset = HardNegativeMutationDataset(sampled_df, model.protein_encoder.tokenizer, model.text_encoder.tokenizer, fluorescent_texts)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                collate_fn=lambda b: collate_fn(b, model.protein_encoder.tokenizer, model.text_encoder.tokenizer),
                                num_workers=12, pin_memory=True)
        
        pbar = tqdm(dataloader)
        optimizer.zero_grad()
        for i, (p_inputs, t_inputs, n_inputs) in enumerate(pbar):
            p_inputs = {k: v.to(device) for k, v in p_inputs.items()}
            t_inputs = {k: v.to(device) for k, v in t_inputs.items()}
            n_inputs = {k: v.to(device) for k, v in n_inputs.items()}
            
            with torch.cuda.amp.autocast():
                # --- STAGE 1: Global Contrastive ---
                outputs_p = model(p_inputs, t_inputs)
                t_repr, p_repr = outputs_p[0], outputs_p[1]
                
                p_repr = normalize(p_repr, dim=-1)
                t_repr = normalize(t_repr, dim=-1)
                
                logits = torch.matmul(p_repr, t_repr.T) / model.temperature
                labels = torch.arange(p_repr.shape[0]).to(device)
                loss_global = cross_entropy(logits, labels)

                # --- STAGE 2: Reranking with Cross-Attention ---
                # Positive pairs (Correct mapping)
                pos_rerank_scores = model.rerank(p_inputs, t_inputs) 
                # Negative pairs (Protein + Incorrect Description)
                neg_rerank_scores = model.rerank(p_inputs, n_inputs) 
                
                rerank_logits = torch.cat([pos_rerank_scores, neg_rerank_scores])
                rerank_labels = torch.cat([torch.ones_like(pos_rerank_scores), torch.zeros_like(neg_rerank_scores)])
                loss_rerank = bce_loss(rerank_logits, rerank_labels)
                
                # Margin loss between pos score and neg score for the same protein
                loss_margin = margin_loss_fn(pos_rerank_scores, neg_rerank_scores, torch.ones_like(pos_rerank_scores))
                
                # Final loss with balanced weights
                loss = loss_global + loss_margin + 2.0 * loss_rerank
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            pbar.set_description(f"Ep {epoch} L: {loss.item()*accumulation_steps:.3f} G: {loss_global.item():.3f} R: {loss_rerank.item():.3f}")

    # Save
    print("Saving optimized model...")
    torch.save(model.state_dict(), "weights/ProTrek_optimized_v2.pt")
    print("Optimization complete.")

if __name__ == "__main__":
    main()

