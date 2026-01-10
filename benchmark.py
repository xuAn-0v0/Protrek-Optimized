import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

# Add ProTrek to path
ROOT_DIR = "/songjian/zixuan/ProTrek"
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel

def load_model(weights_path, device):
    model_dir = "weights/ProTrek_650M"
    config = {
        "protein_config": f"{model_dir}/esm2_t33_650M_UR50D",
        "text_config": f"{model_dir}/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "structure_config": f"{model_dir}/foldseek_t30_150M",
        "from_checkpoint": f"{model_dir}/ProTrek_650M.pt",
        "use_reranker": True 
    }
    model = ProTrekTrimodalModel(**config).to(device)
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: {weights_path} not found, using base model.")
    model.eval()
    return model

def get_rerank_score(model, seq, text, device):
    p_in = model.protein_encoder.tokenizer([seq], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    t_in = model.text_encoder.tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        score = model.rerank(p_in, t_in).item()
    return score

def get_global_sim(model, seq, text, device):
    p_in = model.protein_encoder.tokenizer([seq], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    t_in = model.text_encoder.tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        t_repr, p_repr, _ = model(p_in, t_in)
        p_repr = F.normalize(p_repr, dim=-1)
        t_repr = F.normalize(t_repr, dim=-1)
        sim = torch.matmul(p_repr, t_repr.T).item()
    return sim

def get_shuffled_seq(seq):
    seq_list = list(seq)
    import random
    random.shuffle(seq_list)
    return "".join(seq_list)

def run_benchmark(weights_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(weights_path, device)

    # 1. THE HARD SET (Fluorescent Family)
    hard_set = [
        {"name": "GFP", "seq": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK", "text": "Green fluorescent protein. Transduce blue chemiluminescence into green fluorescent light."},
        {"name": "RFP", "seq": "MRSSKNVIKEFMRFKVRMEGTVNGHEFEIEGEGEGRPYEGHNTVKLKVTKGGPLPFAWDILSPQFQYGSKVYVKHPADIPDYKKLSFPEGFKWERVMNFEDGGVVTVTQDSSLQDGCFIYKVKFIGVNFPSDGPVMQKKTMGWEASTERLYPRDGVLKGEIHKALKLKDGGHYLVEFKSIYMAKKPVQLPGYYYVDSKLDITSHNEDYTIVEQYERTEGRHHLFL", "text": "Red fluorescent protein. Play a role in photoprotection and convert blue light into longer wavelengths."},
        {"name": "YFP", "seq": "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFGYGLQCFARYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK", "text": "Yellow fluorescent protein. A genetic mutant of GFP with shifted emission spectrum."},
        {"name": "CFP", "seq": "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSWGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK", "text": "Cyan fluorescent protein. Emission shifted towards blue/cyan."}
    ]

    print("\n" + "="*60)
    print("TEST 1: THE HARD SET (Fluorescent Family Discrimination)")
    print("="*60)
    
    hard_results = []
    for target in hard_set:
        print(f"\nQuery Protein: {target['name']}")
        scores = []
        for candidate in hard_set:
            r_score = get_rerank_score(model, target['seq'], candidate['text'], device)
            scores.append({
                "cand": candidate['name'],
                "rerank": r_score,
                "is_correct": target['name'] == candidate['name']
            })
        
        # Sort by rerank score
        scores.sort(key=lambda x: x['rerank'], reverse=True)
        
        for s in scores:
            mark = "[CORRECT]" if s['is_correct'] else "         "
            print(f"  -> {s['cand']}: Rerank={s['rerank']:.4f} {mark}")
        
        top1_correct = scores[0]['is_correct']
        # Calculate gap: correct score minus the best incorrect score
        correct_score = next(s['rerank'] for s in scores if s['is_correct'])
        best_incorrect_score = next(s['rerank'] for s in scores if not s['is_correct'])
        gap = correct_score - best_incorrect_score
        hard_results.append({"correct": top1_correct, "gap": gap})

    # 3. DECOY TEST (Anti-Overfitting)
    print("\n" + "="*60)
    print("TEST 3: DECOY TEST (Sensitivity to Sequence Order)")
    print("="*60)
    decoy_results = []
    for target in hard_set:
        shuffled_seq = get_shuffled_seq(target['seq'])
        orig_score = get_rerank_score(model, target['seq'], target['text'], device)
        decoy_score = get_rerank_score(model, shuffled_seq, target['text'], device)
        is_safe = orig_score > (decoy_score + 2.0) # Expect a significant drop
        print(f"  {target['name']}: Original={orig_score:.4f}, Shuffled={decoy_score:.4f} {'[SAFE]' if is_safe else '[RISKY]'}")
        decoy_results.append(is_safe)

    # 2. THE GENERAL SET (Robustness)
    print("\n" + "="*60)
    print("TEST 2: THE GENERAL SET (Robustness Check)")
    print("="*60)
    
    data_path = "protrek_data.tsv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, sep="\t").sample(n=50, random_state=42)
        gen_correct = 0
        for _, row in tqdm(df.iterrows(), total=50):
            # Test if correct text is better than a random one
            random_text = "A completely unrelated protein function description."
            s_pos = get_rerank_score(model, row['protein_sequence'], row['description'], device)
            s_neg = get_rerank_score(model, row['protein_sequence'], random_text, device)
            if s_pos > s_neg:
                gen_correct += 1
        print(f"\nGeneral Robustness (Pos > Random Neg): {gen_correct}/50 ({gen_correct*2}%)")
    else:
        print("\nprotrek_data.tsv not found, skipping general set.")

    # Summary
    hard_acc = sum([1 for r in hard_results if r['correct']]) / len(hard_results)
    avg_gap = np.mean([r['gap'] for r in hard_results])
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Hard Set Accuracy (Top-1): {hard_acc*100:.1f}%")
    print(f"Hard Set Average Gap:     {avg_gap:.4f}")
    if 'gen_correct' in locals():
        print(f"General Robustness:       {gen_correct*2:.1f}%")

if __name__ == "__main__":
    w_path = "weights/ProTrek_optimized_v2.pt"
    run_benchmark(w_path)

