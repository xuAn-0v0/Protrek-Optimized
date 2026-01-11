# ProTrek Optimization: Enhanced Granularity & Locality

This repository contains an optimized version of the ProTrek protein language model. The primary goal of this optimization is to solve the **"Representation Granularity"** and **"Spatial Locality"** issues, enabling the model to distinguish between highly similar protein variants (e.g., Green vs. Red Fluorescent Proteins).

## 🚀 Key Modifications

### 1. Two-Stage Retrieval Architecture
We introduced a **Multi-layer Cross-Attention Reranker** (`model/ProTrek/reranker.py`) to complement the global embedding search.
- **Stage 1 (Recall)**: Fast candidate screening using global vectors.
- **Stage 2 (Rerank)**: Fine-grained alignment between text query tokens and protein residues to capture local functional motifs.

### 2. In-Family Competitive Training
A specialized training strategy was implemented in `finetune.py`:
- **Hard Negative Mining**: Forces the model to distinguish between proteins in the same family by using "In-family" mismatched pairs (e.g., GFP sequence paired with RFP description) as negatives.
- **Efficiency**: Utilizes **Layer Freezing** (freezing the first 24 layers of ESM2-650M) and **LoRA** to achieve high-performance fine-tuning within 3 hours on an A100 GPU.

### 3. Comprehensive Evaluation Suite
We developed a new evaluation script `benchmark.py` that includes:
- **Family Matrix Test**: Evaluates discrimination across GFP, RFP, YFP, and CFP.
- **Decoy Test**: Shuffles sequences to ensure the model understands structural logic rather than just amino acid composition.
- **Robustness Check**: Ensures no performance degradation on general protein tasks.

## 📊 Results & Comparison

| Metric | Original ProTrek (Baseline) | Optimized ProTrek | Status |
| :--- | :--- | :--- | :--- |
| **Hard Set Accuracy** | 25.0% (Colorblind) | **75.0% ~ 100%** | ✨ Major Leap |
| **Discrimination Gap** | -0.0637 | **+1.3289 ~ +7.96** | ✨ Significant |
| **Anti-Overfitting (Decoy)** | [RISKY] | **[SAFE]** | ✅ Logic Verified |
| **General Robustness** | 96.0% | **88.0% ~ 98.0%** | ✅ Stable |

## 🛠️ Execution Guide

### Files & Environment
- **Weights**: Optimized weights are saved in `weights/ProTrek_optimized_v2.pt`, which can be downloaded in this link https://drive.google.com/file/d/1A1mKw_7OV9zwGrCCkSH6_q5AKfhIGRAu/view?usp=drive_link.
- **Data**: Training requires `protrek_data.tsv` (uses a 100k sample subset for efficiency).

### Commands
1. **Training**: To replicate the fine-tuning process:
   ```bash
   python finetune.py
   ```
2. **Evaluation**: To run the full benchmark suite:
   ```bash
   python benchmark.py
   ```
3. **Comparison Demo**: To see the performance jump on a specific case (RFP vs. GFP):
   ```bash
   python comparison_demo.py
   ```

## 📂 Core Optimized Files
- `model/ProTrek/reranker.py`: The new Reranker module.
- `model/ProTrek/protrek_trimodal_model.py`: Integrated two-stage retrieval logic.
- `finetune.py`: Accelerated family-competitive training engine.
- `benchmark.py`: Evaluation system.
- `comparison_demo.py`: Direct visual comparison between model versions.
