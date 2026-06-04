# Hotel Review Rating Prediction — From TF-IDF to ModernBERT

> Predicting hotel star ratings (1–5★) from customer review text, progressing from classical NLP baselines to state-of-the-art Transformer fine-tuning with ordinal regression — reaching **R² = 0.75 and 70% exact accuracy**.

**University:** Université d'Avignon — Master 1 Approaches Neuronales  
**Environment:** Python 3.10 · PyTorch · HuggingFace Transformers · scikit-learn · CUDA · Jupyter Notebook

---

## Problem Statement

Given a hotel review in plain English, predict the customer's **star rating from 1 to 5**. This is an **ordinal regression** problem: the classes are ordered (3★ is worse than 4★, which is worse than 5★), so treating it as standard multi-class classification loses information.

---

## Dataset

| Split | Samples |
|-------|---------|
| Train | 18,491 |
| Validation | 1,000 |
| Test | 1,000 |

**Class distribution (train):**

| Rating | Count | Proportion |
|--------|-------|-----------|
| 1★ | 1,310 | 7.1% |
| 2★ | 1,630 | 8.8% |
| 3★ | 1,982 | 10.7% |
| 4★ | 5,465 | 29.6% |
| 5★ | 8,104 | **43.8%** |

The dataset is **heavily imbalanced** — 4★ and 5★ together account for 73% of reviews.

---

## Approaches & Results

| Version | Method | R² | Accuracy | Notes |
|---------|--------|-----|----------|-------|
| Baseline 1 | TF-IDF (ngram 1–4, 20k features) + MLP Regression | 0.68 | 63% | Classical pipeline, no deep encoder |
| Baseline 2 | Sentence Embeddings (MiniLM-L6, 384d) + MLP Classification | — | 59.1% | Dense embeddings, CrossEntropy |
| v1 | ModernBERT + CrossEntropy | ~0.65 | ~62% | Deep fine-tuning, ignores ordinality |
| v2 | ModernBERT + CORN + class weights + battle royale seeds | 0.7308 | 68.50% | Ordinal regression, best single model |
| v3 | v2 + dual-head + ensemble 3 models | 0.7492 | 68.70% | +0.02 R² from ensemble |
| v4a | v2 + oversampling sqrt | 0.6918 | 63.80% | Oversampling degrades results |
| **v4b** | **v2 + FP16 + multi-GPU + 10 epochs** | **0.75** | **70%** | **Best result** |

**Best model metrics (v4b — test set):**

| Metric | Value |
|--------|-------|
| R² Score | **0.75** |
| MAE | **~0.32** |
| Exact Accuracy | **70%** |
| Accuracy ±1★ | **~98%** |

---

## Architecture: ModernBERT + CORN Head

### Why CORN (Conditional Ordinal Regression Network)?

Standard CrossEntropy treats 1★ vs 5★ the same cost as 4★ vs 5★. CORN models the ordinal structure by decomposing the prediction into **4 binary comparisons**:

| Question | Prediction |
|----------|-----------|
| Note > 1? | yes/no |
| Note > 2? | yes/no |
| Note > 3? | yes/no |
| Note > 4? | yes/no |

**Final prediction = number of "yes" + 1**  
Example: `[yes, yes, yes, no]` → **4★**

### Model Pipeline

```
Review text
    ↓
ModernBERT-base  (answerdotai/ModernBERT-base, 768 dims)
    ↓
Mean Pooling  (weighted average over all tokens)
    ↓
Linear(768 → 256) + GELU + Dropout(0.3) + LayerNorm
    ↓
Linear(256 → 4)   ← 4 CORN logits
    ↓
Sigmoid → count > 0.5 → note (1–5)
```

### Class Weights (handling imbalance)

| Rating | Weight |
|--------|--------|
| 1★ | 1.745 |
| 2★ | 1.402 |
| 3★ | 1.153 |
| 4★ | 0.418 |
| 5★ | 0.282 |

An error on a 1★ review costs ~6× more than a 5★ error, forcing the model to learn minority classes.

### Training Configuration (v4b)

| Parameter | Value |
|-----------|-------|
| Base model | `answerdotai/ModernBERT-base` |
| BERT LR | 2e-5 |
| Head LR | 1e-3 |
| Scheduler | Cosine with 6% warmup |
| Batch size | 16 |
| Epochs | 10 (early stopping, patience=3) |
| Precision | FP16 (mixed precision) |
| Seed | 2024 (identified by Battle Royale) |
| Label smoothing | 0.05 |

### Technical Optimizations

- **FP16 Mixed Precision** — ~1.5× training speedup via `torch.amp.autocast`
- **Multi-GPU DataParallel** — batch split across 2 GPUs
- **Gradient Checkpointing** — reduces VRAM usage by recomputing activations during backward pass
- **Gradient Clipping** (max norm=1.0) — prevents exploding gradients during fine-tuning

---

## Lessons Learned

| Finding | Impact |
|---------|--------|
| CORN > CrossEntropy for ordinal labels | +0.08 R² |
| Class weights in loss > oversampling | Stable training |
| Mean pooling > CLS token for ModernBERT | +1–2% accuracy |
| Cosine scheduler with warmup | Essential for convergence |
| Ensemble of 3 models | +0.02 R² but 3× training time |
| Oversampling | −0.04 R² (degrades results) |

---

## Project Structure

```
TP3/
├── tp3.ipynb                      # Baseline: TF-IDF + MLP regression
├── tp3.2_classification.ipynb     # Sentence embeddings + classification
├── tp4.ipynb                      # ModernBERT experiments (v1/v2/v3)
├── TP4_Hotel_Rating_Prediction.ipynb  # Final: ModernBERT CORN v4b (documented)
├── tp5.ipynb                      # ModernBERT + chunking + corrector
├── data/
│   ├── train_hotel.csv
│   ├── valid_hotel.csv
│   └── test_hotel.csv
├── bert_hotel_rating_model_v4.pth # Saved best checkpoint
└── requirements.txt
```

---

## Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
jupyter notebook TP4_Hotel_Rating_Prediction.ipynb
```

```
# requirements.txt
torch>=2.0
transformers>=4.40
scikit-learn
pandas
numpy
matplotlib
seaborn
```

---

## Key Insights

1. **Ordinal regression is not optional** — hotel ratings have a natural order; ignoring it wastes signal and hurts calibration.
2. **The battle royale found seed 2024** — random initialization matters more than expected in fine-tuning scenarios.
3. **TF-IDF at R²=0.68 is a strong baseline** — transformer fine-tuning adds ~7 points R², which requires significant compute.
4. **98% of predictions are within ±1 star** — the model rarely makes catastrophic errors (confusing 1★ with 5★).

---

*Master 1 — Approches Neuronales — Université d'Avignon*
