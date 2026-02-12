# 📰 Semi-Supervised Nepali News Classification with BERT-GCN

A hybrid deep learning system combining **BERT embeddings** with **Graph Convolutional Networks (GCN)** for semi-supervised text classification, achieving **95.68% accuracy** using only **10% labeled training data**.

---

## 🎯 Key Results

| Model | Test Accuracy | Test F1 | Training Data | Learning Type |
|--------|--------------|----------|---------------|---------------|
| **BERT Baseline** | 87.77% | 87.74% | 10% labeled | Inductive |
| **BERT-GCN** | **95.68%** | **95.64%** | 10% labeled + 85% unlabeled features | Transductive |
| **Improvement** | **+7.91%** | **+7.90%** | — | — |

**Dataset:** 122,817 Nepali news articles across 10 categories

---

## 🏗️ Architecture Overview

```
Stage 1: BERT Fine-tuning (10% labeled data)
        ↓
Stage 2: Embedding Extraction (all 122,817 documents)
        ↓
Stage 3: Heterogeneous Graph Construction
        ↓ (Document nodes + Word nodes + NPMI edges)
Stage 4: GCN Training (transductive learning)
```

---

## 📊 Dataset

- **Total Documents:** 122,817  
- **Categories (10):**  
  art, crime, economy, education, entertainment, global, health, politics, science & technology, sports  
- **Language:** Nepali  

### Split

- **Training:** 10% (12,281 documents)  
- **Validation:** 5% (6,141 documents)  
- **Test:** 85% (104,395 documents)

---

# 🔧 Technical Details

## Stage 1: BERT Classifier

**Model:** `google/muril-base-cased` (Multilingual BERT for Indic languages)

### Architecture

- Freeze embeddings + first **8/12** BERT layers  
- Fine-tune last **4 layers**
- Classification head: `768 → 256 → 128 → 10`

### Training

- Batch size: 8  
- Learning rate: 1e-5  
- Epochs: 20 (early stopping at epoch 12)  
- Trainable parameters: **29.2M (12.3% of total)**  

**Result:** 87.77% test accuracy

---

## Stage 2: Embedding Extraction

- Extract **768-dimensional CLS embeddings**
- For all **122,817 documents**
- Processing time: **29 minutes**

**Output:**

```
node_features.npy   # Shape: (122,817 × 768)
```

---

## Stage 3: Graph Construction

### Heterogeneous Graph

### Nodes (142,817 total)

- 122,817 document nodes (BERT embeddings)
- 20,000 word nodes (zero-initialized)

### Edges (35.3M total)

- Word-Word: 2.5M edges (NPMI > 0.2)
- Document-Word: 30.4M edges (TF-IDF)

### NPMI Parameters

- Window size: 15  
- Minimum word frequency: 3  
- Minimum co-occurrence: 3  
- Threshold: 0.2  

### Graph Normalization

D^(-1/2) A D^(-1/2)

---

## Stage 4: GCN Training

### Architecture

- 2-layer GCN  
- `768 → 256 → 10`
- Dropout: 0.5  
- Parameters: 199K  

### Mini-batch Sampling

- Batch size: 512 documents  
- Neighbor sampling: `[10, 5]`  
- Learning rate: 0.01  

### Training

- Epochs: 200 (best at epoch 90)  
- ~19 seconds per epoch  
- Validation F1: 92.22%  

**Final Result:** 95.68% test accuracy

---

# 📈 Per-Category Performance

| Category | BERT F1 | GCN F1 | Improvement |
|------------|----------|----------|-------------|
| Art | 76.11% | 82.38% | +6.27% |
| Crime | 93.83% | 98.75% | +4.92% |
| Economy | 87.43% | 96.44% | +9.01% |
| Education | 90.14% | 97.63% | +7.49% |
| Entertainment | 81.59% | 88.07% | +6.48% |
| Global | 87.61% | 97.76% | +10.15% |
| Health | 86.71% | 98.00% | +11.29% |
| Politics | 89.37% | 98.31% | +8.94% |
| Science & Tech | 83.49% | 95.74% | +12.25% |
| Sports | 97.13% | 99.04% | +1.91% |

---

# 🚀 Installation

```bash
git clone https://github.com/yourusername/bert-gcn-nepali-news.git
cd bert-gcn-nepali-news

python -m venv .venv
source .venv/bin/activate

pip install torch transformers scikit-learn scipy pandas numpy tqdm
```

---

# 💻 Usage

## 1️⃣ BERT Training

```bash
python train_bert.py
```

Outputs:
- `best_bert_only_model.pt`
- `train_mask.npy`
- `val_mask.npy`
- `test_mask.npy`

---

## 2️⃣ Extract Embeddings

```bash
python extract_embeddings.py
```

Outputs:
- `node_features.npy`
- `labels.npy`
- `embedding_metadata.json`

---

## 3️⃣ Build Graph

```bash
python build_graph.py
```

Outputs:
- `heterogeneous_adj.npz`
- `heterogeneous_features.npy`
- `heterogeneous_labels.npy`
- `node_mapping.json`

---

## 4️⃣ Train GCN

```bash
python train_gcn.py
```

Outputs:
- `best_gcn_minibatch.pt`
- `gcn_minibatch_results.json`

---

# 📊 Experimental Setup

- All 122,817 documents have labels
- 85% labels hidden using boolean masks
- Test labels used only for final evaluation
- Standard semi-supervised benchmarking protocol
