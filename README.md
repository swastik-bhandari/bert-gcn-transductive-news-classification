# Semi-Supervised News Classification Using BERT-GCN Architecture

## Executive Summary

This report presents a **transductive semi-supervised** news classification system combining BERT embeddings with Graph Convolutional Networks (GCN).

- **Dataset:** 122,817 Nepali news documents (10 categories)
- **Labeled (mask=True):** 15% (10% train, 5% val)
- **Unlabeled (mask=False during training):** 85%
- **Learning Setup:** Transductive (all documents in graph, only 15% labels used for loss)

### Results

- **BERT-only (Inductive):** 87.77% accuracy  
- **BERT-GCN (Transductive):** 95.68% accuracy  
- **Improvement:** +7.91 percentage points  
- **Weighted F1:** 95.64%  
- **Macro F1:** 95.21%

---

## 1. Experimental Setup

All 122,817 documents have ground-truth labels.

Boolean masks simulate semi-supervised learning:

```python
labels = np.load('labels.npy')

train_mask  # 10%
val_mask    # 5%
test_mask   # 85%
```

- Loss computed only on `train_mask=True`
- Validation on `val_mask=True`
- Test evaluation on `test_mask=True`
- Test documents are included in graph but their labels are hidden during training

### Learning Paradigms

**BERT (Inductive):**
- Trained on 15% labeled data
- Test documents unseen during training

**GCN (Transductive):**
- Graph includes ALL documents
- Uses features of unlabeled documents
- Labels hidden, but features and edges participate in training

---

## 2. Stage 1: BERT Baseline

### Model

- `google/muril-base-cased`
- Freeze embeddings + first 8 layers
- Fine-tune last 4 layers
- Classifier: 768 → 256 → 128 → 10
- Trainable params: 29.2M (12.3%)

### Training

- Train: 12,281 docs (10%)
- Val: 6,141 docs (5%)
- Test: 104,395 docs (85%)
- LR: 1e-5
- Batch size: 8
- Epochs: 20 (early stopping at 12)

### Test Performance

- Accuracy: **87.77%**
- Weighted F1: **87.74%**

Weakest: Art (76.11% F1)  
Strongest: Sports (97.13% F1)

---

## 3. Stage 2: BERT Embedding Extraction

After BERT training, embeddings extracted for **ALL 122,817 documents**.

- CLS embeddings (768-dim)
- Batch size: 32

Outputs:
- `node_features.npy` (122,817 × 768)
- `labels.npy`
- `embedding_metadata.json`

---

## 4. Stage 3: Graph Construction

### Graph Structure

**Nodes:**
- 122,817 document nodes
- 20,000 word nodes
- Total: 142,817

**Edges:**
- Word–Word (NPMI > 0.2): 2,476,469
- Document–Word (TF-IDF): 15,178,639 non-zero entries
- Total edges: 35,453,033
- Density: ~0.17%

### NPMI Settings

- Window size: 15
- Min frequency: 3
- Threshold: 0.2

### Normalization

```
D^(-1/2) A D^(-1/2)
```

### Node Features

- Document nodes: BERT embeddings
- Word nodes: Zero vectors (learned via message passing)

---

## 5. GCN Training (Transductive)

### Setup

- Graph includes ALL documents
- Loss computed only on 10% train nodes
- Neighbor sampling includes unlabeled nodes

```python
train_nodes = np.where(train_mask[:n_docs])[0]
```

### Architecture

- 2-layer GCN
- 768 → 256 → 10
- Dropout: 0.5
- Parameters: 199K

### Sampling

- Batch size: 512
- Neighbors: [10, 5]

### Final Test Results

- Accuracy: **95.68%**
- Weighted F1: **95.64%**
- Macro F1: **95.21%**
- Improvement over BERT: **+7.91 pp**
- Relative error reduction: 59.9%

---

## Per-Category F1 (GCN)

| Category | F1 | Improvement |
|----------|----|------------|
| Art | 82.38% | +6.27 |
| Crime | 98.75% | +4.92 |
| Economy | 96.44% | +9.01 |
| Education | 97.63% | +7.49 |
| Entertainment | 88.07% | +6.48 |
| Global | 97.76% | +10.15 |
| Health | 98.00% | +11.29 |
| Politics | 98.31% | +8.94 |
| Science & Tech | 95.74% | +12.25 |
| Sports | 99.04% | +1.91 |

Largest gains: Science, Health, Global.

---

## 6. Key Insight

The improvement comes primarily from **transductive learning**:

- BERT: learns from 15%, must generalize to unseen 85%
- GCN: learns from 15% labels but uses features and graph structure of all 122,817 documents

GCN sees test document embeddings and their graph connections during training — but not their labels.

---

## 7. Limitations

- Controlled simulation (all documents have labels)
- Transductive (cannot handle new documents without rebuilding graph)
- Static graph
- Sampling approximation

---

## 8. Conclusion

- BERT-GCN achieves **95.68% accuracy** using only 15% labeled data.
- Major improvement comes from **transductive learning**.
- Ideal for fixed corpus classification.
- Less suitable for streaming or real-time systems.

---

## Saved Artifacts

- `best_bert_only_model.pt`
- `node_features.npy`
- `heterogeneous_adj.npz`
- `best_gcn_minibatch.pt`
- `gcn_minibatch_results.json`
