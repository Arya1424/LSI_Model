# Legal Statute Identification (LSI)

*A hybrid NLP and linguistic feature engineering framework for Indian legal statute prediction*

---

## Overview

This repository presents an enhanced framework for **Legal Statute Identification (LSI)**, extending the LeSICiN baseline with targeted improvements in linguistic modeling and inference efficiency.

The system integrates **phrase normalization**, **stylistic feature engineering**, and **efficient retrieval mechanisms** to improve both predictive performance and runtime efficiency.

- **Macro-F1 (Test):** 39.79%  
- **Relative Improvement:** ~11.34% over LeSICiN baseline  
- **Inference Speedup:** ~6.7×  

---

## Key Contributions

### 1. Trie-Based Phrase Normalization

Legal concepts often appear in multiple surface forms (e.g., *“attempt to murder”*, *“trying to kill”*).  
A Trie-based annotator maps such variations to canonical tags (e.g., `<IPC_307>`), reducing vocabulary fragmentation and improving semantic consistency.

---

### 2. Orwellian Clarity Features

Inspired by principles of clear writing, the model incorporates lightweight linguistic features to capture document style:

- Average Sentence Length  
- Lexical Density  
- Passive Voice Ratio  

These features complement semantic embeddings and improve interpretability.

---

### 3. KD-Tree Accelerated Retrieval

To reduce inference cost, a **KD-Tree** is built over statute embeddings, enabling sublinear retrieval:

- Reduces complexity from **O(N)** to **O(log N + k)**  
- Enables real-time inference (<30 ms per case)  

---

## Dataset

The model is evaluated on the **Indian Legal Statute Identification (ILSI)** dataset:

- **Total documents:** 66,090  
- **Number of statutes:** 100 (most frequent IPC sections)  
- **Average document length:** 1232 words  
- **Average labels per case:** 3.78  

### Splits

- **Train:** 42,884  
- **Validation:** 10,203  
- **Test:** 13,043  

### Dataset Link

> Add dataset link here: **[[Dataset URL](https://zenodo.org/records/6053791)]**

---

## Model Architecture

### Preprocessing
- Trie-based phrase annotation  
- Tokenization and sentence segmentation  
- Orwell feature extraction (3-dimensional vector)

### Encoding
- Hierarchical BiLSTM with Attention (HAN)  
- Style feature projection and concatenation  
- Final document embedding generation  

### Training
- Binary Cross-Entropy with logits  
- Adam optimizer  
- Learning rate scheduling and early stopping  

### Inference Pipeline
1. Encode input document into embedding  
2. Retrieve top-k statute candidates using KD-Tree  
3. Score candidates  
4. Apply threshold (τ = 0.65)  

---

## Results

| Split          | Loss   | Macro-P | Macro-R | Macro-F1 |
|----------------|--------|--------|--------|----------|
| Train          | 0.0140 | —      | —      | 0.7126   |
| Validation     | 0.0154 | 0.4007 | 0.3590 | 0.3756   |
| Test           | 0.0088 | 0.4222 | 0.3826 | 0.3979   |

### Highlights

- Significant improvement over LeSICiN baseline  
- Balanced precision and recall  
- Reduced inference time with minimal accuracy trade-off  

---

## Repository Structure
configs/
│── data_path.json
└── hyperparams.json

model/
│── basicmodules.py
│── model.py
└── submodules.py

modules/
│── kdtree_retrieval.py
│── orwell_simplifier.py
│── trie_annotator.py
└── data_helper.py

helper.py
run.py


---

## Limitations

- Phrase dictionary currently limited (~24 legal concepts)  
- Passive voice detection relies on heuristic rules and may miss edge cases  
- Model trained on Indian legal data; cross-jurisdiction generalization is limited  
- Requires periodic retraining to account for evolving legal language  

---

## Contributors

- **Aryasree M**  
- **Daksha P Jain**  
- **Shristi Bose**  
- **Jahnavi Raja**  
