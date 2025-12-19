
# ⚖️ Enhanced Legal Statute Identification (LSI)

*A hybrid NLP + linguistic-engineering model for Indian legal statute prediction*


## 📌 Overview

This repository contains an improved framework for **Legal Statute Identification (LSI)** that enhances the LeSICiN baseline through three major innovations:

1. **Trie-based Legal Phrase Normalization**
2. **Orwellian Linguistic Clarity Features**
3. **KD-Tree Accelerated Statute Retrieval**

The model achieves a **Test Macro-F1 of 39.79%**, a *~11.34% relative improvement* over the widely-used LeSICiN baseline (28.45%), while reducing inference time by **6.7×**.

---

## ✨ Key Contributions

### 🔹 1. Trie-Based Phrase Normalization

Legal terminology often appears in diverse forms (e.g., *“attempt to murder”*, *“trying to kill”*).
A custom Trie maps these surface variations to canonical tags such as `<IPC_307>`, reducing vocabulary noise and improving concept consistency.

### 🔹 2. Orwellian Clarity Features

Inspired by George Orwell’s principles on clear writing, three stylistic features are computed:

* **Average Sentence Length**
* **Lexical Density**
* **Passive Voice Ratio**

These capture document complexity and add interpretability to the model.

### 🔹 3. KD-Tree Accelerated Retrieval

Instead of evaluating similarity with all 100+ statutes at inference, we use a **KD-Tree** over statute embeddings to retrieve the top-k candidates in **O(log N)** time.
➡️ This enables real-time inference (<30 ms per case).

---

## 🏛️ Dataset

The model is evaluated on the **Indian Legal Statute Identification (ILSI)** dataset:

* **66,090** criminal case documents
* **100** most frequently cited IPC sections
* **Avg document length:** 1232 words
* **Avg labels per case:** 3.78

### Splits

* **Train:** 42,884
* **Validation:** 10,203
* **Test:** 13,043

---

## 📐 Model Architecture

The pipeline integrates textual, structural, and linguistic signals:

### 📝 Preprocessing

* Trie-based annotation of key legal phrases
* Tokenization + sentence segmentation
* Orwell feature extraction (3-dim vector → projected to d/4)

### 🧠 Encoding

* Hierarchical BiLSTM + Attention (HAN)
* Style vector concatenation
* Final document embedding

### 🎯 Training

* Binary cross-entropy with logits
* Adam optimizer
* Early stopping and LR scheduling

### ⚡ Inference

1. Encode document → embedding
2. Query KD-Tree for top-k statute candidates
3. Score candidates
4. Apply threshold (τ=0.65) for final predictions

---

## 📊 Results

| Split          | Loss   | Macro-P | Macro-R | Macro-F1   |
| -------------- | ------ | ------- | ------- | ---------- |
| **Train**      | 0.0140 | –       | –       | **0.7126** |
| **Validation** | 0.0154 | 0.4007  | 0.3590  | **0.3756** |
| **Test**       | 0.0088 | 0.4222  | 0.3826  | **0.3979** |

### 🔍 Highlights

* **+39.86% relative improvement** over LeSICiN
* **6.7× faster inference** via KD-Tree retrieval
* Strong precision–recall balance
* Minimal overfitting

---

## 📂 Repository Structure

```
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

```

---

## 🚀 Getting Started

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the model

```bash
python run.py
```
---


## ⚠️ Limitations

* Phrase dictionary (24 legal concepts) requires ongoing expansion
* Passive voice detection uses heuristics → may miss edge cases
* Trained on Indian legal text → cross-jurisdiction generalization unclear
* Requires periodic retraining due to legal language drift

---
---

## 🤝 Contributors

* **Aryasree M**
* **Daksha P Jain**
* **Shristi Bose**
* **Jahnavi Raja**

---
