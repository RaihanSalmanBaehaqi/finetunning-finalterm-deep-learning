# 🤗 Fine-tuning BERT for Natural Language Inference (NLI)

## 📋 Overview

Repository ini berisi implementasi **fine-tuning BERT** untuk task **Natural Language Inference (NLI)** menggunakan dataset **MNLI (Multi-Genre NLI)**. Task NLI bertujuan untuk menentukan hubungan logika antara dua kalimat: **premise** dan **hypothesis**.

| Item | Detail |
|------|--------|
| **Model** | `bert-base-uncased` |
| **Dataset** | MNLI (Multi-Genre NLI) |
| **Task** | 3-class Classification |
| **Labels** | Entailment, Neutral, Contradiction |

---

## 👤 Identitas

| Field | Value |
|-------|-------|
| **Nama** | [Nama Lengkap Anda] |
| **NIM** | [NIM Anda] |
| **Kelas** | TK-46-02 |
| **Mata Kuliah** | Deep Learning |

---

## 📚 Dataset: MNLI

### Tentang MNLI

**Multi-Genre Natural Language Inference (MNLI)** adalah dataset benchmark untuk NLI yang berisi pasangan kalimat dari berbagai genre teks.

### Label Categories

| Label | ID | Description | Example |
|-------|----|-----------| --------|
| **Entailment** | 0 | Hypothesis logically follows from premise | P: "A man is playing guitar" → H: "Someone is making music" |
| **Neutral** | 1 | Hypothesis might be true, but not guaranteed | P: "A man is playing guitar" → H: "The man is a professional musician" |
| **Contradiction** | 2 | Hypothesis contradicts premise | P: "A man is playing guitar" → H: "Nobody is playing any instrument" |

### Data Splits

| Split | Samples | Description |
|-------|---------|-------------|
| Train | 392,702 | Training data |
| Validation (Matched) | 9,815 | Same genres as training |
| Validation (Mismatched) | 9,832 | Different genres from training |

---

## 🏗️ Model Architecture

- **Base Model:** `bert-base-uncased`
- **Architecture:** Encoder-only (Bidirectional)
- **Input:** Two sentences (premise + hypothesis)
- **Output:** 3-class classification

### Input Format

```
[CLS] premise [SEP] hypothesis [SEP]
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Max Length | 256 |
| Epochs | 3 |
| Learning Rate | 2e-5 |
| Train Batch Size | 16 |
| Eval Batch Size | 32 |
| Weight Decay | 0.01 |
| Optimizer | AdamW |
| FP16 | Enabled (CUDA) |

---

## 📊 Results

### Validation Metrics

| Split | Loss | Accuracy | Macro-F1 |
|-------|------|----------|----------|
| Matched | - | ~84% | ~84% |
| Mismatched | - | ~84% | ~84% |

> **Note:** Hasil akan diupdate setelah training selesai.

---

## 📁 Repository Structure

```
finetuning-bert-nli/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── notebooks/
│   └── finetune_bert_mnli.ipynb       # MNLI training notebook
├── reports/
│   └── report_mnli.md                 # Experiment report
└── models/
    └── (saved model files)            # Best model checkpoints
```

---

## 🚀 How to Run

### 1. Clone Repository
```bash
git clone https://github.com/[username]/finetuning-bert-nli.git
cd finetuning-bert-nli
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run on Google Colab

1. Upload notebook ke Google Colab
2. Mount Google Drive
3. Set project directory:
```python
PROJECT_DIR = "/content/drive/MyDrive/finetuning-bert-nli"
```
4. Run all cells

---

## 🔍 Key Differences from Text Classification

| Aspect | Text Classification | NLI |
|--------|--------------------|----|
| **Input** | Single text | Two sentences (premise + hypothesis) |
| **Tokenization** | `tokenizer(text)` | `tokenizer(premise, hypothesis)` |
| **Max Length** | 128 | 256 (longer for 2 sentences) |
| **Semantic Task** | Topic/Emotion | Logical relationship |

---

## 📓 Notebook Contents

| Section | Description |
|---------|-------------|
| 0-1 | Mount Drive & Setup |
| 2-3 | Install & Import |
| 4 | Configuration |
| 5-6 | Load Dataset & Inspect |
| 7 | Tokenization (Sentence Pairs) |
| 8-9 | Data Collator & Metrics |
| 10-11 | Load Model & Trainer Setup |
| 12 | Training |
| 13 | Evaluation (Matched & Mismatched) |
| 14-15 | Save Model & Analysis |
| 16 | Inference Demo |

---

## 🔗 Related Repositories

- **Task 1 (Classification):** `finetuning-bert-text-classification` - AG News & GoEmotions
- **Task 2:** `finetuning-t5-question-answering` - T5 for QA
- **Task 3:** `finetuning-phi2-text-summarization` - Phi-2 for Summarization

---

## 📜 License

This project is created for educational purposes as part of Deep Learning course final exam (UAS).
