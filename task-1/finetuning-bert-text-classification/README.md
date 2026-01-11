# 🤗 Fine-tuning BERT for Text Classification

## 📋 Overview

Repository ini berisi implementasi **fine-tuning model BERT** untuk task **Text Classification** sebagai bagian dari UAS Deep Learning. Project ini mengeksplorasi arsitektur **Transformer Encoder** untuk menyelesaikan dua jenis klasifikasi teks:

1. **Multi-class Classification** (AG News) - Klasifikasi berita ke 4 kategori
2. **Multi-label Classification** (GoEmotions) - Deteksi multi-emosi dalam teks

> **Note:** Task NLI (MNLI) disubmit di repository terpisah: `finetuning-bert-nli`

---

## 👤 Identitas

| Field | Value |
|-------|-------|
| **Nama** | [Nama Lengkap Anda] |
| **NIM** | [NIM Anda] |
| **Kelas** | TK-46-02 |
| **Mata Kuliah** | Deep Learning |

---

## 📚 Datasets

### 1. AG News (Multi-class)
- **Source:** `sh0416/ag_news` (HuggingFace)
- **Task:** 4-class news topic classification
- **Labels:** World (0), Sports (1), Business (2), Sci/Tech (3)
- **Size:** 120,000 train / 7,600 test

### 2. GoEmotions (Multi-label)
- **Source:** `google-research-datasets/go_emotions` (HuggingFace)
- **Task:** Multi-label emotion detection
- **Labels:** 28 emotion categories
- **Size:** ~58,000 samples

---

## 🏗️ Model Architecture

- **Base Model:** `bert-base-uncased`
- **Framework:** HuggingFace Transformers
- **Architecture Type:** Encoder-only (Bidirectional)

---

## ⚙️ Training Configuration

| Parameter | AG News | GoEmotions |
|-----------|---------|------------|
| Max Length | 128 | 128 |
| Epochs | 3 | 3 |
| Learning Rate | 2e-5 | 2e-5 |
| Batch Size (Train) | 16 | 16 |
| Batch Size (Eval) | 32 | 32 |
| Weight Decay | 0.01 | 0.01 |
| Optimizer | AdamW | AdamW |
| FP16 | ✅ (if CUDA) | ✅ (if CUDA) |

---

## 📊 Results

### AG News (Multi-class Classification)

| Split | Loss | Accuracy | Macro-F1 |
|-------|------|----------|----------|
| Validation | - | ~94% | ~94% |
| Test | - | ~94% | ~94% |

### GoEmotions (Multi-label Classification)

| Split | Loss | Micro-F1 | Macro-F1 |
|-------|------|----------|----------|
| Validation | - | ~50% | ~40% |
| Test | - | ~50% | ~40% |

> **Note:** Hasil akan diupdate setelah training selesai.

---

## 📁 Repository Structure

```
finetuning-bert-text-classification/
├── README.md                              # Project documentation
├── requirements.txt                       # Python dependencies
├── notebooks/
│   ├── finetune_bert_ag_news.ipynb       # AG News training notebook
│   └── finetune_bert_go_emotions.ipynb   # GoEmotions training notebook
├── reports/
│   ├── report_ag_news.md                 # AG News experiment report
│   └── report_go_emotions.md             # GoEmotions experiment report
└── models/
    └── (saved model files)               # Best model checkpoints
```

---

## 🚀 How to Run

### 1. Clone Repository
```bash
git clone https://github.com/[username]/finetuning-bert-text-classification.git
cd finetuning-bert-text-classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run on Google Colab (Recommended)

1. Upload notebook ke Google Colab
2. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Set project directory:
```python
PROJECT_DIR = "/content/drive/MyDrive/finetuning-bert-text-classification"
```

4. Run all cells

---

## 📓 Notebooks

| Notebook | Description | Dataset |
|----------|-------------|---------|
| `finetune_bert_ag_news.ipynb` | Multi-class text classification | AG News |
| `finetune_bert_go_emotions.ipynb` | Multi-label emotion detection | GoEmotions |

---

## 🔍 Key Implementation Details

### AG News (Single-Label)
- Uses `CrossEntropyLoss` (automatic)
- Prediction: `argmax` of logits
- Metrics: Accuracy, Macro-F1

### GoEmotions (Multi-Label)
- Uses `BCEWithLogitsLoss`
- Prediction: `sigmoid > 0.5` threshold
- Metrics: Micro-F1, Macro-F1
- Requires one-hot encoding for labels

---

## 📈 Training Pipeline

```
1. Load Dataset (HuggingFace Datasets)
         ↓
2. Preprocessing & Tokenization
         ↓
3. Train/Validation Split
         ↓
4. Load Pre-trained BERT
         ↓
5. Fine-tuning with Trainer API
         ↓
6. Evaluation (Metrics + Confusion Matrix)
         ↓
7. Save Best Model
         ↓
8. Inference Demo
```

---

## 🛠️ Technologies Used

- **PyTorch** - Deep learning framework
- **Transformers** - HuggingFace transformers library
- **Datasets** - HuggingFace datasets library
- **Evaluate** - HuggingFace evaluation metrics
- **scikit-learn** - ML utilities & metrics
- **Accelerate** - Training optimization

---

## 📝 Reports

Detailed experiment reports are available in the `reports/` folder:
- [AG News Report](reports/report_ag_news.md)
- [GoEmotions Report](reports/report_go_emotions.md)

---

## 🔗 Related Repositories

- **Task 1 (NLI):** `finetuning-bert-nli` - BERT for Natural Language Inference
- **Task 2:** `finetuning-t5-question-answering` - T5 for Question Answering
- **Task 3:** `finetuning-phi2-text-summarization` - Phi-2 for Text Summarization

---

## 📜 License

This project is created for educational purposes as part of Deep Learning course final exam (UAS).

---

## 🙏 Acknowledgments

- HuggingFace for providing pre-trained models and datasets
- Google Colab for free GPU resources
- Course instructors for guidance
