# 🤗 Fine-tuning BERT for Natural Language Inference (NLI)

## 📋 Overview

Repository ini berisi implementasi **fine-tuning BERT** untuk task **Natural Language Inference (NLI)** menggunakan dataset **MNLI (Multi-Genre NLI)**. Task NLI bertujuan untuk menentukan hubungan logika antara dua kalimat: **premise** dan **hypothesis**.

| Item | Detail |
|------|--------|
| **Model** | `bert-base-uncased` |
| **Dataset** | MNLI (Multi-Genre NLI) via GLUE |
| **Task** | 3-class Classification |
| **Labels** | Entailment, Neutral, Contradiction |
| **Best Accuracy** | **84.67%** (Matched) / **84.74%** (Mismatched) |

---

## 🎯 Highlights

| Achievement | Value |
|-------------|-------|
| 🏆 Matched Accuracy | **84.67%** (matches BERT paper!) |
| 🏆 Mismatched Accuracy | **84.74%** (exceeds BERT paper!) |
| 🏆 Macro-F1 | **84.63%** |
| ⚡ Training Time | ~113 minutes |
| 📊 Training Samples | 392,702 |
| 🔧 Total Parameters | ~109M (BERT-base) |

---

## 👤 Identitas Tim

* RAIHAN SALMAN BAEHAQI (1103220180)
* JAKA KELANA WIJAYA (1103223048)

---

## 📚 Dataset: MNLI

### Tentang MNLI

**Multi-Genre Natural Language Inference (MNLI)** adalah dataset benchmark untuk NLI yang berisi pasangan kalimat dari berbagai genre teks, dikembangkan oleh NYU sebagai bagian dari GLUE benchmark.

### Label Categories

| Label | ID | Description | Example |
|-------|----|-----------| --------|
| 🟢 **Entailment** | 0 | Hypothesis MUST be true if premise is true | P: "A man is playing guitar" → H: "Someone is making music" |
| 🟡 **Neutral** | 1 | Hypothesis MIGHT be true (not enough info) | P: "A man is playing guitar" → H: "The man is a professional musician" |
| 🔴 **Contradiction** | 2 | Hypothesis CANNOT be true if premise is true | P: "A man is playing guitar" → H: "Nobody is playing any instrument" |

### Data Splits

| Split | Samples | Description |
|-------|---------|-------------|
| Train | 392,702 | Training data from multiple genres |
| Validation (Matched) | 9,815 | Same genres as training |
| Validation (Mismatched) | 9,832 | Different genres from training |
| Test (Matched) | 9,796 | Hidden labels (GLUE leaderboard) |
| Test (Mismatched) | 9,847 | Hidden labels (GLUE leaderboard) |

### Class Distribution

| Class | Samples | Percentage |
|-------|---------|------------|
| Entailment | ~130,900 | 33.3% |
| Neutral | ~130,900 | 33.3% |
| Contradiction | ~130,900 | 33.3% |

> ✅ Dataset ini **perfectly balanced** - setiap kelas memiliki jumlah sampel yang sama.

### Genres in MNLI

| Type | Genres |
|------|--------|
| **Matched** | Fiction, Government, Telephone, Travel, Letters |
| **Mismatched** | 9/11, Face-to-face, Slate, Verbatim, OUP |

---

## 🗂️ Model Architecture

| Property | Value |
|----------|-------|
| **Base Model** | `bert-base-uncased` |
| **Architecture** | Encoder-only (Bidirectional) |
| **Layers** | 12 |
| **Hidden Size** | 768 |
| **Attention Heads** | 12 |
| **Total Parameters** | ~109M |
| **Input** | Two sentences (premise + hypothesis) |
| **Output** | 3-class classification |

### Input Format

```
[CLS] premise [SEP] hypothesis [SEP]
  ↓      ↓       ↓       ↓        ↓
  0      0       0       1        1   ← Token Type IDs
```

---

## ⚙️ Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max Length | 256 | Accommodate 2 sentences |
| Epochs | 3 | Standard for BERT fine-tuning |
| Learning Rate | 2e-5 | Recommended for BERT |
| Train Batch Size | 16 | Balance speed & memory |
| Eval Batch Size | 32 | Larger for faster eval |
| Weight Decay | 0.01 | Regularization |
| Warmup Ratio | 0.1 | 10% warmup steps |
| Optimizer | AdamW | Standard for transformers |
| FP16 | ✅ Enabled | Mixed precision |
| Seed | 42 | Reproducibility |

---

## 📊 Results

### Validation Metrics

| Split | Loss | Accuracy | Macro-F1 |
|-------|------|----------|----------|
| **Matched** | 0.5522 | **84.67%** | **84.63%** |
| **Mismatched** | 0.5337 | **84.74%** | **84.69%** |

> ✅ Matched ≈ Mismatched menunjukkan **excellent generalization** across genres!

### Per-Class Performance (Validation Matched)

| Class | Precision | Recall | F1-Score | Support | Status |
|-------|-----------|--------|----------|---------|--------|
| 🟢 Entailment | 89.69% | 84.28% | **86.90%** | 3,479 | 🥇 Best |
| 🔴 Contradiction | 86.04% | 86.68% | **86.36%** | 3,213 | 🥈 Great |
| 🟡 Neutral | 78.36% | 83.03% | **80.63%** | 3,123 | 🥉 Hardest |

### Training Statistics

| Metric | Value |
|--------|-------|
| ⏱️ Training Time | **113.06 minutes** |
| 📉 Final Training Loss | **0.3825** |
| 📊 Training Samples | 392,702 |
| 🎯 Best Epoch | 3 |

### Why Neutral is the Hardest?

| Challenge | Explanation |
|-----------|-------------|
| Semantically ambiguous | Neither clearly follows nor contradicts |
| Requires world knowledge | Need to know what's NOT stated |
| Between two extremes | Boundary between entailment & contradiction |

---

## 🏆 Comparison with Benchmarks

| Model | MNLI-Matched | MNLI-Mismatched | Source |
|-------|--------------|-----------------|--------|
| **BERT-base (Ours)** | **84.67%** ✅ | **84.74%** ✅ | This repository |
| BERT-base (Paper) | 84.6% | 83.4% | Devlin et al. 2019 |
| RoBERTa-base | 87.6% | 87.4% | Liu et al. 2019 |
| ALBERT-base | 84.6% | 84.2% | Lan et al. 2020 |
| DistilBERT | 82.2% | 81.5% | Sanh et al. 2019 |
| DeBERTa-v3-base | 90.5% | 90.2% | He et al. 2021 |

> 🎉 **Our implementation matches/exceeds the original BERT paper!**
> - Matched: 84.67% vs 84.6% (+0.07%)
> - Mismatched: 84.74% vs 83.4% (+1.34%)

---

## 📁 Repository Structure

```
finetuning-bert-nli/
├── 📄 README.md                           # Project documentation
├── 📄 requirements.txt                    # Python dependencies
├── 📓 notebooks/
│   └── finetune_bert_mnli.ipynb          # MNLI training notebook (19 sections)
├── 📊 reports/
│   └── report_mnli.md                    # Detailed experiment report
└── 🤖 models/
    └── (saved model checkpoints)         # Best model saved here
```

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended) ⭐

1. **Upload notebook** ke Google Colab
2. **Enable GPU runtime:**
   ```
   Runtime → Change runtime type → GPU (T4)
   ```
3. **Mount Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. **Run all cells:**
   ```
   Runtime → Run all (Ctrl+F9)
   ```

### Option 2: Local Environment

```bash
# 1. Clone repository
git clone https://github.com/[username]/finetuning-bert-nli.git
cd finetuning-bert-nli

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Jupyter
jupyter notebook notebooks/
```

### ⏱️ Expected Training Time

| Hardware | Time |
|----------|------|
| Tesla T4 (Colab) | ~113 minutes |
| RTX 3090 | ~60 minutes |
| CPU only | ~10+ hours ❌ |

---

## 🔑 Key Differences from Text Classification

| Aspect | Text Classification (AG News) | NLI (MNLI) |
|--------|-------------------------------|------------|
| **Input** | Single text | Two sentences |
| **Tokenization** | `tokenizer(text)` | `tokenizer(premise, hypothesis)` |
| **Max Length** | 128 | 256 |
| **Classes** | 4 (topics) | 3 (logical relations) |
| **Task** | What is it about? | Does B follow from A? |
| **Accuracy** | 94.75% | 84.67% |
| **Difficulty** | ⭐⭐ Medium | ⭐⭐⭐ Hard |
| **Reasoning** | Surface patterns | Semantic inference |

### Why NLI is Harder?

1. **Semantic reasoning required** - not just pattern matching
2. **Two inputs** - must understand relationship between sentences
3. **Neutral class** - inherently ambiguous category
4. **World knowledge** - often needed for inference

---

## 📓 Notebook Contents

| Section | Description |
|---------|-------------|
| 0-1 | Mount Drive & Setup Project Directory |
| 2-3 | Install Dependencies & Import Libraries |
| 4 | Configuration (model, hyperparameters) |
| 5-6 | Load Dataset & EDA |
| 7 | Prepare Dataset Splits (train, val_matched, val_mismatched) |
| 8 | Tokenization (Sentence Pairs) ⭐ |
| 9 | Rename Label Column |
| 10 | Data Collator & Metrics |
| 11-12 | Load Model & Sanity Checks |
| 13 | Training Arguments & Trainer |
| 14 | Training (~113 min) 🚀 |
| 15 | Evaluation (Matched & Mismatched) |
| 16 | Detailed Analysis & Confusion Matrix |
| 17 | Save Model |
| 18 | Inference Demo |
| 19 | Summary |

### ⚠️ Critical: Sentence Pair Tokenization

```python
# NLI requires TWO sentences as input
def tokenize_function(batch):
    return tokenizer(
        batch["premise"],      # First sentence
        batch["hypothesis"],   # Second sentence
        truncation=True,
        max_length=256
    )
```

---

## 🎯 Inference Examples

### Example 1: Entailment ✅
```
Premise:    "A man is playing a guitar on stage."
Hypothesis: "Someone is making music."
Prediction: Entailment (92.3% confidence)
```

### Example 2: Neutral 🤷
```
Premise:    "A woman is reading a book."
Hypothesis: "The woman is reading a novel by Stephen King."
Prediction: Neutral (78.5% confidence)
```

### Example 3: Contradiction ❌
```
Premise:    "The restaurant is empty."
Hypothesis: "The restaurant is crowded with customers."
Prediction: Contradiction (96.1% confidence)
```

---

## 💡 Potential Improvements

| Improvement | Expected Impact | Difficulty |
|-------------|-----------------|------------|
| Use RoBERTa-base | +3% accuracy | ⭐ Easy |
| Use DeBERTa-v3-base | +5-6% accuracy | ⭐ Easy |
| Increase epochs to 4 | +0.3-0.5% accuracy | ⭐ Easy |
| Data augmentation | +1-2% accuracy | ⭐⭐ Medium |
| Ensemble models | +1-2% accuracy | ⭐⭐ Medium |

---

## 📝 Report

Detailed experiment report available:
- [📊 report_mnli.md](reports/report_mnli.md) - Complete analysis with confusion matrix, error analysis, and benchmarks

---

## 🔗 Related Repositories

| Repository | Task | Model | Dataset |
|------------|------|-------|---------|
| [`finetuning-bert-text-classification`](https://github.com/[username]/finetuning-bert-text-classification) | Classification | BERT | AG News, GoEmotions |
| **This repo** | NLI | BERT | MNLI |
| [`finetuning-t5-question-answering`](https://github.com/[username]/finetuning-t5-question-answering) | QA | T5 | SQuAD |
| [`finetuning-phi2-text-summarization`](https://github.com/[username]/finetuning-phi2-text-summarization) | Summarization | Phi-2 | CNN/DailyMail |

---

## 📜 License

This project is created for **educational purposes** as part of Deep Learning course final exam (UAS).

---

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/) for pre-trained models and datasets
- [Google Colab](https://colab.research.google.com/) for free GPU resources
- Course instructors for guidance and support

---

## 📚 References

1. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
2. Williams, A., et al. (2018). "A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference" (MNLI)
3. Wang, A., et al. (2018). "GLUE: A Multi-Task Benchmark and Analysis Platform"
4. HuggingFace Transformers Documentation

---
