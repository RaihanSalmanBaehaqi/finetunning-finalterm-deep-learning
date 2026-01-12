# 🎓 UAS Deep Learning - Task 1: Fine-tuning BERT for Text Understanding

<div align="center">

![BERT](https://img.shields.io/badge/Model-BERT--base--uncased-blue?style=for-the-badge&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-Educational-red?style=for-the-badge)

**Fine-tuning Transformer Encoder (BERT) untuk berbagai task NLU**

[📰 Text Classification](#-text-classification) • [😊 Emotion Detection](#-emotion-detection) • [🔗 NLI](#-natural-language-inference) • [📊 Results](#-results-summary)

</div>

---

## 📋 Overview

Repository ini berisi implementasi lengkap **Task 1 UAS Deep Learning** yang mengeksplorasi arsitektur **Transformer Encoder (BERT)** untuk menyelesaikan berbagai task **Natural Language Understanding (NLU)**:

| # | Task | Dataset | Type | Classes | Best Metric |
|---|------|---------|------|---------|-------------|
| 1 | 📰 News Classification | AG News | Multi-class | 4 | **94.75% Accuracy** |
| 2 | 😊 Emotion Detection | GoEmotions | Multi-label | 28 | **57.49% Micro-F1** |
| 3 | 🔗 Natural Language Inference | MNLI | 3-class NLI | 3 | **84.67% Accuracy** |

---

## 👤 Identitas Tim

* RAIHAN SALMAN BAEHAQI (1103220180)
* JAKA KELANA WIJAYA (1103223048)

---

## 🎯 Highlights & Achievements

| Achievement | Value | Status |
|-------------|-------|--------|
| 🏆 AG News Accuracy | **94.75%** | Exceeds BERT paper (94.2%) |
| 🏆 GoEmotions Micro-F1 | **57.49%** | Matches benchmark (58.0%) |
| 🏆 MNLI Matched Accuracy | **84.67%** | Matches BERT paper (84.6%) |
| 🏆 MNLI Mismatched Accuracy | **84.74%** | Exceeds BERT paper (83.4%) |
| ⚡ Total Training Time | ~159 minutes | On Tesla T4 GPU |
| 📊 Total Samples Trained | ~544,000 | Across all datasets |
| 🔧 Model Parameters | ~109M | BERT-base-uncased |

---

## 📁 Repository Structure

```
task-1/
│
├── 📄 README.md                                    # This file
│
├── 📂 finetuning-bert-text-classification/         # Text Classification Tasks
│   ├── 📄 README.md                                # Project documentation
│   ├── 📄 requirements.txt                         # Dependencies
│   ├── 📓 notebooks/
│   │   ├── finetune_bert_ag_news.ipynb            # AG News (Multi-class)
│   │   └── finetune_bert_go_emotions.ipynb        # GoEmotions (Multi-label)
│   ├── 📊 reports/
│   │   ├── report-ag-news                         # AG News experiment report
│   │   │    └── report_ag_news.md  
│   │   └── report-go-emotions                     # GoEmotions experiment report
│   │         └── report_go_emotions.md 
│   └── 🤖 models/                                  # Saved model checkpoints
│
└── 📂 finetuning-bert-nli/                         # NLI Task
    ├── 📄 README.md                                # Project documentation
    ├── 📄 requirements.txt                         # Dependencies
    ├── 📓 notebooks/
    │   └── finetune_bert_mnli.ipynb               # MNLI (3-class NLI)
    ├── 📊 reports/
    │   └── report_mnli.md                         # MNLI experiment report
    └── 🤖 models/                                  # Saved model checkpoints
```

---

## 📚 Tasks Overview

### 📰 Task 1A: AG News Classification

**Multi-class Text Classification** - Klasifikasi artikel berita ke 4 kategori topik.

| Property | Value |
|----------|-------|
| **Dataset** | AG News (`sh0416/ag_news`) |
| **Task Type** | Multi-class (Single-label) |
| **Classes** | 4 (World, Sports, Business, Sci/Tech) |
| **Train/Val/Test** | 108K / 12K / 7.6K |
| **Loss Function** | CrossEntropyLoss |
| **Prediction** | argmax(logits) |

**Results:**

| Metric | Validation | Test |
|--------|------------|------|
| Accuracy | 94.83% | **94.75%** |
| Macro-F1 | 94.81% | **94.76%** |

**Per-Class F1:** Sports (98.87%) > World (95.94%) > Sci/Tech (92.38%) > Business (91.83%)

---

### 😊 Task 1B: GoEmotions Detection

**Multi-label Emotion Classification** - Deteksi multiple emosi dalam teks Reddit.

| Property | Value |
|----------|-------|
| **Dataset** | GoEmotions (`google-research-datasets/go_emotions`) |
| **Task Type** | Multi-label |
| **Classes** | 28 emotions |
| **Train/Val/Test** | 43K / 5.4K / 5.4K |
| **Loss Function** | BCEWithLogitsLoss |
| **Prediction** | sigmoid(logits) > 0.5 |

**Results:**

| Metric | Validation | Test |
|--------|------------|------|
| Micro-F1 | 57.43% | **57.49%** |
| Macro-F1 | 39.91% | **39.50%** |

**Top Performers:** gratitude (91.5%), amusement (81.0%), love (80.9%), admiration (71.1%)

**⚠️ Challenge:** 5 rare emotions with F1=0% (grief, pride, relief, nervousness, embarrassment)

---

### 🔗 Task 1C: MNLI (Natural Language Inference)

**3-class NLI** - Menentukan hubungan logika antara premise dan hypothesis.

| Property | Value |
|----------|-------|
| **Dataset** | MNLI (`nyu-mll/glue`, config: `mnli`) |
| **Task Type** | 3-class Classification |
| **Classes** | Entailment, Neutral, Contradiction |
| **Train/Val** | 393K / 9.8K (matched) + 9.8K (mismatched) |
| **Loss Function** | CrossEntropyLoss |
| **Input Format** | `[CLS] premise [SEP] hypothesis [SEP]` |

**Results:**

| Split | Accuracy | Macro-F1 |
|-------|----------|----------|
| Matched | **84.67%** | **84.63%** |
| Mismatched | **84.74%** | **84.69%** |

**Per-Class F1:** Entailment (86.90%) > Contradiction (86.36%) > Neutral (80.63%)

---

## 📊 Results Summary

### Comparison with Benchmarks

| Task | Our Result | Benchmark | Delta | Status |
|------|------------|-----------|-------|--------|
| AG News | 94.75% | 94.2% (BERT paper) | **+0.55%** | ✅ Exceeds |
| GoEmotions | 57.49% | 58.0% (Paper) | -0.51% | ✅ Matches |
| MNLI-Matched | 84.67% | 84.6% (BERT paper) | **+0.07%** | ✅ Matches |
| MNLI-Mismatched | 84.74% | 83.4% (BERT paper) | **+1.34%** | ✅ Exceeds |

### Training Statistics

| Task | Dataset Size | Training Time | Final Loss |
|------|--------------|---------------|------------|
| AG News | 120K | ~35 min | 0.1747 |
| GoEmotions | 54K | ~10.5 min | 0.1115 |
| MNLI | 393K | ~113 min | 0.3825 |
| **Total** | **~567K** | **~159 min** | - |

---

## ⚙️ Training Configuration

### Common Hyperparameters

| Parameter | AG News | GoEmotions | MNLI |
|-----------|---------|------------|------|
| Model | bert-base-uncased | bert-base-uncased | bert-base-uncased |
| Max Length | 128 | 128 | 256 |
| Epochs | 3 | 3 | 3 |
| Learning Rate | 2e-5 | 2e-5 | 2e-5 |
| Batch Size (Train) | 16 | 16 | 16 |
| Batch Size (Eval) | 32 | 32 | 32 |
| Weight Decay | 0.01 | 0.01 | 0.01 |
| Warmup Ratio | 0.1 | 0.1 | 0.1 |
| Optimizer | AdamW | AdamW | AdamW |
| FP16 | ✅ | ✅ | ✅ |
| Seed | 42 | 42 | 42 |

### Key Implementation Differences

| Aspect | Single-Label (AG News, MNLI) | Multi-Label (GoEmotions) |
|--------|------------------------------|--------------------------|
| Label Encoding | Integer | Multi-hot (float32) |
| Loss Function | CrossEntropyLoss | BCEWithLogitsLoss |
| Activation | Softmax | Sigmoid |
| Prediction | argmax | threshold > 0.5 |
| problem_type | single_label_classification | multi_label_classification |

---

## 🚀 How to Run

### Prerequisites

- Python 3.8+
- Google Colab (recommended) or local GPU
- ~16GB GPU memory (Tesla T4 sufficient)

### Option 1: Google Colab (Recommended) ⭐

1. **Upload folder** ke Google Drive:
   ```
   My Drive/
   ├── finetuning-bert-text-classification/
   └── finetuning-bert-nli/
   ```

2. **Open notebook** di Google Colab

3. **Enable GPU runtime:**
   ```
   Runtime → Change runtime type → GPU (T4)
   ```

4. **Mount Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

5. **Run all cells:**
   ```
   Runtime → Run all (Ctrl+F9)
   ```

### Option 2: Local Environment

```bash
# Clone/download repository
cd task-1

# Install dependencies
pip install -r finetuning-bert-text-classification/requirements.txt
pip install -r finetuning-bert-nli/requirements.txt

# Run Jupyter
jupyter notebook
```

### Execution Order (Recommended)

| Order | Notebook | Time | Notes |
|-------|----------|------|-------|
| 1️⃣ | `finetune_bert_ag_news.ipynb` | ~35 min | Start here (simplest) |
| 2️⃣ | `finetune_bert_go_emotions.ipynb` | ~10.5 min | Multi-label complexity |
| 3️⃣ | `finetune_bert_mnli.ipynb` | ~113 min | Longest, run last |

---

## 🔑 Key Learnings

### 1. Single-Label vs Multi-Label Classification

```python
# Single-Label (AG News, MNLI)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4  # or 3 for MNLI
)
prediction = torch.argmax(logits, dim=-1)

# Multi-Label (GoEmotions)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=28,
    problem_type="multi_label_classification"  # CRITICAL!
)
prediction = (torch.sigmoid(logits) > 0.5).int()
```

### 2. Sentence Pair Input (NLI)

```python
# Text Classification: single text
tokenizer(text, truncation=True, max_length=128)

# NLI: two sentences
tokenizer(premise, hypothesis, truncation=True, max_length=256)
# Result: [CLS] premise [SEP] hypothesis [SEP]
```

### 3. Class Imbalance Handling

- **AG News:** Perfectly balanced → Standard training works well
- **GoEmotions:** Severely imbalanced → Consider weighted loss, threshold tuning
- **MNLI:** Perfectly balanced → Standard training works well

---

## 📈 Visualizations

### AG News - Confusion Matrix

| | World | Sports | Business | Sci/Tech |
|--|-------|--------|----------|----------|
| **World** | 1,807 | 7 | 49 | 37 |
| **Sports** | 3 | 1,883 | 8 | 6 |
| **Business** | 37 | 5 | 1,759 | 99 |
| **Sci/Tech** | 20 | 15 | 117 | 1,748 |

### GoEmotions - Performance Tiers

| Tier | F1 Range | Emotions |
|------|----------|----------|
| 🟢 Excellent | 70-92% | gratitude, amusement, love, admiration |
| 🟡 Good | 50-70% | neutral, fear, joy, remorse, optimism, sadness, surprise |
| 🟠 Moderate | 30-50% | anger, curiosity, desire, disgust, approval, caring |
| 🔴 Poor | 0-30% | annoyance, disappointment, realization |
| ⚫ Zero | 0% | embarrassment, grief, nervousness, pride, relief |

### MNLI - Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Entailment | 89.69% | 84.28% | 86.90% |
| Neutral | 78.36% | 83.03% | 80.63% |
| Contradiction | 86.04% | 86.68% | 86.36% |

---

## 🛠️ Technologies Used

| Technology | Purpose | Version |
|------------|---------|---------|
| PyTorch | Deep learning framework | 2.0+ |
| Transformers | Pre-trained BERT models | 4.35+ |
| Datasets | HuggingFace datasets | 2.14+ |
| Evaluate | Metrics computation | 0.4+ |
| Accelerate | Training optimization | 0.24+ |
| scikit-learn | Classification metrics | 1.3+ |
| Matplotlib/Seaborn | Visualization | - |
| Google Colab | GPU environment | Tesla T4 |

---

## 📝 Reports & Documentation

| Document | Description | Location |
|----------|-------------|----------|
| AG News Report | Detailed analysis & results | `finetuning-bert-text-classification/reports/report_ag_news.md` |
| GoEmotions Report | Multi-label analysis | `finetuning-bert-text-classification/reports/report_go_emotions.md` |
| MNLI Report | NLI analysis & benchmarks | `finetuning-bert-nli/reports/report_mnli.md` |

---

## 💡 Potential Improvements

| Task | Improvement | Expected Impact |
|------|-------------|-----------------|
| AG News | Use RoBERTa-base | +0.3-0.5% accuracy |
| GoEmotions | Per-class threshold tuning | +5-10% Macro-F1 |
| GoEmotions | Weighted BCELoss | +3-5% on rare classes |
| MNLI | Use DeBERTa-v3 | +5-6% accuracy |
| All | Ensemble models | +1-2% overall |

---

## 🔗 Related Tasks

| Task | Repository | Model | Status |
|------|------------|-------|--------|
| **Task 1** | This repository | BERT | ✅ Complete |
| Task 2 | `finetuning-t5-question-answering` | T5 | 📝 Pending |
| Task 3 | `finetuning-phi2-text-summarization` | Phi-2 | 📝 Pending |

---

## 📚 References

1. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
2. Zhang, X., et al. (2015). "Character-level Convolutional Networks for Text Classification" (AG News)
3. Demszky, D., et al. (2020). "GoEmotions: A Dataset of Fine-Grained Emotions"
4. Williams, A., et al. (2018). "A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference" (MNLI)
5. HuggingFace Transformers Documentation

---

## 📜 License

This project is created for **educational purposes** as part of Deep Learning course final exam (UAS) at Telkom University.

---

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/) for pre-trained models and datasets
- [Google Colab](https://colab.research.google.com/) for free GPU resources
- Course instructors for guidance and support
- Original paper authors for benchmark datasets

---
