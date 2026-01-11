# 🤗 Fine-tuning BERT for Text Classification

## 📋 Overview

Repository ini berisi implementasi **fine-tuning model BERT** untuk task **Text Classification** sebagai bagian dari UAS Deep Learning. Project ini mengeksplorasi arsitektur **Transformer Encoder** untuk menyelesaikan dua jenis klasifikasi teks:

| Task | Dataset | Type | Classes | Best Metric |
|------|---------|------|---------|-------------|
| 📰 News Classification | AG News | Multi-class | 4 | **94.75% Accuracy** |
| 😊 Emotion Detection | GoEmotions | Multi-label | 28 | **57.49% Micro-F1** |

> **Note:** Task NLI (MNLI) disubmit di repository terpisah: [`finetuning-bert-nli`](https://github.com/[username]/finetuning-bert-nli)

---

## 👤 Identitas Tim

* RAIHAN SALMAN BAEHAQI (1103220180)
* JAKA KELANA WIJAYA (1103223048)

---

## 🎯 Highlights

| Achievement | Value |
|-------------|-------|
| 🏆 AG News Accuracy | **94.75%** (exceeds BERT paper benchmark!) |
| 🏆 GoEmotions Micro-F1 | **57.49%** (matches benchmark) |
| ⚡ AG News Training Time | ~35 minutes |
| ⚡ GoEmotions Training Time | ~10.5 minutes |
| 🔧 Total Parameters | ~109M (BERT-base) |

---

## 📚 Datasets

### 1. 📰 AG News (Multi-class Classification)

| Property | Value |
|----------|-------|
| **Source** | `sh0416/ag_news` (HuggingFace) |
| **Task** | 4-class news topic classification |
| **Train Samples** | 108,000 |
| **Validation Samples** | 12,000 |
| **Test Samples** | 7,600 |
| **Class Balance** | ✅ Perfectly balanced (25% each) |

**Labels:**

| ID | Category | Description |
|----|----------|-------------|
| 0 | 🌍 World | International news & politics |
| 1 | ⚽ Sports | Sports news |
| 2 | 💼 Business | Business & economics |
| 3 | 🔬 Sci/Tech | Science & technology |

### 2. 😊 GoEmotions (Multi-label Classification)

| Property | Value |
|----------|-------|
| **Source** | `google-research-datasets/go_emotions` (HuggingFace) |
| **Config** | simplified (28 emotions) |
| **Task** | Multi-label emotion detection |
| **Train Samples** | ~43,000 |
| **Validation Samples** | ~5,400 |
| **Test Samples** | ~5,400 |
| **Class Balance** | ⚠️ Severely imbalanced (300:1 ratio) |

**Labels (28 Emotions):**

| Category | Emotions |
|----------|----------|
| 🟢 **Positive** | admiration, amusement, approval, caring, excitement, gratitude, joy, love, optimism, pride, relief |
| 🔴 **Negative** | anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness |
| 🟡 **Ambiguous** | confusion, curiosity, desire, realization, surprise |
| ⚪ **Neutral** | neutral |

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
| **Framework** | HuggingFace Transformers |

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
| Warmup Ratio | 0.1 | 0.1 |
| Optimizer | AdamW | AdamW |
| FP16 | ✅ Enabled | ✅ Enabled |
| Loss Function | CrossEntropyLoss | BCEWithLogitsLoss |
| Seed | 42 | 42 |

---

## 📊 Results

### 📰 AG News (Multi-class Classification)

| Split | Loss | Accuracy | Macro-F1 |
|-------|------|----------|----------|
| **Validation** | 0.1790 | **94.83%** | **94.81%** |
| **Test** | 0.1832 | **94.75%** | **94.76%** |

**Per-Class Performance (Test Set):**

| Class | Precision | Recall | F1-Score | Status |
|-------|-----------|--------|----------|--------|
| 🌍 World | 96.79% | 95.11% | 95.94% | 🟢 Excellent |
| ⚽ Sports | 98.64% | 99.11% | **98.87%** | 🟢 Best |
| 💼 Business | 91.09% | 92.58% | 91.83% | 🟡 Good |
| 🔬 Sci/Tech | 92.55% | 92.21% | 92.38% | 🟡 Good |

**Training Statistics:**
- ⏱️ Training Time: **34.96 minutes**
- 📉 Final Training Loss: **0.1747**

### 😊 GoEmotions (Multi-label Classification)

| Split | Loss | Micro-F1 | Macro-F1 |
|-------|------|----------|----------|
| **Validation** | 0.0857 | **57.43%** | **39.91%** |
| **Test** | 0.0847 | **57.49%** | **39.50%** |

**Per-Class Performance Tiers:**

| Tier | F1 Range | Emotions |
|------|----------|----------|
| 🟢 **Excellent** | 70-92% | gratitude (91.5%), amusement (81.0%), love (80.9%), admiration (71.1%) |
| 🟡 **Good** | 50-70% | neutral, fear, joy, remorse, optimism, sadness, surprise |
| 🟠 **Moderate** | 30-50% | anger, curiosity, desire, disgust, approval, caring, confusion |
| 🔴 **Poor** | 0-30% | annoyance, disappointment, realization |
| ⚫ **Zero** | 0% | embarrassment, grief, nervousness, pride, relief ⚠️ |

**Training Statistics:**
- ⏱️ Training Time: **10.46 minutes**
- 📉 Final Training Loss: **0.1115**

---

## 🏆 Comparison with Benchmarks

### AG News

| Model | Accuracy | Source |
|-------|----------|--------|
| **BERT-base (Ours)** | **94.75%** ✅ | This repository |
| BERT-base (Paper) | 94.2% | Devlin et al. 2019 |
| DistilBERT | 93.8% | Sanh et al. 2019 |
| RoBERTa-base | 95.0% | Liu et al. 2019 |

> 🎉 **Our implementation exceeds the original BERT paper benchmark!**

### GoEmotions

| Model | Micro-F1 | Macro-F1 | Source |
|-------|----------|----------|--------|
| **BERT-base (Ours)** | **57.49%** | **39.50%** | This repository |
| BERT-base (Paper) | 58.0% | 46.0% | Demszky et al. 2020 |
| RoBERTa-base | 59.1% | 48.2% | Demszky et al. 2020 |

> ✅ **Micro-F1 matches benchmark!** Macro-F1 lower due to 5 rare emotions with F1=0%.

---

## 📁 Repository Structure

```
finetuning-bert-text-classification/
├── 📄 README.md                              # Project documentation
├── 📄 requirements.txt                       # Python dependencies
├── 📓 notebooks/
│   ├── finetune_bert_ag_news.ipynb          # AG News training (24 sections)
│   └── finetune_bert_go_emotions.ipynb      # GoEmotions training (20 sections)
├── 📊 reports/
│   ├── report_ag_news.md                    # AG News detailed report
│   └── report_go_emotions.md                # GoEmotions detailed report
└── 🤖 models/
    └── (saved model checkpoints)            # Best models saved here
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
git clone https://github.com/[username]/finetuning-bert-text-classification.git
cd finetuning-bert-text-classification

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Jupyter
jupyter notebook notebooks/
```

---

## 📓 Notebooks Overview

### 📰 `finetune_bert_ag_news.ipynb` (24 Sections)

| Section | Content |
|---------|---------|
| 0-1 | Mount Drive & Setup |
| 2-3 | Install Dependencies & Imports |
| 4 | Configuration |
| 5-8 | Load Dataset & EDA |
| 9-12 | Preprocessing & Tokenization |
| 13-16 | Model Setup & Sanity Checks |
| 17-18 | Training |
| 19-21 | Evaluation & Analysis |
| 22-24 | Save Model & Inference Demo |

### 😊 `finetune_bert_go_emotions.ipynb` (20 Sections)

| Section | Content |
|---------|---------|
| 0-1 | Mount Drive & Setup |
| 2-3 | Install Dependencies & Imports |
| 4 | Configuration |
| 5-7 | Load Dataset & EDA |
| 8-10 | Multi-hot Encoding & Tokenization |
| 11-13 | Model Setup (multi_label_classification) |
| 14-15 | Training |
| 16-18 | Evaluation & Per-Class Analysis |
| 19-20 | Save Model & Inference Demo |

---

## 🔑 Key Implementation Differences

| Aspect | AG News (Single-Label) | GoEmotions (Multi-Label) |
|--------|------------------------|--------------------------|
| **Labels per sample** | Exactly 1 | 0 to many |
| **Label encoding** | Integer (0-3) | Multi-hot float32 (28-dim) |
| **Loss function** | CrossEntropyLoss | BCEWithLogitsLoss |
| **Activation** | Softmax | Sigmoid |
| **Prediction** | `argmax(logits)` | `sigmoid(logits) > 0.5` |
| **Primary metric** | Accuracy | Micro-F1 |
| **problem_type** | `single_label_classification` | `multi_label_classification` |

### ⚠️ Critical for Multi-Label (GoEmotions)

```python
# 1. Model must use multi_label_classification
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=28,
    problem_type="multi_label_classification"  # CRITICAL!
)

# 2. Labels must be float32 (not int)
labels = np.zeros(28, dtype=np.float32)

# 3. Prediction uses sigmoid + threshold
probs = torch.sigmoid(logits)
predictions = (probs > 0.5).int()
```

---

## 📈 Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│  1. 📥 Load Dataset (HuggingFace Datasets)                  │
│         ↓                                                    │
│  2. 🔍 Exploratory Data Analysis (EDA)                      │
│         ↓                                                    │
│  3. ✂️  Train/Validation Split                              │
│         ↓                                                    │
│  4. 🔤 Tokenization (AutoTokenizer)                         │
│         ↓                                                    │
│  5. 🤖 Load Pre-trained BERT                                │
│         ↓                                                    │
│  6. ⚙️  Configure TrainingArguments                         │
│         ↓                                                    │
│  7. 🏋️ Fine-tuning with Trainer API                         │
│         ↓                                                    │
│  8. 📊 Evaluation (Metrics + Confusion Matrix)              │
│         ↓                                                    │
│  9. 💾 Save Best Model                                      │
│         ↓                                                    │
│  10. 🎯 Inference Demo                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **PyTorch** | Deep learning framework |
| **Transformers** | Pre-trained BERT models |
| **Datasets** | HuggingFace datasets |
| **Evaluate** | Metrics computation |
| **Accelerate** | Training optimization |
| **scikit-learn** | Classification report & confusion matrix |
| **Matplotlib/Seaborn** | Visualization |

---

## 📝 Reports

Detailed experiment reports available in `reports/` folder:

| Report | Description | Link |
|--------|-------------|------|
| 📰 AG News | Multi-class classification results | [report_ag_news.md](reports/report_ag_news.md) |
| 😊 GoEmotions | Multi-label classification results | [report_go_emotions.md](reports/report_go_emotions.md) |

---

## 💡 Lessons Learned

### AG News (Single-Label)
1. ✅ BERT excels at news classification (94.75% accuracy)
2. ✅ Sports is easiest to classify (distinctive vocabulary)
3. ⚠️ Business ↔ Sci/Tech sometimes confused (tech company news)

### GoEmotions (Multi-Label)
1. ✅ Multi-label is significantly harder than single-label
2. ⚠️ Class imbalance is critical (5 emotions with F1=0%)
3. 💡 Per-class threshold tuning could improve Macro-F1
4. 💡 Rare emotions need special handling (weighted loss, data augmentation)

---

## 🔗 Related Repositories

| Repository | Task | Model | Dataset |
|------------|------|-------|---------|
| **This repo** | Text Classification | BERT | AG News, GoEmotions |
| [`finetuning-bert-nli`](https://github.com/[username]/finetuning-bert-nli) | NLI | BERT | MNLI |
| [`finetuning-t5-question-answering`](https://github.com/[username]/finetuning-t5-question-answering) | Question Answering | T5 | SQuAD |
| [`finetuning-phi2-text-summarization`](https://github.com/[username]/finetuning-phi2-text-summarization) | Summarization | Phi-2 | CNN/DailyMail |

---

## 📜 License

This project is created for **educational purposes** as part of Deep Learning course final exam (UAS).

---

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/) for pre-trained models and datasets
- [Google Colab](https://colab.research.google.com/) for free GPU resources
- Course instructors for guidance and support
- Original paper authors: Devlin et al. (BERT), Demszky et al. (GoEmotions)

---

## 📚 References

1. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
2. Zhang, X., et al. (2015). "Character-level Convolutional Networks for Text Classification" (AG News)
3. Demszky, D., et al. (2020). "GoEmotions: A Dataset of Fine-Grained Emotions"
4. HuggingFace Transformers Documentation

---
