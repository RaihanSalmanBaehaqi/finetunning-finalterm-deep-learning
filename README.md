# 🧠 Fine-tuning Transformers for NLP Tasks

### Ujian Akhir Semester (UAS) - Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-FFD21E?style=for-the-badge)](https://huggingface.co)
[![Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)

**Eksplorasi Tiga Arsitektur Transformer untuk Natural Language Understanding & Generation**

[📰 Task 1: BERT](#user-content--task-1-bert-text-classification--nli) •
[❓ Task 2: T5](#user-content--task-2-t5-question-answering) •
[📝 Task 3: Phi-2](#user-content--task-3-phi-2-text-summarization) •
[🚀 Quick Start](#user-content--quick-start)

---

## 👤 Identitas Mahasiswa

* RAIHAN SALMAN BAEHAQI (1103220180)
* JAKA KELANA WIJAYA (1103223048)

---

## 📋 Deskripsi Proyek

Repository ini berisi implementasi **komprehensif** untuk Ujian Akhir Semester mata kuliah **Deep Learning** yang mengeksplorasi **tiga arsitektur Transformer berbeda** untuk menyelesaikan berbagai task NLP:

| 🏗️ Architecture | 🤖 Model | 📊 Task | 🎯 Best Result |
|:---------------:|:-------:|:------:|:-------------:|
| **Encoder** | BERT-base | Text Classification & NLI | **94.75%** Accuracy |
| **Encoder-Decoder** | T5-base | Question Answering | **77.59%** F1 Score |
| **Decoder** | Phi-2 | Text Summarization | **7.13%** ROUGE-1 |


```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER ARCHITECTURES                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ENCODER-ONLY          ENCODER-DECODER         DECODER-ONLY            │
│   ┌─────────┐          ┌─────────────────┐      ┌─────────┐             │
│   │ ENCODER │          │ENCODER│ DECODER │      │ DECODER │             │
│   │ (BERT)  │          │  (T5) │  (T5)   │      │ (Phi-2) │             │
│   └────┬────┘          └───┬───┴────┬────┘      └────┬────┘             │
│        │                   │        │                │                   │
│        ▼                   ▼        ▼                ▼                   │
│   Classification      Seq2Seq Generation      Text Generation           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```


## 🏆 Highlights & Pencapaian


| 📰 AG News | 😊 GoEmotions | 🔗 MNLI | ❓ SQuAD | 📝 XSum |
|:----------:|:------------:|:-------:|:--------:|:-------:|
| **94.75%** | **57.49%** | **84.67%** | **77.59%** | **7.13%** |
| Accuracy | Micro-F1 | Accuracy | F1 Score | ROUGE-1 |
| ✅ Exceeds Benchmark | ✅ Matches Benchmark | ✅ Matches Benchmark | ✅ Good | ⚠️ Limited |


### 🎯 Key Achievements

- ✅ **AG News:** Melampaui benchmark BERT paper (94.75% vs 94.2%)
- ✅ **GoEmotions:** Mendekati benchmark paper (57.49% vs 58.0%)
- ✅ **MNLI Matched:** Menyamai benchmark BERT paper (84.67% vs 84.6%)
- ✅ **MNLI Mismatched:** Melampaui benchmark (84.74% vs 83.4%)
- ✅ **SQuAD QA:** Performa solid dengan 60% Exact Match & 77.59% F1
- ⚠️ **XSum:** Performa terbatas karena constraint training (1 epoch, small dataset)

---

## 📁 Struktur Repository

```
finetunning-finalterm-deep-learning/
│
├── 📄 README.md                              ← You are here!
│
├── 📂 task-1/                                ← BERT: Encoder Architecture
│   ├── 📄 README.md
│   │
│   ├── 📂 finetuning-bert-text-classification/
│   │   ├── 📓 notebooks/
│   │   │   ├── finetune_bert_ag_news.ipynb        # 📰 News Classification
│   │   │   └── finetune_bert_go_emotions.ipynb    # 😊 Emotion Detection
│   │   ├── 📊 reports/
│   │   │   ├── reports-ag-news/
│   │   │   │   ├── report_ag_news.md
│   │   │   │   └── *.png
│   │   │   └── reports-go-emotions/
│   │   │       ├── report_go_emotions.md
│   │   │       └── *.png
│   │   └── 📄 requirements.txt
│   │
│   └── 📂 finetuning-bert-nli/
│       ├── 📓 notebooks/
│       │   └── finetune_bert_mnli.ipynb           # 🔗 Natural Language Inference
│       ├── 📊 reports/
│       │   ├── report_mnli.md
│       │   └── *.png
│       └── 📄 requirements.txt
│
├── 📂 task-2/                                ← T5: Encoder-Decoder Architecture
│   └── 📂 finetuning-t5-question-answering/
│       ├── 📓 notebooks/
│       │   └── finetuning_t5_question_answering.ipynb  # ❓ Question Answering
│       ├── 📊 reports/
│       │   ├── report_t5_qa.md
│       │   └── *.png
│       └── 📄 requirements.txt
│
└── 📂 task-3/                                ← Phi-2: Decoder Architecture
    └── 📂 finetuning-phi-2-text-summarization/
        ├── 📓 notebooks/
        │   └── finetuning-phi-2-text-summarization.ipynb  # 📝 Summarization
        ├── 📊 reports/
        │   ├── report_phi2_summarization.md
        │   └── dataset_analysis.png
        └── 📄 requirements.txt
```

---

## 🔵 Task 1: BERT (Text Classification & NLI)


### Architecture: **Encoder-Only (Bidirectional)**

```
                    ┌─────────────────────────────┐
                    │      BERT Encoder           │
     Input ──────►  │  [CLS] Token Token [SEP]   │ ──────► Classification
                    │      Bidirectional          │
                    └─────────────────────────────┘
```


**BERT** (Bidirectional Encoder Representations from Transformers) memproses teks secara **bidirectional**, memungkinkan pemahaman konteks yang lebih baik untuk task klasifikasi dan pemahaman bahasa.

### 📊 Results Overview

| Task | Dataset | Type | Classes | Metric | Result | Status |
|:----:|:-------:|:----:|:-------:|:------:|:------:|:------:|
| 📰 | AG News | Multi-class | 4 | Accuracy | **94.75%** | ✅ Exceeds |
| 😊 | GoEmotions | Multi-label | 28 | Micro-F1 | **57.49%** | ✅ Matches |
| 🔗 | MNLI | 3-class NLI | 3 | Accuracy | **84.67%** | ✅ Matches |

### 📰 Task 1A: AG News Classification

Klasifikasi artikel berita ke **4 kategori**: World, Sports, Business, Sci/Tech

```python
# Example
Input:  "Apple unveils new MacBook Pro with M3 chip at special event"
Output: "Sci/Tech" ✅
```

<details>
<summary><b>📈 Per-Class Performance</b></summary>

| Class | Precision | Recall | F1-Score |
|:-----:|:---------:|:------:|:--------:|
| 🌍 World | 96.79% | 95.11% | 95.94% |
| ⚽ Sports | 98.64% | 99.11% | **98.87%** |
| 💼 Business | 91.09% | 92.58% | 91.83% |
| 🔬 Sci/Tech | 92.55% | 92.21% | 92.38% |

</details>

### 😊 Task 1B: GoEmotions Detection

Deteksi **multiple emosi** dalam teks Reddit (28 kategori emosi)

```python
# Example
Input:  "Thank you so much! This made my day!"
Output: ["gratitude", "joy", "admiration"] ✅
```

<details>
<summary><b>📈 Performance Tiers</b></summary>

| Tier | F1 Range | Emotions |
|:----:|:--------:|:---------|
| 🟢 Excellent | 70-92% | gratitude, amusement, love, admiration |
| 🟡 Good | 50-70% | neutral, fear, joy, remorse, optimism |
| 🟠 Moderate | 30-50% | anger, curiosity, desire, disgust |
| 🔴 Poor | 0-30% | annoyance, disappointment, realization |
| ⚫ Zero | 0% | grief, pride, relief, nervousness, embarrassment |

</details>

### 🔗 Task 1C: MNLI (Natural Language Inference)

Menentukan hubungan logika antara **premise** dan **hypothesis**

```python
# Example
Premise:    "A man is playing guitar on stage"
Hypothesis: "Someone is performing music"
Output:     "Entailment" ✅
```

<details>
<summary><b>📈 Per-Class Performance</b></summary>

| Class | Precision | Recall | F1-Score |
|:-----:|:---------:|:------:|:--------:|
| ✓ Entailment | 89.69% | 84.28% | 86.90% |
| ○ Neutral | 78.36% | 83.03% | 80.63% |
| ✗ Contradiction | 86.04% | 86.68% | 86.36% |

</details>

📂 **Navigate to:** [`task-1/`](task-1/)

---

## 🟢 Task 2: T5 (Question Answering)


### Architecture: **Encoder-Decoder (Seq2Seq)**

```
                    ┌─────────────────────────────────────┐
                    │            T5 Model                  │
     Input ──────►  │   ENCODER    ──►    DECODER         │ ──────► Answer
  (Q + Context)     │  (Understand)      (Generate)       │
                    └─────────────────────────────────────┘
```


**T5** (Text-to-Text Transfer Transformer) menggunakan framework **text-to-text** yang unified, mengubah semua task menjadi format generasi teks.

### 📊 Results

| Dataset | Task | Exact Match | F1 Score | Training Time |
|:-------:|:----:|:-----------:|:--------:|:-------------:|
| SQuAD | Question Answering | **60.00%** | **77.59%** | ~20 min |

### 💬 Example

```python
Context:  "Paris is the capital and largest city of France. 
          It has a population of over 2 million people."
          
Question: "What is the capital of France?"

Generated Answer: "Paris" ✅
```

### 📈 Performance Breakdown

| Category | Count | Percentage |
|:--------:|:-----:|:----------:|
| ✅ Perfect Match | 14/20 | 70% |
| 🟢 Good Match (F1≥0.7) | 2/20 | 10% |
| 🟡 Partial Match | 1/20 | 5% |
| ❌ Poor Match | 3/20 | 15% |

📂 **Navigate to:** [`task-2/finetuning-t5-question-answering/`](task-2/finetuning-t5-question-answering/)

---

## 🟠 Task 3: Phi-2 (Text Summarization)


### Architecture: **Decoder-Only (Causal LM)**

```
                    ┌─────────────────────────────┐
                    │       Phi-2 Decoder          │
     Input ──────►  │   Autoregressive Generation │ ──────► Summary
   (Document)       │        (LoRA Fine-tuned)    │
                    └─────────────────────────────┘
```


**Phi-2** adalah model decoder-only dari Microsoft (2.7B parameters) yang di-finetune menggunakan **LoRA** untuk efisiensi.

### 📊 Results

| Dataset | Metric | Score | Training | Trainable Params |
|:-------:|:------:|:-----:|:--------:|:----------------:|
| XSum | ROUGE-1 | **7.13%** | 1 epoch | 8.4M (0.30%) |
| XSum | ROUGE-2 | **0.21%** | ~1.5 hrs | LoRA r=16 |
| XSum | ROUGE-L | **6.03%** | 4-bit quantized | α=32 |

### ⚠️ Catatan Performa

Skor ROUGE rendah disebabkan oleh:
- **Training terbatas:** Hanya 1 epoch
- **Dataset kecil:** 1.5K samples (vs 204K full XSum)
- **Time constraint:** Limited computational resources

### 💡 LoRA Efficiency

```python
# Full Fine-tuning vs LoRA
Total Parameters:     2,780,428,288 (2.7B)
Trainable (LoRA):         8,421,376 (8.4M)
Efficiency:                   0.30%  ✅
```

📂 **Navigate to:** [`task-3/finetuning-phi-2-text-summarization/`](task-3/finetuning-phi-2-text-summarization/)

---

## 📊 Perbandingan Arsitektur


| Aspect | 🔵 BERT (Encoder) | 🟢 T5 (Enc-Dec) | 🟠 Phi-2 (Decoder) |
|:------:|:-----------------:|:---------------:|:------------------:|
| **Direction** | Bidirectional | Seq2Seq | Autoregressive |
| **Best For** | Understanding | Translation, QA | Generation |
| **Parameters** | 109M | 223M | 2.7B |
| **Output** | Classification | Sequence | Sequence |
| **Pre-training** | MLM + NSP | Span Corruption | Next Token |


### 🎯 Kapan Menggunakan Arsitektur Tertentu?

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE SELECTION GUIDE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Task Type                          Recommended Architecture     │
│  ─────────────────────────────────  ────────────────────────    │
│  Text Classification                → ENCODER (BERT)             │
│  Named Entity Recognition           → ENCODER (BERT)             │
│  Sentiment Analysis                 → ENCODER (BERT)             │
│                                                                  │
│  Machine Translation                → ENCODER-DECODER (T5)       │
│  Question Answering                 → ENCODER-DECODER (T5)       │
│  Summarization (structured)         → ENCODER-DECODER (T5)       │
│                                                                  │
│  Text Generation                    → DECODER (GPT, Phi-2)       │
│  Chatbot/Dialogue                   → DECODER (GPT, Phi-2)       │
│  Creative Writing                   → DECODER (GPT, Phi-2)       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- GPU dengan VRAM 16GB+ (atau Google Colab)
- HuggingFace account (optional, untuk model hosting)

### Option 1: Google Colab (Recommended) ⭐

1. **Clone repository ini**
2. **Upload notebook ke Google Colab**
3. **Enable GPU Runtime:**
   ```
   Runtime → Change runtime type → GPU (T4)
   ```
4. **Run all cells:**
   ```
   Runtime → Run all (Ctrl+F9)
   ```

### Option 2: Local Setup

```bash
# 1. Clone repository
git clone https://github.com/[username]/finetunning-finalterm-deep-learning.git
cd finetunning-finalterm-deep-learning

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies (pilih task)
pip install -r task-1/finetuning-bert-text-classification/requirements.txt
# atau
pip install -r task-2/finetuning-t5-question-answering/requirements.txt
# atau
pip install -r task-3/finetuning-phi-2-text-summarization/requirements.txt

# 4. Run Jupyter
jupyter notebook
```

### ⏱️ Estimated Training Time

| Task | Notebook | Time (T4 GPU) |
|:----:|:--------:|:-------------:|
| 1A | AG News | ~35 min |
| 1B | GoEmotions | ~10 min |
| 1C | MNLI | ~113 min |
| 2 | T5 QA | ~20 min |
| 3 | Phi-2 | ~90 min |
| **Total** | - | **~4.5 hours** |

---

## 🛠️ Tech Stack


| Category | Technologies |
|:--------:|:-------------|
| **Framework** | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) |
| **Models** | ![HuggingFace](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=flat) ![PEFT](https://img.shields.io/badge/PEFT-LoRA-blue?style=flat) |
| **Data** | ![Datasets](https://img.shields.io/badge/🤗_Datasets-FFD21E?style=flat) |
| **Metrics** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat) |
| **Environment** | ![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white) |


---

## 📚 Reports & Documentation

| Task | Report | Visualizations |
|:----:|:------:|:--------------:|
| 1A | [📄 AG News Report](task-1/finetuning-bert-text-classification/reports/reports-ag-news/report_ag_news.md) | Confusion Matrix, Distribution |
| 1B | [📄 GoEmotions Report](task-1/finetuning-bert-text-classification/reports/reports-go-emotions/report_go_emotions.md) | Per-class F1, Distribution |
| 1C | [📄 MNLI Report](task-1/finetuning-bert-nli/reports/report_mnli.md) | Confusion Matrix, Distribution |
| 2 | [📄 T5 QA Report](task-2/finetuning-t5-question-answering/reports/report_t5_qa.md) | Loss Curves, Metrics, Examples |
| 3 | [📄 Phi-2 Report](task-3/finetuning-phi-2-text-summarization/reports/report_phi2_summarization.md) | Dataset Analysis, ROUGE Scores |

---

## 💡 Key Learnings

### 1️⃣ Single-Label vs Multi-Label Classification

```python
# Single-Label (AG News, MNLI) - Exactly ONE class per sample
loss_fn = CrossEntropyLoss()
prediction = torch.argmax(logits, dim=-1)

# Multi-Label (GoEmotions) - MULTIPLE classes possible
loss_fn = BCEWithLogitsLoss()
prediction = (torch.sigmoid(logits) > 0.5).int()
```

### 2️⃣ Sentence Pair Encoding (NLI)

```python
# Single sentence (Classification)
tokenizer(text, max_length=128)
# → [CLS] text [SEP]

# Sentence pair (NLI)
tokenizer(premise, hypothesis, max_length=256)
# → [CLS] premise [SEP] hypothesis [SEP]
```

### 3️⃣ Parameter-Efficient Fine-tuning (LoRA)

```python
# Instead of training 2.7B parameters...
# LoRA trains only 8.4M parameters (0.30%)!

lora_config = LoraConfig(
    r=16,           # Low-rank dimension
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05
)
```

---

## 📖 References

1. Devlin, J., et al. (2019). **"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"**
2. Raffel, C., et al. (2020). **"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"** (T5)
3. Microsoft Research (2023). **"Phi-2: The surprising power of small language models"**
4. Hu, E., et al. (2021). **"LoRA: Low-Rank Adaptation of Large Language Models"**
5. HuggingFace Transformers Documentation

---

## 📜 License

This project is created for **educational purposes** as part of the Deep Learning course final exam (UAS) at **Telkom University**.

---

## 🙏 Acknowledgments


| | |
|:-:|:-:|
| [🤗 HuggingFace](https://huggingface.co/) | Pre-trained models & datasets |
| [Google Colab](https://colab.research.google.com/) | Free GPU resources |
| [Telkom University](https://telkomuniversity.ac.id/) | Academic support |
| Course Instructors | Guidance & feedback |


---
