<div align="center">

# 🤖 Task 2: Fine-tuning Encoder-Decoder Transformer

### T5 for Question Answering | UAS Deep Learning

[![T5](https://img.shields.io/badge/Model-T5--base-green?style=for-the-badge&logo=google&logoColor=white)](https://huggingface.co/google-t5/t5-base)
[![SQuAD](https://img.shields.io/badge/Dataset-SQuAD_v1.1-blue?style=for-the-badge)](https://rajpurkar.github.io/SQuAD-explorer/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.40+-FFD21E?style=for-the-badge)](https://huggingface.co/)

**Generative Question Answering dengan Text-to-Text Transfer Transformer**

[📊 Results](#-results) • [🏗️ Architecture](#️-architecture) • [🚀 Quick Start](#-quick-start) • [📁 Structure](#-directory-structure)

---

### 🎯 Performance Highlights

| Exact Match | F1 Score | Training Time | Parameters |
|:-----------:|:--------:|:-------------:|:----------:|
| **60.00%** | **77.59%** | ~20 min | 223M |

</div>

---

## 📋 Overview

**Task 2** mengeksplorasi arsitektur **Encoder-Decoder Transformer** untuk task **Generative Question Answering**. Berbeda dengan Task 1 yang menggunakan encoder-only (BERT), task ini menggunakan **T5 (Text-to-Text Transfer Transformer)** yang dapat menghasilkan jawaban dalam bentuk teks.

### 🎓 Learning Objectives

| # | Objective | Status |
|:-:|:----------|:------:|
| 1 | Memahami arsitektur encoder-decoder Transformer | ✅ |
| 2 | Implementasi fine-tuning T5 untuk Question Answering | ✅ |
| 3 | Menerapkan text-to-text paradigm | ✅ |
| 4 | Evaluasi dengan SQuAD metrics (EM & F1) | ✅ |
| 5 | Optimasi untuk resource constraints | ✅ |

### 🔄 Task 2 vs Task 1

| Aspect | Task 1 (BERT) | Task 2 (T5) |
|:------:|:-------------:|:-----------:|
| **Architecture** | Encoder-only | **Encoder-Decoder** |
| **Approach** | Extractive (span) | **Generative (text)** |
| **Output** | Start/End positions | **Generated sequence** |
| **Paradigm** | Task-specific head | **Text-to-Text** |
| **Flexibility** | Must extract from context | **Can generate freely** |

---

## 🏗️ Architecture

### T5: Text-to-Text Transfer Transformer

```
┌────────────────────────────────────────────────────────────────────┐
│                     ENCODER-DECODER ARCHITECTURE                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input: "question: What is X? context: ... X is Y ..."            │
│                              │                                      │
│                              ▼                                      │
│                    ┌─────────────────┐                             │
│                    │     ENCODER     │  ← Bidirectional            │
│                    │   (12 layers)   │     Understand context      │
│                    └────────┬────────┘                             │
│                             │                                       │
│                      Hidden States                                  │
│                             │                                       │
│                    ┌────────▼────────┐                             │
│                    │     DECODER     │  ← Autoregressive           │
│                    │   (12 layers)   │     Generate answer         │
│                    └────────┬────────┘                             │
│                             │                                       │
│                             ▼                                       │
│                    Output: "Y" (Answer)                            │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Model Specifications

| Property | Value |
|:---------|:------|
| **Model** | `google-t5/t5-base` |
| **Parameters** | 222,903,552 (~223M) |
| **Encoder Layers** | 12 |
| **Decoder Layers** | 12 |
| **Hidden Size** | 768 |
| **Attention Heads** | 12 |
| **Vocabulary** | 32,128 (SentencePiece) |

---

## 📊 Results

### Training Performance

| Epoch | Train Loss | Val Loss | Status |
|:-----:|:----------:|:--------:|:------:|
| 1 | 0.6357 | 0.0845 | ✅ |
| 2 | 0.0443 | 0.0862 | ✅ |

### Evaluation Metrics

| Metric | Score | Description |
|:------:|:-----:|:------------|
| **Exact Match** | **60.00%** | Perfect string match |
| **F1 Score** | **77.59%** | Token-level overlap |

### Prediction Quality

| Category | Percentage | Description |
|:--------:|:----------:|:------------|
| ✅ Perfect | **70%** | Exactly matches ground truth |
| 🟢 Good | **10%** | F1 ≥ 0.7 |
| 🟡 Partial | **5%** | 0.3 < F1 < 0.7 |
| ❌ Poor | **15%** | F1 ≤ 0.3 |

### Sample Predictions

| Question | Ground Truth | Prediction | Match |
|:---------|:-------------|:-----------|:-----:|
| "In what year did Massachusetts first require children to be educated?" | 1852 | 1852 | ✅ |
| "Why was this organization created?" | coordinate the response | to coordinate the response | 🟢 |

---

## 📁 Directory Structure

```
task-2/
│
├── 📄 README.md                              ← You are here!
│
└── 📂 finetuning-t5-question-answering/      ← Main project
    │
    ├── 📄 README.md                          # Detailed documentation
    ├── 📄 requirements.txt                   # Dependencies
    │
    ├── 📓 notebooks/
    │   └── finetuning_t5_question_answering.ipynb  # Main notebook
    │
    └── 📊 reports/
        ├── 📄 report_t5_qa.md                # Detailed report
        ├── 🖼️ dataset_analysis.png           # Dataset visualization
        ├── 🖼️ Training & Validation Loss.png # Loss curves
        ├── 🖼️ Training_Config.png            # Configuration
        ├── 🖼️ evaluation_metrics.png         # EM & F1 metrics
        ├── 🖼️ Final_Results.png              # Summary
        ├── 🖼️ F1_Distributions.png           # F1 histogram
        ├── 🖼️ Model_Comparison.png           # Benchmarks
        └── 🖼️ Inferences_example.png         # Predictions
```

---

## 📚 Dataset: SQuAD v1.1

**Stanford Question Answering Dataset** - Dataset benchmark untuk extractive QA.

| Split | Original | Used | Percentage |
|:-----:|:--------:|:----:|:----------:|
| Train | 87,599 | 4,379 | 5% |
| Validation | 10,570 | 1,057 | 10% |

### Input-Output Format

```python
# T5 Input Format
input = "question: What is the capital of France? context: Paris is the capital of France..."

# T5 Output
output = "Paris"
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|:----------|:------|
| Batch Size | 4 |
| Epochs | 2 |
| Learning Rate | 3e-4 |
| Optimizer | AdamW |
| Max Input Length | 256 |
| Max Output Length | 32 |
| Mixed Precision | FP16 |
| Warmup Steps | 200 |

---

## 🚀 Quick Start

### Google Colab (Recommended)

1. **Navigate to notebook:**
   ```
   task-2/finetuning-t5-question-answering/notebooks/
   ```

2. **Open in Colab:**
   - Upload `finetuning_t5_question_answering.ipynb`
   - Enable GPU: `Runtime → Change runtime type → GPU`

3. **Run all cells:**
   ```
   Runtime → Run all (Ctrl+F9)
   ```

### Local Setup

```bash
# Navigate to project
cd task-2/finetuning-t5-question-answering

# Install dependencies
pip install -r requirements.txt

# Run Jupyter
jupyter notebook notebooks/finetuning_t5_question_answering.ipynb
```

---

## 👥 Team Information

| Name | NIM | Class |
|:-----|:---:|:-----:|
| Raihan Salman Baehaqi | 1103220180 | TK-46-02 |
| Jaka Kelana Wijaya | 1103223048 | TK-46-02 |

---

## 📖 Documentation

| Document | Description | Link |
|:---------|:------------|:----:|
| **Project README** | Detailed documentation | [📄](finetuning-t5-question-answering/README.md) |
| **Experiment Report** | Full analysis & results | [📊](finetuning-t5-question-answering/reports/report_t5_qa.md) |
| **Training Notebook** | Complete implementation | [📓](finetuning-t5-question-answering/notebooks/finetuning_t5_question_answering.ipynb) |

---

## 🔗 Related Tasks

| Task | Model | Architecture | Dataset | Status |
|:----:|:-----:|:------------:|:-------:|:------:|
| [Task 1](../task-1/) | BERT | Encoder | AG News, GoEmotions, MNLI | ✅ |
| **Task 2** | **T5** | **Encoder-Decoder** | **SQuAD** | ✅ |
| [Task 3](../task-3/) | Phi-2 | Decoder | XSum | ✅ |

---

## 📚 Key Concepts

### Text-to-Text Paradigm

T5 menggunakan format unified untuk semua task:

```python
# Question Answering
"question: {Q} context: {C}" → "{Answer}"

# Translation
"translate English to French: {text}" → "{translation}"

# Summarization
"summarize: {document}" → "{summary}"
```

### Encoder-Decoder vs Encoder-Only

| Aspect | Encoder-Only (BERT) | Encoder-Decoder (T5) |
|:-------|:--------------------|:---------------------|
| **Processing** | Bidirectional | Bidirectional → Autoregressive |
| **Output** | Fixed-size logits | Variable-length sequence |
| **Training** | MLM + Classification | Denoising + Generation |
| **Use Case** | Understanding | Generation |

---

## 📜 License

Educational project for Deep Learning course (UAS) at Telkom University.

---

<div align="center">

**Part of UAS Deep Learning**

*Exploring Encoder-Decoder Transformer Architecture*

[![Task](https://img.shields.io/badge/Task-2_of_3-blue?style=for-the-badge)]()
[![Status](https://img.shields.io/badge/Status-✅_Completed-success?style=for-the-badge)]()

</div>
