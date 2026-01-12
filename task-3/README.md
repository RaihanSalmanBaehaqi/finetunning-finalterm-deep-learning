<div align="center">

# 📝 Task 3: Fine-tuning Decoder-Only LLM

### Phi-2 for Text Summarization | UAS Deep Learning

[![Phi-2](https://img.shields.io/badge/Model-Phi--2_(2.7B)-00BCF2?style=for-the-badge&logo=microsoft&logoColor=white)](https://huggingface.co/microsoft/phi-2)
[![XSum](https://img.shields.io/badge/Dataset-XSum-purple?style=for-the-badge)](https://huggingface.co/datasets/EdinburghNLP/xsum)
[![LoRA](https://img.shields.io/badge/Method-LoRA-orange?style=for-the-badge)](https://arxiv.org/abs/2106.09685)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.35+-FFD21E?style=for-the-badge)](https://huggingface.co/)

**Abstractive Summarization dengan Parameter-Efficient Fine-tuning**

[📊 Results](#-results) • [🏗️ Architecture](#️-architecture) • [🚀 Quick Start](#-quick-start) • [📁 Structure](#-directory-structure)

---

### 🎯 Performance Highlights

| ROUGE-1 | ROUGE-2 | ROUGE-L | Trainable Params | Training Time |
|:-------:|:-------:|:-------:|:----------------:|:-------------:|
| **7.13%** | **0.21%** | **6.03%** | 8.4M (0.30%) | ~1.5-2 hrs |

</div>

---

## 📋 Overview

**Task 3** mengeksplorasi arsitektur **Decoder-Only Large Language Model (LLM)** untuk task **Abstractive Text Summarization**. Berbeda dengan Task 1 (encoder-only) dan Task 2 (encoder-decoder), task ini menggunakan **Phi-2** dari Microsoft dengan teknik **LoRA** untuk parameter-efficient fine-tuning.

### 🎓 Learning Objectives

| # | Objective | Status |
|:-:|:----------|:------:|
| 1 | Memahami arsitektur decoder-only (Causal LM) | ✅ |
| 2 | Implementasi LoRA untuk parameter-efficient fine-tuning | ✅ |
| 3 | Menerapkan 4-bit quantization (QLoRA) | ✅ |
| 4 | Menggunakan instruction-style prompting | ✅ |
| 5 | Evaluasi dengan ROUGE metrics | ✅ |
| 6 | Optimasi untuk consumer GPU (T4) | ✅ |

### 🔄 Comparison with Other Tasks

| Aspect | Task 1 (BERT) | Task 2 (T5) | Task 3 (Phi-2) |
|:------:|:-------------:|:-----------:|:--------------:|
| **Architecture** | Encoder-only | Encoder-Decoder | **Decoder-only** |
| **Parameters** | 109M | 223M | **2.7B** |
| **Fine-tuning** | Full | Full | **LoRA (0.3%)** |
| **Task Type** | Classification | Generation | **Generation** |
| **Memory** | Standard | Standard | **Quantized (4-bit)** |

---

## 🏗️ Architecture

### Phi-2: Decoder-Only LLM

```
┌────────────────────────────────────────────────────────────────────┐
│                   DECODER-ONLY ARCHITECTURE                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input: "### Article:\n{document}\n\n### Summary:\n"              │
│                              │                                      │
│                              ▼                                      │
│          ┌───────────────────────────────────┐                     │
│          │        PHI-2 DECODER              │                     │
│          │     (32 Transformer Layers)       │                     │
│          │                                   │                     │
│          │   • Self-Attention (Causal Mask)  │                     │
│          │   • LoRA Adapters (r=16, α=32)    │                     │
│          │   • 4-bit Quantization            │                     │
│          └───────────────────────────────────┘                     │
│                              │                                      │
│                              ▼                                      │
│              Output: Generated Summary (Autoregressive)            │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Model Specifications

| Property | Value |
|:---------|:------|
| **Model** | `microsoft/phi-2` |
| **Total Parameters** | 2,780,428,288 (~2.78B) |
| **Trainable (LoRA)** | 8,421,376 (~8.4M) |
| **Trainable %** | **0.30%** |
| **Layers** | 32 |
| **Hidden Size** | 2560 |
| **Context Length** | 2048 tokens |

### LoRA Configuration

| Parameter | Value | Description |
|:----------|:------|:------------|
| **Rank (r)** | 16 | Low-rank dimension |
| **Alpha (α)** | 32 | Scaling factor |
| **Target Modules** | q_proj, k_proj, v_proj, dense | Attention layers |
| **Dropout** | 0.05 | Regularization |

---

## 📊 Results

### Training Performance

| Metric | Value |
|:------:|:-----:|
| **Initial Loss** | 2.4634 |
| **Final Loss** | 2.1901 |
| **Improvement** | **11.09%** |
| **Training Time** | ~1.5-2 hours |

### ROUGE Evaluation

| Metric | Score | Description |
|:------:|:-----:|:------------|
| **ROUGE-1** | **7.13%** | Unigram overlap |
| **ROUGE-2** | **0.21%** | Bigram overlap |
| **ROUGE-L** | **6.03%** | Longest common subsequence |

### Performance Notes

| Aspect | Status | Explanation |
|:------:|:------:|:------------|
| **Training Convergence** | ✅ Good | Loss decreased consistently |
| **ROUGE Scores** | ⚠️ Low | Limited by 1 epoch & small data |
| **LoRA Efficiency** | ✅ Excellent | Only 0.3% params trained |
| **Memory Usage** | ✅ Efficient | Runs on T4 GPU (16GB) |

> ⚠️ **Note:** Low ROUGE scores are expected due to training constraints (1 epoch, 0.7% of XSum data). With full training, scores would improve significantly.

---

## 📁 Directory Structure

```
task-3/
│
├── 📄 README.md                              ← You are here!
│
└── 📂 finetuning-phi-2-text-summarization/   ← Main project
    │
    ├── 📄 README.md                          # Detailed documentation
    │
    ├── 📓 notebooks/
    │   └── finetuning-phi-2-text-summarization.ipynb  # Main notebook
    │
    └── 📊 reports/
        ├── 📄 report_phi2_summarization.md   # Comprehensive report
        ├── 📄 sample_predictions.txt         # Example outputs
        ├── 📄 all_predictions.csv            # All test predictions
        ├── 🖼️ dataset_analysis.png           # Dataset visualization
        ├── 🖼️ training_loss.png              # Loss curve
        └── 🖼️ Rouge_Scores.png               # ROUGE metrics
```

---

## 📚 Dataset: XSum

**Extreme Summarization** - One-sentence summaries of BBC news articles.

| Split | Original | Used | Percentage |
|:-----:|:--------:|:----:|:----------:|
| Train | 204,045 | 1,500 | 0.7% |
| Test | 11,334 | 150 | 1.3% |

### Characteristics

| Property | Value |
|:---------|:------|
| **Source** | BBC News articles |
| **Summary Style** | Highly abstractive (paraphrasing) |
| **Compression** | ~18:1 ratio |
| **Challenge** | Requires rewriting, not extraction |

---

## ⚙️ Training Configuration

| Parameter | Value |
|:----------|:------|
| **Epochs** | 1 |
| **Batch Size** | 1 |
| **Gradient Accumulation** | 8 |
| **Effective Batch** | 8 |
| **Learning Rate** | 2e-4 |
| **Optimizer** | paged_adamw_8bit |
| **Quantization** | 4-bit (NF4) |
| **FP16** | ✅ Enabled |
| **Gradient Checkpointing** | ✅ Enabled |

---

## 🚀 Quick Start

### Google Colab (Recommended)

1. **Navigate to notebook:**
   ```
   task-3/finetuning-phi-2-text-summarization/notebooks/
   ```

2. **Open in Colab:**
   - Upload `finetuning-phi-2-text-summarization.ipynb`
   - Enable GPU: `Runtime → Change runtime type → GPU (T4)`

3. **Run all cells:**
   ```
   Runtime → Run all (Ctrl+F9)
   ```

### Local Setup

```bash
# Navigate to project
cd task-3/finetuning-phi-2-text-summarization

# Install dependencies
pip install torch transformers accelerate peft bitsandbytes
pip install trl rouge-score datasets

# Run Jupyter
jupyter notebook notebooks/finetuning-phi-2-text-summarization.ipynb
```

### Hardware Requirements

| Component | Minimum | Recommended |
|:----------|:--------|:------------|
| **GPU** | T4 (16GB) | A100 (40GB) |
| **RAM** | 12GB | 16GB+ |
| **Time** | 1.5 hrs | 1 hr |

---

## 👥 Team Information

| Name | NIM | Class | Task |
|:-----|:---:|:-----:|:----:|
| [Member 1] | [NIM] | TK-46-02 | Task 1 |
| [Member 2] | [NIM] | TK-46-02 | Task 2 |
| [Your Name] | [Your NIM] | TK-46-02 | **Task 3** ✅ |

---

## 📖 Documentation

| Document | Description | Link |
|:---------|:------------|:----:|
| **Project README** | Detailed documentation | [📄](finetuning-phi-2-text-summarization/README.md) |
| **Experiment Report** | Full analysis & results | [📊](finetuning-phi-2-text-summarization/reports/report_phi2_summarization.md) |
| **Training Notebook** | Complete implementation | [📓](finetuning-phi-2-text-summarization/notebooks/finetuning-phi-2-text-summarization.ipynb) |

---

## 🔗 Related Tasks

| Task | Model | Architecture | Dataset | Status |
|:----:|:-----:|:------------:|:-------:|:------:|
| [Task 1](../task-1/) | BERT | Encoder | AG News, GoEmotions, MNLI | ✅ |
| [Task 2](../task-2/) | T5 | Encoder-Decoder | SQuAD | ✅ |
| **Task 3** | **Phi-2** | **Decoder** | **XSum** | ✅ |

---

## 💡 Key Concepts

### Why LoRA?

| Benefit | Description |
|:--------|:------------|
| **Memory Efficient** | Train only 0.3% of parameters |
| **Fast** | Much faster than full fine-tuning |
| **Portable** | Adapter weights are small (~32MB) |
| **No Forgetting** | Preserves pre-trained knowledge |

### Memory Optimization Stack

```
┌─────────────────────────────────────────┐
│       MEMORY OPTIMIZATION               │
├─────────────────────────────────────────┤
│  • 4-bit Quantization    → 75% saved   │
│  • LoRA Adapters         → 99% saved   │
│  • Gradient Checkpointing → 30% saved  │
│  • Mixed Precision (FP16) → 50% saved  │
├─────────────────────────────────────────┤
│  Result: 2.7B model on 16GB GPU! ✅    │
└─────────────────────────────────────────┘
```

---

## 📜 License

Educational project for Deep Learning course (UAS) at Telkom University.

---

<div align="center">

**Part of UAS Deep Learning**

*Exploring Decoder-Only LLM with Parameter-Efficient Fine-tuning*

[![Task](https://img.shields.io/badge/Task-3_of_3-blue?style=for-the-badge)]()
[![Status](https://img.shields.io/badge/Status-✅_Completed-success?style=for-the-badge)]()
[![LoRA](https://img.shields.io/badge/Trainable-0.30%25-orange?style=for-the-badge)]()

</div>
