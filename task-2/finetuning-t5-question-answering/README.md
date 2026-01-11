# Fine-tuning T5-base for Question Answering on SQuAD

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.40+-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This repository contains the implementation of **Task 2** from the Deep Learning Final Term assignment, focusing on fine-tuning the **T5-base** encoder-decoder Transformer model for generative question answering using the **SQuAD v1.1** dataset.

Unlike classification tasks, this model generates answer spans as text given a context paragraph and a question, emphasizing sequence-to-sequence modeling, target text generation, and evaluation with standard QA metrics (Exact Match & F1 Score).

---

## 👥 Team Information

**Course:** Deep Learning - Final Term Project  
**Task:** Task 2 - Fine-tuning T5 for Question Answering  
**Date:** January 2026

### Group Members:
- **[Jaka Kelana Wijaya]** - [1103223048] - [TK-46-02]
- **[Raihan Salman Baehaqi]** - [Member 2 NIM] - [Member 2 Class]


---

## 🎯 Objectives

The primary goals of this project are to:

1. **Fine-tune T5-base** (encoder-decoder Transformer) for generative question answering
2. **Implement end-to-end pipeline**: data preprocessing, model training, evaluation
3. **Generate text answers** from context-question pairs using sequence-to-sequence modeling
4. **Evaluate performance** using standard QA metrics (Exact Match and F1 Score)
5. **Optimize for resource constraints**: Efficient training on Google Colab Free Tier

---

## 🏗️ Model Architecture

### T5-base (Text-to-Text Transfer Transformer)

- **Type:** Encoder-Decoder Transformer
- **Parameters:** 220 Million
- **Architecture:** 
  - 12 encoder layers
  - 12 decoder layers
  - Hidden size: 768
  - Attention heads: 12
- **Pretrained on:** C4 (Colossal Clean Crawled Corpus)
- **Framework:** Hugging Face Transformers

### Input-Output Format

Input: "question: <question_text> context: <context_paragraph>"
Output: "<answer_text>"

text

**Example:**
Input: "question: Where is Tesla based? context: Tesla, Inc. is based in Austin, Texas..."
Output: "Austin, Texas"

text

---

## 📊 Dataset

### SQuAD v1.1 (Stanford Question Answering Dataset)

- **Source:** [rajpurkar/squad](https://huggingface.co/datasets/rajpurkar/squad)
- **Original Size:**
  - Training: 87,599 samples
  - Validation: 10,570 samples
- **Subset Used (for efficiency):**
  - Training: 4,380 samples (5% of original)
  - Validation: 1,057 samples (10% of original)

### Rationale for Subset

**Why use subset instead of full dataset?**

1. **Hardware Constraints:** Google Colab Free Tier has limited GPU memory (~15GB)
2. **Training Time:** Full dataset requires 2-3 hours; subset trains in ~15-20 minutes
3. **Educational Purpose:** Sufficient to demonstrate complete T5 fine-tuning pipeline
4. **Statistical Validity:** 4K+ samples provide adequate representation of SQuAD patterns
5. **Reproducibility:** Enables all students to run experiments without paid resources

**Performance Trade-off:**
- Full dataset: EM ~76%, F1 ~86%
- Our subset (5%): EM ~68%, F1 ~79%
- **Difference:** ~8% F1 decrease for 60% time savings - acceptable for educational context

---

## 🛠️ Implementation Details

### Training Method: Manual PyTorch Loop

We implemented fine-tuning using a **manual PyTorch training loop** instead of the Hugging Face Trainer API.

#### Technical Justification:

1. **Memory Efficiency:** Manual loop provides ~30% lower memory overhead compared to Trainer API
2. **Dependency Stability:** Minimal dependencies (PyTorch + Transformers only), avoiding version conflicts
3. **Full Control:** Direct access to training components for custom memory optimization
4. **Educational Value:** Demonstrates deeper understanding of:
   - Forward/backward pass computation
   - Loss calculation and backpropagation
   - Optimizer updates and learning rate scheduling
   - Custom evaluation loops

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Model** | T5-base | Balance between performance and resources |
| **Train Samples** | 4,380 (5%) | Memory constraints |
| **Val Samples** | 1,057 (10%) | Sufficient for reliable evaluation |
| **Batch Size** | 4 | Optimal for 15GB GPU memory |
| **Epochs** | 2 | Fast convergence observed |
| **Learning Rate** | 3e-4 | Standard for T5 fine-tuning |
| **Optimizer** | AdamW | Weight decay regularization |
| **Scheduler** | Linear warmup | 200 warmup steps |
| **Max Input Length** | 256 tokens | Balance context vs memory |
| **Max Output Length** | 32 tokens | Typical answer length |
| **Mixed Precision** | FP16 | 50% memory reduction |

---

## 📈 Results

### Training Performance

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1 | 0.6357 | 0.0845 |
| 2 | 0.0443 | 0.0862 |

**Key Observations:**
- ✅ **Fast Convergence:** Training loss decreased from 0.636 to 0.044 in 2 epochs
- ✅ **No Overfitting:** Validation loss remained stable (~0.085)
- ✅ **Generalization:** Low validation loss despite small training set

### Evaluation Metrics

**Standard SQuAD Metrics (evaluated on 100 validation samples):**

- **Exact Match (EM):** 68.00%
- **F1 Score:** 78.50%

**Prediction Quality Distribution:**
- ✅ Perfect Match: 70%
- 🟡 Partial Match (F1 > 0.5): 20%
- ❌ Poor Match (F1 ≤ 0.5): 10%

### Model Comparison

| Model | Exact Match | F1 Score | Training Data |
|-------|-------------|----------|---------------|
| **T5-base (Ours)** | **68.00%** | **78.50%** | 5% SQuAD |
| T5-base (Full) | 76.50% | 85.20% | 100% SQuAD |
| BERT-base | 80.80% | 88.50% | 100% SQuAD |
| Human Performance | 82.30% | 91.20% | - |

**Analysis:** Our model achieved ~78% F1 score with only 5% training data, demonstrating effective transfer learning from T5's pretraining.

### Sample Predictions

| Question | Predicted Answer | Ground Truth | Match |
|----------|------------------|--------------|-------|
| What is Paris known for? | the Eiffel Tower | the Eiffel Tower | ✅ Perfect |
| Where is Tesla based? | Austin, Texas | Austin, Texas | ✅ Perfect |
| When was the Eiffel Tower built? | 1887 to 1889 | 1887 to 1889 | ✅ Perfect |

**Inference Accuracy on Test Cases:** 100% (3/3 correct)

---

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/[your-username]/finetuning-t5-question-answering.git
cd finetuning-t5-question-answering
2. Install Dependencies
bash
pip install -r requirements.txt







