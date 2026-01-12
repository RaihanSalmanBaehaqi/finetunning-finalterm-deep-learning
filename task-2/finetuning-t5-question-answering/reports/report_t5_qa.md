# 📊 Task 2 Report: Fine-tuning T5 for Question Answering

## 1. Overview

Laporan ini mendokumentasikan implementasi **fine-tuning T5 (Text-to-Text Transfer Transformer)** untuk task **Extractive Question Answering** menggunakan dataset **SQuAD**. Model dilatih untuk menghasilkan jawaban dari konteks berdasarkan pertanyaan yang diberikan.

| Item | Detail |
|------|--------|
| **Model** | `t5-base` |
| **Dataset** | SQuAD (Stanford Question Answering Dataset) |
| **Task Type** | Extractive Question Answering |
| **Architecture** | Encoder-Decoder (Seq2Seq) |
| **Metrics** | Exact Match (EM), F1 Score |

---

## 2. Task Description

### What is Question Answering?

Question Answering (QA) adalah task NLP dimana model harus menemukan jawaban dari sebuah pertanyaan berdasarkan konteks yang diberikan.

**Input Format:**
```
question: What is the capital of France? context: Paris is the capital and largest city of France...
```

**Output:**
```
Paris
```

### Why T5 for QA?

| Advantage | Description |
|-----------|-------------|
| **Text-to-Text Framework** | Semua task diubah menjadi format text-to-text |
| **Encoder-Decoder** | Cocok untuk sequence generation tasks |
| **Pre-trained** | Sudah dilatih pada berbagai task NLP |
| **Flexibility** | Bisa digunakan untuk berbagai task tanpa modifikasi arsitektur |

---

## 3. Dataset Description

### SQuAD (Stanford Question Answering Dataset)

SQuAD adalah dataset benchmark untuk extractive question answering yang berisi pertanyaan-pertanyaan berdasarkan artikel Wikipedia.

**Dataset Statistics:**

| Split | Samples | Usage |
|-------|---------|-------|
| Train | 4,379 | Model training |
| Validation | 1,057 | Evaluation & metrics |
| **Total** | **5,436** | - |

**Dataset Characteristics:**

| Property | Value |
|----------|-------|
| Source | Wikipedia articles |
| Question Types | Factoid (who, what, when, where, why, how) |
| Answer Type | Extractive (span from context) |
| Language | English |
| Average Context Length | ~150 words |
| Average Answer Length | ~3 words |

**Sample Data:**

| Field | Example |
|-------|---------|
| **Context** | "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary..." |
| **Question** | "What sits on top of the Main Building at Notre Dame?" |
| **Answer** | "a golden statue of the Virgin Mary" |

---

## 4. Model Architecture

### T5-base Specifications

| Property | Value |
|----------|-------|
| **Architecture** | Encoder-Decoder Transformer |
| **Parameters** | **222,903,552** (~223M) |
| **Encoder Layers** | 12 |
| **Decoder Layers** | 12 |
| **Hidden Size** | 768 |
| **Attention Heads** | 12 |
| **Feed-Forward Size** | 3072 |
| **Vocab Size** | 32,128 (SentencePiece) |

### T5 vs BERT for QA

| Aspect | BERT | T5 |
|--------|------|-----|
| **Architecture** | Encoder-only | Encoder-Decoder |
| **Output** | Start/End positions | Generated text |
| **Flexibility** | Task-specific head | Unified text-to-text |
| **QA Approach** | Span extraction | Text generation |
| **Parameters** | ~110M (base) | ~223M (base) |

### Input/Output Format

```
Input:  "question: {question} context: {context}"
Output: "{answer}"
```

---

## 5. Methodology

### 5.1 Data Preprocessing

1. **Format Conversion:** Convert SQuAD format to T5 input format
   ```python
   input_text = f"question: {question} context: {context}"
   target_text = answer
   ```

2. **Tokenization:**
   - Input max length: 256 tokens
   - Target max length: 32 tokens
   - Tokenizer: T5Tokenizer (SentencePiece)

### 5.2 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Batch Size** | 4 | Memory constraints (T5 is large) |
| **Epochs** | 2 | Quick convergence with pre-trained model |
| **Learning Rate** | 3e-4 | Higher than BERT (T5 recommendation) |
| **Optimizer** | AdamW | Standard for transformers |
| **Weight Decay** | 0.01 | Regularization |
| **Warmup Steps** | 200 | Gradual LR increase |
| **Scheduler** | Linear decay | After warmup |
| **FP16** | Enabled | Memory optimization |

### 5.3 Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    T5 QA TRAINING PIPELINE                   │
├─────────────────────────────────────────────────────────────┤
│  1. 📥 Load SQuAD Dataset                                   │
│         ↓                                                    │
│  2. 🔄 Convert to Text-to-Text Format                       │
│         ↓                                                    │
│  3. 🔤 Tokenize (input: 256, target: 32)                    │
│         ↓                                                    │
│  4. 🤖 Load T5-base Model                                   │
│         ↓                                                    │
│  5. ⚙️  Configure Training (batch=4, lr=3e-4)               │
│         ↓                                                    │
│  6. 🏋️ Train for 2 Epochs                                   │
│         ↓                                                    │
│  7. 📊 Evaluate (EM & F1)                                   │
│         ↓                                                    │
│  8. 💾 Save Model                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Results

### 6.1 Training Progress

| Epoch | Training Loss | Validation Loss | Δ Val Loss |
|-------|---------------|-----------------|------------|
| 1 | 0.6357 | 0.0845 | - |
| 2 | 0.0443 | 0.0862 | +0.0017 |

**Training Statistics:**

| Metric | Value |
|--------|-------|
| ⏱️ Training Time | **~15-20 minutes** |
| 📉 Final Training Loss | **0.0443** |
| 📉 Final Validation Loss | **0.0862** |
| 📊 Steps per Epoch | ~1,095 |
| 🔧 Total Parameters | 222,903,552 |
| 💻 Device | CUDA (GPU) |

### 6.2 Evaluation Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| **Exact Match (EM)** | **60.00%** | Percentage of predictions exactly matching ground truth |
| **F1 Score** | **77.59%** | Token-level overlap between prediction and ground truth |

> 📝 **Note:** Evaluation performed on 100 validation samples for efficiency.

### 6.3 Performance Breakdown

| Category | Count | Percentage | Description |
|----------|-------|------------|-------------|
| ✅ **Perfect Match** (EM=1) | 14/20 | **70%** | Exact match with ground truth |
| 🟢 **Good Match** (F1≥0.7) | 2/20 | **10%** | High overlap, minor differences |
| 🟡 **Partial Match** (0.3<F1<0.7) | 1/20 | **5%** | Some correct tokens |
| ❌ **Poor Match** (F1≤0.3) | 3/20 | **15%** | Mostly incorrect |

### 6.4 Training Loss Curve

![Training & Validation Loss](reports/Training%20&%20Validation%20Loss.png)

**Observations:**
- Training loss drops dramatically from epoch 1 to 2 (0.6357 → 0.0443)
- Validation loss remains stable (~0.085)
- No significant overfitting observed

---

## 7. Analysis

### 7.1 Prediction Examples

#### ✅ Perfect Match Example
| Field | Value |
|-------|-------|
| **Question** | "In what year did Massachusetts first require children to be educated?" |
| **Ground Truth** | "1852" |
| **Prediction** | "1852" |
| **F1 Score** | 1.00 |

#### 🟢 Good Match Example
| Field | Value |
|-------|-------|
| **Question** | "Why was this short termed organization created?" |
| **Ground Truth** | "coordinate the response to the embargo" |
| **Prediction** | "to coordinate the response to the embargo" |
| **F1 Score** | 0.92 |

#### ❌ Poor Match Example
| Field | Value |
|-------|-------|
| **Question** | "In 1890, who did the university decide to team up with?" |
| **Ground Truth** | "several regional colleges and universities" |
| **Prediction** | "Shimer College in Mount Carroll, Illinois" |
| **F1 Score** | 0.00 |

### 7.2 Error Analysis

**Common Error Patterns:**

| Error Type | Frequency | Example |
|------------|-----------|---------|
| **Specific vs General** | High | Truth: "several colleges" → Pred: "Shimer College" |
| **Extra Words** | Medium | Truth: "coordinate response" → Pred: "to coordinate response" |
| **Wrong Entity** | Low | Model picks different entity from context |
| **Partial Answer** | Low | Model generates incomplete answer |

**Why Errors Occur:**

1. **Ambiguous Questions:** Multiple valid answers in context
2. **Complex Reasoning:** Multi-hop questions require inference
3. **Long Contexts:** Important info may be truncated (256 token limit)
4. **Specificity Mismatch:** Model may be too specific or too general

### 7.3 F1 Score Distribution

![F1 Distribution](reports/F1_Distributions.png)

**Observations:**
- Majority of predictions have high F1 (>0.7)
- Bimodal distribution: either very good or very poor
- Few predictions in middle range

### 7.4 Key Observations

1. ✅ **Good overall performance** — 77.59% F1, 60% EM
2. ✅ **Fast convergence** — Only 2 epochs needed
3. ✅ **No overfitting** — Val loss stable
4. ⚠️ **Exact Match gap** — EM (60%) << F1 (77.59%) indicates partial matches
5. ⚠️ **Short training** — More epochs might improve performance

---

## 8. Comparison with Benchmarks

### SQuAD 1.1 Leaderboard Comparison

| Model | Exact Match | F1 Score | Parameters |
|-------|-------------|----------|------------|
| Human Performance | 82.3% | 91.2% | - |
| BERT-large | 84.1% | 90.9% | 340M |
| RoBERTa-large | 86.5% | 92.4% | 355M |
| **T5-base (Ours)** | **60.0%** | **77.59%** | **223M** |
| T5-base (Paper) | 83.0% | 89.7% | 223M |
| T5-large | 86.1% | 92.5% | 770M |

**Analysis:**
- Our implementation is below benchmark (60% vs 83% EM)
- Possible reasons:
  - Smaller training subset (4.3K vs 87K full SQuAD)
  - Only 2 epochs (vs typical 3-5 epochs)
  - Limited hyperparameter tuning

### Expected vs Actual Performance

| Metric | Expected (Full Training) | Actual (This Experiment) | Gap |
|--------|--------------------------|--------------------------|-----|
| EM | ~80-83% | 60.00% | -20-23% |
| F1 | ~88-90% | 77.59% | -10-12% |

---

## 9. Technical Implementation

### 9.1 Data Collation Function

```python
def collate_fn(batch):
    """Custom collate function for T5 QA"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
```

### 9.2 Inference Function

```python
def generate_answer(question, context, model, tokenizer, device):
    """Generate answer for a question-context pair"""
    input_text = f"question: {question} context: {context}"
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=256,
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=32,
            num_beams=4,
            early_stopping=True
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
```

### 9.3 SQuAD Metrics Computation

```python
def compute_exact_match(prediction, ground_truth):
    """Compute exact match score"""
    return int(prediction.strip().lower() == ground_truth.strip().lower())

def compute_f1_score(prediction, ground_truth):
    """Compute token-level F1 score"""
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = num_same / len(truth_tokens) if len(truth_tokens) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1
```

---

## 10. Visualizations

### Training Configuration
![Training Config](reports/Training_Config.png)

### Final Results Summary
![Final Results](reports/Final_Results.png)

### Evaluation Metrics
![Evaluation Metrics](reports/Evaluation_Metrics.png)

### Inference Examples
![Inference Examples](reports/Inferences_example.png)

### Model Comparison
![Model Comparison](reports/Model_Comparison.png)

---

## 11. Saved Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| Trained Model | `models/t5_qa_best/` | Fine-tuned T5-base weights |
| Tokenizer | `models/t5_qa_best/` | T5 tokenizer |
| Training History | `reports/training_history.csv` | Loss per epoch |
| Training Config | `reports/training_config.csv` | Hyperparameters |
| Inference Results | `reports/inference_results.csv` | Sample predictions |
| Detailed Predictions | `reports/detailed_predictions.csv` | All evaluation results |
| Evaluation Summary | `reports/evaluation_summary.csv` | EM & F1 scores |

### Visualization Files

| File | Description |
|------|-------------|
| `Training & Validation Loss.png` | Loss curves over epochs |
| `Training_Config.png` | Configuration table |
| `Final_Results.png` | Performance summary |
| `Inferences_example.png` | Sample predictions |
| `Evaluation_Metrics.png` | EM & F1 visualization |
| `F1_Distributions.png` | F1 score histogram |
| `Model_Comparison.png` | Benchmark comparison |
| `dataset_analysis.png` | Dataset statistics |

---

## 12. Potential Improvements

| Improvement | Expected Impact | Difficulty |
|-------------|-----------------|------------|
| **Train on full SQuAD** (87K samples) | +15-20% EM | ⭐ Easy |
| **More epochs** (3-5) | +5-10% EM | ⭐ Easy |
| **Use T5-large** (770M params) | +3-5% EM | ⭐⭐ Medium |
| **Beam search tuning** | +1-2% EM | ⭐ Easy |
| **Learning rate scheduling** | +1-2% EM | ⭐ Easy |
| **Data augmentation** | +2-3% EM | ⭐⭐ Medium |
| **Ensemble multiple models** | +2-3% EM | ⭐⭐⭐ Hard |

### Quick Wins

```python
# 1. Increase epochs
EPOCHS = 4  # Instead of 2

# 2. Better beam search
outputs = model.generate(
    num_beams=8,        # Instead of 4
    length_penalty=0.6,  # Add this
    no_repeat_ngram_size=2
)

# 3. Lower learning rate for stability
LEARNING_RATE = 1e-4  # Instead of 3e-4
```

---

## 13. Key Differences: T5 vs BERT for QA

| Aspect | BERT (Extractive) | T5 (Generative) |
|--------|-------------------|-----------------|
| **Architecture** | Encoder-only | Encoder-Decoder |
| **Output** | Start & end positions | Generated text |
| **Answer Source** | Must be exact span | Can paraphrase |
| **Flexibility** | Limited to extraction | Can generate any text |
| **Training** | Classification head | Seq2Seq generation |
| **Inference** | argmax positions | Beam search |

### T5 Input/Output Example

```
# BERT approach (extractive)
Input:  [CLS] Question [SEP] Context [SEP]
Output: start_position=45, end_position=52

# T5 approach (generative)
Input:  "question: What is X? context: ... X is Y ..."
Output: "Y"
```

---

## 14. Conclusion

### ✅ Achievements

1. **Successfully fine-tuned T5-base** for question answering
2. **Achieved 77.59% F1 score** and 60% Exact Match
3. **Fast training** — only ~15-20 minutes for 2 epochs
4. **No overfitting** — validation loss remained stable
5. **70% perfect predictions** in detailed analysis

### ⚠️ Limitations

1. **Below benchmark performance** (60% vs 83% EM)
2. **Small training subset** (4.3K vs 87K full SQuAD)
3. **Limited epochs** (2 vs recommended 3-5)
4. **Some completely wrong predictions** (15% poor match)

### 🎯 Key Takeaways

1. **T5's text-to-text approach** works well for QA
2. **Encoder-Decoder architecture** enables flexible answer generation
3. **Pre-training is powerful** — good results with minimal fine-tuning
4. **More data = better results** — full SQuAD would significantly improve
5. **Trade-off exists** — T5 is slower but more flexible than BERT

### 📈 Future Work

1. Train on full SQuAD dataset (87K samples)
2. Experiment with T5-large for better performance
3. Try T5-v1.1 or FLAN-T5 variants
4. Implement answer verification/validation
5. Test on other QA datasets (SQuAD 2.0, TriviaQA)

---

## 15. References

1. Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5 Paper)
2. Rajpurkar, P., et al. (2016). "SQuAD: 100,000+ Questions for Machine Comprehension of Text"
3. HuggingFace Transformers Documentation
4. Google Research T5 GitHub Repository

---

## Appendix A: Hardware Specifications

| Component | Specification |
|-----------|---------------|
| GPU | Tesla T4 (16GB VRAM) |
| Platform | Google Colab |
| CUDA Version | 11.8 |
| PyTorch Version | 2.0+ |
| Transformers Version | 4.35+ |

---

## Appendix B: Training Logs

```
Epoch 1/2 [Train]: 100%|██████████| 1095/1095 [08:32<00:00]
Epoch 1/2 [Val]:   100%|██████████| 265/265 [01:45<00:00]

📊 Epoch 1 Results:
  Train Loss: 0.6357
  Val Loss: 0.0845

Epoch 2/2 [Train]: 100%|██████████| 1095/1095 [08:28<00:00]
Epoch 2/2 [Val]:   100%|██████████| 265/265 [01:42<00:00]

📊 Epoch 2 Results:
  Train Loss: 0.0443
  Val Loss: 0.0862

✅ Training Completed!
```

---

## Appendix C: Complete Evaluation Summary

```
┌────────────────────────────────────────────────────────────────┐
│                    TRAINING PERFORMANCE                         │
├────────────────────────────────────────────────────────────────┤
│  Final Training Loss:          0.0443                          │
│  Final Validation Loss:        0.0862                          │
│  Training Samples:             4,379                           │
│  Validation Samples:           1,057                           │
├────────────────────────────────────────────────────────────────┤
│                    EVALUATION METRICS                           │
├────────────────────────────────────────────────────────────────┤
│  Exact Match Score:            60.00%                          │
│  F1 Score:                     77.59%                          │
│  Evaluated Samples:            100                             │
├────────────────────────────────────────────────────────────────┤
│                  PREDICTION QUALITY                             │
├────────────────────────────────────────────────────────────────┤
│  ✅ Perfect Predictions:       14/20 (70%)                     │
│  🟢 Good Predictions:          2/20 (10%)                      │
│  🟡 Partial Predictions:       1/20 (5%)                       │
│  ❌ Poor Predictions:          3/20 (15%)                      │
└────────────────────────────────────────────────────────────────┘
```

---
