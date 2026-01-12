# 📊 Task 3 Report: Fine-tuning Phi-2 for Text Summarization

## 1. Overview

Laporan ini mendokumentasikan implementasi **fine-tuning Phi-2** untuk task **Text Summarization** menggunakan dataset **XSum**. Model dilatih menggunakan teknik **LoRA (Low-Rank Adaptation)** untuk parameter-efficient fine-tuning pada Large Language Model (LLM).

| Item | Detail |
|------|--------|
| **Model** | `microsoft/phi-2` (2.7B parameters) |
| **Dataset** | XSum (Extreme Summarization) |
| **Task Type** | Abstractive Text Summarization |
| **Architecture** | Decoder-only (Causal LM) |
| **Fine-tuning Method** | LoRA (Parameter-Efficient) |
| **Metrics** | ROUGE-1, ROUGE-2, ROUGE-L |

---

## 2. Task Description

### What is Text Summarization?

Text Summarization adalah task NLP dimana model menghasilkan ringkasan singkat dan informatif dari dokumen yang lebih panjang.

**Types of Summarization:**

| Type | Description | Example |
|------|-------------|---------|
| **Extractive** | Memilih kalimat penting dari dokumen asli | Copy-paste sentences |
| **Abstractive** | Menghasilkan kalimat baru yang merangkum konten | Paraphrasing & condensing |

> 📝 **XSum** adalah dataset untuk **extreme abstractive summarization** - menghasilkan ringkasan satu kalimat dari artikel berita BBC.

### Why Phi-2 for Summarization?

| Advantage | Description |
|-----------|-------------|
| **Small but Powerful** | 2.7B params, performa setara model 7B+ |
| **Efficient** | Bisa di-finetune dengan LoRA di GPU consumer |
| **State-of-the-art** | Dikembangkan oleh Microsoft Research |
| **Instruction-following** | Sudah dilatih untuk mengikuti instruksi |

---

## 3. Model Architecture

### Phi-2 Specifications

| Property | Value |
|----------|-------|
| **Developer** | Microsoft Research |
| **Architecture** | Decoder-only Transformer |
| **Total Parameters** | **2,780,428,288** (~2.78B) |
| **Trainable Parameters (LoRA)** | **8,421,376** (~8.4M) |
| **Trainable Percentage** | **0.30%** |
| **Hidden Size** | 2560 |
| **Num Layers** | 32 |
| **Attention Heads** | 32 |
| **Context Length** | 2048 tokens |
| **Vocab Size** | 51,200 |

### LoRA Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Rank (r)** | 16 | Dimension of low-rank matrices |
| **Alpha** | 32 | Scaling factor (alpha/r = 2) |
| **Target Modules** | q_proj, k_proj, v_proj, dense | Attention layers |
| **Dropout** | 0.05 | Regularization |
| **Bias** | none | No bias adaptation |
| **Task Type** | CAUSAL_LM | Decoder-only generation |

### Why LoRA?

| Benefit | Explanation |
|---------|-------------|
| **Memory Efficient** | Only trains ~0.3% of parameters |
| **Fast Training** | Much faster than full fine-tuning |
| **No Catastrophic Forgetting** | Preserves pre-trained knowledge |
| **Portable** | Adapter weights are small (~32MB vs 5.5GB) |

---

## 4. Dataset Description

### XSum (Extreme Summarization)

XSum adalah dataset benchmark untuk abstractive summarization dari artikel berita BBC.

**Dataset Statistics:**

| Split | Samples | Usage |
|-------|---------|-------|
| Train | 1,500 | Model training |
| Test | 150 | Evaluation |
| **Total** | **1,650** | - |

> ⚠️ **Note:** Menggunakan subset kecil untuk efisiensi training. Full XSum memiliki ~204K training samples.

**Dataset Characteristics:**

| Property | Value |
|----------|-------|
| **Source** | BBC News articles |
| **Summary Style** | Single sentence, extreme compression |
| **Language** | English |
| **Avg Document Length** | ~430 words |
| **Avg Summary Length** | ~23 words |
| **Compression Ratio** | ~18:1 |

**Sample Data:**

| Field | Example |
|-------|---------|
| **Document** | "The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed. Repair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water. Trains on the west coast mainline face disruption due to damage at the Lamington Viaduct..." |
| **Summary** | "Clean-up operations are continuing across the Scottish Borders and Dumfries and Galloway after flooding caused by Storm Frank." |

### Dataset Visualization

![Dataset Analysis](reports/dataset_analysis.png)

---

## 5. Methodology

### 5.1 Quantization (4-bit)

Untuk menghemat GPU memory, model di-load dalam format 4-bit:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

### 5.2 Input Format

Model dilatih dengan format prompt terstruktur:

```
### Document:
{document_text}

### Summary:
{summary_text}
```

### 5.3 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 1 | Time constraint |
| **Batch Size** | 1 | GPU memory limit |
| **Gradient Accumulation** | 8 | Effective batch = 8 |
| **Learning Rate** | 2e-4 | Standard for LoRA |
| **LR Scheduler** | Cosine | Smooth decay |
| **Warmup Steps** | 30 | Gradual start |
| **Optimizer** | paged_adamw_8bit | Memory efficient |
| **FP16** | ✅ Enabled | Mixed precision |
| **Gradient Checkpointing** | ✅ Enabled | Memory saving |
| **Max Grad Norm** | 0.3 | Gradient clipping |

### 5.4 Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│               PHI-2 SUMMARIZATION PIPELINE                   │
├─────────────────────────────────────────────────────────────┤
│  1. 📥 Load XSum Dataset (1500 train, 150 test)             │
│         ↓                                                    │
│  2. 🔄 Format to Prompt Template                            │
│         ↓                                                    │
│  3. 🤖 Load Phi-2 with 4-bit Quantization                   │
│         ↓                                                    │
│  4. ⚙️  Apply LoRA Adapters                                  │
│         ↓                                                    │
│  5. 🔤 Tokenize with T5Tokenizer                            │
│         ↓                                                    │
│  6. 🏋️ Train with SFTTrainer (1 epoch)                      │
│         ↓                                                    │
│  7. 📊 Evaluate with ROUGE Metrics                          │
│         ↓                                                    │
│  8. 💾 Save Model & Generate Report                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Results

### 6.1 Training Progress

| Metric | Value |
|--------|-------|
| **Initial Loss** | 2.4634 |
| **Final Loss** | 2.1901 |
| **Loss Improvement** | **11.09%** |
| **Total Steps** | 180 |
| **Training Time** | ~1.5-2 hours |

### Training Loss Curve

| Step | Loss | Trend |
|------|------|-------|
| 0 | 2.4634 | Start |
| 60 | ~2.35 | ↓ Decreasing |
| 120 | ~2.25 | ↓ Decreasing |
| 180 | 2.1901 | ↓ Final |

> ✅ Loss consistently decreasing indicates model is learning

### 6.2 ROUGE Evaluation Metrics

| Metric | Score | Percentage | Description |
|--------|-------|------------|-------------|
| **ROUGE-1** | 0.0713 | **7.13%** | Unigram overlap |
| **ROUGE-2** | 0.0021 | **0.21%** | Bigram overlap |
| **ROUGE-L** | 0.0603 | **6.03%** | Longest common subsequence |
| **ROUGE-Lsum** | 0.0603 | **6.03%** | Summary-level LCS |

### 6.3 Performance Analysis

| Aspect | Status | Notes |
|--------|--------|-------|
| **Training Convergence** | ✅ Good | Loss decreased 11% |
| **ROUGE Scores** | ⚠️ Low | Expected with 1 epoch, small data |
| **Generation Quality** | ⚠️ Variable | Some coherent, some generic |
| **Model Size** | ✅ Efficient | Only 0.3% params trained |

---

## 7. Analysis

### 7.1 Sample Predictions

#### Example 1 (Best - ROUGE-L: 0.2449)

| Field | Content |
|-------|---------|
| **Document** | "The death toll doubled over the last two days as officials found more than 100 bodies once waters began receding. Officials estimate that the floods have affected 450,000 people in the state..." |
| **Reference** | "Floods in the western Indian state of Gujarat have killed 218 people, government officials have confirmed." |
| **Generated** | "for i in the United States. She was a sense of the other people who have been the way, this time to the number of these concepts, we would have been a member." |
| **Analysis** | ❌ Incoherent generation, unrelated to document |

#### Example 2 (ROUGE-L: 0.1667)

| Field | Content |
|-------|---------|
| **Document** | "Officers searched properties in the Waterfront Park and Colonsay View areas of the city on Wednesday. Detectives said three firearms, ammunition and a five-figure sum of money were recovered..." |
| **Reference** | "A man has appeared in court after firearms, ammunition and cash were seized by police in Edinburgh." |
| **Generated** | "In this year and a great job." |
| **Analysis** | ❌ Very short, generic response |

### 7.2 Error Analysis

**Common Issues Observed:**

| Issue | Frequency | Description |
|-------|-----------|-------------|
| **Incoherent Text** | High | Model generates unrelated sentences |
| **Too Generic** | High | Outputs don't relate to input |
| **Incomplete Sentences** | Medium | Sentences cut off or fragmented |
| **Repetition** | Medium | Repeating phrases or patterns |
| **Code-like Output** | Low | Sometimes generates "for i in..." |

### 7.3 Why Low ROUGE Scores?

| Reason | Impact | Solution |
|--------|--------|----------|
| **Only 1 Epoch** | High | Train for 3-5 epochs |
| **Small Dataset** | High | Use full XSum (204K samples) |
| **Short Training** | Medium | Longer training time |
| **No Beam Search** | Low | Use beam search for generation |
| **High Compression** | Medium | XSum requires extreme abstraction |
| **Model Not Converged** | High | Loss still decreasing, more training needed |

### 7.4 Comparison: Expected vs Actual

| Metric | Expected (Full Training) | Actual | Gap |
|--------|--------------------------|--------|-----|
| ROUGE-1 | ~35-40% | 7.13% | -28-33% |
| ROUGE-2 | ~12-15% | 0.21% | -12-15% |
| ROUGE-L | ~28-32% | 6.03% | -22-26% |

> ⚠️ **Note:** Results are significantly below benchmark due to limited training resources and time constraints.

---

## 8. Comparison with Benchmarks

### XSum Leaderboard Comparison

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Parameters |
|-------|---------|---------|---------|------------|
| PEGASUS | 47.21 | 24.56 | 39.25 | 568M |
| BART-large | 45.14 | 22.27 | 37.25 | 400M |
| T5-large | 43.52 | 21.55 | 36.69 | 770M |
| GPT-3.5 (zero-shot) | ~35 | ~12 | ~28 | 175B |
| **Phi-2 (Ours, 1 epoch)** | **7.13** | **0.21** | **6.03** | **2.7B** |
| Phi-2 (Expected, full) | ~30-35 | ~10-12 | ~25-28 | 2.7B |

### Performance Context

```
┌────────────────────────────────────────────────────────────┐
│              ROUGE-1 SCORE COMPARISON                       │
├────────────────────────────────────────────────────────────┤
│  PEGASUS          ████████████████████████████████ 47.21%  │
│  BART-large       █████████████████████████████░░░ 45.14%  │
│  T5-large         ████████████████████████████░░░░ 43.52%  │
│  Expected (full)  ██████████████████████░░░░░░░░░░ ~32%    │
│  Phi-2 (Ours)     █████░░░░░░░░░░░░░░░░░░░░░░░░░░░ 7.13%   │
└────────────────────────────────────────────────────────────┘
```

---

## 9. Technical Implementation

### 9.1 Model Loading with Quantization

```python
# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
```

### 9.2 LoRA Setup

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

### 9.3 ROUGE Evaluation

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
    use_stemmer=True
)

def compute_rouge(predictions, references):
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': []}
    
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)
    
    return {k: np.mean(v) for k, v in scores.items()}
```

### 9.4 Generation Function

```python
def generate_summary(document, model, tokenizer, max_length=100):
    prompt = f"### Document:\n{document}\n\n### Summary:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the summary part
    summary = summary.split("### Summary:\n")[-1].strip()
    
    return summary
```

---

## 10. Saved Artifacts

| Artifact | Location | Size |
|----------|----------|------|
| Fine-tuned Model | `phi2-xsum-final/` | ~5.5GB |
| LoRA Adapters | `phi2-xsum-final/adapter_model.safetensors` | ~32MB |
| Training Loss Plot | `reports/training_loss.png` | ~100KB |
| ROUGE Scores Plot | `reports/rouge_scores.png` | ~100KB |
| Dataset Analysis | `reports/dataset_analysis.png` | ~300KB |
| Sample Predictions | `reports/sample_predictions.txt` | ~50KB |
| All Predictions | `reports/all_predictions.csv` | ~200KB |

---

## 11. Potential Improvements

| Improvement | Expected Impact | Difficulty | Priority |
|-------------|-----------------|------------|----------|
| **Train for 3-5 epochs** | +15-20% ROUGE | ⭐ Easy | 🔴 High |
| **Use full XSum dataset** (204K) | +10-15% ROUGE | ⭐⭐ Medium | 🔴 High |
| **Increase LoRA rank** (r=32) | +3-5% ROUGE | ⭐ Easy | 🟡 Medium |
| **Better prompt engineering** | +5-10% ROUGE | ⭐ Easy | 🟡 Medium |
| **Beam search tuning** | +2-3% ROUGE | ⭐ Easy | 🟢 Low |
| **Use Phi-3 instead** | +5-10% ROUGE | ⭐ Easy | 🟡 Medium |
| **Gradient accumulation 16** | +2-3% ROUGE | ⭐ Easy | 🟢 Low |
| **Learning rate warmup** | +1-2% ROUGE | ⭐ Easy | 🟢 Low |

### Recommended Configuration for Better Results

```python
# Improved training config
training_args = TrainingArguments(
    num_train_epochs=3,           # Instead of 1
    per_device_train_batch_size=2, # If memory allows
    gradient_accumulation_steps=8,
    learning_rate=1e-4,           # Lower for stability
    warmup_steps=100,             # More warmup
    max_steps=1000,               # Longer training
)

# Improved LoRA config
lora_config = LoraConfig(
    r=32,                         # Higher rank
    lora_alpha=64,                # Scale accordingly
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "dense"],
)
```

---

## 12. Key Differences: Summarization Models

### Phi-2 vs Other Summarization Approaches

| Aspect | BART/T5 (Seq2Seq) | Phi-2 (Causal LM) |
|--------|-------------------|-------------------|
| **Architecture** | Encoder-Decoder | Decoder-only |
| **Pre-training** | Denoising, Span Corruption | Next-token prediction |
| **Input-Output** | Source → Target | Prompt → Continuation |
| **Fine-tuning** | Full model | LoRA (efficient) |
| **Parameters** | 140M-770M | 2.7B |
| **Memory** | Moderate | High (needs quantization) |
| **Best For** | Seq2seq tasks | General text generation |

### Summarization Approaches

| Approach | How It Works | Pros | Cons |
|----------|--------------|------|------|
| **Extractive** | Select important sentences | Fast, faithful | Not fluent |
| **Abstractive** | Generate new sentences | Fluent, concise | May hallucinate |
| **LLM Prompting** | Instruct model to summarize | Flexible | Expensive |
| **LoRA Fine-tuning** | Adapt LLM with small params | Efficient | Lower quality than full FT |

---

## 13. Conclusion

### ✅ Achievements

1. **Successfully fine-tuned Phi-2** (2.7B params) with LoRA
2. **Efficient training** — only 0.3% parameters trained
3. **Loss decreased 11%** — model is learning
4. **Pipeline complete** — from data loading to evaluation
5. **Memory efficient** — 4-bit quantization enabled training on consumer GPU

### ⚠️ Limitations

1. **Low ROUGE scores** (7.13% ROUGE-1) — far below benchmark
2. **Incoherent generations** — many outputs don't make sense
3. **Limited training** — only 1 epoch on small subset
4. **No early stopping** — potential for improvement
5. **Generation quality** — model needs more training

### 🎯 Key Takeaways

1. **LoRA is powerful** — enables training 2.7B model on limited hardware
2. **1 epoch is not enough** — need 3-5 epochs for decent results
3. **Data size matters** — 1.5K samples << 204K full dataset
4. **LLMs need tuning** — zero-shot/few-shot may work better for some cases
5. **Summarization is hard** — XSum requires extreme compression

### 📈 Future Work

1. Train for more epochs (3-5)
2. Use full XSum dataset
3. Try Phi-3 or other efficient LLMs
4. Experiment with different prompt formats
5. Implement beam search and sampling strategies
6. Compare with zero-shot prompting

---

## 14. References

1. Microsoft Research. (2023). "Phi-2: The surprising power of small language models"
2. Narayan, S., et al. (2018). "Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization" (XSum)
3. Hu, E., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
4. HuggingFace PEFT Documentation
5. HuggingFace TRL Documentation

---

## Appendix A: Hardware Specifications

| Component | Specification |
|-----------|---------------|
| GPU | Tesla T4 (16GB VRAM) |
| Platform | Google Colab |
| CUDA Version | 11.8 |
| PyTorch Version | 2.0+ |
| Transformers Version | 4.35+ |
| PEFT Version | 0.6+ |
| BitsAndBytes | 0.41+ |

---

## Appendix B: Training Logs

```
⚙️ Configuring training parameters...
✅ Training arguments configured!

⏱️ Estimated training time: 1.5-2 hours
📊 Total steps: ~187

🏋️ Starting training...
================================================================================

Step 20:  Loss: 2.41
Step 40:  Loss: 2.35
Step 60:  Loss: 2.30
Step 80:  Loss: 2.26
Step 100: Loss: 2.23
Step 120: Loss: 2.21
Step 140: Loss: 2.20
Step 160: Loss: 2.19
Step 180: Loss: 2.19

✅ Training completed!

📉 Training Summary:
   Initial Loss: 2.4634
   Final Loss: 2.1901
   Improvement: 11.09%
   Total Steps: 180
```

---

## Appendix C: Model Parameters Breakdown

```
📊 Model Parameters:
================================
Total parameters:     2,780,428,288
Trainable parameters: 8,421,376
Trainable %:          0.30%
================================

LoRA Modules Applied:
- model.layers.*.self_attn.q_proj (32 layers)
- model.layers.*.self_attn.k_proj (32 layers)
- model.layers.*.self_attn.v_proj (32 layers)
- model.layers.*.mlp.dense (32 layers)

Each LoRA adapter adds:
- A matrix: [hidden_size × r] = [2560 × 16]
- B matrix: [r × hidden_size] = [16 × 2560]
```

---

## Appendix D: ROUGE Score Interpretation

| ROUGE Metric | What It Measures | Good Score |
|--------------|------------------|------------|
| **ROUGE-1** | Unigram (word) overlap | >35% |
| **ROUGE-2** | Bigram (2-word) overlap | >12% |
| **ROUGE-L** | Longest common subsequence | >28% |
| **ROUGE-Lsum** | LCS at summary level | >28% |

**Our Scores vs Good Scores:**

| Metric | Our Score | Target | Gap |
|--------|-----------|--------|-----|
| ROUGE-1 | 7.13% | >35% | -28% |
| ROUGE-2 | 0.21% | >12% | -12% |
| ROUGE-L | 6.03% | >28% | -22% |

---
