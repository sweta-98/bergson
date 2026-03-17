# OLMo WMDP Experiment Log

Base model: `allenai/OLMo-2-1124-7B-Instruct`

## Training Runs

### 1. LoRA on forget+retain (adamw_8bit, bf16)
- **Adapter**: `runs/olmo_wmdp_lora/final_adapter`
- **Optimizer**: `runs/olmo_wmdp_lora/checkpoint-308/optimizer.pt` (8-bit quantized)
- **Config**: r=128, alpha=256, targets=[q,k,v,o,gate_proj], dropout=0.1, rslora, lr=1e-4, cosine, 4 epochs, bs=16
- **Data**: `data/wmdp_mixed` (4890 forget + 4890 retain bio)

### 2. LoRA on forget+retain (adamw_torch, bf16)
- **Adapter**: `runs/olmo_wmdp_lora/20260316_092716/final_adapter`
- **Optimizer**: `runs/olmo_wmdp_lora/20260316_092716/checkpoint-612/optimizer.pt` (fp32 exp_avg_sq)
- **Config**: Same as #1 but adamw_torch
- **Data**: `data/wmdp_mixed`

### 3. SFT on forget+retain (adamw_torch, bf16)
- **Model**: `runs/olmo_wmdp_sft/20260316_105701/final_model`
- **Optimizer**: `runs/olmo_wmdp_sft/20260316_105701/checkpoint-612/optimizer.pt` (fp32)
- **Config**: Full finetune, lr=2e-5, gradient_checkpointing, bs=4 x grad_accum=4, 4 epochs
- **Wandb**: https://wandb.ai/eleutherai/huggingface/runs/rlbcc7fc

### 4. LoRA on forget+retain (adamw_torch, fp32 model) — RUNNING (2907194)
- **Script**: `scripts/train_olmo_wmdp_fp32.py`
- **Data**: `data/wmdp_mixed` + pile-10k (19780 examples)
- **Config**: Same LoRA as #1 but fp32 model dtype, adamw_torch

### 5. SFT on forget+retain (adamw_torch, fp32 model) — RUNNING (2907195)
- **Script**: `scripts/train_olmo_wmdp_sft_fp32.py`
- **Data**: `data/wmdp_mixed`
- **Config**: Full finetune, fp32 model, adamw_torch, gradient_checkpointing, bs=2 x grad_accum=8

### 6. LoRA on retain+pile (adamw_torch, bf16) — RUNNING (2907125)
- **Script**: `scripts/train_olmo_retain_pile_lora.py`
- **Data**: `data/wmdp_retain_pile` (4890 retain bio + 10k pile general text, no forget)

### 7. SFT on retain+pile (adamw_torch, bf16) — RUNNING (2907126)
- **Script**: `scripts/train_olmo_retain_pile_sft.py`
- **Data**: `data/wmdp_retain_pile`

## Experiment Grid

Each cell = AUROC for the given comparison.

**Models**: trained on different data splits
- LoRA/SFT (forget+retain) = wmdp_mixed: 4890 forget bio + 4890 retain bio
- LoRA/SFT (retain+pile) = 4890 retain bio + 10k pile general text, no forget data

**Comparisons**:
- forget vs retain = score forget+retain dataset, separate by source label
- pile vs retain = score pile and retain separately, compare distributions

**Normalizers**:
- none = raw gradients
- bergson_adafactor = adafactor 2nd moments estimated from gradient statistics
- bergson_adam = adam 2nd moments estimated from gradient statistics
- opt_8bit_adam = adam 2nd moments from 8-bit training optimizer (LoRA only)
- opt_fp32_adam = adam 2nd moments from fp32 training optimizer

### LoRA (forget+retain) — bf16 TrackStar

| Normalizer | forget vs retain | pile vs retain |
|------------|-----------------|----------------|
| none | 0.497 | 0.596 |
| bergson_adafactor | 0.491 | 0.596 |
| bergson_adam | 0.495 | 0.594 |
| opt_8bit_adam | 0.496 | 0.573 |
| opt_fp32_adam | 0.498 | 0.577 |

### SFT (forget+retain) — bf16 TrackStar

| Normalizer | forget vs retain | pile vs retain |
|------------|-----------------|----------------|
| none | 0.514 | 0.548 |
| bergson_adafactor | 0.498 | 0.538 |
| bergson_adam | RUNNING | RUNNING |
| opt_fp32_adam | 0.514 | 0.548 |

### LoRA (forget+retain) — bf16 Raw Cosine Sim (pre-projection normalizers)

| Normalizer | forget vs retain | pile vs retain |
|------------|-----------------|----------------|
| none | 0.497 | 0.707 |
| bergson_adafactor | 0.504 | 0.668 |
| bergson_adam | 0.505 | 0.671 |
| opt_8bit_adam | 0.503 | 0.594 |
| opt_fp32_adam | 0.502 | 0.614 |

### SFT (forget+retain) — bf16 Raw Cosine Sim (pre-projection normalizers)

| Normalizer | forget vs retain | pile vs retain |
|------------|-----------------|----------------|
| none | 0.534 | 0.674 |
| bergson_adafactor | 0.502 | 0.576 |
| bergson_adam | 0.506 | 0.578 |
| opt_fp32_adam | 0.534 | 0.674 |

### LoRA (retain+pile) — bf16 Raw Cosine Sim (pre-projection normalizers)

| Normalizer | forget vs retain | pile vs retain |
|------------|-----------------|----------------|
| none | 0.079 | 0.372 |
| bergson_adafactor | 0.056 | 0.324 |
| bergson_adam | 0.067 | 0.332 |
| opt_fp32_adam | 0.107 | 0.385 |

### SFT (retain+pile) — bf16 Raw Cosine Sim (pre-projection normalizers)

| Normalizer | forget vs retain | pile vs retain |
|------------|-----------------|----------------|
| none | 0.569 | 0.486 |
| bergson_adafactor | 0.517 | 0.496 |
| bergson_adam | 0.498 | 0.489 |
| opt_fp32_adam | 0.569 | 0.486 |

### LoRA (forget+retain) — fp32 Raw Cosine Sim (pre-projection normalizers)

| Normalizer | forget vs retain | pile vs retain |
|------------|-----------------|----------------|
| none | 0.497 | 0.705 |
| bergson_adafactor | 0.503 | RUNNING |
| bergson_adam | 0.504 | RUNNING |
| opt_8bit_adam | 0.503 | 0.593 |
| opt_fp32_adam | 0.501 | 0.612 |

### SFT (forget+retain) — fp32 Raw Cosine Sim (pre-projection normalizers)

| Normalizer | forget vs retain | pile vs retain |
|------------|-----------------|----------------|
| none | 0.534 | 0.674 |
| bergson_adafactor | RUNNING | RUNNING |
| bergson_adam | RUNNING | RUNNING |
| opt_fp32_adam | 0.534 | 0.674 |

### TrackStar retain+pile models — RUNNING (2908935-2908942)
### fp32 TrackStar — RUNNING
### LoRA fp32 model — BLOCKED on training (2907194)
### SFT fp32 model — BLOCKED on training (2907305)
### LoRA (retain+pile) fp32 — BLOCKED on training (2907885)
### SFT (retain+pile) fp32 — BLOCKED on training (2907886)

## Normalizer Comparison (bergson-fitted vs training optimizer buffer)

- **Script**: `scripts/compare_normalizers.py`
- Bergson adam vs training adam — cosine similarity: mean=0.8418
- Bergson adafactor vs training adam — cosine similarity: mean=0.9444
- Adafactor looks higher because 1D row/col comparison is easier than full matrices

## Preconditioner Eigenvalue Analysis

- **Plots**: `runs/eigenvalue_spectra_all.png`, `runs/query_preconditioner_negative_eigenvalues.png`
- Value preconditioner: fully PSD, 0 negative eigenvalues
- Query preconditioner: 35% negative eigenvalues by count, but negligible magnitude (max |-3.5e-6|)
- Neg/pos mass ratio ~1e-7 per module — numerical noise
