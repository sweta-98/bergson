# OLMo WMDP Experiment Log

Base model: `allenai/OLMo-2-1124-7B-Instruct`
Dataset: `data/wmdp_mixed` (4890 forget + 4890 retain bio examples)

## Training Runs

### 1. Original LoRA (adamw_8bit, bf16)
- **Adapter**: `runs/olmo_wmdp_lora/final_adapter`
- **Optimizer checkpoint**: `runs/olmo_wmdp_lora/checkpoint-308/optimizer.pt` (8-bit quantized, dequantizable)
- **Config**: r=128, alpha=256, targets=[q,k,v,o,gate_proj], dropout=0.1, rslora, lr=1e-4, cosine schedule, 4 epochs, bs=16, warmup=50
- **Wandb**: olmo_wmdp_lora

### 2. Full Adam LoRA (adamw_torch, bf16)
- **Adapter**: `runs/olmo_wmdp_lora/20260316_092716/final_adapter`
- **Optimizer checkpoint**: `runs/olmo_wmdp_lora/20260316_092716/checkpoint-612/optimizer.pt` (full fp32 exp_avg_sq)
- **Config**: Same LoRA as #1 but with `adamw_torch` instead of `adamw_8bit`
- **Sbatch job**: 2892162 (completed)

### 3. FP32 LoRA (pile + bio forget + bio retain, adamw_8bit, fp32 model)
- **Script**: `scripts/train_olmo_wmdp_fp32.py`
- **Config**: Same LoRA config as #1 but fp32 model dtype, dataset = pile-10k + wmdp_mixed (19780 examples)

### 4. Full SFT (adamw_torch, bf16)
- **Model**: `runs/olmo_wmdp_sft/20260316_105701/final_model`
- **Script**: `scripts/train_olmo_wmdp_sft.py`
- **Config**: Full finetune (no LoRA), adamw_torch, bf16, lr=2e-5, gradient_checkpointing, bs=4 x grad_accum=4, 4 epochs
- **Sbatch job**: 2894295 (completed)
- **Wandb**: olmo_wmdp_sft_20260316_105701

## TrackStar Runs

All use the original LoRA adapter (`runs/olmo_wmdp_lora/final_adapter`) and score
forget vs retain on `data/wmdp_mixed`. Normalizers below are **bergson-fitted**
(estimated from gradient statistics on the dataset), not from the training optimizer.

### 1. Bergson adafactor normalizer (original)
- **Run path**: `runs/olmo_wmdp` (or `runs/olmo_wmdp_lora_trackstar`)
- **Scores**: `runs/olmo_wmdp/scores/scores.bin`
- **Config**: `ablations/olmo_wmdp_lora.yaml`
- **Normalizer**: bergson-fitted adafactor
- **Result**: AUROC 0.4914 (no separation between forget/retain)
  - forget: mean=0.000072, std=0.000922
  - retain: mean=0.000095, std=0.000925

### 2. No normalizer
- **Run path**: `runs/olmo_wmdp_lora_trackstar_nonorm`
- **Config**: `ablations/olmo_wmdp_lora_nonorm.yaml`
- **Normalizer**: none
- **Result**: AUROC 0.5100 (no separation)
  - forget: mean=0.000102, std=0.000912
  - retain: mean=0.000066, std=0.000916

### 3. Bergson adam normalizer
- **Run path**: `runs/olmo_wmdp_lora_trackstar_adam`
- **Config**: `ablations/olmo_wmdp_lora_adam.yaml`
- **Normalizer**: bergson-fitted adam
- **Sbatch job**: 2894434 (completed)
- **Result**: AUROC 0.4949 (no separation)
  - forget: mean=0.000073, retain: mean=0.000089

### 4. Training optimizer normalizer (8-bit adam from original training)
- **Run path**: `runs/olmo_wmdp_optimizer_norm_trackstar`
- **Script**: `scripts/trackstar_optimizer_norm.py`
- **Normalizer**: 8-bit adam second moments from `runs/olmo_wmdp_lora/checkpoint-308/optimizer.pt`
- **Saved normalizers**: `runs/optimizer_adam_normalizers/`
- **Result**: AUROC 0.5098 (no separation)
  - forget: mean=0.000076, std=0.000896
  - retain: mean=0.000038, std=0.000890

### 5. SFT optimizer normalizer (fp32 adam from full-adam training)
- **Run path**: `runs/olmo_wmdp_sft_norm_trackstar`
- **Script**: `scripts/trackstar_sft_norm.py`
- **Normalizer**: fp32 adam second moments from `runs/olmo_wmdp_lora/20260316_094031/checkpoint-612/optimizer.pt`
- **Saved normalizers**: `runs/sft_adam_normalizers/`
- **Result**: AUROC 0.5102 (no separation)
  - forget: mean=0.000084, std=0.000899
  - retain: mean=0.000046, std=0.000893

### 6. No normalizer — pile-10k value dataset
- **Run path**: `runs/olmo_trackstar_nonorm_pile`
- **Config**: `ablations/olmo_wmdp_lora_nonorm_pile.yaml`
- **Normalizer**: none
- **Value data**: NeelNanda/pile-10k
- **Result**: N=10000, mean=-0.000224, std=0.000949

### 7. No normalizer — retain-only value dataset
- **Run path**: `runs/olmo_trackstar_nonorm_retain`
- **Config**: `ablations/olmo_wmdp_lora_nonorm_retain.yaml`
- **Normalizer**: none
- **Value data**: data/wmdp_retain
- **Result**: N=4890, mean=0.000094, std=0.000897

### Pile vs Retain (trackstar, no normalizer)
- **AUROC**: 0.5963 (retain=positive class)
- Cohen's d: -0.34 (pile scores shifted negative vs retain)
- Trackstar preconditioners compressed the signal vs raw cosine sim (raw had
  pile mean=-0.0027 vs retain mean=+0.00007, trackstar has -0.00022 vs +0.00009)

### Summary: Normalizer does not affect forget/retain separation
All normalizer variants give AUROC ~0.51 (random) in both trackstar and direct
gradient cosine sim. Confirmed that post-hoc normalizer application on projected
gradients has no effect (projection destroys per-element structure). Trackstar
runs apply normalizer pre-projection and still show no separation. The LoRA adapter
trained on mixed forget+retain data does not produce gradients that distinguish
the two sets — the adapter learns a shared bio-domain representation.

## Raw Cosine Similarity Runs (no preconditioners, no normalizers)

### 1. Mixed dataset (forget + retain)
- **Run path**: `runs/olmo_wmdp_raw`
- **Scores**: `runs/olmo_wmdp_raw/olmo_wmdp_raw.part/scores.bin`
- **Result**: AUROC 0.4967 (no separation)
  - forget: mean=0.000029, std=0.003340
  - retain: mean=0.000069, std=0.003348

### 2. Pile-10k
- **Run path**: `runs/olmo_raw_pile-10k`
- **Scores**: `runs/olmo_raw_pile-10k` (in .part subdir)
- **Result**: N=10000, mean=-0.002667, std=0.003760

### 3. WMDP retain only
- **Run path**: `runs/olmo_raw_wmdp-retain`
- **Scores**: `runs/olmo_raw_wmdp-retain` (in .part subdir)
- **Result**: N=4890, mean=0.000069, std=0.003348

**Key finding**: Pile-10k has negative mean cosine sim (-0.0027) vs near-zero for bio retain (0.00007). The adapter distinguishes "bio domain vs general text" but NOT "hazardous bio vs benign bio."

## Normalizer Comparison (bergson-fitted vs training optimizer buffer)

Compares normalizers estimated by bergson from gradient statistics against the
actual adam second moments accumulated during training.

- **Script**: `scripts/compare_normalizers.py`
- **Training optimizer source**: `runs/olmo_wmdp_lora/checkpoint-308` (8-bit, dequantized)
- **Bergson-fitted source**: `runs/olmo_wmdp/value_preconditioner`
- **Results**:
  - Bergson adam vs training adam — cosine similarity: mean=0.8418
  - Bergson adafactor vs training adam (as adafactor) — cosine similarity: mean=0.9444
  - Note: adafactor looks higher because comparing 1D row/col vectors is easier than full matrices. The materialized adafactor->adam comparison shows lower cosine (~0.61 for lora_B layers).

## Preconditioner Eigenvalue Analysis

- **Plot**: `runs/eigenvalue_spectra_all.png`, `runs/query_preconditioner_negative_eigenvalues.png`

### Value preconditioner
- Fully PSD: 0 negative eigenvalues out of 327,680
- Eigenvalues range: [0.004, 398.3], sharp decay with ~50 dominant dimensions per module

### Query preconditioner
- 35% negative eigenvalues (114,664 / 327,680), consistent across all modules (mean neg frac = 0.350)
- Negative eigenvalues are negligible in magnitude: min=-3.5e-6, mean=-4e-7
- Neg/pos mass ratio ~1e-7 per module — effectively numerical noise, not real signal
- Caused by insufficient query samples (1272 wmdp-bio test questions vs 9780 value examples)

### Mixed and scores preconditioners
- Eigen decompositions not cached (empty dicts)

## Available Artifacts

| Artifact | Path |
|----------|------|
| Original adapter (8-bit adam) | `runs/olmo_wmdp_lora/final_adapter` |
| Full adam adapter | `runs/olmo_wmdp_lora/20260316_092716/final_adapter` |
| 8-bit training optimizer buffer | `runs/olmo_wmdp_lora/checkpoint-308/optimizer.pt` |
| Full adam training optimizer buffer | `runs/olmo_wmdp_lora/20260316_092716/checkpoint-612/optimizer.pt` |
| Training optimizer as bergson normalizers | `runs/sft_adam_normalizers/` |
| Mixed dataset | `data/wmdp_mixed` (4890 forget + 4890 retain) |
| Retain-only dataset | `data/wmdp_retain` (4890 retain) |
| Bergson-fitted normalizers (adafactor) | `runs/olmo_wmdp/value_preconditioner/normalizers.pth` |
| Bergson-fitted preconditioners | `runs/olmo_wmdp/value_preconditioner/preconditioners.pth` |
| Mixed preconditioner | `runs/olmo_wmdp/mixed_preconditioner/` |
| Raw query gradients | `runs/olmo_wmdp_raw_query/gradients.bin` |
| SFT model (full finetune) | `runs/olmo_wmdp_sft/20260316_105701/final_model` |
| Eigenvalue spectra plot | `runs/eigenvalue_spectra_all.png` |
| Query negative eigenvalues plot | `runs/query_preconditioner_negative_eigenvalues.png` |
