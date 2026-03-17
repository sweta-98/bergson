# OLMo WMDP Experiment Log

Base model: `allenai/OLMo-2-1124-7B-Instruct`
f+r = forget+retain, r+p = retain+pile, f+p = forget+pile

## Training Runs

| # | Model | Data | Optimizer | Precision | Status | Path |
|---|-------|------|-----------|-----------|--------|------|
| 1 | LoRA | f+r | adamw_8bit | bf16 | DONE | `runs/olmo_wmdp_lora/final_adapter` |
| 2 | LoRA | f+r | adamw_torch | bf16 | DONE | `runs/olmo_wmdp_lora/20260316_092716/final_adapter` |
| 3 | LoRA | f+r | adamw_torch | fp32 | DONE | `runs/olmo_wmdp_lora_fp32/20260316_235720/final_adapter` |
| 4 | SFT | f+r | adamw_torch | bf16 | DONE | `runs/olmo_wmdp_sft/20260316_105701/final_model` |
| 5 | SFT | f+r | adamw_torch | fp32 | RUNNING (2913380) | — |
| 6 | LoRA | r+p | adamw_torch | bf16 | DONE | `runs/olmo_retain_pile_lora/20260316_233724/final_adapter` |
| 7 | LoRA | r+p | adamw_torch | fp32 | DONE | `runs/olmo_retain_pile_lora_fp32/20260317_004243/final_adapter` |
| 8 | SFT | r+p | adamw_torch | bf16 | DONE | `runs/olmo_retain_pile_sft/20260316_233724/final_model` |
| 9 | SFT | r+p | adamw_torch | fp32 | RUNNING (2913381) | — |
| 10 | LoRA | f+p | adamw_torch | bf16 | DONE | `runs/olmo_forget_pile_lora/20260317_024032/final_adapter` |
| 11 | LoRA | f+p | adamw_torch | fp32 | RUNNING (2909728) | — |
| 12 | SFT | f+p | adamw_torch | bf16 | DONE | `runs/olmo_forget_pile_sft/20260317_024028/final_model` |
| 13 | SFT | f+p | adamw_torch | fp32 | RUNNING (2913382) | — |

## Experiment Grid

Each cell = AUROC. Only showing comparisons matching training data:
- f+r models → forget vs retain
- r+p models → pile vs retain

### bf16 TrackStar — forget vs retain

| Normalizer | LoRA (f+r) | SFT (f+r) |
|------------|-----------|----------|
| none | 0.497 | 0.514 |
| bergson_adafactor | 0.491 | 0.498 |
| bergson_adam | 0.495 | — |
| opt_8bit_adam | 0.496 | — |
| opt_fp32_adam | 0.498 | 0.514 |

### bf16 TrackStar — pile vs retain

| Normalizer | LoRA (r+p) | SFT (r+p) |
|------------|-----------|----------|
| none | 0.435 | 0.474 |
| bergson_adafactor | 0.443 | 0.496 |
| bergson_adam | 0.447 | — |
| opt_fp32_adam | 0.460 | 0.474 |

### bf16 Raw Cosine Sim — forget vs retain

| Normalizer | LoRA (f+r) | SFT (f+r) |
|------------|-----------|----------|
| none | 0.497 | 0.534 |
| bergson_adafactor | 0.504 | 0.502 |
| bergson_adam | 0.505 | 0.506 |
| opt_8bit_adam | 0.503 | — |
| opt_fp32_adam | 0.502 | 0.534 |

### bf16 Raw Cosine Sim — pile vs retain

| Normalizer | LoRA (r+p) | SFT (r+p) |
|------------|-----------|----------|
| none | 0.372 | 0.486 |
| bergson_adafactor | 0.324 | 0.496 |
| bergson_adam | 0.332 | 0.489 |
| opt_fp32_adam | 0.385 | 0.486 |

### fp32 Raw Cosine Sim — forget vs retain

| Normalizer | LoRA (f+r) | SFT (f+r) |
|------------|-----------|----------|
| none | 0.497 | 0.534 |
| bergson_adafactor | 0.503 | 0.502 |
| bergson_adam | 0.504 | 0.506 |
| opt_8bit_adam | 0.503 | — |
| opt_fp32_adam | 0.501 | 0.534 |

### Projection Dim Scaling — SFT (f+r) forget vs retain

| dim | none (bf16) | opt_fp32 (bf16) | none (fp32) | opt_fp32 (fp32) |
|-----|------------|----------------|------------|----------------|
| 8 | 0.517 | 0.534 | 0.517 | 0.534 |
| 16 | 0.534 | 0.535 | 0.533 | 0.534 |
| 32 | 0.534 | 0.534 | 0.534 | 0.534 |
| 64 | 0.569 | 0.534 | 0.568 | 0.534 |
| 128 | 0.565 | 0.534 | 0.564 | 0.534 |
| 256 | OOM | 0.534 | — | — |

## Normalizer Comparison (bergson-fitted vs training optimizer buffer)

- Bergson adam vs training adam — cosine similarity: mean=0.8418
- Bergson adafactor vs training adam — cosine similarity: mean=0.9444

## Preconditioner Eigenvalue Analysis

- Value preconditioner: fully PSD, 0 negative eigenvalues
- Query preconditioner: 35% negative eigenvalues by count, negligible magnitude (max |-3.5e-6|)
- Plots: `runs/eigenvalue_spectra_all.png`, `runs/query_preconditioner_negative_eigenvalues.png`

## bf16 Batching Nondeterminism

Script: `scripts/test_batching_cosine.py`

SDPA attention produces different logits for different batch sizes.
On pythia-160m, ~20% argmax disagreement between batch_size=1 and batch_size=2.
Fixed by defaulting to `attn_implementation="eager"`.

With eager attention, bf16 backward pass still produces batch-size-dependent
per-example parameter gradients due to CUDA matmul precision. This is a
PyTorch/CUDA issue, not bergson — tested without any bergson hooks.

This is not a padding issue, because it occurs even when all sequences are 
length-matched such that there is no padding. It is not a multi-GPU issue,
as it occurs on a single device. 

### Pythia-160m (eager attention, 2x 1024-token sequences, no bergson hooks)

Cosine similarity of sum(separate per-example grads) vs batched grad:

| Precision | sep vs sep | sum(sep) vs batched |
|-----------|-----------|---------------------|
| bf16 | 1.000 | 0.484 |
| fp32+tf32 | 1.000 | 0.967 |
| fp32 | 1.000 | 1.000 |
| autocast bf16 | 1.000 | 0.498 |

### OLMo-2-7B-Instruct (eager attention, 2x ~1024-token sequences, no bergson hooks)

| Precision | sep vs sep | sum(sep) vs batched |
|-----------|-----------|---------------------|
| bf16 | 1.000 | 0.997 |
| fp32+tf32 | 1.000 | 0.9998 |
| fp32 | 1.000 | 1.000 |

The effect is much smaller on OLMo-7B than pythia-160m, likely an issue with Pythia
being trained in FP16 or using a less stable architecture.

However, test_padding_ratio.py uses OLMo on index builds and finds that while the mean
similarity is high, the minimum similarity is very low - 0.086.

## Filtered Fine-tuning Experiment

Tests whether attribution scores can identify high/low-influence training examples.
For each scoring method, selects top 10% and bottom 10% of examples from `data/wmdp_mixed`
(978 examples each), then fine-tunes LoRA adapters on each subset. Evaluates on WMDP Bio.

Training config matches original: LoRA r=128, alpha=256, targets=[q,k,v,o,gate_proj],
lr=1e-4, cosine schedule, 4 epochs, adamw_torch, bf16.

### Score distributions

| Scoring method | Top 10% forget% | Bottom 10% forget% |
|----------------|-----------------|-------------------|
| TrackStar nonorm | 49.2% | 50.7% |
| TrackStar adafactor | 47.1% | 49.9% |
| Raw cosine none | 49.5% | 51.0% |
| TrackStar opt_fp32 adam | 49.3% | 50.7% |
| Raw cosine opt_fp32 adam | 51.1% | 47.8% |

Note: all ~50% forget — scores don't separate forget from retain (consistent with AUROC ~0.50).

### Training runs (RUNNING)

| Scoring | Subset | Dataset |
|---------|--------|---------|
| TrackStar nonorm | top 10% | data/filtered_trackstar_nonorm_top |
| TrackStar nonorm | bottom 10% | data/filtered_trackstar_nonorm_bottom |
| TrackStar adafactor | top 10% | data/filtered_trackstar_adafactor_top |
| TrackStar adafactor | bottom 10% | data/filtered_trackstar_adafactor_bottom |
| Raw cosine none | top 10% | data/filtered_raw_cosine_none_top |
| Raw cosine none | bottom 10% | data/filtered_raw_cosine_none_bottom |
| TrackStar opt_fp32 adam | top 10% | data/filtered_trackstar_opt_fp32_top |
| TrackStar opt_fp32 adam | bottom 10% | data/filtered_trackstar_opt_fp32_bottom |
| Raw cosine opt_fp32 adam | top 10% | data/filtered_raw_cosine_opt_fp32_top |
| Raw cosine opt_fp32 adam | bottom 10% | data/filtered_raw_cosine_opt_fp32_bottom |

### WMDP Bio Accuracy (pending evaluation)

| Model | WMDP Bio Acc |
|-------|-------------|
| Base (no adapter) | pending |
| Original adapter (full mixed) | pending |
| TrackStar nonorm top 10% | pending |
| TrackStar nonorm bottom 10% | pending |
| TrackStar adafactor top 10% | pending |
| TrackStar adafactor bottom 10% | pending |
| Raw cosine none top 10% | pending |
| Raw cosine none bottom 10% | pending |
| TrackStar opt_fp32 top 10% | pending |
| TrackStar opt_fp32 bottom 10% | pending |
| Raw cosine opt_fp32 top 10% | pending |
| Raw cosine opt_fp32 bottom 10% | pending |