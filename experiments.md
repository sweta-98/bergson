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

SDPA attention produces different logits for different batch sizes (~20% argmax
disagreement). Fixed by defaulting to `attn_implementation="eager"`.

With eager attention, bf16 backward pass still produces batch-size-dependent
per-example gradients due to CUDA matmul precision (sum(sep) vs batched cos=0.997
for OLMo-7B). FP32 is perfectly deterministic. FP32+TF32 gives cos=0.9998.

| Precision | OLMo-7B sum(sep) vs batched |
|-----------|---------------------------|
| bf16 eager | 0.997 |
| fp32+tf32 eager | 0.9998 |
| fp32 eager | 1.000 |
