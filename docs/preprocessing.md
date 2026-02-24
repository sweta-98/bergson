# Gradient Preprocessing

Bergson supports several gradient preprocessing operations that affect the quality and meaning of similarity scores. This page explains the operations available, when to apply them to query versus index gradients, and walks through concrete use cases.

## Operations

**Optimizer normalization** (`--normalizer`): Scales each gradient element by an estimate of the inverse standard deviation of that parameter's gradient distribution. Applied elementwise during gradient collection using Adam or Adafactor running statistics. This downweights parameters with high gradient variance and amplifies signal in stable, task-specific directions.

**Unit normalization** (`--unit_normalize`): Normalizes each gradient vector to unit L2 norm before similarity computation, enabling cosine similarity when used with inner product scoring.

**Preconditioning** (`--query_preconditioner_path`, `--index_preconditioner_path`): Applies a per-module matrix transformation derived from a Hessian approximation (second moment matrix of gradients). For inner product scoring, H⁻¹ is applied to the query side. For cosine similarity scoring, H^(-1/2) must be applied to both sides symmetrically.

## Query vs Index Gradients

Every similarity computation involves two sides:

- **Index gradients**: Gradients from the training dataset you want to search.
- **Query gradients**: Gradients from the dataset whose most similar training examples you want to find.

For a similarity score to be meaningful, preprocessing applied to query and index gradients must be consistent.

| Operation | Can apply one-sided? | Notes |
|-----------|---------------------|-------|
| Optimizer normalization | Yes | Apply the same `--normalizer` when collecting both query and index gradients |
| Preconditioning (inner product) | Yes | H⁻¹ applied to query only; relative score rankings are preserved |
| Preconditioning (cosine similarity) | **No** | H^(-1/2) must be applied to **both** sides before unit normalization |
| Unit normalization | **No** | Must be applied consistently to both sides |

**Unit normalization is a non-linear operation and does not commute with preconditioning.** When unit normalization is enabled alongside preconditioning, the preconditioner must be applied to both query and index gradients before normalization. Bergson handles this automatically: when `unit_normalize=True`, it applies H^(-1/2) to the query gradient upfront in the `score` command and applies H^(-1/2) to each index gradient as it is collected during scoring.

## Case Studies

### Cosine similarity with an optimizer normalizer (full gradients)

**Goal:** Rank training examples by cosine similarity to a query, using optimizer-normalized gradients.

Optimizer normalization scales each parameter's gradient by 1/sqrt(v), where v is an exponential moving average of squared gradients. Applied before cosine similarity, this reweights the gradient space by the inverse of per-parameter gradient noise, emphasizing consistent parameter updates over noisy ones.

The normalizer is applied during gradient collection, so the same `--normalizer` must be set when collecting both query and index gradients. Unit normalization is then applied at scoring time to obtain cosine similarity.

```bash
# Reduce query dataset to a single mean gradient with optimizer normalization
bergson reduce runs/query \
    --model EleutherAI/pythia-160m \
    --dataset query_data \
    --projection_dim 0 \
    --normalizer adafactor \
    --method mean \
    --skip_preconditioners

# Score: collect training gradients with the same normalizer, unit normalize for cosine similarity
bergson score runs/scores \
    --query_path runs/query \
    --model EleutherAI/pythia-160m \
    --dataset training_data \
    --projection_dim 0 \
    --normalizer adafactor \
    --unit_normalize
```

Both commands use `--projection_dim 0` to preserve the full gradient, and the same `--normalizer` to ensure consistent per-parameter scaling. The `score` command applies unit normalization to both the loaded query gradient and each training gradient, giving cosine similarity in the optimizer-normalized space.

### Inner product with an optimizer normalizer (full gradients)

**Goal:** Rank training examples by inner product with a query gradient in optimizer-normalized space, approximating the classic influence function.

The influence function estimates the change in query loss from upweighting a training example as `∂L_q/∂ε_t ≈ -g_q H⁻¹ g_t^T`, where H is the Hessian. The optimizer normalizer provides a diagonal approximation to H^(-1/2), so applying it to both query and index gradients approximates the full influence inner product.

Unlike cosine similarity, inner product preserves gradient magnitude, so training examples with larger gradients contribute more to the score.

```bash
# Reduce query dataset to a single mean gradient with optimizer normalization
bergson reduce runs/query \
    --model EleutherAI/pythia-160m \
    --dataset query_data \
    --projection_dim 0 \
    --normalizer adafactor \
    --method mean \
    --skip_preconditioners

# Score: inner product (no --unit_normalize)
bergson score runs/scores \
    --query_path runs/query \
    --model EleutherAI/pythia-160m \
    --dataset training_data \
    --projection_dim 0 \
    --normalizer adafactor
```

**Inner product vs cosine similarity:** Use inner product when gradient magnitude carries information (larger gradients indicate stronger relevance). Use cosine similarity to compare direction independently of magnitude, which is more robust when examples differ systematically in gradient norm (e.g., due to different sequence lengths or loss scales).

### Randomly projected gradients with reduce and score

**Goal:** Select training examples most similar to a query set using random projection, keeping full-batch scoring tractable for large models.

Random projections (Johnson-Lindenstrauss) approximately preserve inner products and cosine similarities while reducing gradient dimensionality by orders of magnitude. For large models, full gradients may be gigabytes per example; projecting to a few thousand dimensions makes the `reduce → score` pipeline tractable while retaining most of the signal.

`reduce` aggregates all query gradients into a single vector (mean or sum) without storing any per-example gradients. `score` then collects each training gradient on-the-fly and scores it against the precomputed query vector, avoiding the need to build or store a full training gradient index.

```bash
# Reduce query dataset to a single mean gradient vector
bergson reduce runs/query \
    --model EleutherAI/pythia-160m \
    --dataset query_data \
    --projection_dim 4096 \
    --method mean \
    --skip_preconditioners

# Score training data against the reduced query
bergson score runs/scores \
    --query_path runs/query \
    --model EleutherAI/pythia-160m \
    --dataset training_data \
    --projection_dim 4096
```

Both commands must use the same `--projection_dim` and identical model configuration so that both sides are projected into the same random subspace. The random projection matrix is derived deterministically from the model architecture and the projection dimension.

**Note on preprocessing order:** Optimizer normalization must be applied during gradient collection (set `--normalizer` at both `reduce` and `score` time). It cannot be applied after the mean-reduction in `reduce`, since applying normalizer to the mean gradient is not the same as normalizing each gradient then taking the mean.

### Randomly projected gradients with unit normalization, preconditioners, build, and score

**Goal:** Compute preconditioner-weighted cosine similarity using random projections. This is the approach used by the `trackstar` command.

When combining preconditioning with cosine similarity, the preconditioner must be applied before unit normalization to both query and index gradients. Bergson applies H^(-1/2) to the query gradient at the start of `score`, and H^(-1/2) to each index gradient as it is collected. The resulting score is:

```
g_q_p = g_q @ H^(-1/2)
g_t_p = g_t @ H^(-1/2)
score(q, t) = (g_q_p / ‖g_q_p‖) · (g_t_p / ‖g_t_p‖)
```

This is cosine similarity in the H⁻¹-weighted inner product space — the same geometry used by the influence function.

```bash
# Step 1: Compute normalizers and preconditioners on the query dataset
bergson preconditioners runs/query_precond \
    --model EleutherAI/pythia-160m \
    --dataset query_data \
    --projection_dim 4096

# Step 2: Compute normalizers and preconditioners on the training dataset
bergson preconditioners runs/index_precond \
    --model EleutherAI/pythia-160m \
    --dataset training_data \
    --projection_dim 4096

# Step 3: Build per-example query gradient index
# The query normalizer (from runs/query_precond) is applied during collection
bergson build runs/query \
    --model EleutherAI/pythia-160m \
    --dataset query_data \
    --projection_dim 4096 \
    --processor_path runs/query_precond \
    --skip_preconditioners

# Step 4: Score training data against query
# H^(-1/2) is applied to both query and index gradients, then unit normalized
bergson score runs/scores \
    --query_path runs/query \
    --model EleutherAI/pythia-160m \
    --dataset training_data \
    --projection_dim 4096 \
    --processor_path runs/index_precond \
    --skip_preconditioners \
    --unit_normalize \
    --query_preconditioner_path runs/query_precond \
    --index_preconditioner_path runs/index_precond
```

This pipeline is also available as the `trackstar` command, which automates the four steps above. See `bergson trackstar --help` for the full argument list.

**Why H^(-1/2) on both sides?** For inner product scoring, applying H⁻¹ to one side only is sufficient since the relative ordering of `g_q H⁻¹ g_t^T` is preserved. For cosine similarity, the unit normalization would undo a one-sided application: normalizing `g_t` to unit norm discards the preconditioner's geometry. Applying H^(-1/2) symmetrically to both sides before normalization preserves the preconditioned structure and ensures the normalization operates in the correct space.

**Mixing query and index preconditioners:** When query and index datasets come from different distributions, `--mixing_coefficient` (default 0.99) interpolates between their second moment matrices:

```
H_mixed = α * H_query + (1 - α) * H_index
```

Values close to 1.0 weight the query distribution more heavily; values close to 0.0 weight the index distribution. Adjust this when the query dataset is small (causing noisy H_query estimates) or when the query and index distributions diverge significantly.
