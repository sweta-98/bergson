# Gradient Debug

Inspect and debug gradient data from bergson experiments. Use this when experiments produce unexpected results (e.g., all scores identical, 0% accuracy) to check whether gradients were computed correctly.

## Usage

```
/gradient-debug [path] [options]
```

Arguments:
- `path` - Path to gradient store (index or eval_grads directory)

Options:
- `--check-zeros` - Check for zero/corrupted gradients
- `--similarity` - Compute pairwise similarity between samples
- `--compare PATH` - Compare gradients between two stores

## Instructions

### Check gradient health:

```python
import json
import numpy as np

path = 'runs/attribute_preservation/index'  # or user-specified path

with open(f'{path}/info.json') as f:
    info = json.load(f)

grads = np.fromfile(f'{path}/gradients.bin', dtype=np.float16)
grads = grads.reshape(info['num_grads'], -1).astype(np.float32)

print(f'Shape: {grads.shape}')
print(f'Min: {grads.min():.6f}')
print(f'Max: {grads.max():.6f}')
print(f'Mean: {grads.mean():.6f}')
print(f'Std: {grads.std():.6f}')
print(f'Non-zero entries: {np.count_nonzero(grads):,} / {grads.size:,}')

# Check for corruption
if grads.std() == 0:
    print('WARNING: All gradients are identical (likely corrupted)')
if np.count_nonzero(grads) == 0:
    print('ERROR: All gradients are zero!')
```

### Check pairwise similarity:

```python
# Sample a few gradients and compute cosine similarity
for i in range(min(5, len(grads))):
    for j in range(i+1, min(5, len(grads))):
        cos_sim = np.dot(grads[i], grads[j]) / (
            np.linalg.norm(grads[i]) * np.linalg.norm(grads[j]) + 1e-8
        )
        print(f'Cosine sim [{i}] vs [{j}]: {cos_sim:.4f}')
```

### Check score matrix:

```python
scores_path = 'runs/attribute_preservation/scores_no_precond/scores.npy'
scores = np.load(scores_path)

print(f'Shape: {scores.shape}')
print(f'Min: {scores.min():.4f}')
print(f'Max: {scores.max():.4f}')
print(f'Unique values: {len(np.unique(scores))}')

if len(np.unique(scores)) == 1:
    print('ERROR: All scores identical (gradients likely corrupted)')
if np.isnan(scores).any():
    print(f'WARNING: {np.isnan(scores).sum()} NaN values in scores')
```

## Common Issues

1. **All zeros**: Gradient computation failed silently. Re-run bergson build.
2. **All identical**: Normalization issue or corrupted storage.
3. **NaN scores**: Division by zero in preconditioner inverse. Check preconditioner matrix.
4. **Same top-k for all queries**: Eval gradients are identical (corrupted).

## Fix corrupted data

Delete and rebuild:
```bash
rm -rf runs/attribute_preservation/index runs/attribute_preservation/eval_grads*
rm -rf runs/attribute_preservation/scores_*
# Then re-run the experiment
```
