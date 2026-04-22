# Preconditioner Analysis

Analyze and compare preconditioner strategies for style suppression. Preconditioners transform the gradient space to downweight certain directions (e.g., style) before computing similarity. This skill helps understand what each preconditioner is doing and how much variance it captures.

## Usage

```
/preconditioner-analysis [options]
```

Options:
- `--base-path PATH` - Experiment directory (default: runs/attribute_preservation)
- `--compare A B` - Compare two preconditioners directly
- `--eigenspectrum` - Analyze eigenvalue distribution

## Available Preconditioners

1. **R_between**: Computed from style mean differences
   - `delta = mean(style_A) - mean(style_B)`
   - `R_between = delta @ delta.T` (rank-1)
   - Downweights the "style direction"

2. **H_eval**: Second moment of eval gradients
   - `H_eval = (1/n) * G_eval.T @ G_eval`
   - Downweights directions that vary in eval set
   - Doesn't require style labels

3. **R_mixed**: Combination strategies (from preconditioners.py)

## Instructions

### Inspect R_between computation:

```python
from datasets import load_from_disk
from bergson.data import load_gradients
import numpy as np
import json

base_path = 'runs/attribute_preservation'
train_ds = load_from_disk(f'{base_path}/data/train.hf')
train_styles = np.array(train_ds['style'])

with open(f'{base_path}/index/info.json') as f:
    info = json.load(f)

# Load one module's gradients to inspect
grads = np.fromfile(f'{base_path}/index/gradients.bin', dtype=np.float16)
grads = grads.reshape(info['num_grads'], -1).astype(np.float32)

# Compute style means
styles = list(set(train_styles))
means = {}
for style in styles:
    idx = np.where(train_styles == style)[0]
    means[style] = grads[idx].mean(axis=0)
    print(f'{style}: {len(idx)} samples, mean norm: {np.linalg.norm(means[style]):.4f}')

# Style direction
delta = means[styles[0]] - means[styles[1]]
print(f'Style delta norm: {np.linalg.norm(delta):.4f}')

# How much variance is along style direction?
delta_unit = delta / (np.linalg.norm(delta) + 1e-8)
projections = grads @ delta_unit
print(f'Projection variance: {projections.var():.4f}')
print(f'Total variance: {grads.var():.4f}')
print(f'Style direction explains: {projections.var() / grads.var() * 100:.1f}% of variance')
```

### Compare preconditioner effects:

```python
import torch
from bergson.gradients import GradientProcessor

# Load preconditioners
r_between = GradientProcessor.load(f'{base_path}/r_between')
h_eval = GradientProcessor.load(f'{base_path}/h_eval')

# Compare a specific module
module = list(r_between.preconditioners.keys())[0]
R = r_between.preconditioners[module]
H = h_eval.preconditioners[module]

print(f'Module: {module}')
print(f'R_between rank: {torch.linalg.matrix_rank(R).item()}')
print(f'H_eval rank: {torch.linalg.matrix_rank(H).item()}')

# Eigenvalue analysis
eig_R = torch.linalg.eigvalsh(R)
eig_H = torch.linalg.eigvalsh(H)
print(f'R_between top 5 eigenvalues: {eig_R[-5:].tolist()}')
print(f'H_eval top 5 eigenvalues: {eig_H[-5:].tolist()}')
```

## Key Insight

R_between is computed on **training data**, mixing scientists and creatives in the shakespeare mean:
- `mean_shakespeare` = avg(scientists + creatives)
- `mean_pirate` = avg(business)

This means the "style direction" might also capture some occupation signal. A cleaner approach would compute style means within occupation to isolate pure style variation.
