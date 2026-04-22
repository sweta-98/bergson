# Semantic Metrics

Compute and display metrics for semantic attribution experiments. Use this to quickly check results from existing experiment runs without re-running the full pipeline.

## Usage

```
/semantic-metrics [experiment] [options]
```

Arguments:
- `experiment` - Which experiment: `attribute` (default), `asymmetric`, or path to custom run

Options:
- `--preconditioner NAME` - Specific preconditioner to evaluate (default: all available)
- `--base-path PATH` - Base path for experiment outputs

## Instructions

### For attribute preservation experiment:

```python
from examples.semantic.attribute_preservation import (
    AttributePreservationConfig,
    compute_attribute_metrics,
    print_attribute_metrics,
)

config = AttributePreservationConfig()
base_path = 'runs/attribute_preservation'

# Compute for each available preconditioner
for name in ['no_precond', 'r_between', 'h_eval']:
    precond = name if name != 'no_precond' else None
    metrics = compute_attribute_metrics(config, base_path, precond)
    print_attribute_metrics(metrics, name)

# Also compute majority control if available
from pathlib import Path
if (Path(base_path) / 'data' / 'eval_majority.hf').exists():
    from examples.semantic.attribute_preservation import compute_majority_style_metrics
    majority = compute_majority_style_metrics(config, base_path, None)
    print_attribute_metrics(majority, 'majority_no_precond')
```

### For asymmetric experiment:

```python
from examples.semantic.asymmetric import (
    AsymmetricConfig,
    compute_asymmetric_metrics,
    print_metrics,
)

config = AsymmetricConfig()
base_path = 'runs/asymmetric'

for name in ['no_precond', 'r_between']:
    precond = name if name != 'no_precond' else None
    metrics = compute_asymmetric_metrics(config, base_path, precond)
    print_metrics(metrics, name)
```

## Key Metrics Explained

### Attribute Preservation
- **Fact Accuracy**: Exact semantic match (same person, same fact)
- **Occupation Accuracy**: Same occupation cluster (attribute preservation)
- **Style-Only Match**: Style matches but occupation doesn't (style leakage)
- **Trade-off**: Occ Acc - Style Only (overall quality)

### Asymmetric
- **Fact Accuracy**: Correct fact retrieval
- **Style Leakage**: Matching based on style rather than content
