Create and/or activate the venv:

cd ~/bergson && python3 -m venv .venv && source .venv/bin/activate


I can use /ide to connect to the IDE if disconnected.


python scripts/eval_wmdp_robust.py --model_path ./out/DeepIgnorance_CB --batch_size 8
python scripts/eval_mmlu_stem.py --model_path ./out/DeepIgnorance_CB --batch_size 8

python scripts/eval_wmdp_robust.py --model_path EleutherAI/deep-ignorance-unfiltered --batch_size 8
python scripts/eval_mmlu_stem.py --model_path EleutherAI/deep-ignorance-unfiltered --batch_size 8

python scripts/eval_mmlu_stem.py --model_path ./out/DeepIgnorance_checkpoint --batch_size 8
python scripts/eval_wmdp_robust.py --model_path ./out/DeepIgnorance_checkpoint --batch_size 8

python scripts/eval_mmlu_stem.py --model_path ./runs/lens_unlearn_test --batch_size 8
python scripts/eval_wmdp_robust.py --model_path ./runs/lens_unlearn_test --batch_size 8

### Tuned Lens

1. Download data

```
python -m bergson.bergson.unlearn.create_unlearn_data         
```

2. Train lens

```
python -m bergson.tuned_lens.train
```