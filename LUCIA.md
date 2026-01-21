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
torchrun --nproc_per_node=8 bergson/tuned_lens/train.py --batch_size 4 --gradient_accumulation_steps 1 --upload_to_hf True --hf_repo_id 'EleutherAI/deep-ignorance-unfiltered-lens'
```