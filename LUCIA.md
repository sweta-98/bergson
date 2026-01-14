Create and/or activate the venv:

cd ~/bergson && python3 -m venv .venv && source .venv/bin/activate


I can use /ide to connect to the IDE if disconnected.


python scripts/eval_wmdp_robust.py --model_path ./out/DeepIgnorance_CB --batch_size 8
python scripts/eval_wmdp_robust.py --model_path EleutherAI/deep-ignorance-unfiltered --batch_size 8
