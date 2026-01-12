import os
import sys
import json
import torch
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager


def run_local_evaluation(model, tokenizer, output_dir, step="final", tasks=["wmdp_bio_robust", "mmlu_stem", "wmdp_bio_cloze_verified"]):
    """Run local evaluation using a wrapper script with a surgical fix for the path error"""
    try:
        checkpoint_path = f"{output_dir}/eval_checkpoint_{step}"
        os.makedirs(checkpoint_path, exist_ok=True)

        print(f"Saving temporary checkpoint to {checkpoint_path}...")

        # 1. Save Tokenizer
        tokenizer.save_pretrained(checkpoint_path)

        # 2. Save Adapters
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(checkpoint_path)

        # 3. Save Truncated Base Model Weights
        if hasattr(model_to_save, "get_base_model"):
            model_to_save.get_base_model().save_pretrained(checkpoint_path)
        else:
            model_to_save.base_model.save_pretrained(checkpoint_path)

        # 4. Create the wrapper script
        wrapper_script_content = """
import sys
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
import lm_eval.tasks

# === SURGICAL FIX ===
# The library tries to print task paths and crashes on custom paths.
# We replace the printing function with an empty one to bypass the error cleanly.
lm_eval.tasks.pretty_print_task = lambda *args, **kwargs: None

# === JSON FIX ===
# Handles saving torch/numpy types to JSON
class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.dtype):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            return str(obj)
        except:
            return super().default(obj)

def run():
    checkpoint = sys.argv[1]
    task_name = sys.argv[2]
    output_file = sys.argv[3]
    include_path = sys.argv[4]

    print(f"Loading model manually from {checkpoint}...")
    
    # Load Base Model (Truncated)
    base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint, 
        trust_remote_code=True,
        torch_dtype=torch.float32, 
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Load Adapters
    model = PeftModel.from_pretrained(base_model, checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Wrap in lm_eval HFLM (dtype=None prevents the NeoX crash)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto", dtype=None)

    # Setup Tasks
    tm = TaskManager(verbosity="INFO", include_path=include_path)

    print(f"Starting evaluation for {task_name}...")
    results = simple_evaluate(
        model=lm,
        tasks=[task_name],
        task_manager=tm,
    )

    print(f"Saving results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, cls=SafeEncoder)
    
    print("Evaluation finished successfully.")

if __name__ == "__main__":
    run()
"""
        wrapper_path = os.path.join(output_dir, "run_eval_wrapper.py")
        with open(wrapper_path, "w") as f:
            f.write(wrapper_script_content)

        # 5. Run the wrapper script for each task
        for task in tasks:
            task_results_path = f"{output_dir}/eval_results_{step}_{task}.json"
            include_path = "/home/luciarosequirke/bergson/bergson/unlearn/lm_eval_tasks"

            cmd = [
                "python", wrapper_path,
                checkpoint_path,
                task,
                task_results_path,
                include_path
            ]

            print(f"Running custom eval wrapper for {task}...")
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/luciarosequirke/bergson")

            if result.returncode == 0:
                print(f"✅ {task} evaluation completed successfully")
                if os.path.exists(task_results_path):
                    with open(task_results_path, 'r') as f:
                        print(f"Preview: {f.read(500)}...")
            else:
                print(f"❌ {task} failed")
                print("=" * 20 + " FULL STDERR " + "=" * 20)
                print(result.stderr)
                exit(1)

        print(f"Evaluation round completed for step {step}")

    except Exception as e:
        print(f"Exception during evaluation at step {step}: {e}")
        import traceback
        traceback.print_exc()

    return checkpoint_path