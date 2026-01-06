import argparse
import os
import json
import wandb
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--wandb_run_id", required=True)
    parser.add_argument("--wandb_project", required=True)
    args = parser.parse_args()

    # Clear stale wandb service socket references
    os.environ.pop("WANDB_SERVICE", None)
    os.environ.pop("_WANDB_SOCK_PATH", None)

    wandb.init(project=args.wandb_project, id=args.wandb_run_id, resume="must")

    results_base = Path(args.results)

    # candidates = [
    #     results_base / "results.json",  # direct
    #     *results_base.glob("*/results.json"),  # nested subdir
    #     *results_base.parent.glob(f"{results_base.name}*.json"),  # timestamped json
    # ]
    candidates = [
        *results_base.glob("*/results_*.json"),  # nested subdir with timestamp
        *results_base.glob("results_*.json"),  # direct with timestamp
        *results_base.parent.glob(f"{results_base.name}*.json"),  # timestamped at parent level
    ]

    results_path = None
    for c in candidates:
        if c.exists():
            results_path = c
            break

    if not results_path:
        raise FileNotFoundError(f"No results.json found for {args.results}")

    with open(results_path) as f:
        results = json.load(f)

    metrics = {}
    for group_name, group_results in results.get("groups", {}).items(): 
        for key in ["acc,none", "acc_norm,none", "acc"]:
            if key in group_results:
                metrics[group_name.replace("-", "_") + "_acc"] = group_results[key]
                break

    eval_metrics = {f"eval/{k}": v for k, v in metrics.items()}
    eval_metrics["eval_step"] = args.step
    wandb.define_metric("eval_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    wandb.log(eval_metrics)

    wandb.finish()

if __name__ == "__main__":
    main()