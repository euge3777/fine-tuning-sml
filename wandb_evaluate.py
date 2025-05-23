import wandb
import json
import pandas as pd

def main():
    metrics_path = "evaluation_metrics.json"
    error_analysis_path = "error_analysis.json"
    project_name = "mistral-lora-eval"
    run_name = "eval-run"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    with open(error_analysis_path, "r", encoding="utf-8") as f:
        error_cases = json.load(f)

    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "metrics_path": metrics_path,
            "error_analysis_path": error_analysis_path,
        }
    )

    wandb.log(metrics)

    if error_cases:
        for case in error_cases:
            if "metrics" in case:
                case.update(case.pop("metrics"))
        df = pd.DataFrame(error_cases)
        wandb.log({"error_analysis": wandb.Table(dataframe=df)})

    print("Logged evaluation metrics and error analysis to wandb.")
    wandb.finish()

if __name__ == "__main__":
    main()