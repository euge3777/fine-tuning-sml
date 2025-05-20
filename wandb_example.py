import wandb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import numpy as np
from datetime import datetime

def demo_wandb_tracking():
    # 1. Start a new run
    wandb.init(
        project="lora-fine-tuning-demo",
        name=f"demo-run-{datetime.now().strftime('%Y%m%d-%H%M')}",
        config={
            "model": "gpt2",
            "lora_r": 16,
            "lora_alpha": 32,
            "learning_rate": 1e-4,
            "epochs": 3
        }
    )
    
    # 2. Log model architecture
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    wandb.watch(model, log="all")  # Log gradients and parameters
    
    # 3. Track training progress (simulated)
    for epoch in range(3):
        # Simulate training metrics
        train_loss = 2.0 * np.exp(-0.5 * epoch) + np.random.normal(0, 0.1)
        val_loss = 2.2 * np.exp(-0.45 * epoch) + np.random.normal(0, 0.1)
        rouge_score = 0.5 + 0.15 * (1 - np.exp(-0.6 * epoch)) + np.random.normal(0, 0.05)
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "rouge_score": rouge_score,
            
            # Track learning rate
            "learning_rate": wandb.config.learning_rate * (0.9 ** epoch),
            
            # Track custom visualizations
            "loss_plot": wandb.plot.line_series(
                xs=[[0, 1, 2], [0, 1, 2]],
                ys=[[2.0, 1.5, 1.0], [2.2, 1.7, 1.2]],
                keys=["train", "val"],
                title="Loss Curves",
                xname="epoch"
            )
        })
        
        # Log example predictions (simulated)
        if epoch == 2:
            example_table = wandb.Table(columns=["question", "prediction", "ground_truth", "rouge_score"])
            example_table.add_data(
                "What is LoRA?",
                "LoRA is a parameter-efficient fine-tuning method.",
                "LoRA (Low-Rank Adaptation) is an efficient fine-tuning approach.",
                0.85
            )
            wandb.log({"prediction_examples": example_table})
    
    # 4. Log final model summary
    wandb.summary.update({
        "best_rouge_score": rouge_score,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    })
    
    # 5. Close the run
    wandb.finish()

if __name__ == "__main__":
    print("Starting wandb demo...")
    print("This will create a sample run to demonstrate wandb features.")
    print("Check your wandb dashboard after running!")
    demo_wandb_tracking() 