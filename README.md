# fine-tuning-sml

taking inspiration from https://medium.com/data-science-collective/i-fine-tuned-an-llm-on-5-years-of-telegram-chats-7bacb66387c8

training logs: 
train_loss: 2.30
This is your final average training loss after 5 epochs. Lower loss generally means the model is fitting your data better. Compared to your previous runs (with higher loss), this is a good sign—your model is learning.

epoch: 5.0
Training completed for 5 full passes over your dataset.

train_samples_per_second: 0.712
About 0.7 samples processed per second (expected for large models on small datasets).

train_steps_per_second: 0.047
About 0.05 optimization steps per second.

train_runtime: 210.8 seconds
Total training time.

Fine-Tuning Mistral-7B with LoRA: Setup & Process Documentation
1. Dataset Preparation
Format: Each example is a JSON line with prompt and response fields.
Example:
File: dataset_articles.jsonl
2. Model & Tokenizer Setup
Base Model: mistralai/Mistral-7B-v0.1
Tokenizer: Loaded from the same base model.
Quantization: 4-bit quantization using bitsandbytes for efficient GPU memory usage.
3. LoRA (Low-Rank Adaptation) Steps
Purpose: LoRA enables efficient fine-tuning by injecting trainable low-rank matrices into certain layers, drastically reducing the number of trainable parameters.
Configuration:
r=16: Rank of the update matrices (higher = more capacity, more memory).
lora_alpha=32: Scaling factor for the LoRA updates.
target_modules=["q_proj", "v_proj"]: Apply LoRA to the query and value projection layers.
lora_dropout=0.05: Dropout applied to LoRA layers during training.
bias="none": No bias adaptation.
task_type=TaskType.CAUSAL_LM: For causal language modeling tasks.
4. Tokenization & Data Collation
Prompt Engineering:
Each prompt is prepended with "Answer concisely:\n" to encourage shorter, focused responses.
Tokenization:
max_length=256 to prevent overly long sequences.
Labels are masked (-100) for the prompt part so only the response is used for loss calculation.
5. Training Configuration
Arguments:
per_device_train_batch_size=4
gradient_accumulation_steps=4
num_train_epochs=5 (increase for small datasets)
learning_rate=2e-4
fp16=False, bf16=True (mixed precision for speed/memory)
output_dir="mistral-output"
Trainer:
Uses Hugging Face Trainer with a DataCollatorForLanguageModeling.
6. Evaluation Metrics
ROUGE (rouge1_f1, rouge2_f1, rougeL_f1):
Measures word and phrase overlap between generated and reference answers.
BLEU:
Measures n-gram overlap (strict, penalizes extra/missing words).
BERTScore:
Measures semantic similarity using a pretrained BERT model.
length_ratio:
Ratio of generated answer length to reference length (should be close to 1).
failure_rate:
Fraction of prompts where the model failed to generate an answer.
7. Evaluation Process
Run evaluate.py:
Computes all metrics and saves results to evaluation_metrics.json.
Saves detailed error cases to error_analysis.json.
Supports a --fast flag to skip BERTScore for quicker runs.
8. Experiment Tracking with wandb
Purpose:
Track, visualize, and compare experiments and evaluation results.
Setup:
Create a wandb account and log in when prompted.
Use wandb_evaluate.py to upload metrics and error analysis after evaluation.
What gets logged:
All evaluation metrics.
Error analysis as a table for easy inspection.
9. Future Improvements
Increase dataset size and diversity for better generalization.
Tune LoRA parameters (r, lora_alpha, target_modules) for your specific task.
Experiment with prompt engineering to further encourage concise, relevant answers.
Use a validation split and early stopping to prevent overfitting.
Try instruction-tuned base models if available.
Refine generation parameters (e.g., max_new_tokens, length_penalty) for more focused outputs.
Automate hyperparameter sweeps using wandb for systematic optimization.
10. Summary Workflow
Prepare dataset in JSONL format.
Configure and run train_lora.py for LoRA fine-tuning.
Run evaluate.py to compute metrics and save results.
Run wandb_evaluate.py to log results to wandb for visualization and tracking.
Review metrics and error analysis to guide further improvements.

B. Install NVIDIA Drivers
Check Device Manager (Windows) or lspci (Linux) to confirm the GPU is visible.
Download and install the latest NVIDIA driver for your GPU model:
NVIDIA Data Center Drivers
For Windows, use the Quadro/Data Center driver (e.g., version 537.70 for V100/A10).
Reboot after installation.
C. Verify GPU Availability
Open a terminal and run:
You should see your GPU listed with driver and CUDA version.
D. Install CUDA-enabled Libraries
PyTorch:
Install the CUDA version matching your driver (e.g., for CUDA 12.1):
bitsandbytes:
E. Test CUDA in Python
Run:
Should print True.