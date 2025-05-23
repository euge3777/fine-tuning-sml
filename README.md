# Fine-Tuning LLM with LoRA on Mistral-7B

## 🚀 Final Training Logs

    Train Loss	2.30
    
    Epochs    	5.0
    
    Samples/sec	0.712
    
    Steps/sec	0.047
    
    Total Training Time	210.8 seconds

## 🛠️ Fine-Tuning Setup
1. Dataset Preparation

    Format: JSONL with prompt and response fields

    Example File: dataset_articles.jsonl

3. Model & Tokenizer

    Base Model: mistralai/Mistral-7B-v0.1: **Create a free Hugging Face account (link below), request access for gated model and retrieve access token with read permissions**
    
    Tokenizer: Loaded from base model
    
    Quantization: 4-bit via bitsandbytes for memory efficiency

5. LoRA Configuration
python
```
{
  "r": 16,
  "lora_alpha": 32,
  "target_modules": ["q_proj", "v_proj"],
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

4. Prompt Engineering & Tokenization

```
Prompt format: "Answer concisely:\n{your prompt}"
    
max_length = 256
    
Labels are masked (-100) for prompts, so loss is only calculated on responses.
```

5. Training Configuration
```
per_device_train_batch_size=4
gradient_accumulation_steps=4
num_train_epochs=5
learning_rate=2e-4
fp16=False
bf16=True
output_dir="mistral-output
```
  Trainer: Hugging Face Trainer with DataCollatorForLanguageModeling

## 📊 Evaluation Metrics
**Metric	Description**

    ROUGE	Measures word/phrase overlap (rouge1_f1, rouge2_f1, rougeL_f1)
    
    BLEU	Measures n-gram overlap (strict)
    
    BERTScore	Semantic similarity using BERT
    
    Length Ratio	Ratio of generated vs reference answer length
    
    Failure Rate	Fraction of empty or failed responses

**Run Evaluation**

powershell

```
python evaluate.py
```

**Optional fast mode**

```
python evaluate.py --fast
```

Outputs: evaluation_metrics.json, error_analysis.json

## 📈 Experiment Tracking with wandb
Setup
powershell
```
wandb login # You would need to create a free account, input access token retrieved
```
Log Metrics
```
python wandb_evaluate.py
```
What’s Logged:
All evaluation metrics
Detailed error table for inspection
![Screenshot 2025-05-23 143949](https://github.com/user-attachments/assets/7b82d167-229c-47f9-a9ea-a60086548e02)

## 🔧 System Setup & Environment
A. Prerequisites
Python 3.10+
Install packages via:
```
pip install -r requirements.txt
```

B. NVIDIA Driver Installation
Windows: Use Quadro/Data Center driver (e.g., v537.70 for V100/A10)

C. Verify GPU Availability
```
nvidia-smi
```
D. CUDA Libraries Installation
Install PyTorch with CUDA support matching your driver version
Install CUDA-enabled bitsandbytes

E. Test CUDA in Python
python
```
import torch
print(torch.cuda.is_available()) # Should return True
```

## 🗂️ Project Structure
```
.
├── dataset_articles.jsonl
├── train_lora.py
├── evaluate.py
├── wandb_evaluate.py
├── mistral-output/
├── requirements.txt
└── README.md
```

## 🔄 Sample Workflow

1. Prepare your dataset in JSONL format

   **How I Prepared My Dataset:**

    Created a file (articles.txt) containing a list of Q&A pairs in JSON format, with each item having a ```prompt``` and a ```response```.
      
      Converted this list into a JSONL file ```(dataset_articles.jsonl)``` using a Python script ```(prepare_data.py)```.
      
      The script reads the JSON array and writes each Q&A pair as a single line in the new file.
      
      Result: The final dataset is in **JSONL format**, with one prompt-response pair per line, ready for fine-tuning language models.

2. Run:
```
python train_lora.py
```
3. Evaluation:
```
python evaluate.py
```
4. Upload results to Weights & Biases:
```
python wandb_evaluate.py
```

💡 Future Improvements
1. Increase dataset size & diversity

2. Tune LoRA parameters (r, alpha, target_modules)

3. Refine prompts to improve generation quality

4. Use validation splits with early stopping

5. Explore instruction-tuned base models

6. Automate hyperparameter sweeps using wandb


## 📚 References & Further Reading

[Fine-Tuning LLM on Telegram Chats](https://medium.com/data-science-collective/i-fine-tuned-an-llm-on-5-years-of-telegram-chats-7bacb66387c8)

[Experimental Insights](https://www.encora.com/insights/fine-tuning-small-language-models-experimental-insights)

[Weights & Biases Page](https://wandb.ai/site/)

[Hugging Face Page](https://huggingface.co/models)
