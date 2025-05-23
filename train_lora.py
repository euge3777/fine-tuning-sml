import os
import torch
import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from huggingface_hub import login

# Log in with your token login(token="")
print("✅ Successfully logged in to Hugging Face.")

DATA_PATH = "dataset_articles.jsonl"
OUTPUT_DIR = "mistral-lora"
MISTRAL_OUTPUT = "mistral-output"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    return Dataset.from_list(lines)

dataset = load_jsonl(DATA_PATH)

def tokenize(example):
    # Encourage concise answers
    instruction = "Answer concisely:\n"
    prompt = instruction + example["prompt"] + "\n"
    full_input = prompt + example["response"]
    input_ids = tokenizer(
        full_input,
        truncation=True,
        padding="max_length",
        max_length=256
    )["input_ids"]

    prompt_ids = tokenizer(prompt, truncation=True, max_length=256)["input_ids"]
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
    labels = labels[:256]
    input_ids = input_ids[:256]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids)
    }

tokenized_dataset = dataset.map(tokenize, batched=False)

training_args = TrainingArguments(
    output_dir=MISTRAL_OUTPUT,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=5,  # Increased epochs for better learning
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)