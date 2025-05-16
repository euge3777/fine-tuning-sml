import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch
from typing import Dict, Sequence
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_dataset(data_path: str, tokenizer, max_length: int = 512) -> Dataset:
    """Prepare and tokenize the dataset."""
    dataset = load_dataset("json", data_files={"train": data_path})
    
    def tokenize(examples: Dict[str, Sequence[str]]) -> Dict[str, Sequence]:
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_dataset

def main():
    # Configuration
    MODEL_NAME = "gpt2"
    OUTPUT_DIR = "./lora-gpt2-output"
    TRAIN_DATA_PATH = "train_data.jsonl"
    
    # Load tokenizer and model
    logger.info(f"Loading model {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 8-bit quantization for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # increased rank
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],  # targeting more modules
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    logger.info("Applying LoRA adapters")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    logger.info("Preparing dataset")
    tokenized_dataset = prepare_dataset(TRAIN_DATA_PATH, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size of 16
        num_train_epochs=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
    )
    
    # Train
    logger.info("Starting training")
    trainer.train()
    
    # Save the final model
    output_dir = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
if __name__ == "__main__":
    main()
