import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch
import logging
from dataclasses import dataclass
import wandb
from evaluate import ModelEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = None
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.03

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

class LoRATrainer:
    def __init__(self, config: LoRAConfig, base_model: str = "meta-llama/Meta-Llama-3-7B"):
        self.config = config
        self.base_model = base_model
        self.setup_wandb()

    def setup_wandb(self):
        wandb.init(
            project="lora-fine-tuning",
            config={
                "r": self.config.r,
                "lora_alpha": self.config.lora_alpha,
                "lora_dropout": self.config.lora_dropout,
                "learning_rate": self.config.learning_rate,
                "num_train_epochs": self.config.num_train_epochs,
                "batch_size": self.config.per_device_train_batch_size,
                "model": "Llama-3-7B"
            }
        )

    def prepare_model(self):
        logger.info(f"Loading base model {self.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        lora_config = LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        logger.info("Applying LoRA adapters")
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def generate_response(self, prompt: str) -> str:
        formatted_prompt = f"{prompt}\nBot:"
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                num_beams=4,
                early_stopping=True
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_response.split("Bot:")[-1].strip() if "Bot:" in full_response else full_response.strip()

    def prepare_dataset(self, data_path: str, validation_split: float = 0.1):
        dataset = load_dataset("json", data_files={"train": data_path})

        def tokenize_function(examples):
            texts = [f"{text}\nBot:" for text in examples["text"]]
            tokenized = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_attention_mask=True
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            for i in range(len(tokenized["labels"])):
                text = examples["text"][i]
                bot_start = text.find("Bot:") + 4
                if bot_start > 4:
                    bot_start_tokens = len(self.tokenizer.encode(text[:bot_start]))
                    tokenized["labels"][i][:bot_start_tokens] = -100
            return tokenized

        tokenized_dataset = dataset["train"].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )

        dataset = tokenized_dataset.train_test_split(
            test_size=validation_split,
            shuffle=True,
            seed=42
        )

        return dataset

    def train(self, train_data_path: str, output_dir: str = "./lora-output"):
        self.prepare_model()
        dataset = self.prepare_dataset(train_data_path)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            fp16=True,
            logging_steps=1,
            eval_steps=50,
            save_steps=50,
            save_total_limit=2,
            load_best_model_at_end=True,
            warmup_ratio=self.config.warmup_ratio,
            report_to="wandb",
            remove_unused_columns=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        logger.info("Starting training")
        trainer.train()

        final_output_dir = os.path.join(output_dir, "final")
        os.makedirs(final_output_dir, exist_ok=True)

        logger.info(f"Saving model to {final_output_dir}")
        self.model.save_pretrained(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)

        evaluator = ModelEvaluator(self.base_model, final_output_dir)
        eval_metrics = evaluator.evaluate_batch(dataset["test"])
        wandb.log(eval_metrics)

        return eval_metrics

def main():
    configs = [
        LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.1),
        LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.05),
    ]

    best_metrics = None
    best_config = None

    for config in configs:
        logger.info(f"\nTrying configuration: {config}")
        trainer = LoRATrainer(config)
        metrics = trainer.train("train_data.jsonl")

        if best_metrics is None or metrics['rougeL_f1'] > best_metrics['rougeL_f1']:
            best_metrics = metrics
            best_config = config

    logger.info("\nBest configuration found:")
    logger.info(f"r: {best_config.r}")
    logger.info(f"lora_alpha: {best_config.lora_alpha}")
    logger.info(f"lora_dropout: {best_config.lora_dropout}")
    logger.info("\nBest metrics:")
    for metric, value in best_metrics.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
