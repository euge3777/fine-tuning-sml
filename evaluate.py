import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
import json
import logging
from tqdm import tqdm
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, base_model: str, lora_path: str, skip_bertscore: bool = False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(lora_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()
        self.skip_bertscore = skip_bertscore
        if not skip_bertscore:
            try:
                self.bertscore = evaluate.load('bertscore')
            except Exception:
                self.bertscore = None
        else:
            self.bertscore = None

    def generate_response(self, prompt: str, max_new_tokens: int = 128) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()

    def calculate_metrics(self, prediction: str, reference: str) -> dict:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(prediction, reference)
        smoothie = SmoothingFunction().method4
        bleu = sentence_bleu(
            [reference.split()],
            prediction.split(),
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie
        )
        # Use preloaded BERTScore
        if self.bertscore is not None:
            try:
                bertscore_results = self.bertscore.compute(
                    predictions=[prediction],
                    references=[reference],
                    lang="en"
                )
                bert_f1 = float(np.mean(bertscore_results['f1']))
            except Exception:
                bert_f1 = 0.0
        else:
            bert_f1 = 0.0

        return {
            'rouge1_f1': rouge_scores['rouge1'].fmeasure,
            'rouge2_f1': rouge_scores['rouge2'].fmeasure,
            'rougeL_f1': rouge_scores['rougeL'].fmeasure,
            'bleu': bleu,
            'bertscore_f1': bert_f1,
            'length_ratio': len(prediction.split()) / max(1, len(reference.split()))
        }

    def evaluate_batch(self, eval_data):
        all_metrics = []
        failed = 0
        for item in tqdm(eval_data, desc="Evaluating"):
            prompt = item["prompt"]
            reference = item["response"].strip()
            try:
                prediction = self.generate_response(prompt)
                metrics = self.calculate_metrics(prediction, reference)
                all_metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Failed to evaluate example: {str(e)}")
                failed += 1
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
        avg_metrics['failure_rate'] = failed / len(eval_data)
        return avg_metrics

    def analyze_errors(self, eval_data, threshold=0.5):
        error_cases = []
        for item in eval_data:
            prompt = item["prompt"]
            reference = item["response"].strip()
            try:
                prediction = self.generate_response(prompt)
                metrics = self.calculate_metrics(prediction, reference)
                if any(v < threshold for k, v in metrics.items() if k != 'length_ratio'):
                    error_cases.append({
                        "prompt": prompt,
                        "expected": reference,
                        "generated": prediction,
                        "metrics": metrics
                    })
            except Exception as e:
                error_cases.append({
                    "prompt": prompt,
                    "error": str(e)
                })
        return error_cases

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Skip BERTScore for faster evaluation")
    args = parser.parse_args()

    base_model = "mistralai/Mistral-7B-v0.1"
    lora_path = "./mistral-lora"
    data_path = "dataset_articles.jsonl"

    with open(data_path, "r", encoding="utf-8") as f:
        eval_data = [json.loads(line) for line in f]

    evaluator = ModelEvaluator(base_model, lora_path, skip_bertscore=args.fast)

    logger.info("Starting evaluation...")
    metrics = evaluator.evaluate_batch(eval_data)
    logger.info("\nEvaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

    # Save metrics for wandb_evaluate.py
    with open("evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("\nRunning error analysis...")
    error_cases = evaluator.analyze_errors(eval_data)
    with open("error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(error_cases, f, indent=2)
    logger.info(f"\nFound {len(error_cases)} problematic examples")
    logger.info("Error analysis saved to error_analysis.json")

if __name__ == "__main__":
    main()