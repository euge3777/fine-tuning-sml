from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import evaluate
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np
from typing import Dict, List
import json
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, base_model: str, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.base_model = base_model
        
    def generate_response(self, prompt: str) -> str:
        """Generate a response for a given prompt"""
        # Tokenize the prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Generate response
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )
        
        # Decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the bot's response
        if "Bot:" in response:
            response = response.split("Bot:")[-1].strip()
        
        return response
    
    def evaluate_batch(self, dataset):
        """Evaluate the model on a batch of examples"""
        metrics = {
            'rouge1_f1': 0,
            'rouge2_f1': 0,
            'rougeL_f1': 0,
            'bleu': 0,
            'bertscore': 0
        }
        
        total = 0
        for example in dataset:
            # Generate response
            response = self.generate_response(example['text'])
            
            # Extract ground truth
            ground_truth = example['text'].split('Bot:')[-1].strip()
            
            # Calculate metrics
            rouge = evaluate.load('rouge')
            bleu = evaluate.load('bleu')
            bertscore = evaluate.load('bertscore')
            
            # ROUGE scores
            rouge_scores = rouge.compute(
                predictions=[response],
                references=[ground_truth],
                use_stemmer=True
            )
            
            # BLEU score
            bleu_score = bleu.compute(
                predictions=[response.split()],
                references=[[ground_truth.split()]],
                max_order=4
            )
            
            # BERTScore
            bertscore_scores = bertscore.compute(
                predictions=[response],
                references=[ground_truth],
                model_type='microsoft/deberta-xlarge-mnli'
            )
            
            # Update metrics
            metrics['rouge1_f1'] += rouge_scores['rouge1']
            metrics['rouge2_f1'] += rouge_scores['rouge2']
            metrics['rougeL_f1'] += rouge_scores['rougeL']
            metrics['bleu'] += bleu_score['bleu']
            metrics['bertscore'] += sum(bertscore_scores['f1']) / len(bertscore_scores['f1'])
            
            total += 1
        
        # Average metrics
        for metric in metrics:
            metrics[metric] /= total
        
        return metrics

    def calculate_metrics(self, prediction: str, reference: str) -> Dict[str, float]:
        # Calculate ROUGE scores
        rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = rouge_scorer.score(prediction, reference)
        
        # Calculate BLEU score
        bleu_score = sentence_bleu(
            [reference.split()],
            prediction.split(),
            weights=(0.25, 0.25, 0.25, 0.25)
        )
        
        # Calculate BERTScore
        bertscore = evaluate.load('bertscore')
        bertscore_results = bertscore.compute(
            predictions=[prediction],
            references=[reference],
            lang="en"
        )
        
        # Compile metrics
        metrics = {
            'rouge1_f1': rouge_scores['rouge1'].fmeasure,
            'rouge2_f1': rouge_scores['rouge2'].fmeasure,
            'rougeL_f1': rouge_scores['rougeL'].fmeasure,
            'bleu': bleu_score,
            'bertscore_f1': np.mean(bertscore_results['f1']),
            'bertscore_precision': np.mean(bertscore_results['precision']),
            'bertscore_recall': np.mean(bertscore_results['recall']),
        }
        
        # Add length ratio metric
        metrics['length_ratio'] = len(prediction.split()) / len(reference.split())
        
        return metrics

    def evaluate_batch(self, eval_data: List[Dict]) -> Dict[str, float]:
        all_metrics = []
        failed_generations = 0
        
        for item in tqdm(eval_data, desc="Evaluating"):
            prompt = f"Context: {item['context']}\nUser: {item['question']}\nBot:"
            try:
                generated_response = self.generate_response(prompt)
                response = generated_response.split("Bot:", 1)[1].strip()
                metrics = self.calculate_metrics(response, item['answer'])
                all_metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Failed to evaluate example: {str(e)}")
                failed_generations += 1
        
        # Calculate average metrics
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        # Add failure rate metric
        avg_metrics['failure_rate'] = failed_generations / len(eval_data)
        
        return avg_metrics

    def analyze_errors(self, eval_data: List[Dict], threshold: float = 0.5) -> List[Dict]:
        """Analyze specific examples where the model performs poorly"""
        error_analysis = []
        
        for item in eval_data:
            prompt = f"Context: {item['context']}\nUser: {item['question']}\nBot:"
            generated_response = self.generate_response(prompt)
            
            try:
                response = generated_response.split("Bot:", 1)[1].strip()
                metrics = self.calculate_metrics(response, item['answer'])
                
                # If any metric is below threshold, add to error analysis
                if any(v < threshold for k, v in metrics.items() if k != 'length_ratio'):
                    error_analysis.append({
                        'question': item['question'],
                        'expected': item['answer'],
                        'generated': response,
                        'metrics': metrics
                    })
            except Exception as e:
                error_analysis.append({
                    'question': item['question'],
                    'error': str(e)
                })
        
        return error_analysis

def main():
    # Configuration
    base_model = "gpt2"
    model_path = "./lora-gpt2-output/final"
    
    # Load test data
    with open('test_data.jsonl', 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    # Initialize evaluator
    evaluator = ModelEvaluator(base_model, model_path)
    
    # Run evaluation
    logger.info("Starting evaluation...")
    metrics = evaluator.evaluate_batch(test_data)
    
    # Print results
    logger.info("\nEvaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Run error analysis
    logger.info("\nRunning error analysis...")
    error_cases = evaluator.analyze_errors(test_data)
    
    # Save error analysis
    with open('error_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(error_cases, f, indent=2)
    
    logger.info(f"\nFound {len(error_cases)} problematic examples")
    logger.info("Error analysis saved to error_analysis.json")

if __name__ == "__main__":
    main() 