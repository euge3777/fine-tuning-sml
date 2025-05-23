import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_response(prompt: str, model, tokenizer, max_new_tokens: int = 128):
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

def main():
    # Paths from your train_lora.py
    base_model_name = "mistralai/Mistral-7B-v0.1"
    lora_path = "./mistral-lora"

    # Load tokenizer and model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    logger.info(f"Model device: {next(model.parameters()).device}")

    print("\nModel loaded! Ask a question (type 'quit' to exit):")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            break
        prompt = user_input
        try:
            response = generate_response(prompt, model, tokenizer)
            print(f"Bot: {response}")
        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()