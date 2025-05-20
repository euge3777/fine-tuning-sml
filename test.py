from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_response(prompt: str, context: str, model, tokenizer, max_length: int = 512):
    # Prepare the input with context
    full_prompt = f"Context: {context}\nUser: {prompt}\nBot:"
    logger.info(f"Generated prompt: {full_prompt}")
    
    # Tokenize with proper padding and attention mask
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    
    logger.info(f"Input shape: {inputs['input_ids'].shape}")
    
    try:
        # Generate with simpler parameters
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=50,  # Generate up to 50 new tokens
            do_sample=False,  # Use greedy decoding
            pad_token_id=tokenizer.eos_token_id
        )
        
        logger.info(f"Output shape: {outputs.shape}")
        
        # Get only the newly generated tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        logger.info(f"Generated text: {generated_text}")
        
        return generated_text.strip()
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}", exc_info=True)
        return f"Error generating response: {str(e)}"

def main():
    # Article context
    context = """The James Webb Space Telescope (JWST) has made a groundbreaking discovery: 
    it has detected signs of water vapor in the atmosphere of a small, rocky exoplanet called GJ 486 b. 
    This planet, located 26 light-years from Earth, is slightly larger than our planet and orbits a red dwarf star.
    The discovery marks the first time water vapor has been potentially identified in the atmosphere of a rocky planet 
    outside our solar system. The planet's surface temperature is estimated to be around 800 degrees Fahrenheit (430 Celsius),
    making it too hot for liquid water to exist on its surface. Scientists believe studying such planets can help us 
    understand how rocky planets like Earth form and evolve."""
    
    # Load the base model and tokenizer
    base_model_name = "gpt2"
    adapter_path = "./lora-gpt2-output/final"
    
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Load the LoRA adapter
    logger.info("Loading fine-tuned adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()  # Set to evaluation mode
    
    # Print model device
    logger.info(f"Model device: {next(model.parameters()).device}")
    
    print("\nModel loaded! You can now ask questions about the article.")
    print("Example questions:")
    print("1. What did the James Webb Space Telescope discover?")
    print("2. How far is GJ 486 b from Earth?")
    print("3. What is the surface temperature of the planet?")
    print("\nType 'quit' to exit.")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            break
            
        try:
            response = generate_response(user_input, context, model, tokenizer)
            print(f"Bot: {response}")
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
