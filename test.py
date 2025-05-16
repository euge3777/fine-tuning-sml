from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("./lora-gpt2-finetuned")
tokenizer = AutoTokenizer.from_pretrained("./lora-gpt2-finetuned")

prompt = "User: How are you?\nBot:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
output = model.generate(**inputs, max_new_tokens=30, do_sample=True)
print(tokenizer.decode(output[0]))
