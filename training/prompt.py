from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import torch

model_name = sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

comma_token_id = tokenizer.encode(',', add_special_tokens=False)[0]

while True:
    prompt = input("Enter your prompt: ")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=128, do_sample=True, early_stopping=True, eos_token_id=comma_token_id)
    responses = tokenizer.decode(output[0], skip_special_tokens=True).split('.')
    print(responses[0] + '.')
