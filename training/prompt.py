from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("my_fine_tuned_model")
model = AutoModelForCausalLM.from_pretrained("my_fine_tuned_model")

while True:
    prompt = input("Enter your prompt: ")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=128, do_sample=True)
    responses = tokenizer.decode(output[0], skip_special_tokens=True).split('.')
    print()
    print(responses[0] + '.')
    print(responses[1] + '.')
    print()
