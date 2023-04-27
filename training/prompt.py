from transformers import AutoTokenizer, AutoModelWithLMHead

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("path/to/trained/model")
model = AutoModelWithLMHead.from_pretrained("path/to/trained/model")

# Define function to generate prompts
def generate_prompt():
    prompt = input("Enter your prompt: ")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=50, do_sample=True)
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated)

# Call generate_prompt() function
while True:
    generate_prompt()
