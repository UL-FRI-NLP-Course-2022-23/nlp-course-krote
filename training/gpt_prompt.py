import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pylev
from sentence_transformers import SentenceTransformer, util
import sys

MODEL = sys.argv[1]
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

SAVE_PATH = MODEL

# Load the trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained(SAVE_PATH)
tokenizer = GPT2TokenizerFast.from_pretrained(SAVE_PATH)

# Set the model to evaluation mode
model.eval()

while True:
    # Prompt the user for input
    user_input = input("Enter your text (or 'q' to quit): ")

    if user_input.lower() == "q":
        break

    # Tokenize the input text
    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    # Generate the output
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)

    # Decode the generated output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Print the generated text
    print("Generated Text:")
    print(generated_text)
