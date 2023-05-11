import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import pylev
from sentence_transformers import SentenceTransformer, util
import sys

MODEL = sys.argv[1]

# Initializing models and tokenizers
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = T5TokenizerFast.from_pretrained(f"./{MODEL}")
model = T5ForConditionalGeneration.from_pretrained(f"./{MODEL}").to(device)


# Calculates the levenschtein distance to see how much the model varied the original text
def get_distance(a, b):
    return pylev.levenshtein(a.split(), b.split())


# Main loop
while True:
    text = input("Enter text to paraphrase: ")
    tokenized_input = tokenizer(text, return_tensors="pt").to(device)
    encoded_output = model.generate(**tokenized_input, max_length=1000)[0]
    decoded = tokenizer.decode(encoded_output, skip_special_tokens=True)
    print("Response:", decoded)
