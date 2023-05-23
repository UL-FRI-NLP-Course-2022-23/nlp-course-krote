import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import pylev
from sentence_transformers import SentenceTransformer, util
import sys
import json
from random import shuffle, seed
from tqdm import tqdm


seed(10)

MODEL = sys.argv[1]
SAVE_PATH = sys.argv[2]

with open('../data/sentences_openai.json', 'r', encoding='utf-8') as r:
    data = json.load(r)

# Select NUM_SAMPLES random samples from data after shuffling
NUM_SAMPLES = 100
shuffle(data)
data = data[:NUM_SAMPLES]
data = list(map(lambda x: x['original'], data))

# Initializing models and tokenizers
device = torch.device(
    "cuda:0" if torch.cuda.is_available() and False else "cpu")
tokenizer = T5TokenizerFast.from_pretrained(f"./{MODEL}")
model = T5ForConditionalGeneration.from_pretrained(f"./{MODEL}").to(device)

results = []

for sentence in tqdm(data, desc="Processing sentences"):
    tokenized_input = tokenizer(sentence, return_tensors="pt").to(device)
    encoded_output = model.generate(**tokenized_input, max_length=100)[0]
    decoded = tokenizer.decode(encoded_output, skip_special_tokens=True)

    # Select the last sentence that the model blurted out
    if '.' in decoded:
        out = list(filter(lambda x: len(x) > 4, decoded.split('.')))[-1]
    else:
        out = decoded

    out = out if out[-1] == '.' else out + '.'  # Add period if not present

    results.append({
        'original': sentence,
        'paraphrase': out,
    })


with open(SAVE_PATH, 'w', encoding='utf-8') as w:
    json.dump(results, w, indent=4, ensure_ascii=False)

print(len(results))
