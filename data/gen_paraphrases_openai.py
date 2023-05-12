import os
import openai
import json
from tqdm import tqdm
import time

openai.api_key = os.getenv("OPENAI_API_KEY")
stop = 14107

prompt = """
I am a Slovene paraphrasing bot. I will read a Slovene sentence and rewrite it with a the same meaning but in different words.

Sentence: {}
Paraphrased:
"""

data = []

try:
    with open("sentences_openai.json", "r") as f:
        data = json.load(f)
except:
    pass

with open("sentences.txt", "r") as sentences:
    for i, s in tqdm(enumerate(sentences), total=stop):
        if i >= stop:
            break

        if i < len(data):
            continue

        # print(s.strip())
        # print(prompt.format(s.strip()))

        time_start = time.time()
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt.format(s.strip()),
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n"],
        )

        # print(response)
        d = {
            "original": s.strip(),
            "paraphrased": None,
        }

        if response and response["choices"] and response["choices"][0].text:
            paraphrased = response["choices"][0].text.strip()
            d["paraphrased"] = paraphrased
            # print(paraphrased)
            # print()
        else:
            # print('No response')
            # print()
            pass

        data.append(d)

        with open("sentences_openai.json", "w") as f:
            json.dump(data, f, indent=4)

        # Wait until 1 second has passed since the start of the iteration
        time_end = time.time()
        time.sleep(max(0, 2 - (time_end - time_start)))


# Count all the non-null paraphrases in data
count = sum([1 for d in data if d["paraphrased"] is not None])
print()
print("Total paraphrased:", count)
