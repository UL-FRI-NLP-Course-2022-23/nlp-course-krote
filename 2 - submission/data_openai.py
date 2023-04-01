import openai
import os

# Replace with your OpenAI API key
openai.api_key = os.getenv('OPENAPI_KEY')


def generate_paraphrase(sentence, model="text-davinci-003"):
    prompt = f"Paraphrase the Slovene sentence given below. The output must be in Slovene and should only contain 3 sentences, each one in a separate line. They must not be enumerated or contain any quote chars. They must all differ from each other and the original sentence. The sentence to paraphrase is: \"{sentence}\""

    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7,
    )

    result = response.choices[0].text.strip().split('\n')
    return result


def main():
    slovene_sentences = [
        "Študiram na fakulteti za računalništvo in informatiko.",
        "Jaz sem programer.",
        "Kako ti gre danes?",
    ]

    paraphrased_sentences = []

    for sentence in slovene_sentences:
        paraphrase = generate_paraphrase(sentence)
        paraphrased_sentences.append(paraphrase)
        p = "\n".join(paraphrase)
        print(f"Original: {sentence}\nParaphrase:\n{p}\n")


if __name__ == "__main__":
    main()
