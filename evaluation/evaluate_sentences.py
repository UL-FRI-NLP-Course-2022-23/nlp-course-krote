import json
import random

# Define path to sentences file here
FILE_PATH = "../data/backtranslation_openai.json"
RANDOM_SAMPLE_SIZE = 10

# Output results file path
OUTPUT_FILE_PATH = "../data/results/results_backtranslation_openai.json"


def load_sentences(path):
    """Load sentence pairs from file"""
    with open(path, "r", encoding="utf_8") as f:
        return json.load(f)


def ask_for_ratings():
    """Ask user to rate sentence pairs and validate user input"""

    while True:
        # Rate meaning preservation
        meaning_rating = input("Rate meaning preservation (1-3): ")
        # Rate fluency (grammatical and idiomatic correctness)
        fluency_rating = input("Rate fluency (1-3): ")
        # Rate paraphrase
        diversity_rating = input("Rate diversity (1-3): ")

        ratings = {"meaning": float(meaning_rating),
                   "fluency": float(fluency_rating), "diversity": float(diversity_rating)}

        # Validate ratings are number and not string
        try:
            # Validate every rating is between 1 and 4
            if all(1 <= float(rating) <= 4 for rating in ratings.values()):
                break
            else:
                print("Wrong format. Please enter a number between 1 and 3.\n")
        except ValueError:
            print("Wrong format. Please enter a number between 1 and 3.\n")

    return ratings


def evaluate_sentences(input_file=FILE_PATH, output_file=OUTPUT_FILE_PATH):
    """Evaluate sentence pairs"""
    sentence_pairs = load_sentences(input_file)
    # Get random 100 sentence pairs
    random_pairs = random.sample(sentence_pairs, RANDOM_SAMPLE_SIZE)

    try:
        # Load existing evaluation results
        with open(output_file, "r", encoding="utf_8") as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # File is empty
        results = []

    with open(output_file, "w", encoding="utf_8") as f:
        for pair in random_pairs:
            original = pair["original"]
            paraphrased = pair["paraphrased"]

            print(f"Original: {original}")
            print(f"Paraphrased: {paraphrased}\n")

            ratings = ask_for_ratings()

            # Add sentence pair to evaluation results
            ratings["original"] = original
            ratings["paraphrased"] = paraphrased

            # Add sentence pair to evaluation results
            results.append(ratings)

            print("\n")

        # Write ratings to file in JSON format
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    evaluate_sentences()
