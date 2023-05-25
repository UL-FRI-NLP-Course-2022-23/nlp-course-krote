import os
import json
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "../data/results"

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def compute_avg_scores(data):
    total_meaning = total_fluency = total_diversity = 0
    for item in data:
        total_meaning += item.get('meaning', 0)
        total_fluency += item.get('fluency', 0)
        total_diversity += item.get('diversity', 0)

    avg_meaning = total_meaning / len(data)
    avg_fluency = total_fluency / len(data)
    avg_diversity = total_diversity / len(data)

    return avg_meaning, avg_fluency, avg_diversity


def compare_models(directory):
    data = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.json'):
            file_path = os.path.join(directory, file_name)
            json_data = load_json(file_path)
            avg_meaning, avg_fluency, avg_diversity = compute_avg_scores(
                json_data)
            # assuming model name is file name without extension
            model_name = os.path.splitext(file_name)[0]
            data.append({
                'Model Name': model_name,
                'Average Meaning Score': avg_meaning,
                'Average Fluency Score': avg_fluency,
                'Average Diversity Score': avg_diversity,
            })

    df = pd.DataFrame(data)
    return df


def plot_scores(df):
    df.set_index('Model Name', inplace=True)
    df.plot(kind='bar', rot=0, figsize=(10, 6))
    plt.ylabel('Score')
    plt.title('Average Scores for each Paraphrasing Model')
    plt.tight_layout()
    plt.savefig('scores.png')


# usage

if __name__ == '__main__':
    df = compare_models(RESULTS_DIR)
    print(df)
    plot_scores(df)
