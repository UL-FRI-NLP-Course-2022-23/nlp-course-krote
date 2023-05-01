from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
    AutoConfig,
    EvalPrediction,
)
from datasets import load_metric
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

from datasets import Dataset
import torch
import sys
import json

from pynvml import *

from datasets import load_metric

rouge = load_metric("rouge")


def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    rouge_scores = rouge.compute(predictions=predictions, references=labels)
    return rouge_scores


dataset_path = sys.argv[1]
model_name = sys.argv[2]


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def tokenize_function(examples):
    return tokenizer(
        examples["input"], padding="max_length", truncation=True, max_length=128
    )


tokenizer = AutoTokenizer.from_pretrained("cjvt/gpt-sl-base")
print_gpu_utilization()
model = AutoModelWithLMHead.from_pretrained("cjvt/gpt-sl-base").to("cuda")
print_gpu_utilization()

tmp = []
with open(dataset_path, "r") as f:
    tmp = json.load(f)

tmp = list(
    filter(lambda x: x["paraphrased"] is not None and x["original"] is not None, tmp)
)
tmp_train, tmp_eval = train_test_split(tmp, test_size=0.2)

# Modify the input format to include the "paraphrase" prompt
train_data = {
    "input": ["Originalno: " + x["original"] for x in tmp_train],
    "output": ["Parafrazirano: " + x["paraphrased"] for x in tmp_train],
}
eval_data = {
    "input": ["Originalno: " + x["original"] for x in tmp_eval],
    "output": ["Parafrazirano: " + x["paraphrased"] for x in tmp_eval],
}

train_dataset = Dataset.from_dict(train_data)
eval_dataset = Dataset.from_dict(eval_data)
print(len(train_data["input"]), len(eval_data["input"]))

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Increase the number of training epochs
num_epochs = 30

training_args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="no",
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=num_epochs,
    weight_decay=1e-2,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",  # Choose a specific ROUGE variant
    greater_is_better=True,
    save_strategy="no",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.model.save_pretrained(model_name)
tokenizer.save_pretrained(model_name)
