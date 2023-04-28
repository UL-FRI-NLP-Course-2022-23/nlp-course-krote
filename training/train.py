from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
    AutoConfig,
)
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


from datasets import Dataset
import torch
import sys
import json

from pynvml import *


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

class CustomLossWrapper(nn.Module):
    def __init__(self, model, repetition_penalty):
        super().__init__()
        self.model = model
        self.repetition_penalty = torch.tensor(repetition_penalty, device=model.device)

    def forward(self, input_ids, labels=None, **kwargs):
        outputs = self.model(input_ids, labels=labels, **kwargs)
        loss = outputs[0]

        if self.repetition_penalty > 1:
            input_tokens = input_ids[:, -1]
            for token in input_tokens:
                token_count = (input_ids == token).sum(dim=1).clamp(1)
                penalty = (token_count * torch.log(self.repetition_penalty)).sum()
                loss += penalty

        return loss, outputs[1:]



tokenizer = AutoTokenizer.from_pretrained("cjvt/gpt-sl-base")
print_gpu_utilization()
model = AutoModelWithLMHead.from_pretrained("cjvt/gpt-sl-base").to('cuda')
model = CustomLossWrapper(model, 1.2)
print_gpu_utilization()

tmp = []
with open(dataset_path, "r") as f:
    tmp = json.load(f)

# tmp = tmp[:100]

tmp = list(
    filter(lambda x: x["paraphrased"] is not None and x["original"] is not None, tmp)
)
tmp_train, tmp_eval = train_test_split(tmp, test_size=0.2)
train_data = {
    "input": [x["original"] for x in tmp_train],
    "output": [x["paraphrased"] for x in tmp_train],
}
eval_data = {
    "input": [x["original"] for x in tmp_eval],
    "output": [x["paraphrased"] for x in tmp_eval],
}

train_dataset = Dataset.from_dict(train_data)
eval_dataset = Dataset.from_dict(eval_data)
print(len(train_data["input"]), len(eval_data["input"]))


tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
training_args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained(model_name)
tokenizer.save_pretrained(model_name)
