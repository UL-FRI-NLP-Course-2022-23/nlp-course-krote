from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from sklearn.model_selection import train_test_split


from datasets import Dataset
import json


def tokenize_function(examples):
    return tokenizer(
        examples["input"], padding="max_length", truncation=True, max_length=512
    )


tokenizer = AutoTokenizer.from_pretrained("cjvt/gpt-sl-base")
model = AutoModelWithLMHead.from_pretrained("cjvt/gpt-sl-base")

tmp = []
with open("../data/sentences_openai.json", "r") as f:
    tmp = json.load(f)

tmp = list(
    filter(lambda x: x["paraphrased"] is not None and x["original"] is not None, tmp)
)
tmp_train, tmp_eval = train_test_split(tmp, test_size=0.5)
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
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
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

model.save_pretrained("my_fine_tuned_model")
tokenizer.save_pretrained("my_fine_tuned_model")
