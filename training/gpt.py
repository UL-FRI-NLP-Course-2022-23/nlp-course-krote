from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import json
import shutil
import subprocess
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from transformers.optimization import AdamW
from tqdm import tqdm
import sys
import os
from pytorch_lightning.callbacks import Callback

TRAIN_DATA_PATH = sys.argv[1]
SAVE_PATH = sys.argv[2]


class ParaphraseGenerator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model_name = "cjvt/gpt-sl-base"
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.batch_size = 1
        self.lr = 5e-5

    def encode_text(self, data):
        for item in tqdm(data):
            source = self.tokenizer(
                item["original"],
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            target = self.tokenizer(
                item["paraphrased"],
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            source_ids = source.input_ids.squeeze()
            target_ids = target.input_ids.squeeze()
            yield source_ids, target_ids

    def to_tensor(self, source_ids, target_ids):
        source_ids = torch.stack(source_ids)
        target_ids = torch.stack(target_ids)
        data = TensorDataset(source_ids, target_ids)
        return random_split(data, [len(data), 0])[0]

    def prepare_data(self):
        with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as r:
            data = json.load(r)

        # Train test split
        train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
        print("Train data size:", len(train_data))
        print("Test data size:", len(test_data))

        source_ids, target_ids = list(zip(*tuple(self.encode_text(train_data))))
        self.train_ds = self.to_tensor(source_ids, target_ids)

        source_ids, target_ids = list(zip(*tuple(self.encode_text(test_data))))
        self.test_ds = self.to_tensor(source_ids, target_ids)

    def forward(self, batch):
        source_ids, target_ids = batch
        return self.model(input_ids=source_ids, labels=target_ids)

    def training_step(self, batch, batch_idx):
        loss = self(batch)[0]
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)[0]
        self.log("val_loss", loss)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=12,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=12,
        )

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)


class SaveModelCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        SAVE_PATH_FINAL = SAVE_PATH_FINAL = f"{SAVE_PATH}_epoch{trainer.current_epoch}"
        # Create dir if not exists
        if not os.path.exists(SAVE_PATH_FINAL):
            os.makedirs(SAVE_PATH_FINAL)
        pl_module.tokenizer.save_vocabulary(SAVE_PATH_FINAL)
        pl_module.tokenizer.save_pretrained(SAVE_PATH_FINAL)
        pl_module.model.save_pretrained(SAVE_PATH_FINAL)

        print(f"Saved model and tokenizer at epoch {trainer.current_epoch}")


trainer = pl.Trainer(
    default_root_dir="logs",
    min_epochs=1,
    max_epochs=10,
    val_check_interval=0.5,
    accumulate_grad_batches=8,
    logger=TensorBoardLogger("logs/", name="paraphrase", version=0),
    callbacks=[SaveModelCallback()],
)

para_model = ParaphraseGenerator()
trainer.fit(para_model)

print("Saving model...")
SAVE_PATH_FINAL = f"{SAVE_PATH}_final"
# Create dir if not exists
if not os.path.exists(SAVE_PATH_FINAL):
    os.makedirs(SAVE_PATH_FINAL)
para_model.tokenizer.save_pretrained(SAVE_PATH_FINAL)
para_model.model.save_pretrained(SAVE_PATH_FINAL)

print("Zipping the model...")
shutil.make_archive(SAVE_PATH_FINAL, "zip", SAVE_PATH_FINAL)

print("Uploading the model...")
upload_command = f"curl --upload-file {SAVE_PATH_FINAL}.zip https://transfer.sh/"
process = subprocess.Popen(upload_command, stdout=subprocess.PIPE, shell=True)
(output, _) = process.communicate()
model_url = output.decode("utf-8").strip()

print("Model uploaded successfully.")
print("Model URL:", model_url)