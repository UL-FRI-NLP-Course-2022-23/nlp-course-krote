from sklearn.model_selection import train_test_split
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import json
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader

# from transformers.optimization import AdamW
from torch.optim import AdamW
from tqdm import tqdm
import sys
import os
from pytorch_lightning.callbacks import Callback

TRAIN_DATA_PATH = sys.argv[1]
SAVE_PATH = sys.argv[2]


class ParaphraseGenerator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model_name = "cjvt/t5-sl-small"
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.batch_size = 16
        self.lr = 4e-5

    def encode_text(self, data):
        for item in tqdm(data):
            # tokenizing original and paraphrase:
            source = self.tokenizer(
                item["original"],
                max_length=160,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            target = self.tokenizer(
                item["paraphrased"],
                max_length=200,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            yield source["input_ids"], target["input_ids"]

    def to_tensor(self, source_ids, target_ids):
        source_ids = torch.cat(source_ids, dim=0)
        target_ids = torch.cat(target_ids, dim=0)
        data = TensorDataset(source_ids, target_ids)
        return random_split(data, [len(data), 0])[0]

    def prepare_data(self):
        with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as r:
            data = json.load(r)

        # Train test split
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        print("Train data size:", len(train_data))
        print("Test data size:", len(test_data))

        source_ids, target_ids = list(zip(*tuple(self.encode_text(train_data))))
        self.train_ds = self.to_tensor(source_ids, target_ids)

        source_ids, target_ids = list(zip(*tuple(self.encode_text(test_data))))
        self.test_ds = self.to_tensor(source_ids, target_ids)

    def forward(self, batch, batch_idx):
        source_ids, target_ids = batch[:2]
        return self.model(input_ids=source_ids, labels=target_ids)

    def training_step(self, batch, batch_idx):
        loss = self(batch, batch_idx)[0]
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch, batch_idx)[0]
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


class SaveCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch > 0:
            current_epoch = str(pl_module.current_epoch)
            fn = f"epoch_{current_epoch}"
            new_path = f"{SAVE_PATH}/{fn}/"
            if fn not in os.listdir(SAVE_PATH):
                os.mkdir(new_path)
            pl_module.tokenizer.save_vocabulary(new_path)
            pl_module.model.save_pretrained(new_path)


trainer = pl.Trainer(
    default_root_dir="logs",
    min_epochs=8,
    max_epochs=64,
    val_check_interval=0.25,
    accumulate_grad_batches=4,
    callbacks=[SaveCallback()],
    logger=TensorBoardLogger("logs/", name="paraphrase", version=0),
)

para_model = ParaphraseGenerator()
trainer.fit(para_model)

print("Saving model...")
SAVE_PATH_FINAL = f"{SAVE_PATH}_final"
# Create dir if not exists
if not os.path.exists(SAVE_PATH_FINAL):
    os.makedirs(SAVE_PATH_FINAL)
para_model.tokenizer.save_vocabulary(SAVE_PATH_FINAL)
para_model.tokenizer.save_pretrained(SAVE_PATH_FINAL)
para_model.model.save_pretrained(SAVE_PATH_FINAL)

print("Done!")
