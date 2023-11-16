#!/usr/bin/env python
# coding: utf-8
import functools
from pathlib import Path
from typing import Tuple, Any, Literal, List

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger
from wandb.old.summary import np

import utils

# In[4]:

pl.seed_everything(42)

# In[ ]:


INPUT_SIZE = 9  # Number of features (bands)
HIDDEN_SIZE = 64  # Number of features in the hidden state
NUM_LAYERS = 2  # Number of stacked LSTM layers
NUM_CLASSES = 3  # Number of classes (crops)

EPOCHS = 10

# In[ ]:


ROOT_DIR = Path.cwd()
LOG_DIR: Path = ROOT_DIR / "logs"
CKPT_DIR: Path = ROOT_DIR / "ckpts"

# In[2]:


wandb_logger = WandbLogger(
    project='crop-classification',
)


# In[4]:


class CropLSTM(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            num_classes: int
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # (batch_size, seq_length, hidden_size)

        out = self.fc(out[:, -1, :])
        return out


class BiLSTMAttention(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            n_layers: int,
            dropout: float,
            bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )

        # Attention Layer
        self.attention = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # LSTM output shape: (batch_size, seq_length, num_directions * hidden_size)
        lstm_out, (hidden, cell) = self.lstm(x)

        attn_weights = torch.tanh(self.attention(lstm_out))
        soft_attn_weights = torch.softmax(attn_weights, 1)

        new_hidden_state = torch.bmm(lstm_out.transpose(1, 2), soft_attn_weights).squeeze(2)

        y_pred = self.fc(new_hidden_state)

        return self.act(y_pred), soft_attn_weights


# In[5]:


Mode = Literal["train", "val", "test"]


class CropClassifier(pl.LightningModule):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            num_classes: int
    ) -> None:
        super().__init__()
        self.model = CropLSTM(input_size, hidden_size, num_layers, num_classes)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._model_step(batch, mode="train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._model_step(batch, mode="val")

    def _model_step(self, batch: Any, mode: Mode) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_function(logits, y)

        self.log(
            f"{mode}/loss",
            value=loss,
            on_step=True if mode == "train" else None,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# In[6]:
end_datetime = "07-01"  # july 1st
# you can go up to 10-15 (october 15th)
X_18, y_18, y_18_indices = utils.X_y(2018, end_datetime=end_datetime, return_indices=True)
X_19, y_19, y_19_indices = utils.X_y(2019, end_datetime=end_datetime, return_indices=True)
X_20, y_20, y_20_indices = utils.X_y(2020, end_datetime=end_datetime, return_indices=True)

train_ds = utils.torch_dataset(
    np.vstack((X_18, X_20)), np.hstack((y_18, y_20)), n_bands=INPUT_SIZE
)


class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            data_split: tuple[float, float, float],
            use_transforms: bool = False,
            val_batch_size_multiplier: int = 2,
            num_workers: int = 0,
            pin_memory: bool = True,
    ) -> None:
        super().__init__()

        self.batch_size: int = batch_size  # needs to be set for BatchSizeFinder callback
        self.use_transforms: bool = use_transforms

        self.data_split: tuple[float, float, float] = data_split
        assert sum(self.data_split) == 1.0, "Data split must sum to 1.0"

        self.train = None
        self.val = None
        self.test = None

        self.val_batch_size_multiplier: int = val_batch_size_multiplier
        self.dataloader_partial = functools.partial(
            torch.utils.data.DataLoader,
            num_workers=num_workers,
            pin_memory=pin_memory,
            # batch_size=self.batch_size,  # Keep option to set dynamically by BatchSizeFinder callback
        )

    def setup(self, stage: str | None = None) -> None:
        print("Building dataset ...")

        dataset = ...

        # Split dataset
        train_size = int(self.data_split[0] * len(dataset))
        val_size = int(self.data_split[1] * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train, self.val, self.test = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader_partial(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader_partial(self.val, batch_size=self.batch_size)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader_partial(self.test, batch_size=self.batch_size)


dm = DataModule(batch_size=32, data_split=(0.9, 0.1, 0.0))

# In[8]:


model = CropClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
# wandb_logger.watch(model=model, log="all", log_graph=True, log_freq=100)


# In[9]:

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    logger=wandb_logger,
    accelerator="gpu",
    devices=-1,
    benchmark=True,
    log_every_n_steps=1,
    # precision="bf16-mixed",

    # Debugging
    fast_dev_run=False,
    overfit_batches=0,
    profiler=None,
)
callbacks: List[pl.Callback] = [
    pl.callbacks.ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        dirpath=CKPT_DIR,
        save_top_k=1,
        save_last=True,
    )
]

# In[10]:


# model = torch.compile(
#     model,
#     fullgraph=True,
#     mode="max-autotune",
#     dynamic=False,
# )


# In[20]:


trainer.fit(
    model,
    datamodule=dm,
    # train_dataloaders=train_dataloader, 
    # val_dataloaders=val_dataloader
)

# In[ ]:
