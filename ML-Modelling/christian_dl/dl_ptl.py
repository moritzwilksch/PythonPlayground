#%%
import re

import nltk
import pandas as pd
import polars as pl
import pytorch_lightning as pl
import ray
import torch
import torch.functional as F
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from nltk.stem import PorterStemmer
from ray.util.multiprocessing import Pool
from sklearn.model_selection import train_test_split
from torchmetrics.functional import precision
from torchtext.vocab import build_vocab_from_iterator

#%%
pool = Pool()

ps = PorterStemmer()
stopwords = nltk.corpus.stopwords.words("german")

# load data
df = pd.read_json(
    "/home/data-v3.json",
)


def prep_text(s: str) -> str:
    words = re.split(r"\s", s)
    words = [
        ps.stem(w).lower()
        for w in words
        if w.lower() not in stopwords and len(w) > 2 and w != " "
    ]
    return " ".join(words)


preped_text = pool.map(prep_text, df["text"].to_list())

df["text"] = preped_text
df_subset = df[["text", "relevant"]]


#%%
class ASDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_vocab(data):
    for text in data:
        yield tokenizer(text)


tokenizer = torchtext.data.get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(
    build_vocab(df_subset["text"]), specials=["<UNK>"], min_freq=10
)
vocab.set_default_index(vocab["<UNK>"])

#%%
vocab(tokenizer("hello world"))
#%%
# sequence_of_ids = [vocab(tokenizer(text)) for text in df_subset["text"]]
@ray.remote
def text_to_ids(text):
    return vocab(tokenizer(text))


sequence_of_ids = [text_to_ids.remote(txt) for txt in df_subset["text"]]
sequence_of_ids = ray.get(sequence_of_ids)

xtrain, xtest, ytrain, ytest = train_test_split(
    sequence_of_ids, df_subset["relevant"], test_size=0.2, random_state=42
)

ray.shutdown()
#%%


def pad_collate(batch):
    xx = []
    yy = []

    for x, y in batch:
        xx.append(torch.Tensor(x))
        yy.append(y)

    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)

    return xx_pad, torch.Tensor(yy).view(-1, 1)


train_ds = ASDataset(xtrain, ytrain.values)
test_ds = ASDataset(xtest, ytest.values)

train_dataloader = torch.utils.data.DataLoader(
    train_ds, batch_size=128, shuffle=True, collate_fn=pad_collate, num_workers=4
)
test_dataloader = torch.utils.data.DataLoader(
    test_ds, batch_size=512, shuffle=False, collate_fn=pad_collate
)

#%%


class ASModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(vocab) + 1, 64)
        self.rnn = nn.GRU(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.loss = nn.BCELoss()
        self.relu = nn.ReLU()
        # self.train_precision = torchmetrics.Precision(num_classes=2, threshold=0.5)
        # self.val_precision = torchmetrics.Precision(num_classes=2, threshold=0.5)

    def forward(self, x):
        x = self.emb(x.long())
        out, hidden = self.rnn(x)
        x = hidden.squeeze()
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = torch.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        train_precision = precision(y_hat, y.int())
        self.log("train_precision", train_precision)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        val_precision = precision(y_hat, y.int())
        self.log("val_precision", val_precision)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


#%%
model = ASModel()
trainer = pl.Trainer(max_epochs=10)

trainer.fit(model, train_dataloader, test_dataloader)
