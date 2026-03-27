import s3fs
import os
from dotenv import load_dotenv
import duckdb
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder

from torchTextClassifiers.model.components import (
    AttentionConfig, ClassificationHead,
    TextEmbedder, TextEmbedderConfig,
)
from torchTextClassifiers.tokenizers import HuggingFaceTokenizer, WordPieceTokenizer
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers

from torchTextClassifiers.model import TextClassificationModule
import pytorch_lightning as pl
import torch
from torchTextClassifiers.dataset import TextClassificationDataset

import mlflow

TRAIN_FRAC = 1
VAL_FRAC = 1

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
RUN_ID = os.environ["MLFLOW_RUN_ID"]

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
load_dotenv(override=True)

con = duckdb.connect(database=":memory:")
con.execute("INSTALL httpfs;")
con.execute("LOAD httpfs;")

query = "SELECT * FROM read_parquet('https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet')"
df = con.sql(query).df()

n_classes = df['code'].nunique()


def train_val_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df[:n_train]
    val = df[n_train: n_train + n_val]
    test = df[n_train + n_val:]
    return train, val, test


train, val, test = train_val_test_split(df)

X_train, y_train = train['label'].to_numpy(), train['code'].to_numpy()
X_val, y_val = val['label'].to_numpy(),   val['code'].to_numpy()
X_test, y_test = test['label'].to_numpy(),  test['code'].to_numpy()

logger.info(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")


tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=125)
tokenizer.train(X_train)

logger.info("Output tensor size:", tokenizer.tokenize(X_train[0]).input_ids.shape)
logger.info("Tokens:", tokenizer.tokenizer.convert_ids_to_tokens(
    tokenizer.tokenize(X_train[0]).input_ids.squeeze(0)
))
logger.info("Vocabulary size:", tokenizer.vocab_size)

embedding_dim = 96

# Self-attention configuration: 1 layer, 4 attention heads
attention_config = AttentionConfig(
    n_layers=1, n_head=4, n_kv_head=4,
    sequence_len=tokenizer.output_dim,
)

text_embedder_config = TextEmbedderConfig(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=embedding_dim,
    padding_idx=tokenizer.padding_idx,
    attention_config=attention_config,
)

text_embedder = TextEmbedder(text_embedder_config=text_embedder_config)
text_embedder.init_weights()
text_embedder


classification_head = ClassificationHead(
    input_dim=embedding_dim,
    num_classes=n_classes,
)
logger.info(f"Input dim: {embedding_dim} → Output classes: {n_classes}")


model_config = ModelConfig(
    embedding_dim=embedding_dim,
    num_classes=n_classes,
    attention_config=attention_config,
)

ttc = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
)

# Fit on all codes so that val/test codes are not unknown to the encoder
encoder = LabelEncoder()
encoder.fit(df['code'].to_numpy())

X_train, y_train = train['label'].to_numpy(), encoder.transform(train['code'].to_numpy())
X_val, y_val = val['label'].to_numpy(),   encoder.transform(val['code'].to_numpy())
X_test, y_test = test['label'].to_numpy(),  encoder.transform(test['code'].to_numpy())

logger.info(f"Example raw code:     {train['code'][0]}")
logger.info(f"Example encoded code: {encoder.transform(train['code'][:1])[0]}")


X_train_small = X_train[:int(len(X_train) * TRAIN_FRAC)]
y_train_small = y_train[:int(len(y_train) * TRAIN_FRAC)]
X_val_small = X_val[:int(len(X_val) * VAL_FRAC)]
y_val_small = y_val[:int(len(y_val) * VAL_FRAC)]


training_config = TrainingConfig(
    lr=1e-3,
    batch_size=256,
    num_epochs=5,
    patience_early_stopping=3,
    num_workers=8,
)

ttc.train(
    X_train=X_train_small,
    y_train=y_train_small,
    X_val=X_val,
    y_val=y_val,
    training_config=training_config,
    verbose=True,
)


