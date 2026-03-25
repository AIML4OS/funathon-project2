# %%

import s3fs
import os
from dotenv import load_dotenv
import polars as pl
from tokenizers import Tokenizer
from torchTextClassifiers.tokenizers import WordPieceTokenizer
import logging

logger = logging.getLogger(__name__)

load_dotenv(override=True)

# %%
OUTPUT_PATH = "projet-ape/synthetic_data_test/naive/NAF2025_FR/"
FILE_NAME = "generation_gpt-oss-120b_temp14_French_fewshot6_exhaustive.parquet"

fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    token=os.environ["AWS_SESSION_TOKEN"]
)


with fs.open(os.path.join(OUTPUT_PATH, FILE_NAME), 'rb') as f:
    df = pl.read_parquet(f)

print(df.head())
print(len(df))


# %%
n_classes = df['code'].n_unique()
print(n_classes)

# %%
def train_val_test_split(
    df: pl.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    # Mélange aléatoire
    df = df.sample(fraction=1.0, shuffle=True, seed=seed)

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = df[:n_train]
    val   = df[n_train : n_train + n_val]
    test  = df[n_train + n_val :]

    return train, val, test


train, val, test = train_val_test_split(df)

X_train, y_train = train['label'], train['code']
X_val, y_val = val['label'], val['code']
X_test, y_test = test['label'], test['code']

# %%
tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=128)
training_corpus = train['label'].to_list()
tokenizer.train(training_corpus)

# %%
model_config = ModelConfig(
    embedding_dim=50,
    num_classes=n_classes
)

classifier = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config
)

# %%

training_config = TrainingConfig(
    num_epochs=20,
    batch_size=4,
    lr=1e-3,
    patience_early_stopping=5,
    num_workers=0,  # Use 0 for simple examples to avoid multiprocessing issues
)

classifier.train(
    X_train, y_train, 
    training_config=training_config,
    X_val=X_val, y_val=y_val,
    verbose=True
)

result = classifier.predict(X_test)
predictions = result["prediction"].squeeze().numpy()  # Extract predictions from dictionary
confidence = result["confidence"].squeeze().numpy()  # Extract confidence scores
logger.info(f"Predictions: {predictions}")
logger.info(f"Confidence: {confidence}")
logger.info(f"True labels: {y_test}")
# %%
