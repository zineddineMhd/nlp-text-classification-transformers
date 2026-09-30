# RITAL sentiment-classification experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # Movies — XLM-RoBERTa et Twitter-RoBERTa (CardiffNLP)
#
# Ce notebook teste **deux modèles transformers déjà fine-tunés pour le sentiment** :
#
# 1. **cardiffnlp/twitter-roberta-base-sentiment-latest**
# 2. **cardiffnlp/twitter-xlm-roberta-base-sentiment**
#
# ## Pourquoi ces modèles ?
# - `cardiffnlp/twitter-roberta-base-sentiment-latest` est un modèle RoBERTa spécialisé pour le sentiment sur tweets en anglais. citeturn469686search0turn469686search2
# - `cardiffnlp/twitter-xlm-roberta-base-sentiment` est une variante XLM-RoBERTa utilisée pour la classification de sentiment et listée parmi les modèles de text classification sur Hugging Face. citeturn469686search11
#
# ## Important
# - On teste ici **des modèles déjà fine-tunés**, pas `xlm-roberta-base` brut, car `xlm-roberta-base` seul est un modèle pré-entraîné général sans tête de classification sentiment prête à l’emploi. citeturn469686search1turn469686search5
# - Pour votre dataset de critiques de films en anglais, ce test est surtout **comparatif**.
# - Exécuter de préférence sur **GPU Colab**.

# %% [markdown]
# ## 1. Installation des dépendances

# %%
# Si besoin sur Colab, décommente :
# NOTEBOOK: !pip install -q transformers datasets accelerate scikit-learn

# %%
# NOTEBOOK-ONLY COMMAND REMOVED: from google.colab import drive
# NOTEBOOK-ONLY COMMAND REMOVED: drive.mount('/content/drive')

# %% [markdown]
# ## 2. Imports

# %%
from pathlib import Path
import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# %% [markdown]
# ## 3. Seed

# %%
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)

# %% [markdown]
# ## 4. Chargement des données

# %%
DATA_DIR = Path("/content/drive/MyDrive/projet tal/movies1000/movies1000")
print("DATA_DIR existe :", DATA_DIR.exists())
if DATA_DIR.exists():
    print("Contenu :", [p.name for p in DATA_DIR.iterdir()])

# %%
def load_movies_from_folder(data_dir: Path) -> pd.DataFrame:
    rows = []

    for label_folder, label in [("pos", "P"), ("neg", "N")]:
        folder = data_dir / label_folder
        if not folder.exists():
            continue

        for file_path in folder.glob("*.txt"):
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            rows.append({
                "doc_id": file_path.name,
                "label": label,
                "text": text,
            })

    if not rows:
        raise ValueError("Aucun fichier trouvé. Vérifie DATA_DIR.")

    return pd.DataFrame(rows).sort_values("doc_id").reset_index(drop=True)

df = load_movies_from_folder(DATA_DIR)
print(df.shape)
df.head()

# %% [markdown]
# ## 5. Split train / validation

# %%
train_df, valid_df = train_test_split(
    df[["doc_id", "label", "text"]].copy(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

print("Train :", train_df.shape)
print("Valid :", valid_df.shape)
print(train_df["label"].value_counts().sort_index())

# %% [markdown]
# ## 6. Préparer les datasets Hugging Face

# %%
label2id = {"N": 0, "P": 1}
id2label = {0: "N", 1: "P"}

hf_train_df = train_df[["text", "label"]].copy()
hf_valid_df = valid_df[["text", "label"]].copy()

hf_train_df["label_id"] = hf_train_df["label"].map(label2id)
hf_valid_df["label_id"] = hf_valid_df["label"].map(label2id)

hf_train = Dataset.from_pandas(
    hf_train_df[["text", "label_id"]].rename(columns={"label_id": "label"})
)
hf_valid = Dataset.from_pandas(
    hf_valid_df[["text", "label_id"]].rename(columns={"label_id": "label"})
)

hf_train, hf_valid

# %% [markdown]
# ## 7. Fonction de métriques

# %%
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    y_true = np.array([id2label[int(x)] for x in labels])
    y_pred = np.array([id2label[int(x)] for x in preds])

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="P"),
        "recall": recall_score(y_true, y_pred, pos_label="P"),
        "f1": f1_score(y_true, y_pred, pos_label="P"),
    }

def print_report(y_true, y_pred, title):
    print(f"\n===== {title} =====")
    print(classification_report(y_true, y_pred, digits=4))
    metrics = pd.DataFrame({
        "score": {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, pos_label="P"),
            "recall": recall_score(y_true, y_pred, pos_label="P"),
            "f1": f1_score(y_true, y_pred, pos_label="P"),
        }
    })
    display(metrics)

# %% [markdown]
# ## 8. Fonction utilitaire générique

# %%
def run_transformer_experiment(
    model_name: str,
    run_name: str,
    hf_train,
    hf_valid,
    valid_df,
    num_epochs: int = 2,
    learning_rate: float = 2e-5,
    batch_size_train: int = 8,
    batch_size_eval: int = 16,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
        )

    tokenized_train = hf_train.map(tokenize_fn, batched=True)
    tokenized_valid = hf_valid.map(tokenize_fn, batched=True)

    if "text" in tokenized_train.column_names:
        tokenized_train = tokenized_train.remove_columns(["text"])
    if "text" in tokenized_valid.column_names:
        tokenized_valid = tokenized_valid.remove_columns(["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    output_dir = f"/content/drive/MyDrive/projet tal/{run_name}_output"

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size_train,
        per_device_eval_batch_size=batch_size_eval,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_output = trainer.train()
    eval_output = trainer.evaluate()

    pred_output = trainer.predict(tokenized_valid)
    pred_ids = np.argmax(pred_output.predictions, axis=1)
    pred_labels = [id2label[int(x)] for x in pred_ids]

    print_report(valid_df["label"], pred_labels, run_name)

    return {
        "tokenizer": tokenizer,
        "trainer": trainer,
        "train_output": train_output,
        "eval_output": eval_output,
        "pred_labels": pred_labels,
    }

# %% [markdown]
# ## Partie A — CardiffNLP Twitter-RoBERTa

# %%
TWITTER_ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# %%
twitter_roberta_results = run_transformer_experiment(
    model_name=TWITTER_ROBERTA_MODEL,
    run_name="twitter_roberta_cardiffnlp",
    hf_train=hf_train,
    hf_valid=hf_valid,
    valid_df=valid_df,
    num_epochs=2,
    learning_rate=2e-5,
    batch_size_train=8,
    batch_size_eval=16,
)

# %% [markdown]
# ## Partie B — CardiffNLP Twitter-XLM-RoBERTa

# %%
TWITTER_XLMR_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# %%
# Décommente pour lancer l'expérience
twitter_xlmr_results = run_transformer_experiment(
     model_name=TWITTER_XLMR_MODEL,
     run_name="twitter_xlmr_cardiffnlp",
     hf_train=hf_train,
     hf_valid=hf_valid,
     valid_df=valid_df,
     num_epochs=2,
     learning_rate=2e-5,
     batch_size_train=8,
     batch_size_eval=16,
 )

# %% [markdown]
# ## 9. Tableau récapitulatif

# %%
summary_rows = [
    {"model": "cardiffnlp/twitter-roberta-base-sentiment-latest", "accuracy": None, "precision": None, "recall": None, "f1": None},
    {"model": "cardiffnlp/twitter-xlm-roberta-base-sentiment", "accuracy": None, "precision": None, "recall": None, "f1": None},
]

summary_df = pd.DataFrame(summary_rows)
summary_df

# %% [markdown]
# ## 10. Charger le fichier test

# %%
TEST_FILE = Path("/content/drive/MyDrive/projet tal/testSentiment.txt")

def load_test_one_review_per_line(test_file: Path) -> pd.DataFrame:
    lines = test_file.read_text(encoding="utf-8", errors="ignore").split("\n")
    rows = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line:
            rows.append({
                "doc_id": i,
                "text": line
            })
    return pd.DataFrame(rows)

test_df = load_test_one_review_per_line(TEST_FILE)
print(test_df.shape)
test_df.head()

# %% [markdown]
# ### 10.A Soumission avec Twitter-RoBERTa

# %%
# Décommente après avoir entraîné twitter_roberta_results
twitter_roberta_trainer = twitter_roberta_results["trainer"]
twitter_roberta_tokenizer = twitter_roberta_results["tokenizer"]

hf_test = Dataset.from_pandas(test_df[["text"]].copy())

def tokenize_test_twitter_roberta(batch):
     return twitter_roberta_tokenizer(batch["text"], truncation=True)

tokenized_test = hf_test.map(tokenize_test_twitter_roberta, batched=True)
if "text" in tokenized_test.column_names:
     tokenized_test = tokenized_test.remove_columns(["text"])

test_preds_output = twitter_roberta_trainer.predict(tokenized_test)
test_pred_ids = np.argmax(test_preds_output.predictions, axis=1)
test_pred_labels = [id2label[int(x)] for x in test_pred_ids]

submission_twitter_roberta = pd.DataFrame({"label": test_pred_labels})
display(submission_twitter_roberta.head(10))

output_path = "/content/drive/MyDrive/projet tal/submission_movies_twitter_roberta.csv"
submission_twitter_roberta.to_csv(output_path, index=False)
print(f"Fichier enregistré : {output_path}")

# %% [markdown]
# ### 10.B Soumission avec Twitter-XLM-RoBERTa

# %%
# Décommente après avoir entraîné twitter_xlmr_results
# twitter_xlmr_trainer = twitter_xlmr_results["trainer"]
# twitter_xlmr_tokenizer = twitter_xlmr_results["tokenizer"]

# hf_test = Dataset.from_pandas(test_df[["text"]].copy())

# def tokenize_test_twitter_xlmr(batch):
#     return twitter_xlmr_tokenizer(batch["text"], truncation=True)

# tokenized_test = hf_test.map(tokenize_test_twitter_xlmr, batched=True)
# if "text" in tokenized_test.column_names:
#     tokenized_test = tokenized_test.remove_columns(["text"])

# test_preds_output = twitter_xlmr_trainer.predict(tokenized_test)
# test_pred_ids = np.argmax(test_preds_output.predictions, axis=1)
# test_pred_labels = [id2label[int(x)] for x in test_pred_ids]

# submission_twitter_xlmr = pd.DataFrame({"label": test_pred_labels})
# display(submission_twitter_xlmr.head(10))

# output_path = "/content/drive/MyDrive/projet tal/submission_movies_twitter_xlmr.csv"
# submission_twitter_xlmr.to_csv(output_path, index=False)
# print(f"Fichier enregistré : {output_path}")

# %% [markdown]
# ## 11. Conclusion attendue
#
# Après exécution, compare :
# - Twitter-RoBERTa
# - Twitter-XLM-RoBERTa
# - votre meilleur SVM optimisé
#
# et garde seulement le meilleur pour la soumission finale.