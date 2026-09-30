# RITAL sentiment-classification experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # Movies — Modèles NLP avancés
#
# Ce notebook compare **trois approches plus avancées** pour la classification des critiques de films (`P/N`) :
#
# 1. **FastText**
# 2. **DistilBERT**
# 3. **BERT**
#
# ## Objectif
# Comparer ces modèles à votre meilleure baseline classique :
# - **TF-IDF + LinearSVC optimisé**
#
# ## Conseils
# - Exécuter ce notebook sur **Google Colab**
# - Pour DistilBERT et BERT, activer de préférence **GPU**
# - Garder le notebook baseline et le notebook improvements séparés

# %% [markdown]
# ## 1. Installation des dépendances
#
# - `fasttext` pour le modèle FastText
# - `transformers` et `datasets` pour DistilBERT / BERT
# - `accelerate` pour faciliter l'entraînement avec Hugging Face

# %%
# Si besoin sur Colab, décommente cette cellule
# NOTEBOOK: !pip install -q fasttext transformers datasets accelerate scikit-learn

# %%
# NOTEBOOK-ONLY COMMAND REMOVED: from google.colab import drive
# NOTEBOOK-ONLY COMMAND REMOVED: drive.mount('/content/drive')

# %% [markdown]
# ## 2. Imports

# %%
from pathlib import Path
import os
import re
import string
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import matplotlib.pyplot as plt

# %% [markdown]
# ## 3. Fixer la seed

# %%
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)

# %% [markdown]
# ## 4. Chargement des données
#
# Adapte `DATA_DIR` à ton Drive.
#
# Structure attendue :
#
# ```text
# movies1000/
#     pos/
#     neg/
# ```
#
# ou bien :
#
# ```text
# movies1000/
#     movies1000/
#         pos/
#         neg/
# ```

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
# ## 5. Vérification rapide

# %%
print(df["label"].value_counts().sort_index())
df["n_words"] = df["text"].str.split().str.len()
display(df.groupby("label")["n_words"].agg(["mean", "median", "std", "min", "max"]))

# %% [markdown]
# ## 6. Préprocessing modéré
#
# On réutilise ici le préprocessing standard qui avait bien marché :
# - minuscules
# - suppression ponctuation
# - suppression chiffres
# - normalisation espaces

# %%
def preprocess_standard(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text_standard"] = df["text"].apply(preprocess_standard)
df[["text", "text_standard"]].head(3)

# %% [markdown]
# ## 7. Split train / validation

# %%
train_df, valid_df = train_test_split(
    df[["doc_id", "label", "text", "text_standard"]].copy(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

print("Train :", train_df.shape)
print("Valid :", valid_df.shape)

# %% [markdown]
# ## 8. Fonction utilitaire d'évaluation

# %%
def evaluate_predictions(y_true, y_pred, positive_label="P"):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=positive_label),
        "recall": recall_score(y_true, y_pred, pos_label=positive_label),
        "f1": f1_score(y_true, y_pred, pos_label=positive_label),
    }
    return pd.DataFrame({"score": metrics})

def print_report(y_true, y_pred, title):
    print(f"\n===== {title} =====")
    print(classification_report(y_true, y_pred, digits=4))
    display(evaluate_predictions(y_true, y_pred))

# %% [markdown]
# ##**version un fichier plusieur reviews**

# %%
from pathlib import Path
import pandas as pd

DATA_DIR = Path("/content/drive/MyDrive/projet tal/movies1000/movies1000")

def load_movies_one_review_per_line(data_dir: Path) -> pd.DataFrame:
    rows = []

    for label_folder, label in [("pos", "P"), ("neg", "N")]:
        folder = data_dir / label_folder

        if not folder.exists():
            continue

        for file_path in folder.glob("*.txt"):
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # On considère qu'une review = une ligne non vide
            reviews = [line.strip() for line in content.splitlines() if line.strip()]

            for i, review in enumerate(reviews):
                rows.append({
                    "source_file": file_path.name,
                    "review_id": f"{file_path.stem}_{i}",
                    "label": label,
                    "text": review,
                })

    if not rows:
        raise ValueError("Aucune review trouvée. Vérifie DATA_DIR et la structure des fichiers.")

    df = pd.DataFrame(rows).reset_index(drop=True)
    return df

df = load_movies_one_review_per_line(DATA_DIR)

print(df.shape)
display(df.head(10))
print(df["label"].value_counts())

# %%
import re
import string

def preprocess_standard(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text_standard"] = df["text"].apply(preprocess_standard)
display(df[["text", "text_standard"]].head(10))

# %%
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(
    df[["source_file", "review_id", "label", "text", "text_standard"]].copy(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

print("Train :", train_df.shape)
print("Valid :", valid_df.shape)
print("\nRépartition train :")
print(train_df["label"].value_counts())
print("\nRépartition valid :")
print(valid_df["label"].value_counts())

# %% [markdown]
# # Partie A — FastText
#
# FastText est un bon compromis :
# - plus avancé que TF-IDF
# - plus léger que BERT
# - rapide à entraîner
# - adapté à la classification de texte

# %% [markdown]
# ## 9. Préparer les fichiers FastText

# %%
def label_to_fasttext(label: str) -> str:
    return f"__label__{label}"

FASTTEXT_DIR = Path("/content/drive/MyDrive/projet tal/fasttext_tmp")
FASTTEXT_DIR.mkdir(parents=True, exist_ok=True)

fasttext_train_path = FASTTEXT_DIR / "movies_train_fasttext.txt"
fasttext_valid_path = FASTTEXT_DIR / "movies_valid_fasttext.txt"

def write_fasttext_file(dataframe: pd.DataFrame, output_path: Path, text_col: str = "text_standard"):
    with output_path.open("w", encoding="utf-8") as f:
        for _, row in dataframe.iterrows():
            label = label_to_fasttext(row["label"])
            text = str(row[text_col]).replace("\n", " ").strip()
            f.write(f"{label} {text}\n")

write_fasttext_file(train_df, fasttext_train_path, text_col="text_standard")
write_fasttext_file(valid_df, fasttext_valid_path, text_col="text_standard")

print("Train file :", fasttext_train_path)
print("Valid file :", fasttext_valid_path)

# %% [markdown]
# ## 10. Entraîner FastText

# %%
import fasttext

fasttext_model = fasttext.train_supervised(
    input=str(fasttext_train_path),
    lr=0.5,
    epoch=30,
    wordNgrams=2,
    dim=100,
    loss="softmax",
    thread=4,
)

fasttext_model

# %% [markdown]
# ## 11. Évaluer FastText

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

texts_valid = valid_df["text_standard"].astype(str).tolist()
labels_pred, scores = fasttext_model.predict(texts_valid, k=1)
preds = [lab[0].replace("__label__", "") for lab in labels_pred]

print(classification_report(valid_df["label"], preds, digits=4))

metrics_valid = pd.DataFrame({
    "score": {
        "accuracy": accuracy_score(valid_df["label"], preds),
        "precision": precision_score(valid_df["label"], preds, pos_label="P"),
        "recall": recall_score(valid_df["label"], preds, pos_label="P"),
        "f1": f1_score(valid_df["label"], preds, pos_label="P"),
    }
})

display(metrics_valid)

# %% [markdown]
# ## 12. Petit tuning FastText

# %%
fasttext_param_grid = [
    {"lr": 0.5, "epoch": 25, "wordNgrams": 2, "dim": 100},
    {"lr": 0.3, "epoch": 30, "wordNgrams": 2, "dim": 100},
    {"lr": 0.5, "epoch": 25, "wordNgrams": 3, "dim": 100},
    {"lr": 0.5, "epoch": 30, "wordNgrams": 2, "dim": 200},
]

fasttext_results = []

texts_valid = valid_df["text_standard"].astype(str).tolist()
y_true = valid_df["label"].tolist()

for params in fasttext_param_grid:
    model = fasttext.train_supervised(
        input=str(fasttext_train_path),
        lr=params["lr"],
        epoch=params["epoch"],
        wordNgrams=params["wordNgrams"],
        dim=params["dim"],
        loss="softmax",
        thread=4,
    )

    labels, scores = model.predict(texts_valid, k=1)
    preds = [lab[0].replace("__label__", "") for lab in labels]

    res = {
        **params,
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, pos_label="P"),
        "recall": recall_score(y_true, preds, pos_label="P"),
        "f1": f1_score(y_true, preds, pos_label="P"),
    }
    fasttext_results.append(res)

fasttext_results_df = (
    pd.DataFrame(fasttext_results)
    .sort_values("f1", ascending=False)
    .reset_index(drop=True)
)

fasttext_results_df

# %%
from pathlib import Path
import pandas as pd
import fasttext

# 1) Charger le test
TEST_FILE = Path("/content/testSentiment (1).txt")

def load_test_one_review_per_line(test_file: Path) -> pd.DataFrame:
    raw_text = test_file.read_text(encoding="utf-8", errors="ignore")
    lines = raw_text.split("\n")

    print("Nombre total de lignes dans le fichier test :", len(lines))

    non_empty_lines = [line.strip() for line in lines if line.strip()]
    print("Nombre de lignes non vides (= samples utilisés) :", len(non_empty_lines))

    rows = []
    for i, line in enumerate(non_empty_lines):
        rows.append({
            "doc_id": i,
            "text": line
        })

    return pd.DataFrame(rows)

test_df = load_test_one_review_per_line(TEST_FILE)

print("\nShape du DataFrame test :", test_df.shape)
print("Nombre final de samples test :", len(test_df))

# 2) Préprocessing identique à l'entraînement
test_df["text_standard"] = test_df["text"].apply(preprocess_standard)


# 4) Prédictions sur le test
texts_test = test_df["text_standard"].astype(str).tolist()
labels, scores = fasttext_model.predict(texts_test, k=1)
test_pred_fasttext = [lab[0].replace("__label__", "") for lab in labels]

# 5) Création du fichier de soumission
submission_fasttext = pd.DataFrame({
    "label": test_pred_fasttext
})

print("\nNombre de prédictions générées :", len(submission_fasttext))
display(submission_fasttext.head(10))
print(submission_fasttext["label"].value_counts())

# 6) Sauvegarde CSV
output_path = "/content/drive/MyDrive/projet tal/submission-movies-fasttext-1.csv"
submission_fasttext.to_csv(output_path, index=False)

print(f"\nFichier enregistré : {output_path}")

# %%
submission_fasttext.to_csv(output_path, index=False, header=False)

# %% [markdown]
# # Partie B — DistilBERT
#
# DistilBERT est une version plus légère de BERT :
# - plus rapide
# - moins coûteuse
# - souvent très performante

# %% [markdown]
# ## 13. Imports transformers

# %%
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# %% [markdown]
# ## 14. Préparer les données pour Hugging Face

# %%
label2id = {"N": 0, "P": 1}
id2label = {0: "N", 1: "P"}

hf_train_df = train_df[["text", "label"]].copy()
hf_valid_df = valid_df[["text", "label"]].copy()

hf_train_df["label_id"] = hf_train_df["label"].map(label2id)
hf_valid_df["label_id"] = hf_valid_df["label"].map(label2id)

hf_train = Dataset.from_pandas(hf_train_df[["text", "label_id"]].rename(columns={"label_id": "label"}))
hf_valid = Dataset.from_pandas(hf_valid_df[["text", "label_id"]].rename(columns={"label_id": "label"}))

hf_train, hf_valid

# %% [markdown]
# ## 15. Fonction de métriques pour transformers

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

# %% [markdown]
# ## 16. DistilBERT — Tokenizer + tokenisation

# %%
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"

distilbert_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)

def tokenize_distilbert(batch):
    return distilbert_tokenizer(batch["text"], truncation=True)

tokenized_train_distil = hf_train.map(tokenize_distilbert, batched=True)
tokenized_valid_distil = hf_valid.map(tokenize_distilbert, batched=True)

data_collator_distil = DataCollatorWithPadding(tokenizer=distilbert_tokenizer)

# %% [markdown]
# ## 17. DistilBERT — Modèle

# %%
distilbert_model = AutoModelForSequenceClassification.from_pretrained(
    DISTILBERT_MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

# %% [markdown]
# ## 18. DistilBERT — Entraînement

# %%
distil_output_dir = "/content/drive/MyDrive/projet tal/distilbert_movies_output"

distil_training_args = TrainingArguments(
    output_dir=distil_output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    save_total_limit=1,
)

# %%
distil_trainer = Trainer(
    model=distilbert_model,
    args=distil_training_args,
    train_dataset=tokenized_train_distil,
    eval_dataset=tokenized_valid_distil,
    data_collator=data_collator_distil,
    compute_metrics=compute_metrics,
)

# %%
distil_trainer.train()

# %% [markdown]
# ## 19. DistilBERT — Évaluation

# %%
# Décommente après distil_trainer.train()
distil_eval = distil_trainer.evaluate()
print(distil_eval)

# %%
# Décommente après entraînement pour récupérer les prédictions détaillées
distil_preds_output = distil_trainer.predict(tokenized_valid_distil)
distil_preds = np.argmax(distil_preds_output.predictions, axis=1)
distil_pred_labels = [id2label[int(x)] for x in distil_preds]

print_report(valid_df["label"], distil_pred_labels, "DistilBERT")

# %% [markdown]
# # Partie C — BERT
#
# Même logique que DistilBERT, mais avec un modèle plus lourd.
# Ici on utilise `bert-base-uncased`, cohérent avec des reviews en anglais.

# %% [markdown]
# ## 20. BERT — Tokenizer + tokenisation

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
BERT_MODEL_NAME = "bert-base-uncased"

bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

def tokenize_bert(batch):
    return bert_tokenizer(batch["text"], truncation=True)

tokenized_train_bert = hf_train.map(tokenize_bert, batched=True)
tokenized_valid_bert = hf_valid.map(tokenize_bert, batched=True)

data_collator_bert = DataCollatorWithPadding(tokenizer=bert_tokenizer)

# %% [markdown]
# ## 21. BERT — Modèle

# %%
bert_model = AutoModelForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

# %% [markdown]
# ## 22. BERT — Entraînement

# %%
bert_output_dir = "/content/drive/MyDrive/projet tal/bert_movies_output"

bert_training_args = TrainingArguments(
    output_dir=bert_output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    save_total_limit=1,
)

# %%
bert_trainer = Trainer(
    model=bert_model,
    args=bert_training_args,
    train_dataset=tokenized_train_bert,
    eval_dataset=tokenized_valid_bert,
    data_collator=data_collator_bert,
    compute_metrics=compute_metrics,
)

bert_trainer.train()

# %% [markdown]
# ## 23. BERT — Évaluation

# %%
# Décommente après bert_trainer.train()
bert_eval = bert_trainer.evaluate()
print(bert_eval)

# %%
# Décommente après entraînement pour récupérer les prédictions détaillées
bert_preds_output = bert_trainer.predict(tokenized_valid_bert)
bert_preds = np.argmax(bert_preds_output.predictions, axis=1)
bert_pred_labels = [id2label[int(x)] for x in bert_preds]

print_report(valid_df["label"], bert_pred_labels, "BERT")

# %%
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import Dataset

# 1) Charger le test
TEST_FILE = Path("/content/drive/MyDrive/projet tal/testSentiment.txt")

def load_test_one_review_per_line(test_file: Path) -> pd.DataFrame:
    raw_text = test_file.read_text(encoding="utf-8", errors="ignore")
    lines = raw_text.split("\n")

    print("Nombre total de lignes dans le fichier test :", len(lines))

    non_empty_lines = [line.strip() for line in lines if line.strip()]
    print("Nombre de lignes non vides (= samples utilisés) :", len(non_empty_lines))

    rows = []
    for i, line in enumerate(non_empty_lines):
        rows.append({
            "doc_id": i,
            "text": line
        })

    return pd.DataFrame(rows)

test_df = load_test_one_review_per_line(TEST_FILE)

print("\nShape du DataFrame test :", test_df.shape)
print("Nombre final de samples test :", len(test_df))

# 2) Créer le dataset Hugging Face
hf_test = Dataset.from_pandas(test_df[["text"]].copy())

# 3) Tokenisation BERT
def tokenize_test_bert(batch):
    return bert_tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
    )

tokenized_test_bert = hf_test.map(tokenize_test_bert, batched=True)

if "text" in tokenized_test_bert.column_names:
    tokenized_test_bert = tokenized_test_bert.remove_columns(["text"])

# 4) Prédictions avec BERT
bert_preds_output = bert_trainer.predict(tokenized_test_bert)
bert_pred_ids = np.argmax(bert_preds_output.predictions, axis=1)

# 5) Conversion vers labels P/N
test_pred_bert = [id2label[int(x)] for x in bert_pred_ids]

# 6) Création du fichier de soumission sans nom de colonne
submission_bert = pd.DataFrame(test_pred_bert)

print("\nNombre de prédictions générées :", len(submission_bert))
display(submission_bert.head(10))
print(submission_bert[0].value_counts())

# 7) Sauvegarde CSV sans header
output_path = "/content/drive/MyDrive/projet tal/submission-movies-bert-1.csv"
submission_bert.to_csv(output_path, index=False, header=False)

print(f"\nFichier enregistré : {output_path}")

# %% [markdown]
# ## 24. Tableau récapitulatif à remplir

# %%
summary_rows = [
    {"model": "FastText", "accuracy": None, "precision": None, "recall": None, "f1": None},
    {"model": "DistilBERT", "accuracy": None, "precision": None, "recall": None, "f1": None},
    {"model": "BERT", "accuracy": None, "precision": None, "recall": None, "f1": None},
]

summary_df = pd.DataFrame(summary_rows)
summary_df

# %% [markdown]
# ## 25. Modèle final retenu
#
# À compléter après comparaison.
#
# Exemple :
# - si FastText est proche du meilleur SVM mais plus simple, tu peux le garder comme modèle NLP léger
# - si DistilBERT surpasse clairement les autres, il devient le meilleur modèle avancé
# - si BERT n'apporte pas de gain net par rapport à DistilBERT, DistilBERT peut rester le meilleur compromis

# %% [markdown]
# ## 26. Prochaine étape
#
# Quand tu auras exécuté ce notebook, montre-moi :
# - les résultats `fasttext_results_df`
# - les scores de DistilBERT
# - les scores de BERT
#
# et on décidera :
# - quel modèle garder
# - lequel mettre dans le notebook de soumission avancée
# - comment l'expliquer dans le rapport

# %% [markdown]
# **test** roberta xlm

# %%
# Si besoin sur Colab, décommente :
# NOTEBOOK: !pip install -q transformers datasets accelerate scikit-learn

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

# %%
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)

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
# ## Partie B — CardiffNLP Twitter-XLM-RoBERTa

# %%
#TWITTER_XLMR_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
TWITTER_ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# %%
'''# Décommente pour lancer l'expérience
twitter_xlmr_results = run_transformer_experiment(
     model_name=TWITTER_XLMR_MODEL,
     run_name="twitter_xlmr_cardiffnlp",
     hf_train=hf_train,     hf_valid=hf_valid,
     valid_df=valid_df,
     num_epochs=2,
     learning_rate=2e-5,
     batch_size_train=8,
     batch_size_eval=16,
 )'''
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

# %%
# Décommente après avoir entraîné twitter_roberta_results
twitter_roberta_trainer = twitter_roberta_results["trainer"]
twitter_roberta_tokenizer = twitter_roberta_results["tokenizer"]

hf_test = Dataset.from_pandas(test_df[["text"]].copy())

def tokenize_test_twitter_roberta(batch):
    return twitter_roberta_tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
    )

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

# %%
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "/content/drive/MyDrive/projet tal/twitter_roberta_cardiffnlp_output/checkpoint-400"

# si tu ne connais pas le bon checkpoint, regarde le contenu du dossier parent
print(Path("/content/drive/MyDrive/projet tal/twitter_roberta_cardiffnlp_output").exists())

twitter_roberta_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

twitter_roberta_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
twitter_roberta_model.to(device)
twitter_roberta_model.eval()

print("Modèle rechargé depuis :", MODEL_DIR)

# %%
import torch
import pandas as pd
from tqdm.auto import tqdm

texts_test = test_df["text"].astype(str).tolist()

def predict_in_batches(model, tokenizer, texts, batch_size=8, max_length=512):
    all_pred_ids = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]

        encodings = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )

        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        all_pred_ids.extend(pred_ids)

    return all_pred_ids

test_pred_ids = predict_in_batches(
    model=twitter_roberta_model,
    tokenizer=twitter_roberta_tokenizer,
    texts=texts_test,
    batch_size=8,
    max_length=512,
)

test_pred_labels = [id2label[int(x)] for x in test_pred_ids]

submission_twitter_roberta = pd.DataFrame(test_pred_labels)

print("Nombre de prédictions :", len(submission_twitter_roberta))
display(submission_twitter_roberta.head(10))
print(submission_twitter_roberta[0].value_counts())

output_path = "/content/drive/MyDrive/projet tal/submission_movies_twitter_roberta.csv"
submission_twitter_roberta.to_csv(output_path, index=False, header=False)

print(f"Fichier enregistré : {output_path}")