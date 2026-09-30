# RITAL sentiment-classification experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
#
# # Movie 7 — Head + Tail avec Twitter-RoBERTa + Ensemble SVM
#
# Ce notebook teste une amélioration ciblée du meilleur pipeline précédent :
#
# - **SVM** : meilleur modèle classique (`LinearSVC`, `C=5`)
# - **Twitter-RoBERTa** : mêmes meilleurs hyperparamètres qu'avant
# - **Nouvelle idée** : au lieu de tronquer uniquement le début du review, on prend :
#   - les **256 premiers tokens**
#   - les **256 derniers tokens**
# - On compare :
#   1. **RoBERTa standard**
#   2. **RoBERTa head+tail**
#   3. **Ensemble SVM + RoBERTa standard**
#   4. **Ensemble SVM + RoBERTa head+tail**
#
# Objectif : voir si conserver une partie de la fin du review améliore la prédiction.

# %%

# Si besoin sur Colab, décommente :
# NOTEBOOK: !pip install -q transformers datasets accelerate scikit-learn

# %%

from pathlib import Path
import os
import random
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)

# %%
# NOTEBOOK-ONLY COMMAND REMOVED: from google.colab import drive
# NOTEBOOK-ONLY COMMAND REMOVED: drive.mount('/content/drive')

# %% [markdown]
# ## Configuration

# %%

# ===== Chemins =====
DATA_DIR = Path("/content/drive/MyDrive/projet tal/movies1000/movies1000")
TEST_FILE = Path("/content/drive/MyDrive/projet tal/testSentiment.txt")
OUTPUT_DIR = Path("/content/drive/MyDrive/projet tal/movie7_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== Labels =====
label2id = {"N": 0, "P": 1}
id2label = {0: "N", 1: "P"}

# ===== Meilleur SVM observé =====
SVM_NGRAM_RANGE = (1, 2)
SVM_MIN_DF = 3
SVM_MAX_DF = 0.95
SVM_SUBLINEAR_TF = True
SVM_C = 5.0

# ===== Meilleure config RoBERTa observée =====
ROBERTA_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ROBERTA_RUN_NAME = "twitter_roberta_cardiffnlp_head_tail"
ROBERTA_NUM_EPOCHS = 3
ROBERTA_LEARNING_RATE = 2e-5
ROBERTA_BATCH_TRAIN = 8
ROBERTA_BATCH_EVAL = 16
ROBERTA_WEIGHT_DECAY = 0.01
ROBERTA_MAX_LENGTH = 512

# ===== Head + Tail =====
HEAD_TOKENS = 256
TAIL_TOKENS = 256

# ===== Ensemble =====
THRESHOLD_GRID = np.round(np.arange(0.50, 1.001, 0.01), 2)

# ===== Options =====
MAKE_TEST_SUBMISSIONS = True
SAVE_TRAINED_MODELS = True

print("DATA_DIR existe :", DATA_DIR.exists())
print("TEST_FILE existe :", TEST_FILE.exists())

# %% [markdown]
# ## Chargement des données

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

def load_test_file(test_file: Path) -> pd.DataFrame | None:
    if not test_file.exists():
        return None
    lines = test_file.read_text(encoding="utf-8", errors="ignore").split("\n")
    rows = [{"text": line.strip()} for line in lines if line.strip() != ""]
    return pd.DataFrame(rows)

df = load_movies_from_folder(DATA_DIR)
test_df = load_test_file(TEST_FILE)

print("Train total :", df.shape)
if test_df is not None:
    print("Test :", test_df.shape)

display(df.head())

# %% [markdown]
# ## Split validation

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
# ## Métriques et utilitaires

# %%

def softmax_np(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def compute_binary_metrics(y_true, y_pred, pos_label="P"):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=pos_label),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label),
        "f1": f1_score(y_true, y_pred, pos_label=pos_label),
    }

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

def cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

# %% [markdown]
# ## SVM

# %%

def train_svm(train_texts, train_labels):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=SVM_NGRAM_RANGE,
            min_df=SVM_MIN_DF,
            max_df=SVM_MAX_DF,
            sublinear_tf=SVM_SUBLINEAR_TF,
        )),
        ("clf", LinearSVC(C=SVM_C))
    ])
    model.fit(train_texts, train_labels)
    return model

svm_model = train_svm(train_df["text"], train_df["label"])
valid_pred_svm = svm_model.predict(valid_df["text"])

print_report(valid_df["label"], valid_pred_svm, "LinearSVC (validation)")

# %% [markdown]
# ## Jeux Hugging Face pour RoBERTa

# %%

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
# ## Tokenization standard vs head+tail

# %%

def encode_head_tail_text(text, tokenizer, max_length=512, head_tokens=256, tail_tokens=256):
    # Tokenisation brute sans tokens spéciaux
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False
    )

    # Cas simple : le texte tient déjà dans la limite standard
    # pour une séquence simple, on réserve 2 tokens spéciaux : <s> ... </s>
    if len(token_ids) <= max_length - 2:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "is_truncated": 0,
        }

    # Cas head+tail :
    # format RoBERTa pair = <s> head </s></s> tail </s>
    # donc 4 tokens spéciaux au total
    available = max_length - 4

    # Répartition head / tail
    head_len = min(head_tokens, available // 2)
    tail_len = min(tail_tokens, available - head_len)

    # Ajustement de sécurité
    if head_len + tail_len > available:
        tail_len = available - head_len

    head_ids = token_ids[:head_len]
    tail_ids = token_ids[-tail_len:]

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    input_ids = [bos_id] + head_ids + [eos_id, eos_id] + tail_ids + [eos_id]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "is_truncated": 1,
    }

def tokenize_dataset(dataset, tokenizer, mode="standard", max_length=512, head_tokens=256, tail_tokens=256):
    if mode == "standard":
        def tokenize_fn(batch):
            encoded = tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            encoded["is_truncated"] = [
                int(len(tokenizer.encode(t, add_special_tokens=False, truncation=False)) > (max_length - 2))
                for t in batch["text"]
            ]
            return encoded
    elif mode == "head_tail":
        def tokenize_fn(batch):
            outputs = {"input_ids": [], "attention_mask": [], "is_truncated": []}
            for text in batch["text"]:
                enc = encode_head_tail_text(
                    text=text,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    head_tokens=head_tokens,
                    tail_tokens=tail_tokens,
                )
                outputs["input_ids"].append(enc["input_ids"])
                outputs["attention_mask"].append(enc["attention_mask"])
                outputs["is_truncated"].append(enc["is_truncated"])
            return outputs
    else:
        raise ValueError("mode doit être 'standard' ou 'head_tail'.")

    tokenized = dataset.map(tokenize_fn, batched=True)
    cols_to_remove = [c for c in tokenized.column_names if c in ["text", "__index_level_0__"]]
    if cols_to_remove:
        tokenized = tokenized.remove_columns(cols_to_remove)
    return tokenized

# %% [markdown]
# ## Entraînement RoBERTa standard / head+tail

# %%

def run_roberta_experiment(
    model_name: str,
    run_name: str,
    hf_train,
    hf_valid,
    valid_df,
    mode: str = "standard",
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size_train: int = 8,
    batch_size_eval: int = 16,
    max_length: int = 512,
    head_tokens: int = 256,
    tail_tokens: int = 256,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_train = tokenize_dataset(
        hf_train, tokenizer, mode=mode,
        max_length=max_length, head_tokens=head_tokens, tail_tokens=tail_tokens
    )
    tokenized_valid = tokenize_dataset(
        hf_valid, tokenizer, mode=mode,
        max_length=max_length, head_tokens=head_tokens, tail_tokens=tail_tokens
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    output_dir = str(OUTPUT_DIR / f"{run_name}_{mode}_output")

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size_train,
        per_device_eval_batch_size=batch_size_eval,
        num_train_epochs=num_epochs,
        weight_decay=ROBERTA_WEIGHT_DECAY,
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
    logits = pred_output.predictions
    probs = softmax_np(logits)
    pred_ids = np.argmax(logits, axis=1)
    pred_labels = np.array([id2label[int(x)] for x in pred_ids])

    trunc_rate = float(np.mean(tokenized_valid["is_truncated"])) if "is_truncated" in tokenized_valid.column_names else np.nan

    print_report(valid_df["label"], pred_labels, f"{run_name} [{mode}]")

    return {
        "tokenizer": tokenizer,
        "trainer": trainer,
        "train_output": train_output,
        "eval_output": eval_output,
        "pred_labels": pred_labels,
        "pred_probs": probs,
        "tokenized_valid": tokenized_valid,
        "trunc_rate_valid": trunc_rate,
        "mode": mode,
    }

roberta_standard_results = run_roberta_experiment(
    model_name=ROBERTA_MODEL_NAME,
    run_name=ROBERTA_RUN_NAME,
    hf_train=hf_train,
    hf_valid=hf_valid,
    valid_df=valid_df,
    mode="standard",
    num_epochs=ROBERTA_NUM_EPOCHS,
    learning_rate=ROBERTA_LEARNING_RATE,
    batch_size_train=ROBERTA_BATCH_TRAIN,
    batch_size_eval=ROBERTA_BATCH_EVAL,
    max_length=ROBERTA_MAX_LENGTH,
    head_tokens=HEAD_TOKENS,
    tail_tokens=TAIL_TOKENS,
)

cleanup()

roberta_headtail_results = run_roberta_experiment(
    model_name=ROBERTA_MODEL_NAME,
    run_name=ROBERTA_RUN_NAME,
    hf_train=hf_train,
    hf_valid=hf_valid,
    valid_df=valid_df,
    mode="head_tail",
    num_epochs=ROBERTA_NUM_EPOCHS,
    learning_rate=ROBERTA_LEARNING_RATE,
    batch_size_train=ROBERTA_BATCH_TRAIN,
    batch_size_eval=ROBERTA_BATCH_EVAL,
    max_length=ROBERTA_MAX_LENGTH,
    head_tokens=HEAD_TOKENS,
    tail_tokens=TAIL_TOKENS,
)

# %% [markdown]
# ## Comparaison RoBERTa standard vs head+tail

# %%

valid_pred_roberta_standard = roberta_standard_results["pred_labels"]
valid_pred_roberta_headtail = roberta_headtail_results["pred_labels"]

roberta_comparison_df = pd.DataFrame([
    {
        "variant": "RoBERTa standard",
        **compute_binary_metrics(valid_df["label"], valid_pred_roberta_standard),
        "trunc_rate_valid": roberta_standard_results["trunc_rate_valid"],
    },
    {
        "variant": "RoBERTa head_tail",
        **compute_binary_metrics(valid_df["label"], valid_pred_roberta_headtail),
        "trunc_rate_valid": roberta_headtail_results["trunc_rate_valid"],
    },
]).sort_values(["f1", "accuracy"], ascending=False).reset_index(drop=True)

display(roberta_comparison_df)

# %% [markdown]
# ## Recherche du meilleur seuil d'ensemble

# %%

def search_best_threshold(y_true, pred_svm, pred_roberta, probs_roberta, tag="ensemble"):
    threshold_rows = []
    best = None

    conf = probs_roberta.max(axis=1)

    for thr in THRESHOLD_GRID:
        pred_ensemble = np.where(conf >= thr, pred_roberta, pred_svm)
        metrics = compute_binary_metrics(y_true, pred_ensemble)

        row = {
            "tag": tag,
            "threshold": float(thr),
            **metrics
        }
        threshold_rows.append(row)

        if best is None or metrics["f1"] > best["f1"]:
            best = {
                "tag": tag,
                "threshold": float(thr),
                "pred_valid": pred_ensemble,
                **metrics
            }

    table = pd.DataFrame(threshold_rows).sort_values(["f1", "accuracy"], ascending=False).reset_index(drop=True)
    return best, table

best_ens_standard, ens_standard_table = search_best_threshold(
    y_true=valid_df["label"].values,
    pred_svm=valid_pred_svm,
    pred_roberta=valid_pred_roberta_standard,
    probs_roberta=roberta_standard_results["pred_probs"],
    tag="ensemble_standard",
)

best_ens_headtail, ens_headtail_table = search_best_threshold(
    y_true=valid_df["label"].values,
    pred_svm=valid_pred_svm,
    pred_roberta=valid_pred_roberta_headtail,
    probs_roberta=roberta_headtail_results["pred_probs"],
    tag="ensemble_head_tail",
)

display(ens_standard_table.head(5))
display(ens_headtail_table.head(5))

print("Meilleur seuil ensemble standard :", best_ens_standard["threshold"])
print("F1 ensemble standard :", best_ens_standard["f1"])
print()
print("Meilleur seuil ensemble head+tail :", best_ens_headtail["threshold"])
print("F1 ensemble head+tail :", best_ens_headtail["f1"])

# %% [markdown]
# ## Tableau final de comparaison

# %%

final_comparison_df = pd.DataFrame([
    {
        "model": "SVM",
        **compute_binary_metrics(valid_df["label"], valid_pred_svm),
    },
    {
        "model": "RoBERTa standard",
        **compute_binary_metrics(valid_df["label"], valid_pred_roberta_standard),
    },
    {
        "model": "RoBERTa head_tail",
        **compute_binary_metrics(valid_df["label"], valid_pred_roberta_headtail),
    },
    {
        "model": f"Ensemble standard (thr={best_ens_standard['threshold']:.2f})",
        "accuracy": best_ens_standard["accuracy"],
        "precision": best_ens_standard["precision"],
        "recall": best_ens_standard["recall"],
        "f1": best_ens_standard["f1"],
    },
    {
        "model": f"Ensemble head_tail (thr={best_ens_headtail['threshold']:.2f})",
        "accuracy": best_ens_headtail["accuracy"],
        "precision": best_ens_headtail["precision"],
        "recall": best_ens_headtail["recall"],
        "f1": best_ens_headtail["f1"],
    },
]).sort_values(["f1", "accuracy"], ascending=False).reset_index(drop=True)

display(final_comparison_df)

best_row = final_comparison_df.iloc[0]
print("=" * 80)
print("MEILLEUR MODELE SUR VALIDATION")
print("=" * 80)
print("Modèle :", best_row["model"])
print(f"Accuracy : {best_row['accuracy']:.4f}")
print(f"Precision: {best_row['precision']:.4f}")
print(f"Recall   : {best_row['recall']:.4f}")
print(f"F1       : {best_row['f1']:.4f}")

# %% [markdown]
# ## Sauvegarde optionnelle du meilleur modèle validation

# %%

if SAVE_TRAINED_MODELS:
    if best_row["model"].startswith("RoBERTa head_tail") or best_row["model"].startswith("Ensemble head_tail"):
        best_variant = roberta_headtail_results
        best_variant_name = "best_roberta_head_tail_validation_model"
    else:
        best_variant = roberta_standard_results
        best_variant_name = "best_roberta_standard_validation_model"

    save_dir = OUTPUT_DIR / best_variant_name
    best_variant["trainer"].save_model(str(save_dir))
    best_variant["tokenizer"].save_pretrained(str(save_dir))
    print("Modèle et tokenizer sauvegardés dans :", save_dir)

# %% [markdown]
# ## Réentraînement complet pour les soumissions test

# %%

def train_full_svm(df_full: pd.DataFrame):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=SVM_NGRAM_RANGE,
            min_df=SVM_MIN_DF,
            max_df=SVM_MAX_DF,
            sublinear_tf=SVM_SUBLINEAR_TF,
        )),
        ("clf", LinearSVC(C=SVM_C))
    ])
    model.fit(df_full["text"], df_full["label"])
    return model

def train_full_roberta(df_full: pd.DataFrame, tokenizer, mode="standard"):
    df_roberta = df_full[["text", "label"]].copy()
    df_roberta["label_id"] = df_roberta["label"].map(label2id).astype(int)

    hf_full = Dataset.from_pandas(
        df_roberta[["text", "label_id"]].rename(columns={"label_id": "label"})
    )

    tokenized_full = tokenize_dataset(
        hf_full, tokenizer, mode=mode,
        max_length=ROBERTA_MAX_LENGTH, head_tokens=HEAD_TOKENS, tail_tokens=TAIL_TOKENS
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_MODEL_NAME,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / f"{ROBERTA_RUN_NAME}_{mode}_full_output"),
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="epoch",
        learning_rate=ROBERTA_LEARNING_RATE,
        per_device_train_batch_size=ROBERTA_BATCH_TRAIN,
        per_device_eval_batch_size=ROBERTA_BATCH_EVAL,
        num_train_epochs=ROBERTA_NUM_EPOCHS,
        weight_decay=ROBERTA_WEIGHT_DECAY,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_full,
        data_collator=data_collator,
    )

    trainer.train()
    return trainer

def predict_roberta_on_test(trainer, tokenizer, test_df, mode="standard"):
    hf_test = Dataset.from_pandas(test_df[["text"]].copy())
    tokenized_test = tokenize_dataset(
        hf_test, tokenizer, mode=mode,
        max_length=ROBERTA_MAX_LENGTH, head_tokens=HEAD_TOKENS, tail_tokens=TAIL_TOKENS
    )

    output = trainer.predict(tokenized_test)
    logits = output.predictions
    probs = softmax_np(logits)
    pred_ids = np.argmax(logits, axis=1)
    pred_labels = np.array([id2label[int(x)] for x in pred_ids])

    return pred_labels, probs

def save_submission(labels, path: Path):
    sub = pd.DataFrame({"label": labels})
    sub.to_csv(path, index=False)
    print("Soumission enregistrée :", path)
    print(sub["label"].value_counts())
    return sub

# %% [markdown]
# ## Génération des soumissions test

# %%

if MAKE_TEST_SUBMISSIONS and (test_df is not None):
    # 1) Full SVM
    full_svm = train_full_svm(df)
    test_pred_svm = full_svm.predict(test_df["text"])

    # 2) Choix de la meilleure variante RoBERTa selon la validation
    if best_row["model"].startswith("RoBERTa head_tail") or best_row["model"].startswith("Ensemble head_tail"):
        best_mode = "head_tail"
        best_threshold = best_ens_headtail["threshold"]
    else:
        best_mode = "standard"
        best_threshold = best_ens_standard["threshold"]

    full_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    full_roberta_trainer = train_full_roberta(df, full_tokenizer, mode=best_mode)
    test_pred_roberta, test_probs_roberta = predict_roberta_on_test(
        full_roberta_trainer,
        full_tokenizer,
        test_df,
        mode=best_mode
    )

    # 3) Ensemble avec meilleur seuil trouvé sur validation
    test_roberta_conf = test_probs_roberta.max(axis=1)
    test_pred_ensemble = np.where(
        test_roberta_conf >= best_threshold,
        test_pred_roberta,
        test_pred_svm,
    )

    sub_svm = save_submission(test_pred_svm, OUTPUT_DIR / "submission_movie7_svm.csv")
    sub_roberta = save_submission(test_pred_roberta, OUTPUT_DIR / f"submission_movie7_roberta_{best_mode}.csv")
    sub_ensemble = save_submission(test_pred_ensemble, OUTPUT_DIR / f"submission_movie7_ensemble_{best_mode}.csv")

    display(sub_svm.head())
    display(sub_roberta.head())
    display(sub_ensemble.head())
else:
    print("Soumissions non générées : vérifie MAKE_TEST_SUBMISSIONS et TEST_FILE.")

# %% [markdown]
# ## Commentaire final

# %%

print("Interprétation :")
if roberta_comparison_df.iloc[0]["variant"] == "RoBERTa head_tail":
    print("- La stratégie head+tail améliore RoBERTa par rapport à la troncature standard.")
else:
    print("- La stratégie head+tail n'améliore pas RoBERTa sur ce split, la version standard reste meilleure.")

if final_comparison_df.iloc[0]["model"].startswith("Ensemble"):
    print("- Le meilleur système final reste un ensemble SVM + RoBERTa.")
else:
    print("- Le meilleur système final est un modèle seul.")
