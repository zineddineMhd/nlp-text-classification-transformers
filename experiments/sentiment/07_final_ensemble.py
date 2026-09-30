# RITAL sentiment-classification experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # Movie 10 — Comparaison complète SVM / RoBERTa / DeBERTa + ensembles + soumissions test
#
# Ce notebook entraîne et compare :
#
# ## Modèles individuels
# 1. **SVM**
# 2. **RoBERTa standard**
# 3. **RoBERTa head+tail 256/256**
# 4. **RoBERTa head+tail 384/128**
# 5. **DeBERTa-v3-base standard**
#
# Tous les modèles Transformer utilisent la meilleure configuration trouvée :
#
# - `epochs = 3`
# - `learning_rate = 2e-5`
# - `warmup_ratio = 0.06`
# - `weight_decay = 0.01`
#
# ## Ensembles testés
# - **Ensemble A** : `SVM + meilleur RoBERTa global + DeBERTa`
# - **Ensemble B** : `SVM + RoBERTa standard + meilleur RoBERTa head+tail + DeBERTa`
# - **Ensemble C** : `SVM + tous les RoBERTa + DeBERTa`
#
# Pour chaque ensemble :
# - on calcule un **soft vote pondéré**
# - on cherche le meilleur **seuil de décision**
# - on compare les performances sur la validation
#
# ## Sortie finale
# Le notebook génère les **soumissions test pour tous les modèles et tous les ensembles**, avec des noms explicites.

# %%
# NOTEBOOK-ONLY COMMAND REMOVED: from google.colab import drive
# NOTEBOOK-ONLY COMMAND REMOVED: drive.mount('/content/drive')

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

# %% [markdown]
# ## Configuration

# %%
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)

DATA_DIR = Path("/content/drive/MyDrive/projet tal/movies1000/movies1000")
TEST_FILE = Path("/content/drive/MyDrive/projet tal/testSentiment.txt")
OUTPUT_DIR = Path("/content/drive/MyDrive/projet tal/movie10_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

label2id = {"N": 0, "P": 1}
id2label = {0: "N", 1: "P"}

# SVM best params
SVM_NGRAM_RANGE = (1, 2)
SVM_MIN_DF = 3
SVM_MAX_DF = 0.95
SVM_SUBLINEAR_TF = True
SVM_C = 5.0

# Transformer best config
TRANSFORMER_EPOCHS = 3
TRANSFORMER_LR = 2e-5
TRANSFORMER_WEIGHT_DECAY = 0.01
TRANSFORMER_WARMUP_RATIO = 0.06
TRANSFORMER_BATCH_TRAIN = 8
TRANSFORMER_BATCH_EVAL = 16
TRANSFORMER_MAX_LENGTH = 512

# Model names
ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEBERTA_MODEL = "microsoft/deberta-v3-base"

# Head-tail configs
HEAD_256 = 256
TAIL_256 = 256
HEAD_384 = 384
TAIL_128 = 128

# Ensemble search
THRESHOLD_GRID = np.round(np.arange(0.30, 0.701, 0.01), 2)

MAKE_TEST_SUBMISSIONS = True
SAVE_TRAINED_MODELS = True

print("DATA_DIR existe :", DATA_DIR.exists())
print("TEST_FILE existe :", TEST_FILE.exists())
print("OUTPUT_DIR :", OUTPUT_DIR)

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

def load_test_file(test_file: Path):
    if not test_file.exists():
        return None
    lines = test_file.read_text(encoding="utf-8", errors="ignore").split("\n")
    rows = [{"text": line.strip()} for line in lines if line.strip() != ""]
    return pd.DataFrame(rows)

df = load_movies_from_folder(DATA_DIR)
test_df = load_test_file(TEST_FILE)

print("Corpus total :", df.shape)
if test_df is not None:
    print("Corpus test :", test_df.shape)

display(df.head())

# %% [markdown]
# ## Split train / validation

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
# ## Utilitaires

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
# ## Jeux Hugging Face

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
valid_score_svm_raw = svm_model.decision_function(valid_df["text"])

print_report(valid_df["label"], valid_pred_svm, "LinearSVC (validation)")

# %% [markdown]
# ## Tokenization standard et head+tail

# %%
def encode_head_tail_text(text, tokenizer, max_length=512, head_tokens=256, tail_tokens=256):
    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)

    if len(token_ids) <= max_length - 2:
        encoded = tokenizer(text, truncation=True, max_length=max_length, padding=False)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "is_truncated": 0,
        }

    available = max_length - 4
    head_len = min(head_tokens, available)
    tail_len = min(tail_tokens, max(0, available - head_len))

    if head_len + tail_len > len(token_ids):
        tail_len = max(0, len(token_ids) - head_len)

    head_ids = token_ids[:head_len]
    tail_ids = token_ids[-tail_len:] if tail_len > 0 else []

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    if tail_len > 0:
        input_ids = [bos_id] + head_ids + [eos_id, eos_id] + tail_ids + [eos_id]
    else:
        input_ids = [bos_id] + head_ids + [eos_id]

    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "is_truncated": 1}

def tokenize_dataset(dataset, tokenizer, mode="standard", max_length=512, head_tokens=256, tail_tokens=256):
    if mode == "standard":
        def tokenize_fn(batch):
            encoded = tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)
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
# ## Entraînement Transformer générique

# %%
def run_transformer(
    model_name: str,
    run_name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    batch_train: int = 8,
    batch_eval: int = 16,
    max_length: int = 512,
    mode: str = "standard",
    head_tokens: int = 256,
    tail_tokens: int = 256,
    save_model: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    local_train_df = train_df[["text", "label"]].copy()
    local_valid_df = valid_df[["text", "label"]].copy()

    local_train_df["label_id"] = local_train_df["label"].map(label2id)
    local_valid_df["label_id"] = local_valid_df["label"].map(label2id)

    hf_train_local = Dataset.from_pandas(
        local_train_df[["text", "label_id"]].rename(columns={"label_id": "label"})
    )
    hf_valid_local = Dataset.from_pandas(
        local_valid_df[["text", "label_id"]].rename(columns={"label_id": "label"})
    )

    tokenized_train = tokenize_dataset(
        hf_train_local, tokenizer, mode=mode,
        max_length=max_length, head_tokens=head_tokens, tail_tokens=tail_tokens
    )
    tokenized_valid = tokenize_dataset(
        hf_valid_local, tokenizer, mode=mode,
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

    output_dir = str(OUTPUT_DIR / f"{run_name}_output")

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_train,
        per_device_eval_batch_size=batch_eval,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=1,
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_output = trainer.evaluate()

    pred_output = trainer.predict(tokenized_valid)
    logits = pred_output.predictions
    probs = softmax_np(logits)
    pred_ids = np.argmax(logits, axis=1)
    pred_labels = np.array([id2label[int(x)] for x in pred_ids])

    trunc_rate = float(np.mean(tokenized_valid["is_truncated"])) if "is_truncated" in tokenized_valid.column_names else np.nan
    print_report(valid_df["label"], pred_labels, run_name)

    if save_model:
        save_dir = OUTPUT_DIR / f"{run_name}_saved"
        trainer.save_model(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        print("Modèle sauvegardé :", save_dir)

    return {
        "model_name": model_name,
        "run_name": run_name,
        "tokenizer": tokenizer,
        "trainer": trainer,
        "eval_output": eval_output,
        "pred_labels": pred_labels,
        "pred_probs": probs,
        "pred_pos_proba": probs[:, 1],
        "trunc_rate_valid": trunc_rate,
        "mode": mode,
        "head_tokens": head_tokens,
        "tail_tokens": tail_tokens,
    }

# %% [markdown]
# ## Entraîner les quatre Transformers

# %%
RESULTS = {}

roberta_standard_result = run_transformer(
    model_name=ROBERTA_MODEL,
    run_name="roberta_standard_bestcfg",
    train_df=train_df,
    valid_df=valid_df,
    num_epochs=TRANSFORMER_EPOCHS,
    learning_rate=TRANSFORMER_LR,
    weight_decay=TRANSFORMER_WEIGHT_DECAY,
    warmup_ratio=TRANSFORMER_WARMUP_RATIO,
    batch_train=TRANSFORMER_BATCH_TRAIN,
    batch_eval=TRANSFORMER_BATCH_EVAL,
    max_length=TRANSFORMER_MAX_LENGTH,
    mode="standard",
    save_model=True,
)
RESULTS["RoBERTa_standard"] = roberta_standard_result

cleanup()

roberta_256_256_result = run_transformer(
    model_name=ROBERTA_MODEL,
    run_name="roberta_headtail_256_256_bestcfg",
    train_df=train_df,
    valid_df=valid_df,
    num_epochs=TRANSFORMER_EPOCHS,
    learning_rate=TRANSFORMER_LR,
    weight_decay=TRANSFORMER_WEIGHT_DECAY,
    warmup_ratio=TRANSFORMER_WARMUP_RATIO,
    batch_train=TRANSFORMER_BATCH_TRAIN,
    batch_eval=TRANSFORMER_BATCH_EVAL,
    max_length=TRANSFORMER_MAX_LENGTH,
    mode="head_tail",
    head_tokens=HEAD_256,
    tail_tokens=TAIL_256,
    save_model=True,
)
RESULTS["RoBERTa_256_256"] = roberta_256_256_result

cleanup()

roberta_384_128_result = run_transformer(
    model_name=ROBERTA_MODEL,
    run_name="roberta_headtail_384_128_bestcfg",
    train_df=train_df,
    valid_df=valid_df,
    num_epochs=TRANSFORMER_EPOCHS,
    learning_rate=TRANSFORMER_LR,
    weight_decay=TRANSFORMER_WEIGHT_DECAY,
    warmup_ratio=TRANSFORMER_WARMUP_RATIO,
    batch_train=TRANSFORMER_BATCH_TRAIN,
    batch_eval=TRANSFORMER_BATCH_EVAL,
    max_length=TRANSFORMER_MAX_LENGTH,
    mode="head_tail",
    head_tokens=HEAD_384,
    tail_tokens=TAIL_128,
    save_model=True,
)
RESULTS["RoBERTa_384_128"] = roberta_384_128_result

cleanup()

deberta_result = run_transformer(
    model_name=DEBERTA_MODEL,
    run_name="deberta_v3_base_bestcfg",
    train_df=train_df,
    valid_df=valid_df,
    num_epochs=TRANSFORMER_EPOCHS,
    learning_rate=TRANSFORMER_LR,
    weight_decay=TRANSFORMER_WEIGHT_DECAY,
    warmup_ratio=TRANSFORMER_WARMUP_RATIO,
    batch_train=TRANSFORMER_BATCH_TRAIN,
    batch_eval=TRANSFORMER_BATCH_EVAL,
    max_length=TRANSFORMER_MAX_LENGTH,
    mode="standard",
    save_model=True,
)
RESULTS["DeBERTa_v3_base"] = deberta_result

# %% [markdown]
# ## Tableau des modèles individuels

# %%
individual_results_df = pd.DataFrame([
    {
        "model": "SVM",
        **compute_binary_metrics(valid_df["label"], valid_pred_svm),
    },
    *[
        {
            "model": name,
            **compute_binary_metrics(valid_df["label"], res["pred_labels"]),
            "trunc_rate_valid": res["trunc_rate_valid"],
        }
        for name, res in RESULTS.items()
    ]
]).sort_values(["f1", "accuracy"], ascending=False).reset_index(drop=True)

display(individual_results_df)

# %% [markdown]
# ## Identifier le meilleur RoBERTa global et le meilleur head+tail

# %%
roberta_names = ["RoBERTa_standard", "RoBERTa_256_256", "RoBERTa_384_128"]
headtail_names = ["RoBERTa_256_256", "RoBERTa_384_128"]

best_roberta_name = max(
    roberta_names,
    key=lambda n: compute_binary_metrics(valid_df["label"], RESULTS[n]["pred_labels"])["f1"]
)
best_headtail_name = max(
    headtail_names,
    key=lambda n: compute_binary_metrics(valid_df["label"], RESULTS[n]["pred_labels"])["f1"]
)

print("Meilleur RoBERTa global :", best_roberta_name)
print("Meilleur RoBERTa head+tail :", best_headtail_name)

# %% [markdown]
# ## Fonctions pour ensembles pondérés

# %%
def minmax_scale_from_validation(values):
    v = np.asarray(values, dtype=float)
    vmin, vmax = v.min(), v.max()
    if vmax - vmin < 1e-12:
        return np.full_like(v, 0.5, dtype=float), vmin, vmax
    return (v - vmin) / (vmax - vmin), vmin, vmax

def minmax_apply(values, vmin, vmax):
    v = np.asarray(values, dtype=float)
    if vmax - vmin < 1e-12:
        return np.full_like(v, 0.5, dtype=float)
    return np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)

svm_score_valid_scaled, svm_valid_min, svm_valid_max = minmax_scale_from_validation(valid_score_svm_raw)

def build_soft_vote_score(components):
    total_weight = sum(w for w, _ in components)
    score = np.zeros_like(components[0][1], dtype=float)
    for w, s in components:
        score += w * s
    return score / total_weight

def evaluate_soft_vote(y_true, score, threshold_grid, ensemble_name):
    rows = []
    best = None
    for thr in threshold_grid:
        pred = np.where(score >= thr, "P", "N")
        metrics = compute_binary_metrics(y_true, pred)
        row = {"ensemble": ensemble_name, "threshold": float(thr), **metrics}
        rows.append(row)
        if best is None or metrics["f1"] > best["f1"] or (
            metrics["f1"] == best["f1"] and metrics["accuracy"] > best["accuracy"]
        ):
            best = {
                "ensemble": ensemble_name,
                "threshold": float(thr),
                "pred_valid": pred,
                **metrics
            }
    table = pd.DataFrame(rows).sort_values(["f1", "accuracy"], ascending=False).reset_index(drop=True)
    return best, table

# %% [markdown]
# ## Construire et évaluer les ensembles

# %%
# ============================================================
# Recherche plus large de combinaisons d'ensembles SANS DeBERTa
# ============================================================

from itertools import product

svm_pos_valid = svm_score_valid_scaled
roberta_std_valid = RESULTS["RoBERTa_standard"]["pred_pos_proba"]
roberta_256_valid = RESULTS["RoBERTa_256_256"]["pred_pos_proba"]
roberta_384_valid = RESULTS["RoBERTa_384_128"]["pred_pos_proba"]

model_scores = {
    "SVM": svm_pos_valid,
    "RoBERTa_standard": roberta_std_valid,
    "RoBERTa_256_256": roberta_256_valid,
    "RoBERTa_384_128": roberta_384_valid,
}

# Poids à tester
WEIGHT_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

def evaluate_many_ensembles(y_true, model_scores, threshold_grid, weight_grid):
    all_rows = []
    best_result = None

    # ------------------------------------------------
    # 1) Ensembles à 2 modèles
    # ------------------------------------------------
    pair_configs = [
        ("SVM", "RoBERTa_standard"),
        ("SVM", "RoBERTa_256_256"),
        ("SVM", "RoBERTa_384_128"),
        ("RoBERTa_standard", "RoBERTa_256_256"),
        ("RoBERTa_standard", "RoBERTa_384_128"),
        ("RoBERTa_256_256", "RoBERTa_384_128"),
    ]

    for m1, m2 in pair_configs:
        for w1, w2 in product(weight_grid, repeat=2):
            # éviter les doublons inutiles
            if w1 == 0 and w2 == 0:
                continue

            ensemble_name = f"{m1}({w1}) + {m2}({w2})"
            score = build_soft_vote_score([
                (w1, model_scores[m1]),
                (w2, model_scores[m2]),
            ])

            best_local, _ = evaluate_soft_vote(y_true, score, threshold_grid, ensemble_name)

            row = {
                "ensemble": ensemble_name,
                "threshold": best_local["threshold"],
                "accuracy": best_local["accuracy"],
                "precision": best_local["precision"],
                "recall": best_local["recall"],
                "f1": best_local["f1"],
            }
            all_rows.append(row)

            if (best_result is None) or (best_local["f1"] > best_result["f1"]) or (
                best_local["f1"] == best_result["f1"] and best_local["accuracy"] > best_result["accuracy"]
            ):
                best_result = {
                    "ensemble": ensemble_name,
                    "threshold": best_local["threshold"],
                    "accuracy": best_local["accuracy"],
                    "precision": best_local["precision"],
                    "recall": best_local["recall"],
                    "f1": best_local["f1"],
                    "score_valid": score,
                    "pred_valid": best_local["pred_valid"],
                    "models": [m1, m2],
                    "weights": [w1, w2],
                }

    # ------------------------------------------------
    # 2) Ensembles à 3 modèles
    # ------------------------------------------------
    triple_configs = [
        ("SVM", "RoBERTa_standard", "RoBERTa_256_256"),
        ("SVM", "RoBERTa_standard", "RoBERTa_384_128"),
        ("SVM", "RoBERTa_256_256", "RoBERTa_384_128"),
        ("RoBERTa_standard", "RoBERTa_256_256", "RoBERTa_384_128"),
        ("SVM", "RoBERTa_standard", best_headtail_name),
        ("SVM", best_roberta_name, best_headtail_name),
    ]

    # enlever doublons éventuels
    triple_configs = list(dict.fromkeys(triple_configs))

    for m1, m2, m3 in triple_configs:
        for w1, w2, w3 in product(weight_grid, repeat=3):
            if w1 == 0 and w2 == 0 and w3 == 0:
                continue

            ensemble_name = f"{m1}({w1}) + {m2}({w2}) + {m3}({w3})"
            score = build_soft_vote_score([
                (w1, model_scores[m1]),
                (w2, model_scores[m2]),
                (w3, model_scores[m3]),
            ])

            best_local, _ = evaluate_soft_vote(y_true, score, threshold_grid, ensemble_name)

            row = {
                "ensemble": ensemble_name,
                "threshold": best_local["threshold"],
                "accuracy": best_local["accuracy"],
                "precision": best_local["precision"],
                "recall": best_local["recall"],
                "f1": best_local["f1"],
            }
            all_rows.append(row)

            if (best_result is None) or (best_local["f1"] > best_result["f1"]) or (
                best_local["f1"] == best_result["f1"] and best_local["accuracy"] > best_result["accuracy"]
            ):
                best_result = {
                    "ensemble": ensemble_name,
                    "threshold": best_local["threshold"],
                    "accuracy": best_local["accuracy"],
                    "precision": best_local["precision"],
                    "recall": best_local["recall"],
                    "f1": best_local["f1"],
                    "score_valid": score,
                    "pred_valid": best_local["pred_valid"],
                    "models": [m1, m2, m3],
                    "weights": [w1, w2, w3],
                }

    all_results_df = pd.DataFrame(all_rows).sort_values(
        ["f1", "accuracy", "precision", "recall"],
        ascending=False
    ).reset_index(drop=True)

    return best_result, all_results_df

best_ensemble_search, ensemble_search_df = evaluate_many_ensembles(
    y_true=valid_df["label"].values,
    model_scores=model_scores,
    threshold_grid=THRESHOLD_GRID,
    weight_grid=WEIGHT_GRID,
)

display(ensemble_search_df.head(30))

print("=" * 80)
print("MEILLEUR ENSEMBLE TROUVÉ")
print("=" * 80)
print("Ensemble   :", best_ensemble_search["ensemble"])
print("Threshold  :", best_ensemble_search["threshold"])
print("Accuracy   :", best_ensemble_search["accuracy"])
print("Precision  :", best_ensemble_search["precision"])
print("Recall     :", best_ensemble_search["recall"])
print("F1         :", best_ensemble_search["f1"])
print("Models     :", best_ensemble_search["models"])
print("Weights    :", best_ensemble_search["weights"])

# %%
# ============================================================
# UNE SEULE CELLULE :
# - entraîne SVM sur tout le train
# - entraîne RoBERTa 256/256 sur tout le train
# - entraîne RoBERTa 384/128 sur tout le train
# - prédit sur le vrai test
# - construit le meilleur ensemble
# - sauvegarde la soumission finale
# ============================================================

from pathlib import Path
import gc
import numpy as np
import pandas as pd

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

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = Path("/content/drive/MyDrive/projet tal/movies1000/movies1000")
TEST_FILE = Path("/content/drive/MyDrive/projet tal/testSentiment.txt")
OUTPUT_DIR = Path("/content/drive/MyDrive/projet tal/movie10_outputs/final_best_submission")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

label2id = {"N": 0, "P": 1}
id2label = {0: "N", 1: "P"}

SVM_NGRAM_RANGE = (1, 2)
SVM_MIN_DF = 3
SVM_MAX_DF = 0.95
SVM_SUBLINEAR_TF = True
SVM_C = 5.0

ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
TRANSFORMER_EPOCHS = 3
TRANSFORMER_LR = 2e-5
TRANSFORMER_WEIGHT_DECAY = 0.01
TRANSFORMER_WARMUP_RATIO = 0.06
TRANSFORMER_BATCH_TRAIN = 8
TRANSFORMER_BATCH_EVAL = 16
TRANSFORMER_MAX_LENGTH = 512

HEAD_256 = 256
TAIL_256 = 256
HEAD_384 = 384
TAIL_128 = 128

BEST_THRESHOLD = 0.52

# -----------------------------
# OUTILS
# -----------------------------
def cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

def softmax_np(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def build_soft_vote_score(components):
    total_weight = sum(w for w, _ in components)
    score = np.zeros_like(components[0][1], dtype=float)
    for w, s in components:
        score += w * s
    return score / total_weight

def load_movies_from_folder(data_dir: Path) -> pd.DataFrame:
    rows = []
    for label_folder, label in [("pos", "P"), ("neg", "N")]:
        folder = data_dir / label_folder
        for file_path in folder.glob("*.txt"):
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            rows.append({
                "doc_id": file_path.name,
                "label": label,
                "text": text,
            })
    return pd.DataFrame(rows).sort_values("doc_id").reset_index(drop=True)

def load_test_file(test_file: Path):
    lines = test_file.read_text(encoding="utf-8", errors="ignore").split("\n")
    rows = [{"text": line.strip()} for line in lines if line.strip() != ""]
    return pd.DataFrame(rows)

def encode_head_tail_text(text, tokenizer, max_length=512, head_tokens=256, tail_tokens=256):
    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)

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
        }

    available = max_length - 4
    head_len = min(head_tokens, available)
    tail_len = min(tail_tokens, max(0, available - head_len))

    if head_len + tail_len > len(token_ids):
        tail_len = max(0, len(token_ids) - head_len)

    head_ids = token_ids[:head_len]
    tail_ids = token_ids[-tail_len:] if tail_len > 0 else []

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    if tail_len > 0:
        input_ids = [bos_id] + head_ids + [eos_id, eos_id] + tail_ids + [eos_id]
    else:
        input_ids = [bos_id] + head_ids + [eos_id]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

def tokenize_dataset(dataset, tokenizer, mode="standard", max_length=512, head_tokens=256, tail_tokens=256):
    if mode == "standard":
        def tokenize_fn(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
            )
    elif mode == "head_tail":
        def tokenize_fn(batch):
            outputs = {"input_ids": [], "attention_mask": []}
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
            return outputs
    else:
        raise ValueError("mode doit être 'standard' ou 'head_tail'.")

    tokenized = dataset.map(tokenize_fn, batched=True)
    cols_to_remove = [c for c in tokenized.column_names if c in ["text", "__index_level_0__"]]
    if cols_to_remove:
        tokenized = tokenized.remove_columns(cols_to_remove)
    return tokenized

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

def scale_scores_01(x):
    x = np.asarray(x, dtype=float)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.full_like(x, 0.5)
    return (x - mn) / (mx - mn)

def train_full_transformer(df_full, model_name, run_name, mode="standard", head_tokens=256, tail_tokens=256):
    local_df = df_full[["text", "label"]].copy()
    local_df["label_id"] = local_df["label"].map(label2id)
    hf_full = Dataset.from_pandas(
        local_df[["text", "label_id"]].rename(columns={"label_id": "label"})
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_full = tokenize_dataset(
        hf_full,
        tokenizer,
        mode=mode,
        max_length=TRANSFORMER_MAX_LENGTH,
        head_tokens=head_tokens,
        tail_tokens=tail_tokens,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / f"{run_name}_output"),
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="epoch",
        learning_rate=TRANSFORMER_LR,
        per_device_train_batch_size=TRANSFORMER_BATCH_TRAIN,
        per_device_eval_batch_size=TRANSFORMER_BATCH_EVAL,
        num_train_epochs=TRANSFORMER_EPOCHS,
        weight_decay=TRANSFORMER_WEIGHT_DECAY,
        warmup_ratio=TRANSFORMER_WARMUP_RATIO,
        report_to="none",
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_full,
        data_collator=data_collator,
    )

    trainer.train()
    return trainer, tokenizer

def predict_transformer_on_test(trainer, tokenizer, test_df, mode="standard", head_tokens=256, tail_tokens=256):
    hf_test = Dataset.from_pandas(test_df[["text"]].copy())

    tokenized_test = tokenize_dataset(
        hf_test,
        tokenizer,
        mode=mode,
        max_length=TRANSFORMER_MAX_LENGTH,
        head_tokens=head_tokens,
        tail_tokens=tail_tokens,
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

# -----------------------------
# CHARGEMENT DONNÉES
# -----------------------------
df = load_movies_from_folder(DATA_DIR)
test_df = load_test_file(TEST_FILE)

print("Train total :", df.shape)
print("Test total  :", test_df.shape)

# -----------------------------
# 1) SVM FULL
# -----------------------------
print("\n=== Entraînement SVM ===")
full_svm = train_full_svm(df)
test_pred_svm = full_svm.predict(test_df["text"])
test_svm_raw = full_svm.decision_function(test_df["text"])
test_svm_pos = scale_scores_01(test_svm_raw)

# -----------------------------
# 2) RoBERTa 256/256 FULL
# -----------------------------
print("\n=== Entraînement RoBERTa 256/256 ===")
full_roberta_256_trainer, full_roberta_256_tok = train_full_transformer(
    df,
    ROBERTA_MODEL,
    "roberta_headtail_256_256_final",
    mode="head_tail",
    head_tokens=HEAD_256,
    tail_tokens=TAIL_256,
)

test_pred_roberta_256, test_probs_roberta_256 = predict_transformer_on_test(
    full_roberta_256_trainer,
    full_roberta_256_tok,
    test_df,
    mode="head_tail",
    head_tokens=HEAD_256,
    tail_tokens=TAIL_256,
)

cleanup()

# -----------------------------
# 3) RoBERTa 384/128 FULL
# -----------------------------
print("\n=== Entraînement RoBERTa 384/128 ===")
full_roberta_384_trainer, full_roberta_384_tok = train_full_transformer(
    df,
    ROBERTA_MODEL,
    "roberta_headtail_384_128_final",
    mode="head_tail",
    head_tokens=HEAD_384,
    tail_tokens=TAIL_128,
)

test_pred_roberta_384, test_probs_roberta_384 = predict_transformer_on_test(
    full_roberta_384_trainer,
    full_roberta_384_tok,
    test_df,
    mode="head_tail",
    head_tokens=HEAD_384,
    tail_tokens=TAIL_128,
)

cleanup()

# -----------------------------
# 4) MEILLEUR ENSEMBLE
# SVM(2.0) + RoBERTa_256_256(0.5) + RoBERTa_384_128(0.5)
# threshold = 0.52
# -----------------------------
print("\n=== Construction du meilleur ensemble ===")
best_ensemble_score_test = build_soft_vote_score([
    (2.0, test_svm_pos),
    (0.5, test_probs_roberta_256[:, 1]),
    (0.5, test_probs_roberta_384[:, 1]),
])

test_pred_best_ensemble = np.where(
    best_ensemble_score_test >= BEST_THRESHOLD,
    "P",
    "N",
)

# -----------------------------
# 5) SAUVEGARDE
# -----------------------------
sub_best = save_submission(
    test_pred_best_ensemble,
    OUTPUT_DIR / "submission_best_ensemble_movie10.csv"
)

# backups utiles
sub_roberta_256 = save_submission(
    test_pred_roberta_256,
    OUTPUT_DIR / "submission_roberta_256_256_movie10.csv"
)

sub_roberta_384 = save_submission(
    test_pred_roberta_384,
    OUTPUT_DIR / "submission_roberta_384_128_movie10.csv"
)

sub_svm = save_submission(
    test_pred_svm,
    OUTPUT_DIR / "submission_svm_movie10.csv"
)

print("\nFichiers sauvegardés dans :", OUTPUT_DIR)
display(sub_best.head())
display(sub_roberta_256.head())
display(sub_roberta_384.head())
display(sub_svm.head())

# %%
'''def load_test_file(test_file: Path):
    lines = test_file.read_text(encoding="utf-8", errors="ignore").split("\n")
    rows = [{"text": line.strip()} for line in lines if line.strip() != ""]
    return pd.DataFrame(rows)'''

test_df = load_test_file(TEST_FILE)

def predict_transformer_on_test(trainer, tokenizer, test_df, mode="standard", head_tokens=256, tail_tokens=256):
    hf_test = Dataset.from_pandas(test_df[["text"]].copy())

    tokenized_test = tokenize_dataset(
        hf_test,
        tokenizer,
        mode=mode,
        max_length=TRANSFORMER_MAX_LENGTH,
        head_tokens=head_tokens,
        tail_tokens=tail_tokens,
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

# %%
test_pred_roberta_384, test_probs_roberta_384 = predict_transformer_on_test(
    full_roberta_384_trainer,
    full_roberta_384_tok,
    test_df,
    mode="head_tail",
    head_tokens=HEAD_384,
    tail_tokens=TAIL_128,
)

cleanup()

# %%
test_pred_svm = full_svm.predict(test_df["text"])
test_svm_raw = full_svm.decision_function(test_df["text"])
test_svm_pos = scale_scores_01(test_svm_raw)

# %%
test_pred_roberta_256, test_probs_roberta_256 = predict_transformer_on_test(
    full_roberta_256_trainer,
    full_roberta_256_tok,
    test_df,
    mode="head_tail",
    head_tokens=HEAD_256,
    tail_tokens=TAIL_256,
)

cleanup()

# %%
best_ensemble_score_test = build_soft_vote_score([
    (2.0, test_svm_pos),
    (0.5, test_probs_roberta_256[:, 1]),
    (0.5, test_probs_roberta_384[:, 1]),
])

test_pred_best_ensemble = np.where(
    best_ensemble_score_test >= BEST_THRESHOLD,
    "P",
    "N",
)

# %%
sub_best = save_submission(
    test_pred_best_ensemble,
    OUTPUT_DIR / "submission_best_ensemble_movie10.csv"
)

sub_roberta_384 = save_submission(
    test_pred_roberta_384,
    OUTPUT_DIR / "submission_roberta_384_128_movie10.csv"
)

# %% [markdown]
# ## Réentraînement complet sur tout le corpus

# %% [markdown]
# ## Génération de toutes les soumissions test

# %% [markdown]
# ## Commentaire final

# %%
print("Le notebook a généré :")
print("- toutes les soumissions individuelles")
print("- toutes les soumissions d'ensemble")
print()
print("À tester sur la plateforme :")
print("1. meilleur ensemble selon validation")
print("2. meilleur RoBERTa individuel")
print("3. DeBERTa si son score validation est compétitif")