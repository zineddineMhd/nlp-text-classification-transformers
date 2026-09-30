# RITAL contextual authorship-classification experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # Tâche 1 — v14 corrigé
#
# Version corrigée de la meilleure idée de `pres_v14.ipynb` :
#
# - **split par document** avant toute création de chunks
# - **développement local** avec `train / val / test_local`
# - **sauvegarde du meilleur modèle** dans un répertoire paramétrable
# - **prédiction locale** sur `test_local`
# - **prédiction finale** sur le vrai fichier test si vous l’ajoutez
#
# Le but est d'éviter une validation trop optimiste due à un split par chunks.

# %%
# NOTEBOOK-ONLY COMMAND REMOVED: from google.colab import drive
# NOTEBOOK-ONLY COMMAND REMOVED: drive.mount('/content/drive')

# %% [markdown]
# ## 1. Installation

# %%
# NOTEBOOK: !pip install -q transformers datasets accelerate scikit-learn torch

# %% [markdown]
# ## 2. Configuration
# Remplacez simplement les chemins par les vôtres si besoin.

# %%

import os
from pathlib import Path

# === FICHIERS ===
TRAIN_FILE = "/content/drive/MyDrive/projet tal/corpus.tache1.learn.utf8"      # ou votre nom local
#TEST_FILE  = "corpus_tache1_test.utf8"       # optionnel pour la soumission finale

# === DOSSIERS DE SORTIE ===
# Remplacez par un chemin Drive/Colab si vous voulez persister le modèle.
MODEL_SAVE_DIR = "/content/drive/MyDrive/projet tal/rital_tache-doc/models/v14_corrige_best"
OUTPUT_DIR     = "/content/drive/MyDrive/projet tal/rital_tache-doc/runs/v14_corrige"
SUBMISSION_DIR = "/content/drive/MyDrive/projet tal/rital_tache-doc/submissions"

Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(SUBMISSION_DIR).mkdir(parents=True, exist_ok=True)

print("MODEL_SAVE_DIR =", MODEL_SAVE_DIR)
print("OUTPUT_DIR     =", OUTPUT_DIR)
print("SUBMISSION_DIR =", SUBMISSION_DIR)

# %% [markdown]
# ## 3. Imports

# %%

import os, re, codecs, gc, json, random
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from sklearn.metrics import recall_score, precision_score

from sklearn.metrics import (
    f1_score, average_precision_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset as HFDataset

os.environ["WANDB_DISABLED"] = "true"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# %% [markdown]
# ## 4. Fonctions utilitaires

# %%

def read_train_corpus(path):
    entries = []
    with codecs.open(path, "r", "utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"<(\d+):(\d+):([CM])>\s*(.*)", line)
            if m:
                entries.append({
                    "doc": int(m.group(1)),
                    "sent": int(m.group(2)),
                    "label_char": m.group(3),
                    "label": 1 if m.group(3) == "M" else 0,
                    "text": m.group(4),
                })
    return entries

def read_test_corpus(path):
    entries = []
    with codecs.open(path, "r", "utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"<(\d+):(\d+)>\s*(.*)", line)
            if m:
                entries.append({
                    "doc": int(m.group(1)),
                    "sent": int(m.group(2)),
                    "text": m.group(3),
                })
    return entries

def summarize_entries(entries, name="set"):
    n_docs = len({e["doc"] for e in entries})
    cnt = Counter(e["label_char"] for e in entries if "label_char" in e)
    print(f"[{name}] docs={n_docs:,} phrases={len(entries):,}")
    if cnt:
        print(f"  C={cnt.get('C',0):,} | M={cnt.get('M',0):,} | M%={100*cnt.get('M',0)/max(1,sum(cnt.values())):.2f}")

def build_doc_label_profile(entries):
    by_doc = defaultdict(set)
    for e in entries:
        by_doc[e["doc"]].add(e["label_char"])
    pure_c = sum(v == {"C"} for v in by_doc.values())
    pure_m = sum(v == {"M"} for v in by_doc.values())
    mixed  = sum(v == {"C","M"} for v in by_doc.values())
    print(f"Documents: purs C={pure_c}, purs M={pure_m}, mixtes={mixed}")

def build_chunks_from_entries(entries, max_words=350):
    chunks = []
    current_doc = None
    current_label = None
    current_texts = []
    current_words = 0

    for entry in entries:
        words = len(entry["text"].split())
        same_segment = (
            entry["doc"] == current_doc and
            entry["label"] == current_label and
            current_words + words <= max_words
        )
        if same_segment:
            current_texts.append(entry["text"])
            current_words += words
        else:
            if current_texts:
                chunks.append({
                    "text": " ".join(current_texts),
                    "label": current_label,
                    "n_sents": len(current_texts),
                    "doc": current_doc,
                })
            current_doc = entry["doc"]
            current_label = entry["label"]
            current_texts = [entry["text"]]
            current_words = words

    if current_texts:
        chunks.append({
            "text": " ".join(current_texts),
            "label": current_label,
            "n_sents": len(current_texts),
            "doc": current_doc,
        })
    return chunks

def build_context_windows(entries, max_context_words=350):
    by_doc = defaultdict(list)
    for i, entry in enumerate(entries):
        by_doc[entry["doc"]].append((i, entry["text"]))

    windows = []
    for doc_id in sorted(by_doc.keys()):
        sents = sorted(by_doc[doc_id], key=lambda x: x[0])
        for pos, (global_idx, text) in enumerate(sents):
            window = [text]
            word_count = len(text.split())
            left, right = pos - 1, pos + 1

            while word_count < max_context_words:
                added = False
                if left >= 0:
                    w = len(sents[left][1].split())
                    if word_count + w <= max_context_words:
                        window.insert(0, sents[left][1])
                        word_count += w
                        left -= 1
                        added = True
                if right < len(sents):
                    w = len(sents[right][1].split())
                    if word_count + w <= max_context_words:
                        window.append(sents[right][1])
                        word_count += w
                        right += 1
                        added = True
                if not added:
                    break
            windows.append((global_idx, " ".join(window)))

    windows = sorted(windows, key=lambda x: x[0])
    return [w[1] for w in windows]

def label_distribution_from_docs(entries, doc_ids):
    sub = [e for e in entries if e["doc"] in set(doc_ids)]
    cnt = Counter(e["label_char"] for e in sub)
    return cnt, len(sub)

def print_split_stats(name, entries):
    summarize_entries(entries, name=name)
    build_doc_label_profile(entries)

class ChunkDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

    def __len__(self):
        return len(self.texts)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs > 0.5).astype(int)
    f1m = f1_score(labels, preds, average="macro")
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    ap  = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    return {
        "f1_macro": round(f1m * 100, 3),
        "auc": round(auc * 100, 3),
        "ap": round(ap * 100, 3),
    }

def evaluate_probs(y_true, probs, title="eval"):
    preds = (probs > 0.5).astype(int)
    f1m = f1_score(y_true, preds, average="macro")
    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.0
    ap  = average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.0
    cm = confusion_matrix(y_true, preds)
    print(f"\n[{title}] F1_macro={f1m*100:.2f} | AUC={auc*100:.2f} | AP={ap*100:.2f}")
    print("Confusion matrix [[TN, FP],[FN, TP]]:")
    print(cm)
    print(classification_report(y_true, preds, target_names=["Chirac", "Mitterrand"], digits=4))
    return {"f1_macro": f1m, "auc": auc, "ap": ap, "cm": cm}

# %% [markdown]
# ## 5. Chargement du corpus d'entraînement

# %%

train_entries = read_train_corpus(TRAIN_FILE)
summarize_entries(train_entries, "train_full")
build_doc_label_profile(train_entries)

train_df = pd.DataFrame(train_entries)
train_df.head()

# %% [markdown]
# ## 6. Split par document : train / val / test_local

# %%

# On split les doc_id, pas les phrases ni les chunks.
doc_ids = sorted(train_df["doc"].unique())

doc_meta = (
    train_df.groupby("doc")
    .agg(
        n_phrases=("text", "size"),
        n_M=("label", "sum"),
    )
    .reset_index()
)
doc_meta["has_M"] = (doc_meta["n_M"] > 0).astype(int)

# 80 / 10 / 10 via deux splits stratifiés approximatifs sur has_M
docs_train, docs_temp = train_test_split(
    doc_meta["doc"].values,
    test_size=0.20,
    random_state=SEED,
    stratify=doc_meta["has_M"].values,
)

temp_meta = doc_meta.set_index("doc").loc[docs_temp].reset_index()
docs_val, docs_test_local = train_test_split(
    temp_meta["doc"].values,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_meta["has_M"].values,
)

docs_train = set(docs_train)
docs_val = set(docs_val)
docs_test_local = set(docs_test_local)

train_entries_split = [e for e in train_entries if e["doc"] in docs_train]
val_entries_split = [e for e in train_entries if e["doc"] in docs_val]
test_local_entries = [e for e in train_entries if e["doc"] in docs_test_local]

print_split_stats("train_split", train_entries_split)
print_split_stats("val_split", val_entries_split)
print_split_stats("test_local_split", test_local_entries)

# %% [markdown]
# ## 7. Création des chunks train/val et des fenêtres test_local

# %%

MAX_WORDS = 350
MAX_CONTEXT_WORDS = 350

train_chunks = build_chunks_from_entries(train_entries_split, max_words=MAX_WORDS)
val_chunks = build_chunks_from_entries(val_entries_split, max_words=MAX_WORDS)

print(f"Train chunks: {len(train_chunks):,}")
print(f"Val chunks:   {len(val_chunks):,}")
print(f"Avg sents/chunk train: {np.mean([c['n_sents'] for c in train_chunks]):.2f}")
print(f"Avg words/chunk train: {np.mean([len(c['text'].split()) for c in train_chunks]):.2f}")

train_texts = [c["text"] for c in train_chunks]
train_labels = np.array([c["label"] for c in train_chunks])

val_texts = [c["text"] for c in val_chunks]
val_labels = np.array([c["label"] for c in val_chunks])

test_local_window_texts = build_context_windows(test_local_entries, max_context_words=MAX_CONTEXT_WORDS)
test_local_labels = np.array([e["label"] for e in test_local_entries])

print("Local test windows:", len(test_local_window_texts))
print("Local test labels :", len(test_local_labels))
assert len(test_local_window_texts) == len(test_local_labels)

# %% [markdown]
# ## 8. Tokenizer + datasets

# %%

MODEL_NAME = "camembert/camembert-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = ChunkDataset(train_texts, train_labels, tokenizer, max_length=512)
val_dataset   = ChunkDataset(val_texts, val_labels, tokenizer, max_length=512)
print("Datasets prêts.")

# %% [markdown]
# ## 9. Entraînement

# %%

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="auc",
    greater_is_better=True,
    report_to="none",
    fp16=torch.cuda.is_available(),
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

# %% [markdown]
# ## 10. Évaluation sur les chunks de validation

# %%

val_results = trainer.evaluate()
print("\nBEST CHECKPOINT (validation chunks):")
for k, v in val_results.items():
    if k.startswith("eval_"):
        print(f"  {k.replace('eval_', ''):>12s} : {v}")

# %% [markdown]
# ## 11. Sauvegarde du meilleur modèle et du tokenizer

# %%

trainer.save_model(MODEL_SAVE_DIR)
tokenizer.save_pretrained(MODEL_SAVE_DIR)
print("Modèle sauvegardé dans:", MODEL_SAVE_DIR)

# %% [markdown]
# ## 12. Évaluation locale réaliste sur `test_local`
# On applique la stratégie v14 : prédiction de fenêtres de contexte sur un split jamais vu en entraînement.

# %%

test_local_hf = HFDataset.from_dict({"text": test_local_window_texts})

def tokenize_fn(ex):
    return tokenizer(ex["text"], padding="max_length", truncation=True, max_length=512)

test_local_tok = test_local_hf.map(tokenize_fn, batched=True, batch_size=128)
pred_local = trainer.predict(test_local_tok)
logits_local = pred_local.predictions
p_local = torch.nn.functional.softmax(torch.from_numpy(logits_local), dim=-1)[:, 1].numpy()

local_metrics = evaluate_probs(test_local_labels, p_local, title="test_local_windows")

print("\nDistribution des probabilités:")
print(f"  < 0.1:   {np.sum(p_local < 0.1):>6} ({np.mean(p_local < 0.1)*100:.1f}%)")
print(f"  > 0.9:   {np.sum(p_local > 0.9):>6} ({np.mean(p_local > 0.9)*100:.1f}%)")
print(f"  0.1-0.9: {np.sum((p_local >= 0.1) & (p_local <= 0.9)):>6} ({np.mean((p_local >= 0.1) & (p_local <= 0.9))*100:.1f}%)")

# %% [markdown]
# ## 12bis. Analyse d'erreurs sur `test_local`
#
# Les cellules suivantes servent à comprendre **où** la version chunks se trompe :
#
# - erreurs par label ;
# - documents mixtes vs purs ;
# - influence de la longueur de phrase ;
# - influence de la proximité d'une frontière de label ;
# - calibration grossière par zones de probabilité.
#
# Elles n'affectent pas l'entraînement : elles servent uniquement au diagnostic.

# %%

# Table détaillée des prédictions locales phrase par phrase
test_local_df = pd.DataFrame(test_local_entries).copy()
test_local_df["window_text"] = test_local_window_texts
test_local_df["p_mitterrand"] = p_local
test_local_df["pred"] = (test_local_df["p_mitterrand"] > 0.5).astype(int)
test_local_df["correct"] = (test_local_df["pred"] == test_local_df["label"]).astype(int)
test_local_df["error"] = 1 - test_local_df["correct"]
test_local_df["n_words_phrase"] = test_local_df["text"].str.split().str.len()
test_local_df["n_words_window"] = test_local_df["window_text"].str.split().str.len()

# Type de document : pur C, pur M ou mixte
doc_label_sets = (
    pd.DataFrame(train_entries)
    .groupby("doc")["label_char"]
    .agg(lambda s: "".join(sorted(set(s))))
    .to_dict()
)

def doc_type_from_set(v):
    if v == "C":
        return "pur_C"
    if v == "M":
        return "pur_M"
    return "mixte"

test_local_df["doc_type"] = test_local_df["doc"].map(lambda d: doc_type_from_set(doc_label_sets.get(d, "CM")))

# Position et frontière de label dans le document
test_local_df = test_local_df.sort_values(["doc", "sent"]).reset_index(drop=True)
test_local_df["prev_label"] = test_local_df.groupby("doc")["label"].shift(1)
test_local_df["next_label"] = test_local_df.groupby("doc")["label"].shift(-1)
test_local_df["is_boundary"] = (
    ((test_local_df["prev_label"].notna()) & (test_local_df["prev_label"] != test_local_df["label"])) |
    ((test_local_df["next_label"].notna()) & (test_local_df["next_label"] != test_local_df["label"]))
).astype(int)

# Distance à la frontière la plus proche (en nombre de phrases)
dist_to_boundary = []
for doc_id, g in test_local_df.groupby("doc", sort=False):
    idxs = list(g.index)
    boundary_idxs = [i for i in idxs if test_local_df.loc[i, "is_boundary"] == 1]
    if not boundary_idxs:
        dist_to_boundary.extend([999] * len(idxs))
    else:
        for i in idxs:
            dist_to_boundary.append(min(abs(i - b) for b in boundary_idxs))

test_local_df["dist_boundary"] = dist_to_boundary
test_local_df["near_boundary_le_1"] = (test_local_df["dist_boundary"] <= 1).astype(int)
test_local_df["near_boundary_le_3"] = (test_local_df["dist_boundary"] <= 3).astype(int)

print("Table d'analyse prête :", test_local_df.shape)
test_local_df.head()

# %% [markdown]
# ### 12bis.a Vue d'ensemble des erreurs

# %%

overall = pd.DataFrame({
    "n": [len(test_local_df)],
    "accuracy": [test_local_df["correct"].mean()],
    "error_rate": [test_local_df["error"].mean()],
    "mean_p_mitterrand": [test_local_df["p_mitterrand"].mean()],
    "mean_words_phrase": [test_local_df["n_words_phrase"].mean()],
    "mean_words_window": [test_local_df["n_words_window"].mean()],
})
overall

# %% [markdown]
# ### 12bis.b Erreurs par label et par type de document

# %%

by_label = (
    test_local_df.groupby("label_char")
    .agg(
        n=("label", "size"),
        error_rate=("error", "mean"),
        recall_proxy=("correct", "mean"),
        mean_p_mitterrand=("p_mitterrand", "mean"),
        mean_words_phrase=("n_words_phrase", "mean"),
    )
    .reset_index()
)
print("Par label :")
display(by_label)

by_doc_type = (
    test_local_df.groupby("doc_type")
    .agg(
        n=("label", "size"),
        error_rate=("error", "mean"),
        mean_p_mitterrand=("p_mitterrand", "mean"),
        mean_words_phrase=("n_words_phrase", "mean"),
    )
    .reset_index()
    .sort_values("n", ascending=False)
)
print("Par type de document :")
display(by_doc_type)

cross = (
    test_local_df.groupby(["doc_type", "label_char"])
    .agg(
        n=("label", "size"),
        error_rate=("error", "mean"),
        mean_p_mitterrand=("p_mitterrand", "mean"),
    )
    .reset_index()
)
print("Croisement type de document × label :")
display(cross)

# %% [markdown]
# ### 12bis.c Erreurs près des frontières de label

# %%

boundary_stats = (
    test_local_df.groupby("near_boundary_le_1")
    .agg(
        n=("label", "size"),
        error_rate=("error", "mean"),
        mean_p_mitterrand=("p_mitterrand", "mean"),
        mean_words_phrase=("n_words_phrase", "mean"),
    )
    .reset_index()
)
boundary_stats["zone"] = boundary_stats["near_boundary_le_1"].map({0: "loin_frontiere", 1: "a_1_phrase_ou_moins"})
display(boundary_stats[["zone", "n", "error_rate", "mean_p_mitterrand", "mean_words_phrase"]])

boundary_stats3 = (
    test_local_df.groupby("near_boundary_le_3")
    .agg(
        n=("label", "size"),
        error_rate=("error", "mean"),
        mean_p_mitterrand=("p_mitterrand", "mean"),
    )
    .reset_index()
)
boundary_stats3["zone"] = boundary_stats3["near_boundary_le_3"].map({0: "loin_frontiere", 1: "a_3_phrases_ou_moins"})
display(boundary_stats3[["zone", "n", "error_rate", "mean_p_mitterrand"]])

print("Distance moyenne à la frontière :")
display(
    test_local_df.groupby("error")["dist_boundary"].agg(["count", "mean", "median"]).rename(index={0:"correct",1:"error"})
)

# %% [markdown]
# ### 12bis.d Longueur des phrases et confiance du modèle

# %%

test_local_df["len_bin"] = pd.cut(
    test_local_df["n_words_phrase"],
    bins=[0, 5, 10, 20, 40, 1000],
    labels=["1-5", "6-10", "11-20", "21-40", "41+"],
    include_lowest=True,
)

len_stats = (
    test_local_df.groupby("len_bin", observed=False)
    .agg(
        n=("label", "size"),
        error_rate=("error", "mean"),
        mean_p_mitterrand=("p_mitterrand", "mean"),
    )
    .reset_index()
)
display(len_stats)

test_local_df["conf_zone"] = pd.cut(
    test_local_df["p_mitterrand"],
    bins=[-0.001, 0.1, 0.4, 0.6, 0.9, 1.001],
    labels=["<0.1", "0.1-0.4", "0.4-0.6", "0.6-0.9", ">0.9"],
    include_lowest=True,
)

conf_stats = (
    test_local_df.groupby("conf_zone", observed=False)
    .agg(
        n=("label", "size"),
        error_rate=("error", "mean"),
        prop_m_true=("label", "mean"),
    )
    .reset_index()
)
display(conf_stats)

# %% [markdown]
# ### 12bis.e Exemples d'erreurs à inspecter

# %%

# Erreurs Mitterrand ratées (FN) et Chirac confondu avec Mitterrand (FP)
false_negatives = test_local_df[(test_local_df["label"] == 1) & (test_local_df["pred"] == 0)].copy()
false_positives = test_local_df[(test_local_df["label"] == 0) & (test_local_df["pred"] == 1)].copy()

print("Exemples de faux négatifs (vrai Mitterrand, prédit Chirac) :")
display(false_negatives[["doc", "sent", "label_char", "p_mitterrand", "doc_type", "dist_boundary", "n_words_phrase", "text"]].head(15))

print("Exemples de faux positifs (vrai Chirac, prédit Mitterrand) :")
display(false_positives[["doc", "sent", "label_char", "p_mitterrand", "doc_type", "dist_boundary", "n_words_phrase", "text"]].head(15))

# %% [markdown]
# ## 12ter. Inspection du contexte des erreurs
#
# On affiche ici les erreurs avec leur **contexte réel dans le document** pour vérifier l'hypothèse suivante :
#
# - la phrase centrale est correcte seule ou stylistiquement marquée,
# - mais le **contexte dominant autour** pousse le modèle vers l'autre président,
# - surtout dans les **documents mixtes** et près des **frontières de label**.
#
# Ces cellules ne réentraînent pas le modèle : elles servent à **voir** les cas difficiles.

# %%

from collections import defaultdict

def build_entries_by_doc(entries):
    by_doc = defaultdict(list)
    for e in entries:
        by_doc[e["doc"]].append(e)
    for doc_id in by_doc:
        by_doc[doc_id] = sorted(by_doc[doc_id], key=lambda x: x["sent"])
    return by_doc

entries_by_doc_test_local = build_entries_by_doc(test_local_entries)

def show_error_context_examples(
    pred_df,
    entries_by_doc,
    n=12,
    window=3,
    near_boundary_only=False,
    mixed_only=False,
    random_state=42,
):
    df = pred_df[pred_df["error"] == 1].copy()

    if near_boundary_only and "near_boundary_le_1" in df.columns:
        df = df[df["near_boundary_le_1"] == 1]

    if mixed_only and "doc_type" in df.columns:
        df = df[df["doc_type"] == "mixte"]

    print(f"Erreurs retenues: {len(df)}")
    if len(df) == 0:
        print("Aucune erreur à afficher.")
        return

    sample = df.sample(n=min(n, len(df)), random_state=random_state)

    for _, row in sample.iterrows():
        doc_id = int(row["doc"])
        sent_id = int(row["sent"])
        doc_entries = entries_by_doc[doc_id]
        pos = None
        for i, e in enumerate(doc_entries):
            if int(e["sent"]) == sent_id:
                pos = i
                break

        print("=" * 120)
        print(
            f"DOC={doc_id} | SENT={sent_id} | doc_type={row.get('doc_type', 'NA')} | "
            f"dist_frontiere={row.get('dist_to_boundary', 'NA')} | "
            f"TRUE={row['label_char']} | PRED={'M' if row['pred']==1 else 'C'} | "
            f"p(M)={row['p_mitterrand']:.4f}"
        )
        print("-" * 120)

        left = max(0, pos - window)
        right = min(len(doc_entries), pos + window + 1)

        for j in range(left, right):
            e = doc_entries[j]
            marker = ">>> " if j == pos else "    "
            print(f"{marker}[sent={e['sent']:>3}] TRUE={e['label_char']} | {e['text']}")
        print()

print("Fonctions d'inspection du contexte prêtes.")

# %%

# Exemples d'erreurs proches des frontières dans des documents mixtes
show_error_context_examples(
    test_local_df,
    entries_by_doc_test_local,
    n=12,
    window=3,
    near_boundary_only=True,
    mixed_only=True,
    random_state=42,
)

# %% [markdown]
# ## 12quater. Tests rapides sans réentraînement : varier la taille de la fenêtre au test
#
# Ici, on garde **le même modèle déjà entraîné**, et on change seulement la taille de la **fenêtre de contexte** au moment de prédire sur `test_local`.
#
# Cela permet de vérifier rapidement si :
# - un **contexte plus petit** aide près des frontières,
# - le contexte actuel est parfois **trop dominant**,
# - votre hypothèse sur les documents mixtes semble correcte.
#
# Ces tests sont **peu coûteux** car ils ne réentraînent pas le modèle.

# %%

def predict_probs_for_texts_with_trainer(trainer, tokenizer, texts, batch_size=128, max_length=512):
    hf_ds = HFDataset.from_dict({"text": texts})

    def _tok(ex):
        return tokenizer(ex["text"], padding="max_length", truncation=True, max_length=max_length)

    tok_ds = hf_ds.map(_tok, batched=True, batch_size=batch_size)
    pred = trainer.predict(tok_ds)
    logits = pred.predictions
    probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()
    return probs

def eval_fixed_window_sizes(sizes=(150, 200, 250, 350)):
    rows = []
    for size in sizes:
        texts = build_context_windows(test_local_entries, max_context_words=size)
        probs = predict_probs_for_texts_with_trainer(trainer, tokenizer, texts)
        preds = (probs > 0.5).astype(int)
        rows.append({
            "context_words": size,
            "f1_macro": f1_score(test_local_labels, preds, average="macro") * 100,
            "auc": roc_auc_score(test_local_labels, probs) * 100,
            "ap": average_precision_score(test_local_labels, probs) * 100,
            "recall_M": recall_score(test_local_labels, preds, pos_label=1) * 100,
            "precision_M": precision_score(test_local_labels, preds, pos_label=1, zero_division=0) * 100,
            "pred_M_rate": preds.mean() * 100,
        })
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False)

window_grid_results = eval_fixed_window_sizes(sizes=(120, 150, 200, 250, 350))
display(window_grid_results)

# %% [markdown]
# ## 12quinquies. Test adaptatif local (diagnostic) sur les zones critiques
#
# Ici, on fait un **test local diagnostique** :
#
# - si une phrase est dans un **document mixte** ou **près d'une frontière**, on lui donne une **fenêtre plus petite** ;
# - sinon on garde la fenêtre standard.
#
# ⚠️ Important :
# - ce test utilise l'information des labels du `test_local` pour détecter les zones critiques ;
# - il sert donc à **diagnostiquer l'idée**, pas à produire directement une soumission finale.
#
# S'il aide nettement en local, alors cela justifie de concevoir ensuite une version plus réaliste.

# %%

def build_context_windows_adaptive_local(entries, default_words=350, critical_words=200, use_mixed_docs=True, use_near_boundary=True, boundary_k=1):
    # On s'appuie sur les colonnes déjà calculées dans test_local_df
    meta = test_local_df[["doc", "sent", "doc_type", "near_boundary_le_1", "near_boundary_le_3"]].copy()
    meta["critical"] = 0

    if use_mixed_docs:
        meta.loc[meta["doc_type"] == "mixte", "critical"] = 1

    if use_near_boundary:
        col = "near_boundary_le_1" if boundary_k == 1 else "near_boundary_le_3"
        meta.loc[meta[col] == 1, "critical"] = 1

    word_budget = {
        (int(r.doc), int(r.sent)): (critical_words if int(r.critical) == 1 else default_words)
        for r in meta.itertuples(index=False)
    }

    by_doc = defaultdict(list)
    for entry in entries:
        by_doc[entry["doc"]].append(entry)

    windows = []
    for doc_id in sorted(by_doc.keys()):
        sents = sorted(by_doc[doc_id], key=lambda x: x["sent"])
        for pos, entry in enumerate(sents):
            budget = word_budget[(int(entry["doc"]), int(entry["sent"]))]
            window = [entry["text"]]
            word_count = len(entry["text"].split())
            left, right = pos - 1, pos + 1

            while word_count < budget:
                added = False
                if left >= 0:
                    w = len(sents[left]["text"].split())
                    if word_count + w <= budget:
                        window.insert(0, sents[left]["text"])
                        word_count += w
                        left -= 1
                        added = True
                if right < len(sents):
                    w = len(sents[right]["text"].split())
                    if word_count + w <= budget:
                        window.append(sents[right]["text"])
                        word_count += w
                        right += 1
                        added = True
                if not added:
                    break
            windows.append(" ".join(window))
    return windows

def eval_adaptive_local(default_words=350, critical_words=200, use_mixed_docs=True, use_near_boundary=True, boundary_k=1):
    texts = build_context_windows_adaptive_local(
        test_local_entries,
        default_words=default_words,
        critical_words=critical_words,
        use_mixed_docs=use_mixed_docs,
        use_near_boundary=use_near_boundary,
        boundary_k=boundary_k,
    )
    probs = predict_probs_for_texts_with_trainer(trainer, tokenizer, texts)
    preds = (probs > 0.5).astype(int)
    return {
        "default_words": default_words,
        "critical_words": critical_words,
        "mixed_docs": use_mixed_docs,
        "near_boundary": use_near_boundary,
        "boundary_k": boundary_k,
        "f1_macro": f1_score(test_local_labels, preds, average="macro") * 100,
        "auc": roc_auc_score(test_local_labels, probs) * 100,
        "ap": average_precision_score(test_local_labels, probs) * 100,
        "recall_M": recall_score(test_local_labels, preds, pos_label=1) * 100,
        "precision_M": precision_score(test_local_labels, preds, pos_label=1, zero_division=0) * 100,
        "pred_M_rate": preds.mean() * 100,
    }

adaptive_rows = []
for crit in [120, 150, 200, 250]:
    adaptive_rows.append(eval_adaptive_local(default_words=350, critical_words=crit, use_mixed_docs=True, use_near_boundary=True, boundary_k=1))
    adaptive_rows.append(eval_adaptive_local(default_words=350, critical_words=crit, use_mixed_docs=False, use_near_boundary=True, boundary_k=1))
    adaptive_rows.append(eval_adaptive_local(default_words=350, critical_words=crit, use_mixed_docs=True, use_near_boundary=False, boundary_k=1))

adaptive_results = pd.DataFrame(adaptive_rows).sort_values("f1_macro", ascending=False)
display(adaptive_results)

# %% [markdown]
# ## 12sexies. Réentraînement optionnel avec une autre taille de chunk
#
# Cette partie permet de tester de vraies variantes de `v14` en réentraînant le modèle avec :
#
# - `chunk_words = 200`
# - `chunk_words = 250`
# - `chunk_words = 350` (référence)
#
# et éventuellement une autre taille de **fenêtre au test local**.
#
# ⚠️ C'est beaucoup plus coûteux que les cellules précédentes.  
# Lance **une seule expérience à la fois**.

# %%

def run_retrain_experiment(
    chunk_words=250,
    context_words=250,
    model_name=MODEL_NAME,
    num_train_epochs=6,
    output_suffix=None,
    save_model=False,
):
    if output_suffix is None:
        output_suffix = f"cw{chunk_words}_ctx{context_words}"

    print(f"\n=== EXPERIMENT: chunk_words={chunk_words} | context_words={context_words} | model={model_name} ===")

    exp_train_chunks = build_chunks_from_entries(train_entries_split, max_words=chunk_words)
    exp_val_chunks   = build_chunks_from_entries(val_entries_split, max_words=chunk_words)

    exp_train_texts = [c["text"] for c in exp_train_chunks]
    exp_train_labels = np.array([c["label"] for c in exp_train_chunks])

    exp_val_texts = [c["text"] for c in exp_val_chunks]
    exp_val_labels = np.array([c["label"] for c in exp_val_chunks])

    exp_test_local_texts = build_context_windows(test_local_entries, max_context_words=context_words)

    print(f"Train chunks: {len(exp_train_chunks):,} | Val chunks: {len(exp_val_chunks):,}")
    print(f"Avg words/train chunk: {np.mean([len(c['text'].split()) for c in exp_train_chunks]):.1f}")

    exp_tokenizer = AutoTokenizer.from_pretrained(model_name)
    exp_train_dataset = ChunkDataset(exp_train_texts, exp_train_labels, exp_tokenizer, max_length=512)
    exp_val_dataset   = ChunkDataset(exp_val_texts, exp_val_labels, exp_tokenizer, max_length=512)

    exp_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    exp_model.to(device)

    exp_output_dir = os.path.join(OUTPUT_DIR, f"exp_{output_suffix}")
    os.makedirs(exp_output_dir, exist_ok=True)

    exp_args = TrainingArguments(
        output_dir=exp_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=SEED,
    )

    exp_trainer = Trainer(
        model=exp_model,
        args=exp_args,
        train_dataset=exp_train_dataset,
        eval_dataset=exp_val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    exp_trainer.train()
    exp_val_results = exp_trainer.evaluate()

    exp_probs_local = predict_probs_for_texts_with_trainer(exp_trainer, exp_tokenizer, exp_test_local_texts)
    exp_preds_local = (exp_probs_local > 0.5).astype(int)

    result = {
        "chunk_words": chunk_words,
        "context_words": context_words,
        "val_f1_macro": exp_val_results.get("eval_f1_macro"),
        "val_auc": exp_val_results.get("eval_auc"),
        "val_ap": exp_val_results.get("eval_ap"),
        "test_local_f1_macro": f1_score(test_local_labels, exp_preds_local, average="macro") * 100,
        "test_local_auc": roc_auc_score(test_local_labels, exp_probs_local) * 100,
        "test_local_ap": average_precision_score(test_local_labels, exp_probs_local) * 100,
        "test_local_recall_M": recall_score(test_local_labels, exp_preds_local, pos_label=1) * 100,
        "test_local_precision_M": precision_score(test_local_labels, exp_preds_local, pos_label=1, zero_division=0) * 100,
        "pred_M_rate": exp_preds_local.mean() * 100,
        "output_dir": exp_output_dir,
    }

    if save_model:
        exp_model_dir = os.path.join(MODEL_SAVE_DIR, f"exp_{output_suffix}")
        os.makedirs(exp_model_dir, exist_ok=True)
        exp_trainer.save_model(exp_model_dir)
        exp_tokenizer.save_pretrained(exp_model_dir)
        result["saved_model_dir"] = exp_model_dir

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result

print("Fonction run_retrain_experiment(...) prête.")

# %%

# Décommente UNE ligne à la fois.
exp_200 = run_retrain_experiment(chunk_words=200, context_words=200, num_train_epochs=6, output_suffix="chunk200_ctx200")
exp_250 = run_retrain_experiment(chunk_words=250, context_words=250, num_train_epochs=6, output_suffix="chunk250_ctx250")
exp_350 = run_retrain_experiment(chunk_words=350, context_words=350, num_train_epochs=6, output_suffix="chunk350_ctx350")

# Exemple pour comparer ensuite :
pd.DataFrame([exp_200, exp_250, exp_350])

# %% [markdown]
# ## 13. Optionnel — Réentraînement final sur tout le corpus learn
# À lancer seulement quand vous êtes sûrs de la recette.

# %%

RETRAIN_ON_FULL_LEARN = False

if RETRAIN_ON_FULL_LEARN:
    full_chunks = build_chunks_from_entries(train_entries, max_words=MAX_WORDS)
    full_texts = [c["text"] for c in full_chunks]
    full_labels = np.array([c["label"] for c in full_chunks])

    full_dataset = ChunkDataset(full_texts, full_labels, tokenizer, max_length=512)

    final_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    final_model.to(device)

    final_args = TrainingArguments(
        output_dir=OUTPUT_DIR + "_full",
        eval_strategy="no",
        save_strategy="epoch",
        learning_rate=1e-5,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=SEED,
    )

    final_trainer = Trainer(
        model=final_model,
        args=final_args,
        train_dataset=full_dataset,
    )

    final_trainer.train()
    final_trainer.save_model(MODEL_SAVE_DIR + "_full")
    tokenizer.save_pretrained(MODEL_SAVE_DIR + "_full")
    print("Modèle final sauvegardé dans:", MODEL_SAVE_DIR + "_full")

# %% [markdown]
# ## 14. Prédiction finale sur le vrai test du prof
# Cette cellule n'est à lancer que si le fichier test est présent.

# %%

RUN_FINAL_TEST = False

if RUN_FINAL_TEST:
    assert Path(TEST_FILE).exists(), f"Fichier introuvable: {TEST_FILE}"

    # Choisir quel modèle charger
    MODEL_FOR_INFERENCE = MODEL_SAVE_DIR
    if Path(MODEL_SAVE_DIR + "_full").exists():
        MODEL_FOR_INFERENCE = MODEL_SAVE_DIR + "_full"

    infer_tokenizer = AutoTokenizer.from_pretrained(MODEL_FOR_INFERENCE)
    infer_model = AutoModelForSequenceClassification.from_pretrained(MODEL_FOR_INFERENCE).to(device)

    test_entries = read_test_corpus(TEST_FILE)
    print("Test phrases:", len(test_entries))

    test_window_texts = build_context_windows(test_entries, max_context_words=MAX_CONTEXT_WORDS)
    test_hf = HFDataset.from_dict({"text": test_window_texts})

    def tok_final(ex):
        return infer_tokenizer(ex["text"], padding="max_length", truncation=True, max_length=512)

    test_tok = test_hf.map(tok_final, batched=True, batch_size=128)

    infer_trainer = Trainer(model=infer_model)
    pred = infer_trainer.predict(test_tok)
    logits = pred.predictions
    p_mitterrand = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()

    submission_path = Path(SUBMISSION_DIR) / "submission-pres-v14-corrige.csv"
    with open(submission_path, "w", encoding="utf-8") as f:
        for p in p_mitterrand:
            f.write(f"{float(p)}\n")

    print("Soumission écrite dans:", submission_path)
    print(f"Lignes: {len(p_mitterrand)}")
    print(f"Mitterrand prédits (>0.5): {int(np.sum(p_mitterrand > 0.5))}")

# %% [markdown]
# ## 15. Nettoyage

# %%

gc.collect()
torch.cuda.empty_cache()
print("Mémoire nettoyée.")
