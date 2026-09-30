# RITAL contextual authorship-classification experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # Tâche 1 — v14 corrigé — weighted loss + early stopping
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
from torch import nn
from torch.utils.data import Dataset as TorchDataset

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
# Poids de classes calculés sur les chunks d'entraînement
# Convention actuelle: 0 = Chirac, 1 = Mitterrand
class_counts = Counter(train_labels.tolist())
num_classes = 2
n_train = len(train_labels)
class_weights = torch.tensor(
    [n_train / (num_classes * class_counts[i]) for i in range(num_classes)],
    dtype=torch.float,
    device=device,
)

print("Class counts (train chunks):", class_counts)
print("Class weights:", class_weights.detach().cpu().numpy())

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# %%
# Entraînement avec weighted loss + early stopping

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

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    class_weights=class_weights,
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