# RITAL contextual authorship-classification experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # Président – retrain final du meilleur modèle (`train250_ctx250`)
# Ce notebook réentraîne directement le meilleur pipeline retenu :
# - chunks d'entraînement de **250 mots**
# - fenêtres de contexte test de **250 mots**
# - entraînement sur **tout le corpus learn**
# - prédiction sur le **test du prof**
# - sauvegarde du modèle et du CSV de soumission dans **Google Drive**

# %%

import os
import re
import shutil
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# %%

# NOTEBOOK-ONLY COMMAND REMOVED: from google.colab import drive
# NOTEBOOK-ONLY COMMAND REMOVED: drive.mount('/content/drive')

# %%

# ---- chemins à modifier ----
WORK_DIR = "/content/drive/MyDrive/RITAL_president_final"
TRAIN_FILE = "/content/drive/MyDrive/projet tal/corpus.tache1.learn.utf8"
TEST_FILE  = "/content/drive/MyDrive/projet tal/corpus.tache1.test.utf8"

# ---- dossiers de sortie ----
MODEL_DIR = os.path.join(WORK_DIR, "final_model_train250_ctx250")
SUBMISSION_DIR = os.path.join(WORK_DIR, "submissions")
TMP_DIR = os.path.join(WORK_DIR, "tmp_train250_ctx250")

# ---- modèle / hyperparamètres ----
MODEL_NAME = "camembert/camembert-large"
RANDOM_STATE = 42

MAX_LENGTH = 512
CHUNK_WORDS = 250
CONTEXT_WORDS = 250

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
NUM_EPOCHS = 6

PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 32
GRAD_ACCUM = 4

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

print("WORK_DIR:", WORK_DIR)
print("TRAIN_FILE exists:", os.path.exists(TRAIN_FILE))
print("TEST_FILE exists :", os.path.exists(TEST_FILE))
print("MODEL_DIR:", MODEL_DIR)

# %%

TRAIN_LINE_RE = re.compile(r"^<(\d+):(\d+):([CM])>\s*(.*)$")
TEST_LINE_RE  = re.compile(r"^<(\d+):(\d+)>\s*(.*)$")

def read_train_corpus(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            m = TRAIN_LINE_RE.match(line)
            if not m:
                raise ValueError(f"Ligne train non reconnue: {line[:200]}")
            doc_id, sent_id, label, text = m.groups()
            entries.append({
                "doc_id": int(doc_id),
                "sent_id": int(sent_id),
                "label": 0 if label == "C" else 1,
                "text": text.strip(),
            })
    return entries

def read_test_corpus(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            m = TEST_LINE_RE.match(line)
            if not m:
                raise ValueError(f"Ligne test non reconnue: {line[:200]}")
            doc_id, sent_id, text = m.groups()
            entries.append({
                "doc_id": int(doc_id),
                "sent_id": int(sent_id),
                "text": text.strip(),
            })
    return entries

entries = read_train_corpus(TRAIN_FILE)
test_entries_prof = read_test_corpus(TEST_FILE)

print("Train phrases:", len(entries))
print("Train docs   :", len(set(e["doc_id"] for e in entries)))
print("Train labels :", pd.Series([e["label"] for e in entries]).map({0:"C",1:"M"}).value_counts().to_dict())
print("Test phrases :", len(test_entries_prof))
print("Test docs    :", len(set(e["doc_id"] for e in test_entries_prof)))

# %%

def group_entries_by_doc(entries):
    docs = defaultdict(list)
    for e in entries:
        docs[e["doc_id"]].append(e)
    for doc_id in docs:
        docs[doc_id] = sorted(docs[doc_id], key=lambda x: x["sent_id"])
    return docs

def build_chunks_from_entries(entries, max_words=250):
    docs = group_entries_by_doc(entries)
    chunks = []

    for doc_id, doc_entries in docs.items():
        i = 0
        n = len(doc_entries)
        while i < n:
            label = doc_entries[i]["label"]
            segment = []
            while i < n and doc_entries[i]["label"] == label:
                segment.append(doc_entries[i])
                i += 1

            cur = []
            cur_words = 0
            for sent in segment:
                w = len(sent["text"].split())
                if cur and cur_words + w > max_words:
                    chunks.append({
                        "doc_id": doc_id,
                        "label": label,
                        "text": " ".join(x["text"] for x in cur),
                        "n_sents": len(cur),
                    })
                    cur = []
                    cur_words = 0
                cur.append(sent)
                cur_words += w

            if cur:
                chunks.append({
                    "doc_id": doc_id,
                    "label": label,
                    "text": " ".join(x["text"] for x in cur),
                    "n_sents": len(cur),
                })

    return chunks

full_chunks = build_chunks_from_entries(entries, max_words=CHUNK_WORDS)

print("Full train chunks:", len(full_chunks))
print("Avg sents/chunk  :", np.mean([c['n_sents'] for c in full_chunks]))
print("Avg words/chunk  :", np.mean([len(c['text'].split()) for c in full_chunks]))
print("Chunk labels     :", pd.Series([c["label"] for c in full_chunks]).map({0:"C",1:"M"}).value_counts().to_dict())

# %%

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def make_hf_dataset(texts, labels=None):
    data = {"text": texts}
    if labels is not None:
        data["labels"] = labels
    ds = HFDataset.from_dict(data)

    def tok_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    return ds

full_train_texts = [c["text"] for c in full_chunks]
full_train_labels = [c["label"] for c in full_chunks]
full_train_ds = make_hf_dataset(full_train_texts, full_train_labels)

print(full_train_ds)
print(full_train_ds.column_names)

# %%

if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
os.makedirs(TMP_DIR, exist_ok=True)

final_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

final_args = TrainingArguments(
    output_dir=TMP_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    logging_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
    save_only_model=True,
)

final_trainer = Trainer(
    model=final_model,
    args=final_args,
    train_dataset=full_train_ds,
    data_collator=data_collator,
)

train_output = final_trainer.train()
print(train_output)

final_trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print("Modèle final sauvegardé dans :", MODEL_DIR)

# %%

def build_context_windows(entries, max_context_words=250):
    docs = group_entries_by_doc(entries)
    texts = []

    for doc_id in sorted(docs.keys()):
        doc_entries = docs[doc_id]
        sent_texts = [e["text"] for e in doc_entries]
        sent_words = [len(t.split()) for t in sent_texts]

        for i in range(len(doc_entries)):
            parts = [sent_texts[i]]
            total = sent_words[i]
            left = i - 1
            right = i + 1
            turn = 0

            while True:
                added = False

                if turn % 2 == 0:
                    if left >= 0 and total + sent_words[left] <= max_context_words:
                        parts.insert(0, sent_texts[left])
                        total += sent_words[left]
                        left -= 1
                        added = True
                    if right < len(doc_entries) and total + sent_words[right] <= max_context_words:
                        parts.append(sent_texts[right])
                        total += sent_words[right]
                        right += 1
                        added = True
                else:
                    if right < len(doc_entries) and total + sent_words[right] <= max_context_words:
                        parts.append(sent_texts[right])
                        total += sent_words[right]
                        right += 1
                        added = True
                    if left >= 0 and total + sent_words[left] <= max_context_words:
                        parts.insert(0, sent_texts[left])
                        total += sent_words[left]
                        left -= 1
                        added = True

                if not added:
                    break
                turn += 1

            texts.append(" ".join(parts))

    return texts

test_entries_for_windows = [
    {"doc_id": e["doc_id"], "sent_id": e["sent_id"], "text": e["text"], "label": 0}
    for e in test_entries_prof
]

test_texts = build_context_windows(test_entries_for_windows, max_context_words=CONTEXT_WORDS)
print("Test windows:", len(test_texts))

# %%

def predict_probs_for_texts_with_trainer(trainer, tokenizer, texts, batch_size=128, max_length=512):
    hf_ds = HFDataset.from_dict({"text": texts})

    def _tok(ex):
        return tokenizer(
            ex["text"],
            truncation=True,
            max_length=max_length,
        )

    tok_ds = hf_ds.map(_tok, batched=True, batch_size=batch_size, remove_columns=["text"])
    pred = trainer.predict(tok_ds)
    logits = pred.predictions
    probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()
    return probs

test_probs = predict_probs_for_texts_with_trainer(
    final_trainer,
    tokenizer,
    test_texts,
    batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    max_length=MAX_LENGTH,
)

print("Nb probs:", len(test_probs))
print("Min prob:", float(np.min(test_probs)))
print("Max prob:", float(np.max(test_probs)))
print("Mean prob:", float(np.mean(test_probs)))

# %%

submission_path = os.path.join(SUBMISSION_DIR, "submission-pres-train250_ctx250-final.csv")
pd.Series(test_probs).to_csv(submission_path, index=False, header=False)

print("Soumission sauvegardée :", submission_path)

# %% [markdown]
# ## Remarques
# - La soumission contient **P(Mitterrand)** sur chaque ligne.
# - La plateforme du prof décidera ensuite le label avec son seuil.
# - Le modèle final et le CSV sont sauvegardés dans Drive.