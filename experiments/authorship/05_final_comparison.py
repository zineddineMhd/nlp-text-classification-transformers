# RITAL contextual authorship-classification experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # Président — Expériences finales comparatives
#
# Ce notebook exécute **4 expériences locales** sur le corpus d'apprentissage, avec un **split par document** :
#
# 1. `train250_ctx250`
# 2. `train350_ctx250`
# 3. `frontier_aware_base250_ctx250`
# 4. `oversample_minority_train250_ctx250`
#
# Puis il compare les résultats sur un **test local** phrase par phrase avec fenêtres de contexte, choisit le meilleur modèle selon `f1_macro`, et propose ensuite :
#
# - un **réentraînement final sur tout le corpus learn**
# - une **prédiction sur le test du prof**
# - la **sauvegarde d'un CSV de soumission**
#
# > Remplace simplement les chemins dans la cellule de configuration.

# %%
# NOTEBOOK-ONLY COMMAND REMOVED: from google.colab import drive
# NOTEBOOK-ONLY COMMAND REMOVED: drive.mount('/content/drive')

# %%

# =========================
# 0) Install / imports
# =========================

# Décommente si besoin dans Colab :
# NOTEBOOK: !pip -q install transformers datasets accelerate evaluate scikit-learn safetensors

import os
import re
import random
import shutil
from typing import List, Dict

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
)

from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

set_seed(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# %%

# =========================
# 1) Configuration
# =========================

TRAIN_FILE = "/content/drive/MyDrive/projet tal/corpus.tache1.learn.utf8"
TEST_FILE  = "corpus.tache1.test.utf8"   # remplace par le vrai fichier test du prof
WORK_DIR   = "./pres_experiments_final"
MODEL_ROOT = os.path.join(WORK_DIR, "saved_models")
OUTPUT_DIR = os.path.join(WORK_DIR, "outputs")
SUBMISSION_DIR = os.path.join(WORK_DIR, "submissions")

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(MODEL_ROOT, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

MODEL_NAME = "camembert/camembert-large"
MAX_LENGTH = 512

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
RANDOM_STATE = 42

PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRAD_ACCUM = 4
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
NUM_EPOCHS = 6

print("MODEL_NAME =", MODEL_NAME)
print("WORK_DIR   =", WORK_DIR)

# %%

# =========================
# 2) Lecture du corpus
# =========================

LINE_RE = re.compile(r"^<(\d+):(\d+):([CM])>\s*(.*)$")

def read_train_corpus(path: str) -> List[Dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            m = LINE_RE.match(line)
            if not m:
                continue
            doc_id, sent_id, lab, text = m.groups()
            entries.append({
                "doc_id": int(doc_id),
                "sent_id": int(sent_id),
                "label_str": lab,
                "label": 1 if lab == "M" else 0,
                "text": text.strip(),
            })
    return entries

def read_test_corpus(path: str) -> List[Dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            m = LINE_RE.match(line)
            if m:
                doc_id, sent_id, _lab, text = m.groups()
                entries.append({
                    "row_id": idx,
                    "doc_id": int(doc_id),
                    "sent_id": int(sent_id),
                    "text": text.strip(),
                })
            else:
                entries.append({
                    "row_id": idx,
                    "doc_id": 0,
                    "sent_id": idx + 1,
                    "text": line.strip(),
                })
    return entries

entries = read_train_corpus(TRAIN_FILE)
df = pd.DataFrame(entries)
print(df.head())
print()
print("Nb phrases :", len(df))
print("Nb documents :", df['doc_id'].nunique())
print("Répartition labels :")
print(df["label_str"].value_counts())

# %%

# =========================
# 3) Split par document
# =========================

doc_stats = (
    df.groupby("doc_id")
      .agg(
          n_sents=("sent_id", "count"),
          n_M=("label", lambda s: int((s == 1).sum())),
          n_C=("label", lambda s: int((s == 0).sum())),
      )
      .reset_index()
)

doc_stats["doc_type"] = np.select(
    [
        (doc_stats["n_M"] > 0) & (doc_stats["n_C"] == 0),
        (doc_stats["n_C"] > 0) & (doc_stats["n_M"] == 0),
    ],
    ["pur_M", "pur_C"],
    default="mixte"
)

train_docs, temp_docs = train_test_split(
    doc_stats["doc_id"],
    test_size=(1 - TRAIN_RATIO),
    random_state=RANDOM_STATE,
    stratify=doc_stats["doc_type"],
)

temp_doc_stats = doc_stats[doc_stats["doc_id"].isin(temp_docs)]
val_docs, test_docs = train_test_split(
    temp_doc_stats["doc_id"],
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    random_state=RANDOM_STATE,
    stratify=temp_doc_stats["doc_type"],
)

train_docs = set(train_docs.tolist())
val_docs = set(val_docs.tolist())
test_docs = set(test_docs.tolist())

train_entries_split = [e for e in entries if e["doc_id"] in train_docs]
val_entries_split   = [e for e in entries if e["doc_id"] in val_docs]
test_local_entries  = [e for e in entries if e["doc_id"] in test_docs]

print("Docs train:", len(train_docs))
print("Docs val  :", len(val_docs))
print("Docs test :", len(test_docs))
print("Phrases train:", len(train_entries_split))
print("Phrases val  :", len(val_entries_split))
print("Phrases test :", len(test_local_entries))

# %%

# =========================
# 4) Utilitaires chunks / fenêtres
# =========================

def group_entries_by_doc(entries_list: List[Dict]) -> Dict[int, List[Dict]]:
    docs = {}
    for e in entries_list:
        docs.setdefault(e["doc_id"], []).append(e)
    for doc_id in docs:
        docs[doc_id] = sorted(docs[doc_id], key=lambda x: x["sent_id"])
    return docs

def split_into_label_segments(entries_list: List[Dict]) -> List[List[Dict]]:
    docs = group_entries_by_doc(entries_list)
    segments = []
    for _, sents in docs.items():
        cur = [sents[0]]
        for e in sents[1:]:
            if e["label"] == cur[-1]["label"]:
                cur.append(e)
            else:
                segments.append(cur)
                cur = [e]
        segments.append(cur)
    return segments

def build_chunks_from_entries(entries_list: List[Dict], max_words: int = 250) -> List[Dict]:
    segments = split_into_label_segments(entries_list)
    chunks = []
    for seg in segments:
        cur = []
        cur_words = 0
        for e in seg:
            n_words = len(e["text"].split())
            if cur and (cur_words + n_words > max_words):
                chunks.append({
                    "text": " ".join(x["text"] for x in cur),
                    "label": cur[0]["label"],
                    "n_sents": len(cur),
                    "n_words": cur_words,
                })
                cur = [e]
                cur_words = n_words
            else:
                cur.append(e)
                cur_words += n_words
        if cur:
            chunks.append({
                "text": " ".join(x["text"] for x in cur),
                "label": cur[0]["label"],
                "n_sents": len(cur),
                "n_words": cur_words,
            })
    return chunks

def build_chunks_frontier_aware(entries_list: List[Dict], base_words: int = 250, frontier_words: int = 140) -> List[Dict]:
    segments = split_into_label_segments(entries_list)
    chunks = []
    for i, seg in enumerate(segments):
        # heuristique simple: segments courts ou tout segment touchant une frontière => plus petit
        if len(seg) <= 2 or i > 0 or i < len(segments) - 1:
            local_max_words = frontier_words
        else:
            local_max_words = base_words

        cur = []
        cur_words = 0
        for e in seg:
            n_words = len(e["text"].split())
            if cur and (cur_words + n_words > local_max_words):
                chunks.append({
                    "text": " ".join(x["text"] for x in cur),
                    "label": cur[0]["label"],
                    "n_sents": len(cur),
                    "n_words": cur_words,
                })
                cur = [e]
                cur_words = n_words
            else:
                cur.append(e)
                cur_words += n_words
        if cur:
            chunks.append({
                "text": " ".join(x["text"] for x in cur),
                "label": cur[0]["label"],
                "n_sents": len(cur),
                "n_words": cur_words,
            })
    return chunks

def oversample_chunks(chunks: List[Dict], random_state: int = 42) -> List[Dict]:
    dfc = pd.DataFrame(chunks)
    grp0 = dfc[dfc["label"] == 0]
    grp1 = dfc[dfc["label"] == 1]
    if len(grp0) == len(grp1):
        return chunks
    major, minor = (grp0, grp1) if len(grp0) > len(grp1) else (grp1, grp0)
    sampled_minor = minor.sample(n=len(major), replace=True, random_state=random_state)
    out = pd.concat([major, sampled_minor], axis=0).sample(frac=1.0, random_state=random_state)
    return out.to_dict(orient="records")

def build_context_windows(entries_list: List[Dict], max_context_words: int = 250) -> List[str]:
    docs = group_entries_by_doc(entries_list)
    windows = []
    for _, sents in docs.items():
        texts = [e["text"] for e in sents]
        word_counts = [len(t.split()) for t in texts]
        for i in range(len(sents)):
            chosen = [i]
            total_words = word_counts[i]
            left = i - 1
            right = i + 1
            while True:
                added = False
                if left >= 0 and total_words + word_counts[left] <= max_context_words:
                    chosen = [left] + chosen
                    total_words += word_counts[left]
                    left -= 1
                    added = True
                if right < len(sents) and total_words + word_counts[right] <= max_context_words:
                    chosen = chosen + [right]
                    total_words += word_counts[right]
                    right += 1
                    added = True
                if not added:
                    break
            windows.append(" ".join(texts[j] for j in chosen))
    return windows

# %%
# =========================
# 5) Tokenizer / datasets / métriques
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

def make_hf_dataset(texts, labels=None):
    data = {"text": texts}
    if labels is not None:
        data["labels"] = labels.tolist() if hasattr(labels, "tolist") else labels
    ds = HFDataset.from_dict(data)
    ds = ds.map(tokenize_batch, batched=True)
    if "text" in ds.column_names:
        ds = ds.remove_columns(["text"])
    return ds

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs > 0.5).astype(int)
    return {
        "f1_macro": f1_score(labels, preds, average="macro") * 100,
        "auc": roc_auc_score(labels, probs) * 100,
        "ap": average_precision_score(labels, probs) * 100,
    }

def predict_probs_for_texts_with_trainer(trainer, tokenizer, texts, batch_size=128, max_length=512):
    hf_ds = HFDataset.from_dict({"text": texts})
    def _tok(ex):
        return tokenizer(ex["text"], padding="max_length", truncation=True, max_length=max_length)
    tok_ds = hf_ds.map(_tok, batched=True, batch_size=batch_size)
    if "text" in tok_ds.column_names:
        tok_ds = tok_ds.remove_columns(["text"])
    pred = trainer.predict(tok_ds)
    logits = pred.predictions
    probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()
    return probs

def evaluate_on_local_test(trainer, test_entries, context_words: int):
    texts = build_context_windows(test_entries, max_context_words=context_words)
    labels = np.array([e["label"] for e in test_entries])
    probs = predict_probs_for_texts_with_trainer(
        trainer, tokenizer, texts,
        batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        max_length=MAX_LENGTH
    )
    preds = (probs > 0.5).astype(int)
    results = {
        "f1_macro": f1_score(labels, preds, average="macro") * 100,
        "auc": roc_auc_score(labels, probs) * 100,
        "ap": average_precision_score(labels, probs) * 100,
        "recall_M": recall_score(labels, preds, pos_label=1) * 100,
        "precision_M": precision_score(labels, preds, pos_label=1, zero_division=0) * 100,
        "pred_M_rate": preds.mean() * 100,
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "report": classification_report(labels, preds, target_names=["Chirac", "Mitterrand"], digits=4),
    }
    return results, probs, preds, texts, labels

# %%

# =========================
# 6) Fonction générale d'expérience
# =========================

def run_experiment(
    exp_name: str,
    chunk_mode: str = "fixed",
    chunk_words: int = 250,
    context_words: int = 250,
    oversample: bool = False,
    frontier_words: int = 140,
):
    print(f"\n=== EXPERIMENT: {exp_name} ===")
    print(f"chunk_mode={chunk_mode} | chunk_words={chunk_words} | context_words={context_words} | oversample={oversample}")

    if chunk_mode == "fixed":
        train_chunks = build_chunks_from_entries(train_entries_split, max_words=chunk_words)
        val_chunks = build_chunks_from_entries(val_entries_split, max_words=chunk_words)
    elif chunk_mode == "frontier_aware":
        train_chunks = build_chunks_frontier_aware(train_entries_split, base_words=chunk_words, frontier_words=frontier_words)
        val_chunks = build_chunks_frontier_aware(val_entries_split, base_words=chunk_words, frontier_words=frontier_words)
    else:
        raise ValueError("chunk_mode inconnu")

    if oversample:
        train_chunks = oversample_chunks(train_chunks, random_state=RANDOM_STATE)

    print(f"Train chunks: {len(train_chunks):,} | Val chunks: {len(val_chunks):,}")
    print(f"Avg words/train chunk: {np.mean([c['n_words'] for c in train_chunks]):.1f}")
    print("Train labels:", pd.Series([c["label"] for c in train_chunks]).map({0:'C',1:'M'}).value_counts().to_dict())

    train_texts = [c["text"] for c in train_chunks]
    train_labels = [c["label"] for c in train_chunks]
    val_texts = [c["text"] for c in val_chunks]
    val_labels = [c["label"] for c in val_chunks]

    train_ds = make_hf_dataset(train_texts, train_labels)
    val_ds = make_hf_dataset(val_texts, val_labels)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    exp_out = os.path.join(MODEL_ROOT, exp_name)
    if os.path.exists(exp_out):
        shutil.rmtree(exp_out)
    os.makedirs(exp_out, exist_ok=True)

    args = TrainingArguments(
        output_dir=exp_out,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
        save_only_model=True,
    )

    trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    val_results = trainer.evaluate()

    local_results, probs, preds, local_texts, local_labels = evaluate_on_local_test(
        trainer=trainer,
        test_entries=test_local_entries,
        context_words=context_words,
    )

    print("\nBEST CHECKPOINT (validation chunks):")
    for k, v in val_results.items():
        if k.startswith("eval_"):
            print(f"  {k.replace('eval_', ''):>12s} : {v}")

    print(f"\n[test_local] F1_macro={local_results['f1_macro']:.2f} | AUC={local_results['auc']:.2f} | AP={local_results['ap']:.2f}")
    print(local_results["report"])

    pred_df = pd.DataFrame({
        "doc_id": [e["doc_id"] for e in test_local_entries],
        "sent_id": [e["sent_id"] for e in test_local_entries],
        "y_true": local_labels,
        "y_pred": preds,
        "prob_M": probs,
        "text_window": local_texts,
    })
    pred_path = os.path.join(OUTPUT_DIR, f"{exp_name}_local_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    result = {
        "exp_name": exp_name,
        "chunk_mode": chunk_mode,
        "chunk_words": chunk_words,
        "context_words": context_words,
        "oversample": oversample,
        "frontier_words": frontier_words if chunk_mode == "frontier_aware" else None,
        "train_chunks": len(train_chunks),
        "val_chunks": len(val_chunks),
        "avg_train_chunk_words": float(np.mean([c['n_words'] for c in train_chunks])),
        "val_f1_macro": float(val_results.get("eval_f1_macro", np.nan)),
        "val_auc": float(val_results.get("eval_auc", np.nan)),
        "val_ap": float(val_results.get("eval_ap", np.nan)),
        "test_f1_macro": float(local_results["f1_macro"]),
        "test_auc": float(local_results["auc"]),
        "test_ap": float(local_results["ap"]),
        "test_recall_M": float(local_results["recall_M"]),
        "test_precision_M": float(local_results["precision_M"]),
        "pred_path": pred_path,
        "model_dir": exp_out,
    }
    return result, trainer

# %%

# =========================
# 7) Lancer les 4 expériences
# =========================

results = []
trained = {}

res_250_250, trainer_250_250 = run_experiment(
    exp_name="train250_ctx250",
    chunk_mode="fixed",
    chunk_words=250,
    context_words=250,
    oversample=False,
)
results.append(res_250_250)
trained["train250_ctx250"] = trainer_250_250

res_350_250, trainer_350_250 = run_experiment(
    exp_name="train350_ctx250",
    chunk_mode="fixed",
    chunk_words=350,
    context_words=250,
    oversample=False,
)
results.append(res_350_250)
trained["train350_ctx250"] = trainer_350_250

res_frontier, trainer_frontier = run_experiment(
    exp_name="frontier_aware_base250_ctx250",
    chunk_mode="frontier_aware",
    chunk_words=250,
    context_words=250,
    oversample=False,
    frontier_words=140,
)
results.append(res_frontier)
trained["frontier_aware_base250_ctx250"] = trainer_frontier

res_over, trainer_over = run_experiment(
    exp_name="oversample_train250_ctx250",
    chunk_mode="fixed",
    chunk_words=250,
    context_words=250,
    oversample=True,
)
results.append(res_over)
trained["oversample_train250_ctx250"] = trainer_over

results_df = pd.DataFrame(results).sort_values("test_f1_macro", ascending=False).reset_index(drop=True)
display(results_df)
results_df.to_csv(os.path.join(OUTPUT_DIR, "experiment_comparison.csv"), index=False, encoding="utf-8")

# %%

# =========================
# 8) Sélection du meilleur modèle local
# =========================

best_row = results_df.iloc[0].to_dict()
best_exp_name = best_row["exp_name"]

print("Meilleure expérience locale :", best_exp_name)
print(best_row)

best_trainer = trained[best_exp_name]

# %% [markdown]
# ## 9) Réentraînement final sur tout le corpus learn
#
# Cette partie reprend automatiquement la **meilleure recette locale** et la réentraîne sur **tout le corpus learn**.
#
# Ensuite, si `TEST_FILE` est disponible, elle produit un CSV avec une probabilité `P(Mitterrand)` par ligne.

# %%

# =========================
# 9) Reconstruction des chunks sur tout le learn
# =========================

def build_full_train_chunks_from_best(best_row: Dict, all_entries: List[Dict]) -> List[Dict]:
    if best_row["chunk_mode"] == "fixed":
        chunks = build_chunks_from_entries(all_entries, max_words=int(best_row["chunk_words"]))
    else:
        fw = int(best_row["frontier_words"]) if best_row["frontier_words"] is not None and not pd.isna(best_row["frontier_words"]) else 140
        chunks = build_chunks_frontier_aware(all_entries, base_words=int(best_row["chunk_words"]), frontier_words=fw)
    if bool(best_row["oversample"]):
        chunks = oversample_chunks(chunks, random_state=RANDOM_STATE)
    return chunks

full_chunks = build_full_train_chunks_from_best(best_row, entries)
print("Full train chunks:", len(full_chunks))
print("Train labels:", pd.Series([c["label"] for c in full_chunks]).map({0:'C',1:'M'}).value_counts().to_dict())

full_train_texts = [c["text"] for c in full_chunks]
full_train_labels = [c["label"] for c in full_chunks]
full_train_ds = make_hf_dataset(full_train_texts, full_train_labels)

# %%

# =========================
# 10) Réentraînement final sur tout le learn
# =========================

FINAL_MODEL_DIR = os.path.join(MODEL_ROOT, f"FINAL_{best_exp_name}")
if os.path.exists(FINAL_MODEL_DIR):
    shutil.rmtree(FINAL_MODEL_DIR)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

final_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

final_args = TrainingArguments(
    output_dir=FINAL_MODEL_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
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

final_trainer.train()
final_trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

print("Modèle final sauvegardé dans :", FINAL_MODEL_DIR)

# %%

# =========================
# 11) Prédiction sur le test du prof + CSV
# =========================

if os.path.exists(TEST_FILE):
    test_entries_prof = read_test_corpus(TEST_FILE)

    # On ajoute un label dummy uniquement pour réutiliser build_context_windows
    test_entries_for_windows = [
        {"doc_id": e["doc_id"], "sent_id": e["sent_id"], "text": e["text"], "label": 0}
        for e in test_entries_prof
    ]

    best_context_words = int(best_row["context_words"])
    test_texts = build_context_windows(test_entries_for_windows, max_context_words=best_context_words)

    test_probs = predict_probs_for_texts_with_trainer(
        final_trainer,
        tokenizer,
        test_texts,
        batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        max_length=MAX_LENGTH,
    )

    submission_path = os.path.join(SUBMISSION_DIR, f"submission-pres-best-{best_exp_name}.csv")
    pd.Series(test_probs).to_csv(submission_path, index=False, header=False)
    print("Soumission sauvegardée :", submission_path)
else:
    print("TEST_FILE non trouvé. Mets le vrai chemin du test du prof puis relance cette cellule.")