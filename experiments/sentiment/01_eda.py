# RITAL sentiment-classification experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
#
# # EDA — Classification des sentiments (movies1000)
#
# Ce notebook réalise une analyse exploratoire simple et utile pour le rapport :
#
# - structure du corpus
# - répartition des classes
# - longueur des reviews
# - comparaison des longueurs selon le sentiment
# - distribution des longueurs
# - mots fréquents par classe
# - exemples de reviews courts et longs
#
# Les graphiques produits peuvent être réutilisés dans le rapport.

# %%
# NOTEBOOK-ONLY COMMAND REMOVED: from google.colab import drive
# NOTEBOOK-ONLY COMMAND REMOVED: drive.mount('/content/drive')

# %%

# Si besoin sur Colab, décommente :
# NOTEBOOK: !pip install -q pandas matplotlib scikit-learn wordcloud

# %%

from pathlib import Path
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud

# %% [markdown]
# ## 1. Configuration

# %%

DATA_DIR = Path("/content/drive/MyDrive/projet tal/movies1000/movies1000")
OUTPUT_DIR = Path("/content/drive/MyDrive/projet tal/eda_sentiment_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("DATA_DIR existe :", DATA_DIR.exists())
print("OUTPUT_DIR :", OUTPUT_DIR)

# %% [markdown]
# ## 2. Chargement des données

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
print("Taille du corpus :", df.shape)
display(df.head())

# %% [markdown]
# ## 3. Vérification rapide du corpus

# %%

print("Colonnes :", df.columns.tolist())
print("Labels uniques :", sorted(df["label"].unique()))
print("Nombre de documents par classe :")
display(df["label"].value_counts().rename_axis("label").reset_index(name="count"))

# %% [markdown]
# ## 4. Features simples : longueur des reviews

# %%

def count_words(text: str) -> int:
    return len(text.split())

def simple_tokenize(text: str):
    return re.findall(r"[A-Za-z']+", text.lower())

df["n_chars"] = df["text"].str.len()
df["n_words"] = df["text"].apply(count_words)
df["n_lines"] = df["text"].str.count(r"\n") + 1
df["tokens_simple"] = df["text"].apply(simple_tokenize)
df["n_tokens_simple"] = df["tokens_simple"].apply(len)

summary_df = pd.DataFrame({
    "stat": ["n_docs", "avg_chars", "median_chars", "avg_words", "median_words", "avg_lines", "median_lines"],
    "value": [
        len(df),
        df["n_chars"].mean(),
        df["n_chars"].median(),
        df["n_words"].mean(),
        df["n_words"].median(),
        df["n_lines"].mean(),
        df["n_lines"].median(),
    ]
})
display(summary_df)

# %% [markdown]
# ## 5. Répartition des classes

# %%

class_counts = df["label"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
class_counts.plot(kind="bar")
plt.title("Répartition des classes")
plt.xlabel("Label")
plt.ylabel("Nombre de documents")
plt.tight_layout()
plt.show()

class_counts_df = class_counts.rename_axis("label").reset_index(name="count")
display(class_counts_df)

# %% [markdown]
# ## 6. Statistiques de longueur par classe

# %%

length_by_class = df.groupby("label")[["n_chars", "n_words", "n_lines", "n_tokens_simple"]].agg(["mean", "median", "min", "max"])
display(length_by_class)

# %% [markdown]
# ## 7. Histogramme du nombre de mots

# %%

plt.figure(figsize=(8, 5))
plt.hist(df["n_words"], bins=30)
plt.title("Distribution du nombre de mots par review")
plt.xlabel("Nombre de mots")
plt.ylabel("Nombre de reviews")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Histogrammes par classe

# %%

plt.figure(figsize=(8, 5))
plt.hist(df[df["label"] == "P"]["n_words"], bins=30, alpha=0.7, label="Positif")
plt.hist(df[df["label"] == "N"]["n_words"], bins=30, alpha=0.7, label="Négatif")
plt.title("Distribution du nombre de mots selon la classe")
plt.xlabel("Nombre de mots")
plt.ylabel("Nombre de reviews")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Boxplot des longueurs

# %%

data_box = [
    df[df["label"] == "P"]["n_words"].values,
    df[df["label"] == "N"]["n_words"].values,
]

plt.figure(figsize=(6, 5))
plt.boxplot(data_box, tick_labels=["Positif", "Négatif"])
plt.title("Boxplot du nombre de mots par classe")
plt.ylabel("Nombre de mots")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Quantiles de longueur

# %%

quantiles = df["n_words"].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).rename("n_words_quantile")
display(quantiles.to_frame())

# %% [markdown]
# ## 11. Reviews les plus courtes et les plus longues

# %%

short_reviews = df.nsmallest(5, "n_words")[["doc_id", "label", "n_words", "text"]]
long_reviews = df.nlargest(5, "n_words")[["doc_id", "label", "n_words", "text"]]

print("Reviews les plus courtes :")
display(short_reviews)

print("Reviews les plus longues :")
display(long_reviews)

# %% [markdown]
# ## 12. Mots fréquents par classe

# %%

stopwords = set(ENGLISH_STOP_WORDS)

def get_top_words(texts, top_k=30):
    counter = Counter()
    for text in texts:
        tokens = re.findall(r"[A-Za-z']+", text.lower())
        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
        counter.update(tokens)
    return pd.DataFrame(counter.most_common(top_k), columns=["word", "count"])

top_pos = get_top_words(df[df["label"] == "P"]["text"], top_k=20)
top_neg = get_top_words(df[df["label"] == "N"]["text"], top_k=20)

print("Top mots - Positif")
display(top_pos)

print("Top mots - Négatif")
display(top_neg)

# %% [markdown]
# ## 13. Barplots des mots fréquents

# %%

plt.figure(figsize=(8, 6))
plt.barh(top_pos["word"][::-1], top_pos["count"][::-1])
plt.title("Mots fréquents dans les reviews positives")
plt.xlabel("Fréquence")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.barh(top_neg["word"][::-1], top_neg["count"][::-1])
plt.title("Mots fréquents dans les reviews négatives")
plt.xlabel("Fréquence")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 14. Nuages de mots

# %%

pos_text = " ".join(df[df["label"] == "P"]["text"].tolist())
neg_text = " ".join(df[df["label"] == "N"]["text"].tolist())

wc_pos = WordCloud(width=1200, height=600, background_color="white", stopwords=stopwords).generate(pos_text)
wc_neg = WordCloud(width=1200, height=600, background_color="white", stopwords=stopwords).generate(neg_text)

plt.figure(figsize=(12, 5))
plt.imshow(wc_pos)
plt.axis("off")
plt.title("Nuage de mots - Reviews positives")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.imshow(wc_neg)
plt.axis("off")
plt.title("Nuage de mots - Reviews négatives")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 15. Tableau récapitulatif pour le rapport

# %%

report_table = pd.DataFrame([
    {"measure": "Nombre total de documents", "value": len(df)},
    {"measure": "Nombre de documents positifs", "value": int((df['label'] == 'P').sum())},
    {"measure": "Nombre de documents négatifs", "value": int((df['label'] == 'N').sum())},
    {"measure": "Longueur moyenne (mots)", "value": round(df["n_words"].mean(), 2)},
    {"measure": "Longueur médiane (mots)", "value": round(df["n_words"].median(), 2)},
    {"measure": "Longueur max (mots)", "value": int(df["n_words"].max())},
    {"measure": "Longueur min (mots)", "value": int(df["n_words"].min())},
])

display(report_table)

# %% [markdown]
# ## 16. Export optionnel des tableaux

# %%

report_table.to_csv(OUTPUT_DIR / "report_table_summary.csv", index=False)
top_pos.to_csv(OUTPUT_DIR / "top_words_positive.csv", index=False)
top_neg.to_csv(OUTPUT_DIR / "top_words_negative.csv", index=False)
class_counts_df.to_csv(OUTPUT_DIR / "class_counts.csv", index=False)

print("Tableaux exportés dans :", OUTPUT_DIR)

# %% [markdown]
# ## 17. Commentaire final

# %%

print("Commentaires possibles pour le rapport :")
print("- Le corpus est équilibré entre les classes positives et négatives.")
print("- Les reviews présentent une longueur variable, avec certaines critiques très longues.")
print("- Cette variabilité de longueur peut compliquer l'utilisation de modèles Transformers limités en nombre de tokens.")
print("- L'analyse lexicale montre des tendances distinctes entre les reviews positives et négatives.")
