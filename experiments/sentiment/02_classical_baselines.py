# RITAL sentiment-classification experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# ## 1. Imports
#
# On utilise :
# - `pandas` pour manipuler les données ;
# - `scikit-learn` pour les vectoriseurs, les modèles et l'évaluation ;
# - `nltk` pour les **stopwords**, le **stemming** et la **lemmatisation** ;
# - `zipfile` parce que le corpus est fourni dans une archive `.zip`.
#
# ### Pourquoi ajouter `nltk` ?
# Parce qu'on veut comparer plusieurs niveaux de préprocessing :
# 1. **minimal** ;
# 2. **standard** ;
# 3. **agressif** (stopwords + stemming / lemmatisation).

# %% [markdown]
# # Classification des reviews de films — notebook de base
#
# Ce notebook construit une **première baseline solide** pour la classification de sentiments sur les critiques de films (`pos` / `neg`).
#
# ## Objectifs
# - charger les données d'entraînement ;
# - faire une petite exploration du corpus ;
# - appliquer un **préprocessing raisonnable** ;
# - entraîner plusieurs **modèles de base** ;
# - comparer les modèles avec les métriques demandées :
#   - `accuracy`
#   - `precision`
#   - `recall`
#   - `f1`
# - justifier les choix de représentation et de modèles.
#
# ## Idée générale
# Pour un premier projet TAL, on veut une approche :
# 1. **simple**
# 2. **reproductible**
# 3. **forte comme baseline**
#
# C'est pour cela qu'on commence par des pipelines `vectoriseur + classifieur` avec `scikit-learn`, qui sont souvent très performants sur les données textuelles.

# %%

import os
import re
import zipfile
import unicodedata
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import wordpunct_tokenize

for resource in [
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# %% [markdown]
# ## 2. Chargement des données
#
# Le corpus `movies1000.zip` contient deux sous-dossiers :
# - `pos/` : reviews positives
# - `neg/` : reviews négatives
#
# Chaque fichier texte correspond à une review.
#
# ### Choix
# On conserve ici le texte brut au chargement.  
# Le préprocessing sera géré **ensuite dans des fonctions séparées**, ce qui permet de comparer plusieurs variantes sans recharger les données.

# %%
from pathlib import Path
import pandas as pd

DATA_DIR = Path("/content/drive/MyDrive/projet tal/movies1000")

def load_movies_from_folder(data_dir: Path) -> pd.DataFrame:
    rows = []

    for label_folder, label in [("movies1000/pos", "P"), ("movies1000/neg", "N")]:
        folder = data_dir / label_folder
        for file_path in folder.glob("*.txt"):
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            rows.append(
                {
                    "doc_id": file_path.name,
                    "label": label,
                    "text": text,
                }
            )

    df = pd.DataFrame(rows).sort_values("doc_id").reset_index(drop=True)
    return df

df = load_movies_from_folder(DATA_DIR)
print(df.shape)
df.head()

# %%
# NOTEBOOK-ONLY COMMAND REMOVED: from google.colab import drive
# NOTEBOOK-ONLY COMMAND REMOVED: drive.mount('/content/drive')

# %% [markdown]
# ## 3. Exploration rapide du corpus
#
# Avant de lancer les modèles, il faut regarder un minimum les données :
# - taille du corpus ;
# - équilibre des classes ;
# - longueur des documents.
#
# ### Pourquoi ?
# Parce que cela permet de vérifier si le dataset est équilibré.  
# Si une classe domine fortement, l'accuracy peut être trompeuse et il faut alors :
# - utiliser des métriques comme le **F1 macro** ;
# - éventuellement tester `class_weight='balanced'`.
#
# Ici on ajoute donc une **vérification explicite du déséquilibre** avant l'entraînement.

# %%

print("Nombre total de documents :", len(df))
print("\nRépartition des classes :")
class_counts = df["label"].value_counts().sort_index()
print(class_counts)

minority = class_counts.min()
majority = class_counts.max()
imbalance_ratio = majority / minority if minority > 0 else np.inf

print(f"\nTaille classe majoritaire : {majority}")
print(f"Taille classe minoritaire : {minority}")
print(f"Ratio de déséquilibre majoritaire/minoritaire : {imbalance_ratio:.3f}")

if imbalance_ratio <= 1.2:
    print("=> Les classes paraissent bien équilibrées. Pas de correction immédiate nécessaire.")
elif imbalance_ratio <= 1.5:
    print("=> Léger déséquilibre. On surveille surtout precision / recall / f1_macro.")
else:
    print("=> Déséquilibre notable. Il faudra envisager un rééquilibrage ou class_weight='balanced'.")

df["n_chars"] = df["text"].str.len()
df["n_words"] = df["text"].str.split().str.len()

display(df.groupby("label")[["n_chars", "n_words"]].agg(["mean", "median", "std", "min", "max"]))
display(df[["n_chars", "n_words"]].describe())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["label"].value_counts().plot(kind="bar", ax=axes[0], title="Répartition des classes")
axes[0].set_xlabel("Classe")
axes[0].set_ylabel("Nombre de documents")

df.boxplot(column="n_words", by="label", ax=axes[1])
axes[1].set_title("Longueur (nombre de mots) par classe")
axes[1].set_ylabel("Nombre de mots")

plt.suptitle("")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Exemple de documents
#
# Toujours utile pour vérifier :
# - la langue du corpus ;
# - la présence de ponctuation ;
# - la forme générale du texte ;
# - les artefacts éventuels.

# %%

for label in ["P", "N"]:
    example = df[df["label"] == label].iloc[0]
    print(f"\n===== Exemple classe {label} | {example['doc_id']} =====")
    print(example["text"][:700], "...")

# %% [markdown]
# ## 5. Préprocessing
#
# Ici on définit maintenant **trois niveaux principaux** de préprocessing.
#
# ### V1. Préprocessing minimal
# - passage en minuscules ;
# - normalisation des espaces.
#
# **Idée :** garder presque toute l'information.
#
# ### V2. Préprocessing standard
# - minuscules ;
# - suppression des accents ;
# - suppression des URLs ;
# - suppression de la ponctuation et des chiffres ;
# - normalisation des espaces.
#
# **Idée :** enlever une partie du bruit tout en conservant l'essentiel.
#
# ### V3. Préprocessing agressif
# - basé sur la version standard ;
# - tokenisation ;
# - suppression des stopwords ;
# - suppression des tokens très courts ;
# - réduction morphologique.
#
# Comme la réduction morphologique peut être faite de deux manières, on teste ici **deux sous-versions** :
# - **V3-stem** : stemming avec `PorterStemmer`
# - **V3-lemma** : lemmatisation avec `WordNetLemmatizer`
#
# ### Remarque sur les mots rares / trop fréquents
# On ne les retire pas directement dans la fonction de préprocessing.  
# On les filtre plutôt au niveau du vectoriseur avec `min_df` et `max_df`, ce qui est la manière la plus propre dans sklearn.

# %%

def preprocess_minimal(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def strip_accents(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", errors="ignore")

def preprocess_standard(text: str) -> str:
    text = text.lower()
    text = strip_accents(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN

def preprocess_aggressive_stem(text: str) -> str:
    text = preprocess_standard(text)
    tokens = wordpunct_tokenize(text)
    tokens = [
        tok for tok in tokens
        if tok.isalpha() and tok not in STOP_WORDS and len(tok) > 2
    ]
    tokens = [stemmer.stem(tok) for tok in tokens]
    return " ".join(tokens)

def preprocess_aggressive_lemma(text: str) -> str:
    text = preprocess_standard(text)
    tokens = wordpunct_tokenize(text)
    tokens = [
        tok for tok in tokens
        if tok.isalpha() and tok not in STOP_WORDS and len(tok) > 2
    ]
    tagged = pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(tok, get_wordnet_pos(tag)) for tok, tag in tagged]
    return " ".join(lemmas)

df["text_minimal"] = df["text"].apply(preprocess_minimal)
df["text_standard"] = df["text"].apply(preprocess_standard)
df["text_aggr_stem"] = df["text"].apply(preprocess_aggressive_stem)
df["text_aggr_lemma"] = df["text"].apply(preprocess_aggressive_lemma)

preview_cols = ["text", "text_minimal", "text_standard", "text_aggr_stem", "text_aggr_lemma"]
display(df[preview_cols].head(3))

# %% [markdown]
# ## 6. Découpage train / validation
#
# On garde une partie des données pour une **évaluation finale locale**.  
# Le reste servira à la validation croisée pendant la comparaison des modèles.
#
# ### Pourquoi `stratify` ?
# On utilise `stratify=y` pour conserver la même proportion de classes dans `train` et `validation`.  
# C'est particulièrement important dès qu'on veut éviter qu'un éventuel déséquilibre vienne biaiser l'évaluation.

# %%

X_train_raw, X_valid_raw, y_train, y_valid = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=df["label"],
)

X_train_min, X_valid_min, _, _ = train_test_split(
    df["text_minimal"],
    df["label"],
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=df["label"],
)

X_train_std, X_valid_std, _, _ = train_test_split(
    df["text_standard"],
    df["label"],
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=df["label"],
)

X_train_stem, X_valid_stem, _, _ = train_test_split(
    df["text_aggr_stem"],
    df["label"],
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=df["label"],
)

X_train_lemma, X_valid_lemma, _, _ = train_test_split(
    df["text_aggr_lemma"],
    df["label"],
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=df["label"],
)

# %% [markdown]
# ## 7. Pipelines de base
#
# On teste plusieurs baselines principales.
#
# ### 1. CountVectorizer + MultinomialNB
# **Justification :**
# - baseline classique en classification de texte ;
# - très rapide ;
# - simple à interpréter.
#
# ### 2. TF-IDF + LogisticRegression
# **Justification :**
# - excellente baseline pour le texte ;
# - robuste ;
# - compatible avec des vecteurs creux de grande dimension.
#
# ### 3. TF-IDF + LinearSVC
# **Justification :**
# - souvent très performant en classification textuelle ;
# - bien adapté aux espaces de grande dimension.
#
# ### Filtrage des mots rares / trop fréquents
# On l'intègre directement dans le vectoriseur avec :
# - `min_df=2` : ignore les termes trop rares ;
# - `max_df=0.95` : ignore les termes présents dans presque tous les documents.
#
# Cela permet de limiter le bruit sans casser le pipeline.

# %%

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

pipelines = {
    "NB_count_unigram": Pipeline([
        ("vect", CountVectorizer(lowercase=False, ngram_range=(1, 1), min_df=2, max_df=0.95)),
        ("clf", MultinomialNB()),
    ]),
    "LogReg_tfidf_12": Pipeline([
        ("vect", TfidfVectorizer(lowercase=False, ngram_range=(1, 2), min_df=2, max_df=0.95)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ]),
    "LinearSVC_tfidf_12": Pipeline([
        ("vect", TfidfVectorizer(lowercase=False, ngram_range=(1, 2), min_df=2, max_df=0.95)),
        ("clf", LinearSVC(random_state=RANDOM_STATE)),
    ]),
    "LogReg_tfidf_12_balanced": Pipeline([
        ("vect", TfidfVectorizer(lowercase=False, ngram_range=(1, 2), min_df=2, max_df=0.95)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ]),
}

# %% [markdown]
# ## 8. Évaluation par validation croisée
#
# On compare d'abord les modèles sur les textes **prétraités**.
#
# ### Pourquoi `macro` pour precision / recall / f1 ?
# La version `macro` donne le même poids à chaque classe.  
# C'est utile pour éviter de se focaliser seulement sur la classe majoritaire, même si ici le corpus est presque équilibré.

# %%
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, pos_label="P"),
    "recall": make_scorer(recall_score, pos_label="P"),
    "f1": make_scorer(f1_score, pos_label="P"),
}

def evaluate_cv(model_dict, X, y, cv, scoring):
    rows = []
    for name, model in model_dict.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"model": name}
        for metric_name in scoring.keys():
            values = scores[f"test_{metric_name}"]
            row[f"{metric_name}_mean"] = values.mean()
            row[f"{metric_name}_std"] = values.std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)

# %% [markdown]
# ## 9. Comparaison : brut vs V1 vs V2 vs V3
#
# On veut vérifier si le préprocessing aide réellement.
#
# ### Versions comparées
# - `raw` : texte tel quel ;
# - `v1_minimal` : minuscules + espaces ;
# - `v2_standard` : nettoyage plus classique ;
# - `v3_stem` : stopwords + stemming ;
# - `v3_lemma` : stopwords + lemmatisation.
#
# ### Important
# Le préprocessing agressif n'est pas toujours meilleur en sentiment analysis.  
# Il peut réduire le bruit, mais aussi supprimer des indices utiles.  
# On le teste donc expérimentalement au lieu de l'imposer.

# %%

comparison_models = {
    "LogReg_tfidf_12": pipelines["LogReg_tfidf_12"],
    "LinearSVC_tfidf_12": pipelines["LinearSVC_tfidf_12"],
    "LogReg_tfidf_12_balanced": pipelines["LogReg_tfidf_12_balanced"],
}

results_cv_raw = evaluate_cv(comparison_models, df["text"], df["label"], cv, scoring)
results_cv_raw["text_version"] = "raw"

results_cv_min = evaluate_cv(comparison_models, df["text_minimal"], df["label"], cv, scoring)
results_cv_min["text_version"] = "v1_minimal"

results_cv_std = evaluate_cv(comparison_models, df["text_standard"], df["label"], cv, scoring)
results_cv_std["text_version"] = "v2_standard"

results_cv_stem = evaluate_cv(comparison_models, df["text_aggr_stem"], df["label"], cv, scoring)
results_cv_stem["text_version"] = "v3_stem"

results_cv_lemma = evaluate_cv(comparison_models, df["text_aggr_lemma"], df["label"], cv, scoring)
results_cv_lemma["text_version"] = "v3_lemma"

results_cv_all = pd.concat(
    [results_cv_raw, results_cv_min, results_cv_std, results_cv_stem, results_cv_lemma],
    ignore_index=True
).sort_values(["f1_mean", "accuracy_mean"], ascending=False)

display(results_cv_all)

# %% [markdown]
# ## 10. Choix du meilleur pipeline
#
# On choisit le meilleur couple :
# - **version de texte**
# - **pipeline**
# sur la base du `f1_macro` moyen en validation croisée.
#
# ### Pourquoi ?
# Parce que le meilleur modèle dépend souvent du niveau de préprocessing.  
# Ici, on ne suppose pas à l'avance que V3 sera meilleure que V1 ou V2.

# %%

best_row = results_cv_all.iloc[0]
best_model_name = best_row["model"]
best_text_version = best_row["text_version"]
best_pipeline = pipelines[best_model_name]

print("Meilleur pipeline :", best_model_name)
print("Meilleure version du texte :", best_text_version)
display(best_row.to_frame().T)

version_to_train_valid = {
    "raw": (X_train_raw, X_valid_raw),
    "v1_minimal": (X_train_min, X_valid_min),
    "v2_standard": (X_train_std, X_valid_std),
    "v3_stem": (X_train_stem, X_valid_stem),
    "v3_lemma": (X_train_lemma, X_valid_lemma),
}

X_train_best, X_valid_best = version_to_train_valid[best_text_version]

# %% [markdown]
# ## 11. Entraînement du meilleur modèle et évaluation sur le jeu de validation local
#
# On entraîne maintenant le meilleur pipeline sur `train`, puis on évalue sur `valid`.
#
# Les métriques affichées sont :
# - accuracy ;
# - precision_macro ;
# - recall_macro ;
# - f1_macro.
#
# On garde `macro` pour donner le même poids aux deux classes, ce qui reste une bonne pratique même si le corpus semble équilibré.

# %%

best_pipeline.fit(X_train_best, y_train)
y_pred = best_pipeline.predict(X_valid_best)

metrics_local = {
    "accuracy": accuracy_score(y_valid, y_pred),
    "precision_macro": precision_score(y_valid, y_pred, average="macro"),
    "recall_macro": recall_score(y_valid, y_pred, average="macro"),
    "f1_macro": f1_score(y_valid, y_pred, average="macro"),
}

pd.DataFrame([metrics_local]).T.rename(columns={0: "score"})

# %%

print(classification_report(y_valid, y_pred, digits=4))

cm = confusion_matrix(y_valid, y_pred, labels=["P", "N"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["P", "N"])
disp.plot()
plt.show()

# %% [markdown]
# ## 12. Analyse rapide des erreurs
#
# Cette partie est utile dans un vrai projet, parce qu'elle permet de voir :
# - quels documents sont mal classés ;
# - si les erreurs viennent de la négation ;
# - si certains textes sont ambigus ou ironiques ;
# - si le préprocessing a supprimé trop d'information.

# %%

valid_df = pd.DataFrame({
    "text": X_valid_best.values,
    "true_label": y_valid.values,
    "pred_label": y_pred,
})

errors = valid_df[valid_df["true_label"] != valid_df["pred_label"]].copy()
print("Nombre d'erreurs :", len(errors))
display(errors.head(10))

# %% [markdown]
# ## 13. Conclusion intermédiaire
#
# ### Ce que montre ce notebook
# - un pipeline BoW / TF-IDF est déjà une très bonne base ;
# - `MultinomialNB` sert de baseline simple ;
# - `LogisticRegression` et `LinearSVC` sont des choix solides pour des représentations creuses ;
# - les **classes ont été vérifiées explicitement** avant l'entraînement ;
# - le **préprocessing doit être testé**, pas imposé ;
# - la version agressive (stopwords + stemming / lemmatisation) peut aider, mais elle peut aussi supprimer des indices utiles au sentiment.
#
# ### À retenir
# Le meilleur choix est donc empirique : on garde la combinaison qui donne les meilleurs scores de validation, pas celle qui paraît la plus "propre" en théorie.

# %% [markdown]
# ## 14. Cellule utile pour une future soumission
#
# Quand le fichier de test sera disponible, il suffira de :
# 1. charger le test ;
# 2. appliquer **la même version de préprocessing que celle retenue** ;
# 3. prédire ;
# 4. sauvegarder un CSV avec une colonne de labels (`P` / `N`).
#
# La cellule ci-dessous permet de choisir explicitement la fonction de préprocessing.

# %%

PREPROCESSORS = {
    "raw": lambda x: x,
    "v1_minimal": preprocess_minimal,
    "v2_standard": preprocess_standard,
    "v3_stem": preprocess_aggressive_stem,
    "v3_lemma": preprocess_aggressive_lemma,
}

def build_submission(model, texts, text_version="v2_standard", out_path="submission-movie.csv"):
    preprocess_fn = PREPROCESSORS[text_version]
    texts_proc = pd.Series(texts).apply(preprocess_fn)
    preds = model.predict(texts_proc)
    sub = pd.DataFrame({"label": preds})
    sub.to_csv(out_path, index=False)
    return sub

# Exemple futur :
# test_texts = [...]
# submission = build_submission(best_pipeline, test_texts, text_version=best_text_version, out_path="submission-movie.csv")
# submission.head()

# %% [markdown]
# ##**TEST**

# %%
from pathlib import Path
import pandas as pd

TEST_FILE = Path("/content/drive/MyDrive/projet tal/testSentiment.txt")

def load_test_one_review_per_line(test_file: Path) -> pd.DataFrame:
    lines = test_file.read_text(encoding="utf-8", errors="ignore").splitlines()
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
import pandas as pd

test_df["text_standard"] = test_df["text"].apply(preprocess_standard)

best_model = pipelines["LinearSVC_tfidf_12"]
best_model.fit(df["text_standard"], df["label"])

test_pred = best_model.predict(test_df["text_standard"])

print("Nombre de prédictions :", len(test_pred))
print("\nRépartition des classes prédites :")
print(pd.Series(test_pred).value_counts())

submission = pd.DataFrame({
    "label": test_pred
})

display(submission.head(10))

submission.to_csv("/content/drive/MyDrive/projet tal/submission-movie-test.csv", index=False)
print("\nFichier enregistré : /content/drive/MyDrive/projet tal/submission-movie-test.csv")

# %% [markdown]
# Amelioration **A**

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

svm_word_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LinearSVC())
])

param_grid_word = {
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "tfidf__min_df": [1, 2, 3],
    "tfidf__max_df": [0.95, 1.0],
    "tfidf__sublinear_tf": [False, True],
    "clf__C": [0.5, 1.0, 2.0],
}

grid_word = GridSearchCV(
    estimator=svm_word_pipeline,
    param_grid=param_grid_word,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_word.fit(df["text_standard"], df["label"])

print("Best word-level score:", grid_word.best_score_)
print("Best word-level params:", grid_word.best_params_)

# %% [markdown]
# **B**

# %%
svm_char_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char")),
    ("clf", LinearSVC())
])

param_grid_char = {
    "tfidf__ngram_range": [(3, 5), (4, 6), (3, 6)],
    "tfidf__min_df": [1, 2],
    "tfidf__sublinear_tf": [False, True],
    "clf__C": [0.5, 1.0, 2.0],
}

grid_char = GridSearchCV(
    estimator=svm_char_pipeline,
    param_grid=param_grid_char,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_char.fit(df["text_standard"], df["label"])

print("Best char-level score:", grid_char.best_score_)
print("Best char-level params:", grid_char.best_params_)

# %% [markdown]
# LSA / SVD **tronqué**
