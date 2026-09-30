# NLP Text Classification — From Classical ML to Transformers

Comparative NLP project covering two supervised text-classification problems and a progression from **TF-IDF/SVM baselines to Transformer-based systems**.

The work was completed for the M1 MIND RITAL course at Sorbonne Université and is organized around two complementary tasks:

1. **Movie-review sentiment classification** on `movies1000`.
2. **Sentence authorship classification** between Jacques Chirac and François Mitterrand, with document-level context and class imbalance.

The repository emphasizes the experimental path: strong classical baselines, model comparison, error analysis, adaptation to long documents and context-aware Transformer modeling.

## 1. Sentiment classification

The movie corpus contains 2,000 balanced reviews (1,000 positive / 1,000 negative), with highly variable review lengths. This made truncation a central modeling issue for Transformers.

Experiments include:

- TF-IDF + Naive Bayes / Logistic Regression / Linear SVM.
- SVM tuning with word/character n-grams and LSA variants.
- FastText, DistilBERT and BERT.
- XLM-R / Twitter-RoBERTa variants.
- **Head+tail** input strategies for long reviews.
- Ensembles combining the strongest classical and Transformer models.

### Best retained sentiment system

The final ensemble combines an optimized SVM with two head+tail RoBERTa variants:

- **Validation F1: 0.9653**
- **Platform F1: 93.077**

The strongest individual head+tail RoBERTa configuration reached validation F1 0.9435 and platform F1 92.802.

## 2. Contextual sentence authorship classification

The second task predicts whether a sentence was written/spoken by Chirac or Mitterrand. The dataset is imbalanced and contains mixed documents where speaker transitions make isolated sentence classification difficult.

The experiments therefore move from phrase-level modeling to:

- split **by document** to limit leakage,
- CamemBERT-large,
- fixed training chunks,
- local context windows,
- weighted loss and early stopping,
- analysis of speaker-boundary regions,
- detailed error analysis.

The report identifies a **250-word training chunk + 250-word context** strategy as the best overall local compromise. One retained local experiment reached **macro-F1 0.9003**; the best platform submission reported at project time reached **F1 86.895**.

## Repository structure

```text
.
├── experiments/
│   ├── sentiment/
│   │   ├── 01_eda.py
│   │   ├── 02_classical_baselines.py
│   │   ├── 03_svm_improvements.py
│   │   ├── 04_fasttext_distilbert_bert.py
│   │   ├── 05_specialized_transformers.py
│   │   ├── 06_long_document_head_tail.py
│   │   └── 07_final_ensemble.py
│   └── authorship/
│       ├── 01_camembert_baseline.py
│       ├── 02_weighted_loss_early_stopping.py
│       ├── 03_context_and_boundaries.py
│       ├── 04_error_analysis.py
│       ├── 05_final_comparison.py
│       └── 06_retrain_best_250_250.py
├── docs/
│   ├── RESULTS.md
│   └── REPRODUCIBILITY.md
├── requirements.txt
└── README.md
```

The experiment files are clean text exports of the original Jupyter notebooks. Outputs were removed from version control; the verified final metrics are summarized in `docs/RESULTS.md`.

## Key engineering/research lessons

- A well-tuned **Linear SVM** is a strong NLP baseline and remained useful inside the final ensemble.
- For long reviews, model choice alone was insufficient: the **input construction strategy** materially affected performance.
- For authorship classification, respecting document structure and local context mattered more than simply increasing model complexity.
- Class imbalance requires looking beyond accuracy; minority-class precision/recall and F1 changed materially across configurations.
- Error analysis around mixed-document boundaries directly informed the next experiments.

## Limitations

- The project uses course/platform datasets that are not redistributed in this repository.
- Platform scores are specific to the academic evaluation setting and should not be generalized to unrelated corpora.
- The authorship task is a text-classification benchmark; the repository makes no political or biographical claims about the named public figures.
- Several experiments were executed in Colab and contain path/configuration assumptions that require local adaptation.

## Academic context

**Sorbonne Université — Master 1 MIND, RITAL, 2025–2026**

Authors:
- **Wafaa Berrais**
- **Zineddine Mohammedi**

This repository presents the joint academic project and does not invent an undocumented individual contribution split.
