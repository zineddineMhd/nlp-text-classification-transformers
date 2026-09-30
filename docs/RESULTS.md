# Verified results

## Sentiment classification

The retained progression moved from classical TF-IDF models to specialized Transformers and ensembles.

Key verified values from the final report:

- Strongest individual head+tail RoBERTa: **validation F1 0.9435**, **platform F1 92.802**.
- Final ensemble (SVM + two head+tail RoBERTa variants): **validation F1 0.9653**, **platform F1 93.077**.

The final report attributes part of the gain to better handling of long reviews rather than only to a larger model.

## Sentence authorship classification

The main retained family uses CamemBERT-large with document-aware training/evaluation and contextual chunks.

A retained local configuration reached:

- **Macro-F1: 0.9003** on the local test split.
- Mitterrand-class F1: **0.8261** in that experiment.

At report time, the best observed academic-platform submission reached:

- **F1: 86.895**
- Precision: 78.066
- Recall: 96.759

The report retains the 250-word chunk / 250-word context configuration as the strongest overall compromise rather than choosing solely by precision.
