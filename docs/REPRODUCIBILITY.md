# Reproducibility notes

The original experiments were developed in Google Colab/Jupyter. This repository stores clean `# %%` script exports of the selected notebooks for readable version control.

## Data

The course datasets and evaluation-platform files are not redistributed. Configure the dataset paths in each experiment before running.

## Sentiment sequence

1. EDA
2. Classical baselines
3. SVM improvements
4. FastText / DistilBERT / BERT
5. Specialized Transformer models
6. Long-document head+tail experiments
7. Final ensemble

## Authorship sequence

1. CamemBERT baseline / corrected protocol
2. Weighted loss + early stopping
3. Context chunks and boundary analysis
4. Error analysis
5. Final model comparison
6. Retraining of the retained 250/250 configuration

## Compute

Transformer experiments require substantially more memory/compute than the classical baselines. GPU execution is recommended for the CamemBERT/RoBERTa training notebooks.
