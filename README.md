# BBC News Article Clustering Analysis

## Project Overview

This project explores unsupervised learning techniques to cluster BBC news articles based on their content. The primary goal is to apply and evaluate different clustering algorithms, specifically Hierarchical Agglomerative Clustering, to group articles into their respective topics (e.g., business, entertainment, politics, sport, tech).

## Dataset

The analysis uses the [BBC News Classification Dataset](http://mlg.ucd.ie/datasets/bbc.html), which contains 2225 articles, each labeled with one of five topics.

## Methods & Evaluation

1.  **Data Preprocessing:** Text data is processed, likely using techniques like TF-IDF vectorization, to prepare it for clustering.
2.  **Clustering Algorithm:** Hierarchical Agglomerative Clustering is implemented.
3.  **Hyperparameter Tuning:** The script programmatically evaluates various combinations of `linkage` methods ('complete', 'average', 'single', 'ward') and `affinity` metrics ('euclidean', 'manhattan', 'cosine') to find the best performing model.
4.  **Label Mapping:** A frequency-based approach is used to map the generated cluster labels to the ground truth topic labels.
5.  **Evaluation:** Performance is assessed using accuracy scores and confusion matrices based on the mapped labels.


## Requirements

Key Python libraries used include:
* `numpy`
* `pandas` (likely for data handling)
* `scikit-learn` (for TF-IDF, clustering, metrics)
* `matplotlib`/`seaborn` (likely for visualizations, though not explicitly in the evaluation script)
* `nltk` (potentially for text preprocessing like stop words)

## Usage

1.  Ensure you have the required libraries installed (`pip install numpy pandas scikit-learn matplotlib seaborn nltk`).
2.  Download the BBC dataset (if not included or downloaded by the script).
3.  Run the Jupyter Notebook `bbc-topics-1.ipynb` cell by cell to replicate the analysis.

