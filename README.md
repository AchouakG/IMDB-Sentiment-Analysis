# IMDB-Sentiment-Analysis

An MLP binary sentiment classifier for **IMDB movie reviews**,  **implemented using NumPy**.
The pipeline builds sentiment features (VADER + TextBlob), trains a small neural network, and evaluates performance using accuracy, confusion matrix, and other metrics.

## Features used
The model is trained on 6 numeric features:
- VADER: `neg`, `neu`, `pos`, `compound`
- TextBlob: `tb_polarity`, `tb_subjectivity`

## Model
- Input layer: `d_in = number of features`
- One hidden layer: `n_neurons = 16` + ReLU
- Output: 1 neuron + Sigmoid
- Loss: Binary Cross Entropy
- Optimization: mini-batch gradient descent

## Project structure

```
IMDB-Sentiment-Analysis/
├── model/
│ ├── mlp.py # MLPBinary implementation (NumPy)
│ └── train.py # training loop (mini-batch)
├── notebooks/
│ ├── EDA.ipynb
│ ├── feature_eng.ipynb
│ └── model_test.ipynb
├── results/
│ ├── confusion_matrix.png
│ ├── loss_curve.png
│ ├── metrics.txt
│ └── result_discussion.txt
└── README.md

````



## Data

This project uses the **IMDB dataset from Kaggle**.


## Results
Outputs are saved in `results/`:
- `metrics.txt`: accuracy, precision, recall, F1, etc.
- `confusion_matrix.png`: confusion matrix plot.
- `loss_curve.png`: training vs validation loss curve.
- `result_discussion.txt`: short discussion of results.






