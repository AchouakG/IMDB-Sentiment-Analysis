import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from model.mlp import MLPBinary


def train(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, seed=0):
    rand_generator = np.random.default_rng(seed) #  RNG to shuffle training indices each epoch
    loss = {"train_loss": [], "val_loss": [], "val_acc": []}
    n = X_train.shape[0] # training samples (rows)

    for ep in range(1, epochs + 1):
        idx = rand_generator.permutation(n)
        Xs, ys = X_train[idx], y_train[idx]

        for i in range(0, n, batch_size):
            xb = Xs[i:i+batch_size] # current mini-batch
            yb = ys[i:i+batch_size]
            p, cache = model.forward(xb) # predicted probabilities
            model.step(cache, yb)

        p_train = model.predict_probability(X_train)
        p_val   = model.predict_probability(X_val)

        train_loss = model.binary_cross_entropy(y_train, p_train)
        val_loss   = model.binary_cross_entropy(y_val, p_val)

        val_pred = (p_val >= 0.5).astype(np.int32) # convert probabilities to class labels
        val_acc  = float((val_pred == y_val).mean()) # % correct on validation set

        loss["train_loss"].append(train_loss)
        loss["val_loss"].append(val_loss)
        loss["val_acc"].append(val_acc)

        print(f"epoch {ep:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    return loss