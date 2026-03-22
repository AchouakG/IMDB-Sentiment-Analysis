import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import os
from sklearn.metrics import classification_report, confusion_matrix

def train(model, X_train, y_train, X_val, y_val,
        epochs=50, batch_size=64, seed=0, freeze_emb_epochs=0):

    rand_generator = np.random.default_rng(seed)
    loss = {"train_loss": [], "val_loss": [], "val_acc": []}
    n = X_train.shape[0]

    for ep in range(1, epochs + 1):
        idx = rand_generator.permutation(n)
        Xs, ys = X_train[idx], y_train[idx]

        for i in range(0, n, batch_size):
            xb = Xs[i:i+batch_size]
            yb = ys[i:i+batch_size]
            p, cache = model.forward(xb)

            if hasattr(model, 'E') and ep <= freeze_emb_epochs:
                model.step(cache, yb, freeze_embeddings=True)
            else:
                model.step(cache, yb)

        p_train = model.predict_probability(X_train)
        p_val   = model.predict_probability(X_val)

        train_loss = model.binary_cross_entropy(y_train, p_train)
        val_loss   = model.binary_cross_entropy(y_val, p_val)

        val_pred = (p_val >= 0.5).astype(np.int32)
        val_acc  = float((val_pred == y_val).mean())

        loss["train_loss"].append(train_loss)
        loss["val_loss"].append(val_loss)
        loss["val_acc"].append(val_acc)

        print(f"epoch {ep:03d} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    return loss


def save_results(history, model_name, y_val, y_pred, base_path="results"):

    folder = Path(base_path) / model_name
    folder.mkdir(parents=True, exist_ok=True)

    best_acc   = max(history["val_acc"])
    final_loss = history["val_loss"][-1]

    with open(folder / "results.txt", "w") as f:
        f.write(f"Model            : {model_name}\n")
        f.write(f"Best val accuracy: {best_acc:.4f}\n")
        f.write(f"Final val loss   : {final_loss:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_val, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_val, y_pred)))

    print(f"Saved → {folder / 'results.txt'}")