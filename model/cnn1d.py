import numpy as np

class CNN1DBinary:
    def __init__(self, vocab_size, embed_dim=32, n_filters=64, kernel_size=3, learning_rate=0.01, seed=0):
        rng = np.random.default_rng(seed)

        self.E = rng.normal(0, 0.01, (vocab_size, embed_dim)).astype(np.float32)

        self.F = rng.normal(
            0,
            np.sqrt(2 / (kernel_size * embed_dim)),
            (kernel_size, embed_dim, n_filters)
        ).astype(np.float32)
        self.bf = np.zeros((1, n_filters), dtype=np.float32)

        self.W2 = rng.normal(
            0,
            np.sqrt(1 / n_filters),
            (n_filters, 1)
        ).astype(np.float32)
        self.b2 = np.zeros((1, 1), dtype=np.float32)

        self.learning_rate = learning_rate
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.embed_dim = embed_dim

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_grad(z):
        return (z > 0).astype(np.float32)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def binary_cross_entropy(y, p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    def forward(self, X_ids):
        batch, seq_len = X_ids.shape
        k = self.kernel_size

        embeds = self.E[X_ids]

        out_len = seq_len - k + 1
        Z_conv = np.zeros((batch, out_len, self.n_filters), dtype=np.float32)

        for t in range(out_len):
            window = embeds[:, t:t+k, :]
            Z_conv[:, t, :] = (
                window.reshape(batch, -1) @
                self.F.reshape(-1, self.n_filters)
            ) + self.bf

        A_conv = self.relu(Z_conv)

        pool_idx = A_conv.argmax(axis=1)
        pooled = A_conv.max(axis=1)

        z2 = pooled @ self.W2 + self.b2
        p = self.sigmoid(z2)

        cache = (X_ids, embeds, Z_conv, A_conv, pooled, pool_idx, p)
        return p, cache

    def step(self, cache, y):
        X_ids, embeds, Z_conv, A_conv, pooled, pool_idx, p = cache
        batch, seq_len, _ = embeds.shape
        k = self.kernel_size
        out_len = seq_len - k + 1

        dz2 = (p - y) / batch
        dW2 = pooled.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        d_pooled = dz2 @ self.W2.T

        dA_conv = np.zeros_like(A_conv)
        b_idx = np.arange(batch)[:, None]
        f_idx = np.arange(self.n_filters)[None, :]
        dA_conv[b_idx, pool_idx, f_idx] = d_pooled

        dZ_conv = dA_conv * self.relu_grad(Z_conv)

        dF = np.zeros_like(self.F)
        dE = np.zeros_like(self.E)
        dbf = dZ_conv.sum(axis=(0, 1), keepdims=False).reshape(1, -1)

        F_flat = self.F.reshape(-1, self.n_filters)

        for t in range(out_len):
            window = embeds[:, t:t+k, :]
            dz_t = dZ_conv[:, t, :]

            dF += (window.reshape(batch, -1).T @ dz_t).reshape(
                k, self.embed_dim, self.n_filters
            )

            d_window = (dz_t @ F_flat.T).reshape(batch, k, self.embed_dim)
            for ki in range(k):
                np.add.at(dE, X_ids[:, t + ki], d_window[:, ki, :])

        for grad in [dF, dbf, dW2, db2, dE]:
            np.clip(grad, -5, 5, out=grad)

        self.F -= self.learning_rate * dF
        self.bf -= self.learning_rate * dbf
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.E -= self.learning_rate * dE

    def predict_probability(self, X_ids):
        p, _ = self.forward(X_ids)
        return p

    def predict(self, X_ids, threshold=0.5):
        return (self.predict_probability(X_ids) >= threshold).astype(np.int32)