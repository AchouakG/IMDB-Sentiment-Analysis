import numpy as np

class RNNBinary:
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64,
                learning_rate=0.01, seed=0):
        rng = np.random.default_rng(seed)

        self.E  = rng.normal(0, 0.01, (vocab_size, embed_dim)).astype(np.float32)

        # RNN weights
        self.Wx = rng.normal(0, np.sqrt(2/embed_dim),  (embed_dim,  hidden_dim)).astype(np.float32)
        self.Wh = rng.normal(0, np.sqrt(1/hidden_dim), (hidden_dim, hidden_dim)).astype(np.float32)
        self.bh = np.zeros((1, hidden_dim), dtype=np.float32)

        # output weights
        self.W2 = rng.normal(0, np.sqrt(1/hidden_dim), (hidden_dim, 1)).astype(np.float32)
        self.b2 = np.zeros((1, 1), dtype=np.float32)

        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_grad(z):
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def binary_cross_entropy(y, p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    def forward(self, X_ids):
        batch, seq_len = X_ids.shape
        embeds = self.E[X_ids]

        h = np.zeros((batch, self.hidden_dim), dtype=np.float32)
        hs, zs = [], []

        for t in range(seq_len):
            x_t = embeds[:, t, :]
            z_t = (x_t @ self.Wx) + (h @ self.Wh) + self.bh
            h   = self.tanh(z_t)
            hs.append(h)
            zs.append(z_t)

        z2 = h @ self.W2 + self.b2
        p  = self.sigmoid(z2)

        cache = (X_ids, embeds, hs, zs, p)
        return p, cache

    def step(self, cache, y):
        X_ids, embeds, hs, zs, p = cache
        batch, seq_len, embed_dim = embeds.shape

        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dbh = np.zeros_like(self.bh)
        dE  = np.zeros_like(self.E)

        dz2 = (p - y) / batch
        dW2 = hs[-1].T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        dh_next = dz2 @ self.W2.T

        for t in reversed(range(seq_len)):
            dh   = dh_next
            dz_t = dh * self.tanh_grad(zs[t])

            dWx += embeds[:, t, :].T @ dz_t
            dbh += dz_t.sum(axis=0, keepdims=True)
            if t > 0:
                dWh     += hs[t-1].T @ dz_t
                dh_next  = dz_t @ self.Wh.T
            else:
                dh_next  = np.zeros_like(dh_next)

            dx_t = dz_t @ self.Wx.T
            np.add.at(dE, X_ids[:, t], dx_t)

        # gradient clipping
        for grad in [dWx, dWh, dbh, dW2, db2, dE]:
            np.clip(grad, -5, 5, out=grad)

        self.Wx -= self.learning_rate * dWx
        self.Wh -= self.learning_rate * dWh
        self.bh -= self.learning_rate * dbh
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.E  -= self.learning_rate * dE

    def predict_probability(self, X_ids):
        p, _ = self.forward(X_ids)
        return p

    def predict(self, X_ids, threshold=0.5):
        return (self.predict_probability(X_ids) >= threshold).astype(np.int32)