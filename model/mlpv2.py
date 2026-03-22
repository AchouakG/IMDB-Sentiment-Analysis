import numpy as np
from model.mlp import MLPBinary

class MLPBinaryV2(MLPBinary):
    def __init__(self, d_in, n_neurons=16, learning_rate=0.01, momentum=0.9, seed=0):
        super().__init__(d_in, n_neurons, learning_rate, seed)
        self.momentum = momentum
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    def step(self, cache, y):
        X, z1, a1, p = cache
        n = X.shape[0]

        dz2 = (p - y) / n
        dW2 = np.matmul(a1.T, dz2)
        db2 = dz2.sum(axis=0, keepdims=True)
        da1 = np.matmul(dz2, self.W2.T)
        dz1 = da1 * self.relu_grad(z1)
        dW1 = np.matmul(X.T, dz1)
        db1 = dz1.sum(axis=0, keepdims=True)

        # momentum update: accumulate velocity
        self.vW1 = self.momentum * self.vW1 - self.learning_rate * dW1
        self.vW2 = self.momentum * self.vW2 - self.learning_rate * dW2
        self.vb1 = self.momentum * self.vb1 - self.learning_rate * db1
        self.vb2 = self.momentum * self.vb2 - self.learning_rate * db2

        self.W1 += self.vW1
        self.W2 += self.vW2
        self.b1 += self.vb1
        self.b2 += self.vb2