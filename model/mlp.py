import numpy as np

'''
The number of layers is determined by how many weight matrices so in this case i'm
using 2 weight layers (input→hidden, hidden→output)
6 inputs → 16 hidden neurons → 1 output neuron
'''
class MLPBinary:
    def __init__(self, d_in, n_neurons=16, learning_rate=0.01, seed=0):
        rand_generator = np.random.default_rng(seed)
        self.W1 = rand_generator.normal(0, np.sqrt(2/d_in), size=(d_in, n_neurons)).astype(np.float32) # to create symmetry, weights from input --> hidden
        self.b1 = np.zeros((1,n_neurons), dtype=np.float32) # hidden bias
        self.W2 = rand_generator.normal(0, np.sqrt(1/n_neurons), size=(n_neurons, 1)).astype(np.float32) # weights from hidden --> output
        self.b2 = np.zeros((1,1), dtype=np.float32) # output bias
        self.learning_rate = learning_rate
        
    @staticmethod
    def relu(z):
        return np.maximum(0, z)
    
    @staticmethod
    def relu_grad(z): # derivative of relu converts boolean to int
        return (z > 0).astype(np.float32)
    
    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))
    
    @staticmethod
    def binary_cross_entropy(y, p):
        p = np.clip(p, 1e-7, 1-1e-7) # to avoid 0 and not have -inf
        return float(-(y*np.log(p) + (1-y)*np.log(1-p)).mean()) # if y=1 --> -log(p) (p close to 1) , y=0 --> -log(1-p) (p close to 0)
    
    
    def forward(self, X):
        z1= np.matmul(X, self.W1) + self.b1 # matrix multiplication
        a1= self.relu(z1)
        z2= np.matmul(a1, self.W2) + self.b2 # np.matmul((n,16),(16,1)) → (n,1)
        p= self.sigmoid(z2)
        cache = (X, z1, a1, p) # so we can do backpropagation
        return p, cache
        
    
    def step (self, cache, y):
        X, z1, a1, p= cache
        n= X.shape[0]
        dz2= (p-y)/n # ! BCE + sigmoid derivative simplifies: dz2 = (p - y) ?? check the course simplification
        dW2= np.matmul(a1.T, dz2) # .T is the transpose here
        db2= dz2.sum(axis= 0, keepdims=True)
        da1 = np.matmul(dz2, self.W2.T)
        dz1=da1 * self.relu_grad(z1)
        dW1 = np.matmul(X.T, dz1)
        db1 = dz1.sum(axis= 0, keepdims=True)
        
        self.W2 -=self.learning_rate * dW2
        self.b2 -=self.learning_rate * db2
        self.W1 -=self.learning_rate * dW1
        self.b1 -=self.learning_rate * db1
        
    def predict_probability(self, X):
        p, _ = self.forward(X)
        return p
    
    def predict(self, X, threshold = 0.5):
        return (self.predict_probability(X) >= threshold).astype(np.int32) # to get 0 or 1