import numpy as np


def activation_function(function):
    if function.lower() == "sigmoid":
        return lambda x: 1 / (1+np.exp(-x)), lambda x: -np.exp(-x)/(1+np.exp(-x))**2
    elif function.lower() == "tanh":
        return lambda x: np.tanh(x), lambda x: 1 - np.tanh(x)**2
    elif function.lower() == "relu":
        def ReLU(x):
            return np.maximum(x,0)
        def d_ReLU(x):
            return np.where(x>0,1,0)
        return ReLU, d_ReLU
    elif function.lower() == "leaky relu":
        def Leaky_ReLU(x):
            return np.where(x>0,x,0.01*x)
        def d_Leaky_ReLU(x):
            return np.where(x>0,1,0.01)
        return Leaky_ReLU, d_Leaky_ReLU
    elif function.lower() == "softmax":
        def softmax(Z):
            shiftZ = Z - np.max(Z, axis=0, keepdims=True)
            exps = np.exp(shiftZ)
            sum_exps = np.sum(exps, axis=0, keepdims=True)
            A = exps / sum_exps
            return A
        return softmax, None


class Layer:
    def __init__(self, input_size, output_size, a_function):
        self.W = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.B = np.zeros(shape=(output_size, 1), dtype=float)
        self.g, self.d_g = activation_function(a_function)
        self.A, self.Z = None, None

    def forward(self,X):
        assert X.shape[0] == self.W.shape[1], f"input has the wrong shape, {X.shape[0]} != {self.W.shape[1]}"
        self.Z = np.matmul(self.W, X) + self.B
        self.A = self.g(self.Z)
        return self.A
    
class NN:
    def __init__(self, layers, lr):
        self.layers = layers
        self.lr = lr

    def loss(self, X, Y):
        A = self.forward(X)
        loss = -np.mean(np.sum(Y * np.log(A + 1e-9), axis=0)) 
        return loss
    
    def forward(self,X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A
    
    def backward(self,X,Y):
        m = Y.shape[1]

        dZ = self.layers[-1].A - Y
        for i in reversed(range(len(self.layers))):

            if i > 0:
                dW = np.matmul(dZ, self.layers[i-1].A.T) / m
            else:
                dW = np.matmul(dZ, X.T) / m

            dB = np.sum(dZ, axis=1, keepdims=True) / m

            if i > 0:
                dZ = np.matmul(self.layers[i].W.T, dZ) * self.layers[i-1].d_g(self.layers[i-1].Z)
            
            self.layers[i].W -= self.lr * dW
            self.layers[i].B -= self.lr * dB



