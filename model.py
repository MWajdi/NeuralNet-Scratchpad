import numpy as np


def activation_function(function):
    if function.lower() == "sigmoid":
        return lambda x: 1 / (1+np.exp(-x))
    elif function.lower() == "tanh":
        return lambda x: np.tanh(x)
    elif function.lower() == "relu":
        def ReLU(x):
            if x>0:
                return x
            else:
                return 0
        return ReLU
    elif function.lower() == "leaky relu":
        def Leaky_ReLU(x):
            if x>0:
                return x
            else:
                return 0.01*x
        return Leaky_ReLU


class Layer:
    def __init__(self, nb_neurons, a_function, weights_init, biases_init):
        self.w = weights_init
        self.b = biases_init
        self.g = activation_function(a_function)
        self.n = nb_neurons

    def forward(self,x):
        assert x.shape[0] == self.w.shape[1], f"input has the wrong shape, {x.shape[0]} != {self.shape[1]}"
        Z = np.matmul(self.w, x) + self.b
        A = self.g(Z)
        return A
    

