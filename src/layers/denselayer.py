import numpy as np
from src.activation import sigmoid, sigmoid_grad, relu, relu_grad

class DenseLayer():
    
    def __init__(self, nodes_prev, nodes_curr, activation, seed=99):
        np.random.seed(seed)
        self.w = np.random.randn(nodes_curr, nodes_prev) * 0.1
        self.b = np.random.randn(1, nodes_curr) * 0.1
        self.g, self.dg = self._set_activation(activation)
        self.dw = None
        self.db = None
        self.a_prev = None
        self.z_curr = None
    
    def weights(self):
        return self.w, self.b

    def _set_activation(self, str):
        if str == "relu":
            return relu, relu_grad
        elif str == "sigmoid":
            return sigmoid, sigmoid_grad    
        else:
            raise Exception("Activation function is not supported")

    def forward_pass(self, a_prev):
        self.a_prev = a_prev.copy()
        self.z_curr = a_prev @ self.w.T + self.b
        return self.g(self.z_curr)
        
    def back_pass(self, da_curr):
        m = da_curr.shape[0]
        dz_curr = da_curr * self.dg(self.z_curr)
        self.dw = (dz_curr.T @ self.a_prev)/m
        self.db = np.sum(dz_curr,axis=0, keepdims=True)/m
        return dz_curr @ self.w
