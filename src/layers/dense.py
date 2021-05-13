"""
Fully connected (dense) layer implementációja. Tartalmazza az aktivációs függvényt is
"""

import numpy as np
from src.activation import set_activation
from src.layers.layer import Layer

class DenseLayer(Layer):
    
    def __init__(self, nodes_prev, nodes_curr, activation=None, seed=99):
        """
        param int nodes_prev: Előző réteg neuronjainak száma.
        param int nodes_curr: Ennek a rétegnek a neuronjainak száma.
        param str activation: Aktivációs fv adható meg string alakban. Lehet "relu", "sigmoid", "softmax"
        """
        np.random.seed(seed)
        self.w = np.random.randn(nodes_curr, nodes_prev) * 0.1
        self.b = np.random.randn(1, nodes_curr) * 0.1
        self.g, self.dg = set_activation(activation)
        self.dw = None 
        self.db = None
        self.a_prev = None
        self.z_curr = None
    
    def weights(self):
        return self.w, self.b, self.dw, self.db

    def forward_pass(self, a_prev):
        self.a_prev = a_prev.copy()
        self.z_curr = a_prev @ self.w.T + self.b
        return self.g(self.z_curr)
        
    def back_pass(self, da_curr):
        dz_curr = da_curr * self.dg(self.z_curr)
        m = da_curr.shape[0]
        self.dw = (dz_curr.T @ self.a_prev)/m
        self.db = np.sum(dz_curr,axis=0, keepdims=True)/m
        return dz_curr @ self.w
