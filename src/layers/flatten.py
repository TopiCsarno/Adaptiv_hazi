"""
Flatten layer: az inputot "kilapítja" azaz átalakítja egy 1 dimenziós vektorrá.
"""

from src.layers.layer import Layer
import numpy as np 

class FlattenLayer(Layer):

    def __init__(self):
        self.a_prev_shape = None

    def forward_pass(self, a_prev):
        self.a_prev_shape = a_prev.shape
        return np.ravel(a_prev).reshape(a_prev.shape[0], -1)

    def back_pass(self, da_curr):
        return da_curr.reshape(self.a_prev_shape)
