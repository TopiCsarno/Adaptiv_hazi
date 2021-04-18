import numpy as np 

class FlattenLayer():

    def __init__(self):
        self.a_prev_shape = None
        # TODO grad fix
        self.w = None
        self.b = None
        self.dw = None
        self.db = None

    def forward_pass(self, a_prev):
        self.a_prev_shape = a_prev.shape
        return np.ravel(a_prev).reshape(a_prev.shape[0], -1)

    def back_pass(self, da_curr):
        return da_curr.reshape(self.a_prev_shape)
