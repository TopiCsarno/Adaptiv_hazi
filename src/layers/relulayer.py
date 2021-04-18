import numpy as np

class ReluLayer():
    def __init__(self):
        self._z = None
        self.w = None
        self.b = None
        self.dw = None
        self.db = None

    def forward_pass(self, a_prev):
        self._z = np.maximum(0, a_prev)
        return self._z

    def back_pass(self, da_curr):
        dz = np.array(da_curr, copy=True)
        dz[self._z <= 0] = 0
        return dz
