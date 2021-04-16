class SoftmaxLayer():
    def __init__(self):
        self._z = None

    def forward_pass(self, a_prev):
        e = np.exp(a_prev - a_prev.max(axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)  
