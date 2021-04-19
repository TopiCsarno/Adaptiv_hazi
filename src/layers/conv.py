from src.layers.layer import Layer
from src.activation import set_activation
import numpy as np 

class ConvLayer(Layer):

    def __init__(self, filters, kernel_shape, activation=None, seed=99):
        np.random.seed(seed)
        self.w = np.random.randn(*kernel_shape, filters) * 0.1
        self.b = np.random.randn(filters) * 0.1
        self.g, self.dg = set_activation(activation)
        self.dw = None
        self.db = None
        self.a_prev = None
        self.g_out = None

    def weights(self):
        return self.w, self.b, self.dw, self.db

    def forward_pass(self, a_prev):
        self.a_prev = np.array(a_prev, copy=True)

        n, h_in, w_in, _ = a_prev.shape
        h_f, w_f, _, n_f = self.w.shape

        h_out = h_in - h_f + 1
        w_out = w_in - w_f + 1

        output = np.zeros((n, h_out, w_out, n_f))
        for h_start in range(h_out):
            for w_start in range(w_out):
                h_end = h_start + h_f
                w_end = w_start + w_f

                output[:, h_start, w_start, :] = np.sum(
                    a_prev[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                    self.w[np.newaxis, :, :, :],
                    axis=(1, 2, 3)
                ) 
        output += self.b
        self.g_out = self.g(output)
        return self.g_out

    def back_pass(self, da_curr):
        da_curr = da_curr * self.dg(self.g_out)
        n, h_out, w_out, _ = da_curr.shape
        h_f, w_f, _, _ = self.w.shape

        output = np.zeros_like(self.a_prev)
        self.db = da_curr.sum(axis=(0, 1, 2)) / n
        self.dw = np.zeros_like(self.w)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i
                h_end = h_start + h_f
                w_start = j
                w_end = w_start + w_f

                output[:, h_start:h_end, w_start:w_end, :] += np.sum(
                    self.w[np.newaxis, :, :, :, :] *
                    da_curr[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=4
                )

                self.dw += np.sum(
                    self.a_prev[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                    da_curr[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=0
                )
        self.dw /= n
        return output