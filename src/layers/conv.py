"""
N dimenziós konvilúciós réteg implementációja. A megadott kernel mérettől függően számolja ki a konvolúció menetét.
"""
from src.layers.layer import Layer
from src.activation import set_activation
from src.utils import inc
import numpy as np 

class ConvLayer(Layer):

    def __init__(self, filters, kernel_shape, activation=None, seed=99):
        np.random.seed(seed)
        self.D = len(kernel_shape)
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

        # read input and kernel dimentions, generate output shape
        n = a_prev.shape[0]
        d_in = np.array(a_prev.shape[1:-1])

        n_f = self.w.shape[-1]
        d_f = np.array(self.w.shape[0:-2])
        d_out = d_in - d_f + 1
        d_start = np.zeros_like(d_out)
        output = np.zeros((n, *d_out, n_f))

        # convolution loop
        i = len(d_out)-1
        for _ in range(np.prod(d_out)):
            d_end = d_start + d_f

            # array slicing parameters
            s1 = slice(0,n)
            s2 = slice(0,n_f)
            d_slices = [slice(*x) for x in zip(d_start,d_end)]

            # calculate output
            output[(s1, *d_start, s2)] = np.sum(
                a_prev[(s1, *d_slices, s2, np.newaxis)] *
                self.w[np.newaxis, ...],
                axis=tuple([x+1 for x in range(self.D)])
            )

            # increment convol loop
            d_start[i] += 1
            inc(i, d_start, d_out)

        # add bias
        output += self.b
        # activatoin function
        self.g_out = self.g(output)
        return self.g_out

    def back_pass(self, da_curr):
        # activation function gradient
        da_curr = da_curr * self.dg(self.g_out)

        # read input dimentions, generate output shape
        n = da_curr.shape[0]
        d_in = np.array(da_curr.shape[1:-1])
        n_f = self.w.shape[-1]
        d_f = np.array(self.w.shape[0:-2])
        d_start = np.zeros_like(d_in)
        output = np.zeros_like(self.a_prev)

        # dias gradients
        self.db = (1/n) * da_curr.sum(axis=
            tuple([x for x in range(self.D)])
        )

        self.dw = np.zeros_like(self.w)
        i = len(d_in)-1
        for _ in range(np.prod(d_in)):
            d_end = d_start + d_f

            # array slicing parameters
            s1 = slice(0,n)
            s2 = slice(0,n_f)
            d_slices = [slice(*x) for x in zip(d_start, d_end)]
            d_slices2 = [slice(*x) for x in zip(d_start, d_start+1)]

            # calculate output
            output[(s1, *d_slices, s2)] += np.sum(
                self.w[np.newaxis, ...] *
                da_curr[(s1, *d_slices2, np.newaxis, s2)],
                axis=self.D+1
            )

            # weight gradients
            self.dw += np.sum(
                self.a_prev[(s1, *d_slices, s2, np.newaxis)] *
                da_curr[(s1, *d_slices2, np.newaxis, s2)],
                axis=0
            )


            # increment convol loop
            d_start[i] += 1
            inc(i, d_start, d_in)

        self.dw /= n
        return output
