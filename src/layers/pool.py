"""
N dimenziós összevonó réteg implementációja. A pool_size paramétertől függően számolja az összevonás lépéseit.
"""
from src.layers.layer import Layer
from src.utils import inc
import numpy as np

class MaxPoolLayerND(Layer):

    def __init__(self, pool_size, stride=1):
        """
        param tuple pool_size: Az összevonás méreteit adhatjuk meg tuple formátumban pl: (2,2).
        param tuple stride: Lépésköz adható meg egyes dimenziók irányában, vagy megadható int-ként is ha minden irányba azonos lépésközt szeretnénk.
        """
        if (type(pool_size) == int):
            self.D = 1
        else:
            self.D = len(pool_size)
        self.pool_size=pool_size
        self.stride=stride
        self.a_prev = None
        self.cache = {}
        
    def forward_pass(self, a_prev):
        self.a_prev = np.array(a_prev, copy=True)

        n = a_prev.shape[0]
        c = a_prev.shape[-1]
        d_in = np.array(a_prev.shape[1:-1])

        d_pool = np.array(self.pool_size)
        d_out = 1 + (d_in-d_pool) // self.stride
        d_index = np.zeros_like(d_out)
        output = np.zeros((n, *d_out, c))        

        # max pool loop
        i = len(d_out)-1
        for _ in range(np.prod(d_out)):
            d_start = d_index * self.stride
            d_end = d_start + d_pool

            # array slicing params
            s1 = slice(0,n)
            s2 = slice(0,c)
            d_slices = [slice(*x) for x in zip(d_start,d_end)]

            a_prev_slice = a_prev[(s1, *d_slices, s2)]
            self.save_mask(x=a_prev_slice, cords=tuple(d_index))
            output[(s1, *d_index, s2)] = np.max(a_prev_slice,
                axis=tuple([x+1 for x in range(self.D)])
            )

            # increment pool loop
            d_index[i] += 1
            inc(i, d_index, d_out)
        return output

    def back_pass(self, da_curr):
        output = np.zeros(self.a_prev.shape)

        n = da_curr.shape[0]
        c = da_curr.shape[-1]
        d_out = np.array(da_curr.shape[1:-1])
        d_pool = np.array(self.pool_size)
        d_index = np.zeros_like(d_out)
        output = np.zeros(self.a_prev.shape)

        i = len(d_out)-1
        for _ in range(np.prod(d_out)):
            d_start = d_index * self.stride
            d_end = d_start + d_pool
            
            s1 = slice(0,n)
            s2 = slice(0,c)
            d_slices = [slice(*x) for x in zip(d_start,d_end)]
            d_slices2 = [slice(*x) for x in zip(d_index,d_index+1)]

            output[(s1, *d_slices, s2)] += \
                da_curr[(s1, *d_slices2, s2)] * self.cache[tuple(d_index)]

            d_index[i] += 1
            inc(i, d_index, d_out)
        return output

    # A max érték pozícióját menti el az összevont területen belül. Back propogation során ez alapján vissza lehet állítani az az eredeti inputot
    def save_mask(self, x, cords):
        mask = np.zeros_like(x)

        n = x.shape[0]
        c = x.shape[-1]
        dims = np.array(x.shape[1:-1])

        x = x.reshape(n, np.prod(dims) , c)
        idx = np.argmax(x, axis=1)

        n_idx, c_idx = np.indices((n, c))
        np.reshape(mask,(n, np.prod(dims), c))[n_idx, idx, c_idx] = 1
        self.cache[cords] = mask
