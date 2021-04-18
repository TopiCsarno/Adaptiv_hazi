import numpy as np

class MaxPoolLayer():

    def __init__(self, pool_size, stride, seed=99):
        np.random.seed(seed)
        self.pool_size=pool_size
        self.stride=stride
        self.a_prev = None
        self.cache = {}
        # TODO - we dont need these params only for grad descent to work
        self.w = None
        self.b = None
        self.dw = None
        self.db = None
        

    def forward_pass(self, a_prev):
        self.a_prev = np.array(a_prev, copy=True)

        n, h_in, w_in, c = a_prev.shape
        h_pool, w_pool = self.pool_size
        h_out = 1 + (h_in - h_pool) // self.stride
        w_out = 1 + (w_in - w_pool) // self.stride
        output = np.zeros((n, h_out, w_out, c))

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_pool
                w_start = j * self.stride
                w_end = w_start + w_pool
                a_prev_slice = a_prev[:, h_start:h_end, w_start:w_end, :]
                self.save_mask(x=a_prev_slice, cords=(i, j))
                output[:, i, j, :] = np.max(a_prev_slice, axis=(1, 2))
        return output

    def back_pass(self, da_curr):
        output = np.zeros(self.a_prev.shape)
        _, h_out, w_out, _ = da_curr.shape
        h_pool, w_pool = self.pool_size

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_pool
                w_start = j * self.stride
                w_end = w_start + w_pool

                output[:, h_start:h_end, w_start:w_end, :] += \
                    da_curr[:, i:i + 1, j:j + 1, :] * self.cache[(i, j)]  
        return output

    def save_mask(self, x, cords):
        mask = np.zeros_like(x)
        n, h, w, c = x.shape
        x = x.reshape(n, h * w, c)
        idx = np.argmax(x, axis=1)

        n_idx, c_idx = np.indices((n, c))
        np.reshape(mask,(n, h*w, c))[n_idx, idx, c_idx] = 1
        self.cache[cords] = mask


