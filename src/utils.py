import numpy as np

def one_hot_enc(y):
    m = y.shape[0]
    Y = np.zeros((m,np.max(y)+1))
    Y[np.arange(m).reshape(1,m),y.T] = 1
    return Y
    
def generate_batches(x, y, batch_size):
    for i in range(0, x.shape[0], batch_size):
        yield (
            x.take(indices=range(i, min(i + batch_size, x.shape[0])), axis=0),
            y.take(indices=range(i, min(i + batch_size, y.shape[0])), axis=0)
        )