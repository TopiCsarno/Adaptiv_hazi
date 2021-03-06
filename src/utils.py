"""
Segédfüggvények gyűjteménye
"""
import numpy as np

# one-hot-encoding: kategorikus címkéket alakít át one-hot formátumra
def one_hot_enc(y):
    m = y.shape[0]
    Y = np.zeros((m,np.max(y)+1))
    Y[np.arange(m).reshape(1,m),y.T] = 1
    return Y

# tanító adatot mini-batchekre osztja fel adott méret szerint
def generate_batches(x, y, batch_size):
    for i in range(0, x.shape[0], batch_size):
        yield (
            x.take(indices=range(i, min(i + batch_size, x.shape[0])), axis=0),
            y.take(indices=range(i, min(i + batch_size, y.shape[0])), axis=0)
        )

# rekurzív függvény ami beágyazott for loopot helyettesít
def inc(i, a, b):
    if a[i] == b[i]:
        a[i-1] += 1
        a[i] = 0
        inc((i-1), a, b)