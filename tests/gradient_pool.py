# %%
import sys
sys.path.append("../")
from src.layers.pool import MaxPoolLayerND
import numpy as np
import pickle

np.random.seed(69)

x = np.expand_dims(np.array([[1,9,8,4],[4,8,6,7],[4,0,5,9],[7,3,5,4]]), [0,3])

X = np.concatenate((x,x), axis=0)

print(x.shape, X.shape)
# %%

pool_size = (2,2)
stride = 2

layer = MaxPoolLayerND(pool_size, stride)

# %%
da_curr = layer.forward_pass(X)
da_curr

# array([[[[9.],
#          [8.]],

#         [[7.],
#          [9.]]]])

# %%
layer.cache
# %%
layer.back_pass(da_curr)

# array([[[[0.],
#          [9.],
#          [8.],
#          [0.]],

#         [[0.],
#          [0.],
#          [0.],
#          [0.]],

#         [[0.],
#          [0.],
#          [0.],
#          [9.]],

#         [[7.],
#          [0.],
#          [0.],
#          [0.]]]])
# %%
