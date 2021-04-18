# %%

import numpy as np
import pickle

np.random.seed(69)

# with open("data/x.pkl", "rb") as f:
#     x = pickle.load(f)
x = np.expand_dims(np.array([[1,9,8,4],[4,8,6,7],[4,0,5,9],[7,3,5,4]]), [0,3])


X = np.concatenate((x,x), axis=0)

print(x.shape, X.shape)
# %%
from src.layers.poollayer import MaxPoolLayer

pool_size = (2,2)
stride = 2

layer = MaxPoolLayer(pool_size, stride)


# %%

da_curr = layer.forward_pass(x)
da_curr.shape

# %%
layer.cache
# %%
layer.back_prop(da_curr)
# %%
