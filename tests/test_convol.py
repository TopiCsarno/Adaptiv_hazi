# %%
import numpy as np
import pickle

np.random.seed(69)

with open("data/x.pkl", "rb") as f:
    x = pickle.load(f)

# x = np.random.randn(1,6,6,1)*0.1
# print(x.shape)

X = np.concatenate((x,x), axis=0)
print(X.shape)
# %%
from src.layers.convlayer import ConvLayer

kernel_shape = (3,3,1)
filters = 16 
layer = ConvLayer(filters, kernel_shape, seed=69)

out = layer.forward_pass(X)
out.shape
# %%
np.random.seed(69)
da_curr = np.random.randn(*out.shape) * 0.1
da_curr.shape

layer.back_prop(da_curr)[0,0]
# %%
