# %%
import sys
sys.path.append("../")
from src.layers.conv import ConvLayer
import numpy as np
np.random.seed(99)
# %% 
n = 2  # batch size
filters = 4 
kernel_shape = (3, 3, 1)
input_shape = (n, 6, 6, 1)
grads_shape = (n, 4, 4, 4)

# initialize 
input = np.random.randn(*input_shape)
grads = np.random.randn(*grads_shape) * 0.1

# nd conv layer
layer = ConvLayer(filters,kernel_shape)
layer.forward_pass(input)
layer.back_pass(grads)

# array([[[[ 0.03532108],
#          [ 0.0293629 ],
#          [-0.01775786],
#          [-0.02100535],
#          [ 0.00436123],
#          [ 0.02607235]],
#             ....

# %%
