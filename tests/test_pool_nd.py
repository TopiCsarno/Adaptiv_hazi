# %%
import sys
sys.path.append("../")
from src.layers.pool import MaxPoolLayer
from tensorflow import keras
import numpy as np
np.random.seed(99)

n = 1  # batch size
input_shape = (n,4,3)
pool_size = 2
stride = 2

# generate input
input = np.random.randint(10, size = input_shape)

# nd max pool layer
layer = MaxPoolLayer(pool_size, stride)

# keras model
model = keras.Sequential([
    keras.Input(shape=(input_shape[1:])),
    keras.layers.MaxPool1D(pool_size, stride)
])

# compare outputs
print("ND LAYER OURPUT")
print(layer.forward_pass(input))
print("KERAS OUTPUT")
print(model.predict(input))

# %%
a = 1
type(a)

# %%
