# %%
from tensorflow import keras
from src.layers.conv import ConvLayer
import numpy as np
np.random.seed(99)

n = 3  # batch size
filters = 4 
input_shape = (n, 6,1)
kernel_shape = (3,1)

# generate input
input = np.random.randint(10, size=input_shape)

# nd conv layer
layer = ConvLayer(filters,kernel_shape)

# keras model
model = keras.Sequential([
      keras.Input(shape=(input_shape[1:])),
      keras.layers.Conv1D(filters, kernel_shape[:-1])
])

# same weights as our model
model.layers[0].set_weights([layer.w,layer.b])

# compare outputs
print("ND LAYER OUTPUT")
print(layer.forward_pass(input))
print("\nKERAS LAYER OUTPUT")
print(model.predict(input))