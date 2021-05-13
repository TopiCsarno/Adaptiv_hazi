# %%
import sys
sys.path.append("../")
from src.layers.conv import ConvLayerND
from tensorflow import keras
import numpy as np
np.random.seed(99)

n = 3  # batch size
filters = 4 
input_shape = (n, 6,6,1)
kernel_shape = (3,3,1)
stride = (2,1) 

# generate input
input = np.random.randint(10, size=input_shape)

# nd conv layer
layer = ConvLayerND(filters, kernel_shape, stride)

# keras model
model = keras.Sequential([
      keras.Input(shape=(input_shape[1:])),
      keras.layers.Conv2D(filters, kernel_shape[:-1], stride)
])

# same weights as our model
model.layers[0].set_weights([layer.w,layer.b])

# compare outputs
print("ND LAYER OUTPUT")
print(layer.forward_pass(input))
print("\nKERAS OUTPUT")
print(model.predict(input))
# %%
