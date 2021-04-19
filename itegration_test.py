#%%

import numpy as np
from src.layers.denselayer import DenseLayer
from src.layers.convlayer import ConvLayer
from src.layers.flattenlayer import FlattenLayer
from src.layers.poollayer import MaxPoolLayer
from src.layers.relulayer import ReluLayer
from src.model import Model
from src.optimizers import gradient_descent
from src.utils import one_hot_enc
from src.utils import generate_batches

from tensorflow.keras.datasets import fashion_mnist

N_TRAIN_SAMPLES = 2000
N_TEST_SAMPLES = 400
N_CLASSES = 10

((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

X_train = trainX[:N_TRAIN_SAMPLES, :, :]
y_train = trainY[:N_TRAIN_SAMPLES]

X_test = trainX[N_TRAIN_SAMPLES:N_TRAIN_SAMPLES+N_TEST_SAMPLES, :, :]
y_test = trainY[N_TRAIN_SAMPLES:N_TRAIN_SAMPLES+N_TEST_SAMPLES]

X_train = X_train / 255
X_train = np.expand_dims(X_train, axis=3)
y_train = one_hot_enc(y_train)
X_test = X_test / 255
X_test = np.expand_dims(X_test, axis=3)
y_test = one_hot_enc(y_test)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
#%%
layers = [
    ConvLayer(filters=16, kernel_shape=(3,3,1)),
    ReluLayer(),
    MaxPoolLayer(pool_size=(2,2), stride=2),
    ConvLayer(filters=32, kernel_shape=(5 ,5 ,16)),
    ReluLayer(),
    MaxPoolLayer(pool_size=(3,3), stride=3),
    FlattenLayer(),
    DenseLayer(nodes_prev=288, nodes_curr=10, activation="softmax")
]

model = Model(
    layers=layers, 
    optimizer=gradient_descent
)

# %%

# run in debug mode
hist = model.fit(X_train,y_train, epoch=1, lr=0.005, bs=1, debug=True)

# random seed = 99
# Grads for layer:  <class 'src.layers.convlayer.ConvLayer'>
# -0.2699143184847143 -0.174745609968721
# Grads for layer:  <class 'src.layers.convlayer.ConvLayer'>
# -0.0009223090514257201 -0.028849514880122135
# Grads for layer:  <class 'src.layers.denselayer.DenseLayer'>
# 0.009444964391780806 0.09199863806101219
#