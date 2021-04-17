import numpy as np

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def sigmoid_grad(Z):
    sig = sigmoid(Z)
    return sig * (1 - sig)

def relu(Z):
    return np.maximum(0,Z)

def relu_grad(Z):
    return Z > 0