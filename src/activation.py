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

def softmax(Z, axis=1):
    e = np.exp(Z - np.max(Z, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def softmax_grax(Z):
    # TODO
    return 1
