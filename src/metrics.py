import numpy as np

def cost_binary_ce(y, y_hat):
    m = y_hat.shape[0]
    cost = -1 / m * (np.dot(y, np.log(y_hat).T) + np.dot(1 - y, np.log(1 - y_hat).T))
    return np.squeeze(cost)

def accuracy_binary_ce(y, y_hat):
    y_hat_ = y_hat.copy()
    y_hat_[y_hat_ > 0.5] = 1
    y_hat_[y_hat_ <= 0.5] = 0
    return np.sum(y_hat_ == y)/len(y)

def cost_categ_ce(y, y_hat, eps=1e-20):
    m = y_hat.shape[0]
    return - np.sum(y * np.log(np.clip(y_hat, eps, 1.))) / m

def accuracy_categ_ce(y, y_hat):
    m = y_hat.shape[0]
    return np.sum(np.argmax(y_hat,axis=1) == np.argmax(y,axis=1))/m
