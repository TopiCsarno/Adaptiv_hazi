"""
Költségfüggvényeket és pontosság metrikákat tartalmaz:
- BCE = Binary Cross Entropy cost function
- CCE = Categocial Cross Entropy cost function
"""
import numpy as np

def choose_cost_fn(str):
    if (str == "BCE"):
        return cost_binary_init, cost_binary_ce, accuracy_binary_ce
    elif (str == "CCE"):
        return cost_categ_init, cost_categ_ce, accuracy_categ_ce
    else:
        raise Exception("Cost funcion not supported")

def cost_binary_init(y, y_hat):
    return np.divide(1 - y, 1 - y_hat) - np.divide(y, y_hat)

def cost_binary_ce(y, y_hat):
    m = y.shape[0]
    y = y.squeeze()
    y_hat = y_hat.squeeze()
    cost = (-1 / m) * (np.dot(y, np.log(y_hat)) + np.dot(1 - y, np.log(1 - y_hat)))
    return np.squeeze(cost)

def accuracy_binary_ce(y, y_hat):
    m = y_hat.shape[0]
    y_hat_ = y_hat.copy()
    y_hat_[y_hat_ > 0.5] = 1
    y_hat_[y_hat_ <= 0.5] = 0
    return np.sum(y_hat_ == y)/m

def cost_categ_init(y, y_hat):
    return y_hat-y

def cost_categ_ce(y, y_hat, eps=1e-20):
    m = y_hat.shape[0]
    return - np.sum(y * np.log(np.clip(y_hat, eps, 1.))) / m

def accuracy_categ_ce(y, y_hat):
    m = y_hat.shape[0]
    return np.sum(np.argmax(y_hat,axis=1) == np.argmax(y,axis=1))/m
