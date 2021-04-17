import numpy as np

class Model():

    def __init__(self, layers, optimizer):
        self.layers = layers
        self.optimizer = optimizer

    def predict(self, x):
        activation = x
        for layer in self.layers:
            activation = layer.forward_pass(activation)
        return activation
    
    def get_cost(self, y, y_hat):
        m = y_hat.shape[0]
        cost = -1 / m * (np.dot(y, np.log(y_hat).T) + np.dot(1 - y, np.log(1 - y_hat).T))
        return np.squeeze(cost)

    def get_accuracy(self, y, y_hat):
        y_hat_ = y_hat.copy()
        y_hat_[y_hat_ > 0.5] = 1
        y_hat_[y_hat_ <= 0.5] = 0
        return np.sum(y_hat_ == y)/len(y)

    def fit(self, x, y, epoch, lr=0.01, verbose=True):
        history = []
        for i in range(epoch):
            y_hat = self.predict(x)[:,0]

            cost = self.get_cost(y, y_hat)
            accuracy = self.get_accuracy(y, y_hat)
            history.append((cost, accuracy))

            # calculate gradients
            da_prev = np.divide(1 - y, 1 - y_hat) - np.divide(y, y_hat)
            da_prev = da_prev[:,np.newaxis]
            for layer in reversed(self.layers):
                da_curr = da_prev
                da_prev = layer.back_pass(da_curr)
            
            # gradient descent
            self.optimizer(self.layers, lr)

            if(verbose and ((i+1) % 50 == 0 or i==0)):
                print("Iteration: {}\t cost: {:.5f}\taccuracy: {:.2f}%".format(i+1, cost, accuracy*100))

        return history
