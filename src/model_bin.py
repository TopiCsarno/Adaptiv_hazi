import numpy as np

from src.metrics import accuracy_binary_ce, cost_binary_ce
from src.metrics import accuracy_categ_ce, cost_categ_ce
from src.utils import generate_batches

class Model():

    def __init__(self, layers, optimizer):
        self.layers = layers
        self.optimizer = optimizer

    def predict(self, x):
        activation = x
        for layer in self.layers:
            activation = layer.forward_pass(activation)
        return activation

    def fit(self, x, y, epoch, lr=0.01, bs=64, verbose=True):
        history = []
        for i in range(epoch):

            # train on batches
            for (x_batch, y_batch) in generate_batches(x, y, bs):
                y_hat_batch = self.predict(x_batch)[:,0]

                # calc gradients
                da_prev = np.divide(1 - y_batch, 1 - y_hat_batch) - np.divide(y_batch, y_hat_batch)
                da_prev = da_prev[:,np.newaxis]
                for layer in reversed(self.layers):
                    da_curr = da_prev
                    da_prev = layer.back_pass(da_curr)

                # gradient descent
                self.optimizer(self.layers, lr) 
                
            y_hat = self.predict(x)[:,0]
            cost = cost_binary_ce(y, y_hat)
            accuracy = accuracy_binary_ce(y, y_hat)
            history.append((cost, accuracy))

            if(verbose and ((i+1) % 10 == 0 or i==0)):
                print("Iteration: {}\t cost: {:.5f}\taccuracy: {:.2f}%".format(i+1, cost, accuracy*100))

        return history