"""
A Model osztály, ami magába foglalja a neurális háló rétegeit. Választható költség függvény és optimalizációs módszer (jelenleg csak gradient descent)
"""
import numpy as np
from src.metrics import choose_cost_fn
from src.utils import generate_batches

class Model():

    def __init__(self, layers, optimizer, costfn):
        """
        param list layers: neurális háló rétegei egy listában felsorolva 
        param class optimizer: optilamizációs módszer kiválasztása
        param str costfn: lehet "BCE" (Binary Cross Entropy) vagy "CCE" (Categorical Cross Entropy)
        """
        self.layers = layers
        self.optimizer = optimizer
        self.costfn = costfn

    def predict(self, x):
        activation = x
        for layer in self.layers:
            activation = layer.forward_pass(activation)
        return activation

    def fit(self, x, y, epoch, lr=0.01, batch_size=64, verbose=True):
        # set cost function and accuracy function
        init_fn, cost_fn, acc_fn = choose_cost_fn(self.costfn)

        history = []
        for i in range(epoch):

            # train on batches
            for (x_batch, y_batch) in generate_batches(x, y, batch_size):

                # forward pass
                y_hat_batch = self.predict(x_batch)

                # back propogation
                grads = init_fn(y_batch, y_hat_batch)
                for layer in reversed(self.layers):
                    grads = layer.back_pass(grads)

                # gradient descent
                self.optimizer(self.layers, lr) 
                
            # calculate cost, accuracy
            y_hat = self.predict(x)
            cost = cost_fn(y, y_hat)
            accuracy = acc_fn(y, y_hat)
            history.append((cost, accuracy))

            if (verbose):
                if((i+1) % np.ceil(epoch/10) == 0 or i==0 or i+1==epoch):
                    print("Iteration: {}\t cost: {:.5f}\taccuracy: {:.2f}%".format(i+1, cost, accuracy*100))
        return history
