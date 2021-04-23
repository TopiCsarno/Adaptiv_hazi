"""
Optimalizációs módszereket tartalmaz (később ki lehet bővíteni pl: stochastic gradient descent, ADAM)
"""

def gradient_descent(layers, lr=0.01):
    for layer in layers:

        if layer.weights is None:
            continue

        layer.w -= lr*layer.dw
        layer.b -= lr*layer.db
