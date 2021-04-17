def gradient_descent(layers, lr=0.01):
    for layer in layers:
        layer.w -= lr*layer.dw
        layer.b -= lr*layer.db