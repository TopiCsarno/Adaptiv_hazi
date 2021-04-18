def gradient_descent(layers, lr=0.01):
    for layer in layers:

        params = (layer.w, layer.b, layer.dw, layer.db)
        if any(elem is None for elem in params):
            continue

        layer.w -= lr*layer.dw
        layer.b -= lr*layer.db