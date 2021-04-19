
def gradient_descent(layers, lr=0.01, debug=False):
    for layer in layers:

        if layer.weights is None:
            continue

        if(debug):
            print("Grads for layer: ",type(layer))
            print(layer.dw.ravel()[0], layer.db.ravel()[0])

        layer.w -= lr*layer.dw
        layer.b -= lr*layer.db
