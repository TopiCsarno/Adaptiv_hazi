def gradient_descent(layers, lr=0.01, debug=False):
    for layer in layers:

        params = (layer.w, layer.b, layer.dw, layer.db)
        if any(elem is None for elem in params):
            continue

        if(debug):
            print("Grads for layer: ",type(layer))
            print(layer.dw.ravel()[0], layer.db.ravel()[0])

        layer.w -= lr*layer.dw
        layer.b -= lr*layer.db
