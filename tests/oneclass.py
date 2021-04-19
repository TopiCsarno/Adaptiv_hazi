# %%
import numpy as np
from src.layers.dense import DenseLayer
from src.model import Model
from src.optimizers import gradient_descent
from src.metrics import accuracy_binary_ce 

# %%
# Generate Dataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

N_SAMPLES = 2000
TEST_SIZE = 0.2

X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=42)
y = np.expand_dims(y,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# %%
layers = [
    DenseLayer(nodes_prev=2,  nodes_curr=25, activation="relu"),
    DenseLayer(nodes_prev=25, nodes_curr=50, activation="relu"),
    DenseLayer(nodes_prev=50, nodes_curr=50, activation="relu"),
    DenseLayer(nodes_prev=50, nodes_curr=25, activation="relu"),
    DenseLayer(nodes_prev=25, nodes_curr=1, activation="sigmoid")
]

model = Model(
    layers=layers, 
    optimizer=gradient_descent,
    costfn="BCE"
)

# %%
history = model.fit(X_train, y_train, epoch=50, lr=0.001, bs=1)

# %%
accuracy_binary_ce(y=y_test, y_hat=model.predict(X_test))
# %%
import matplotlib.pyplot as plt 

plt.figure(figsize=(12,6))
plt.plot(history)
plt.legend(['cost', 'accuracy'])
plt.show()

# %%
