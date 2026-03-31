
import numpy as np
from shnet import NeuralNet, Linear
from shnet.layers import ReLU, Softmax
from shnet.loss import CrossEntropy
from shnet.optim import Adam
from shnet.train import train



# 🔹 XOR dataset
inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])


# 🔹 Model (non-linear → can learn XOR)
net = NeuralNet([
    Linear(2, 8),
    ReLU(),
    Linear(8, 4),
    ReLU(),
    Linear(4,2)
])


# 🔹 Train
train(
    net,
    inputs,
    targets,
    num_epochs=5000,
    optimizer=Adam(lr=0.001),
    loss=CrossEntropy()
)


# 🔹 Test predictions
print("\n=== Predictions ===")

for x, y in zip(inputs, targets):
    pred = net.forward(x.reshape(1, -1))
    print(f"Input: {x} → Predicted: {pred} → Actual: {y}")