"""
We use an optimizer to adjust the parameters
of our networks based on the gradients computed
during backpropogation
"""
from shnet.nn import NeuralNet

class Optimizer:
    def step(self, net:NeuralNet) -> None:
        raise NotImplementedError

class SGD():
    def __init__(self, lr :float = 0.01) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad

