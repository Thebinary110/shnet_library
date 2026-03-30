"""
Our Neural network will be made up of layers.
Each layer needs to pass it's inputs forward
and proogate gradients backward. For example,
a neural net might look like

inputs --> Linear --> Tanh --> Linear --> output
"""
from typing import Callable

import numpy as np

from shnet.tensor import Tensor


class Layers:
    def __init__(self): ## here we will be needing a constructor
        self.params = {}
        self.grads = {}

    def forward(self, inputs:Tensor) -> Tensor:
        """
        produce outputs corresponding to these inputs
        :param inputs:
        :return:
        """

        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
         Backpropogate the gradients through the layer
        :param grad:
        :param Tensor:
        :return:
        """
        raise NotImplementedError


class Linear(Layers):
    """
    coputes output = inputs @ w + b

    """

    def __init__(self, input_size:int, output_size:int) -> None:
        ## inputs will be (batch_size, input_size)
        ## outputs will be (batch_size, output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs :Tensor) -> Tensor:
        """
        outputs = inputs @ w + b
        """

        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = a * b + c
        dy/da = f'(x) * b
        dy/db = f'(x) * a
        dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da  = f'(x) @ b.T
        then dy/db = a.T @ f'(x)
        and dy/dc = f'(x)

        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @grad
        return grad @ self.params["w"].T

F = Callable[[Tensor], Tensor]

class Activation(Layers):
    """
    An Activation layer just applies a function
    elementwise to its inputs
    """

    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs:Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) anmd x = g(z)
        then dy/dz = f'(x) * g'(z)
        """

        return self.f_prime(self.inputs) * grad


def tanh(x:Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = np.tanh(x)
    return 1 - y ** 2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)