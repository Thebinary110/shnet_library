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
        limit = np.sqrt(2 / input_size)
        self.params["w"] = np.random.randn(input_size, output_size) * limit
        self.params["b"] = np.zeros(output_size)

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

def relu(x:Tensor) -> Tensor:
    return np.maximum(0, x)

def relu_prime(x: Tensor) -> Tensor:
    return (x > 0).astype(float)

def softmax(x: Tensor) -> Tensor:
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

class Tanh(Activation):
    """
    Tanh activation:

    f(x) = tanh(x)

    ---

    Derivative:

    f'(x) = 1 - tanh(x)^2

    ---

    Backprop:

    dL/dx = dL/dy * (1 - tanh(x)^2)

    ---

    Range:

    -1 <= tanh(x) <= 1

    ---

    Interpretation:

    - smooth non-linearity
    - but can suffer from vanishing gradients
    """
    def __init__(self):
        super().__init__(tanh, tanh_prime)

class ReLU(Activation):
    """
    ReLU activation:

    f(x) = max(0, x)

    ---

    Derivative:

    f'(x) = 1  if x > 0
            0  if x <= 0

    ---

    Backprop:

    dL/dx = dL/dy * f'(x)

    ---

    Interpretation:

    - positive inputs → pass gradient
    - negative inputs → block gradient

    ---

    Effect:

    - introduces non-linearity
    - avoids vanishing gradient (better than tanh/sigmoid)

    ---

    Risk:

    "dead neurons" if always negative
    """
    def __init__(self):
        super().__init__(relu, relu_prime)

class Softmax(Layers):
    """
    Softmax converts logits → probabilities

    p_i = exp(z_i) / sum_j exp(z_j)

    ---

    Properties:

    - 0 <= p_i <= 1
    - sum_i p_i = 1

    ---

    Numerical stability trick:

    Instead of:
    exp(z)

    we use:
    exp(z - max(z))

    to avoid overflow

    ---

    Gradient (full form):

    dp_i/dz_j = p_i * (delta_ij - p_j)

    This is a Jacobian matrix:

    J = diag(p) - p p^T

    ---

    BUT:

    When used with CrossEntropy:
    we DO NOT compute this Jacobian explicitly

    because:

    dL/dz = p - y  (simplified)

    ---

    Interpretation:

    Softmax spreads probability mass across classes
    """
    def forward(self, inputs: Tensor) -> Tensor:
        self.outputs = softmax(inputs)
        return self.outputs

    def backward(self, grad: Tensor) -> Tensor:
        return grad  # simplified (works with CrossEntropy)