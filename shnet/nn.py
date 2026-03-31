"""
A neural network is just a collection of layers.
It behaves a lot like a layer itself, although
we'are not going to make it one
"""

from typing import Sequence, Iterator, Tuple
from shnet.tensor import Tensor
from shnet.layers import Layers

class NeuralNet:
    def __init__(self, layers: Sequence[Layers]):
        self.layers = layers

    def forward(self, inputs:Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad:Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad

    def predict(self, inputs):
        outputs = self.forward(inputs)

        ## if batch input
        if outputs.ndim == 2:
            return outputs.argmax(axis=1)

        ## single input
        return outputs.argmax()