"""
A loss functions measures how goof our predictions are,
we can use this to adjust the parameters of our network
"""
from xml.etree.ElementPath import prepare_descendant

import numpy as np
from shnet.tensor import Tensor


class Loss:
    def loss(self, predicted:Tensor, actual:Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted:Tensor, actual:Tensor) ->  float:
        raise NotImplementedError


"""
MSE is mean squared error but we are going to use the total aquared error
"""

class MSE(Loss):
    def loss(self, predicted:Tensor, actual:Tensor) -> float:
        return np.mean((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) ->  Tensor:
        return 2 * (predicted - actual) / predicted.size


class CrossEntropy(Loss):
    """
    if z = logits and p = softmax(z)

    p_i = exp(z_i) / sum_j exp(z_j)

    Loss (CrossEntropy):
    L = - sum_i (y_i * log(p_i))

    where:
    - y = actual (one-hot vector)
    - p = predicted probabilities

    ---

    Key simplification:

    dL/dz = p - y

    Meaning:
    gradient w.r.t logits = (softmax_output - actual)

    ---

    Why this works:

    if we separate:
        softmax → crossentropy
    then gradient becomes complex (Jacobian)

    BUT when combined:
        it simplifies beautifully to:

        dL/dz = p - y

    ---

    Interpretation:

    if prediction is too high → positive gradient → decrease it
    if prediction is too low → negative gradient → increase it
    """
    def loss(self, predicted, actual):
        exp = np.exp(predicted - np.max(predicted, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        eps = 1e-9
        probs = np.clip(probs, eps, 1 - eps)

        return -np.sum(actual * np.log(probs))

    def grad(self, predicted, actual):
        exp = np.exp(predicted - np.max(predicted, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        return probs - actual

