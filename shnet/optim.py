"""
We use an optimizer to adjust the parameters
of our networks based on the gradients computed
during backpropogation
"""
from shnet.nn import NeuralNet
import numpy as np

class Optimizer:
    def step(self, net:NeuralNet) -> None:
        raise NotImplementedError

class SGD():
    def __init__(self, lr :float = 0.01) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad

class Adam(Optimizer):
    """
    Goal:
    Adapt learning rate per parameter using past gradients

    ---

    Given gradient g_t at time t:

    Step 1: First moment (mean)
    m_t = beta1 * m_(t-1) + (1 - beta1) * g_t

    Step 2: Second moment (variance)
    v_t = beta2 * v_(t-1) + (1 - beta2) * (g_t)^2

    ---

    Bias correction:

    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)

    ---

    Parameter update:

    param = param - lr * m_hat / (sqrt(v_hat) + eps)

    ---

    Interpretation:

    - m_t → momentum (direction)
    - v_t → adaptive scaling (step size)

    ---

    Why Adam is powerful:

    - smooth updates (like momentum)
    - adaptive learning rate (like RMSProp)

    ---

    If gradient is large:
    → v increases → step size decreases

    If gradient is small:
    → step size increases

    ---

    Result:
    Stable and fast convergence
    """
    def __init__(self, lr = 0.001, beta1=0.9, beta2 = 0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, net: NeuralNet):
        self.t += 1

        for i, (param, grad) in enumerate(net.params_and_grads()):
            if i not in self.m:
                self.m[i] = np.zeros_like(param)
                self.v[i] = np.zeros_like(param)

            # Update moving averages
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)



