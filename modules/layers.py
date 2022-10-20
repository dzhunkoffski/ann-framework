import numpy as np
from typing import List
from .base import Module


class Linear(Module):
    """
    Applies linear (affine) transformation of data: y = x W^T + b
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        :param in_features: input vector features
        :param out_features: output vector features
        :param bias: whether to use additive bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-1, 1, (out_features, in_features)) / np.sqrt(in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / np.sqrt(in_features) if bias else None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, in_features)
        :return: array of shape (batch_size, out_features)
        """
        if self.bias is not None:
            return input @ self.weight.T + np.full((input.shape[0], self.out_features), self.bias)
        return input @ self.weight.T

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        :return: array of shape (batch_size, in_features)
        """
        return grad_output @ self.weight

    def update_grad_parameters(self, input: np.array, grad_output: np.array):
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        """
        self.grad_weight += grad_output.T @ input
        if self.grad_bias is not None:
            self.grad_bias += np.sum(grad_output, axis=0)

    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.array]:
        if self.bias is not None:
            return [self.weight, self.bias]

        return [self.weight]

    def parameters_grad(self) -> List[np.array]:
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]

        return [self.grad_weight]

    def __repr__(self) -> str:
        out_features, in_features = self.weight.shape
        return f'Linear(in_features={in_features}, out_features={out_features}, ' \
               f'bias={not self.bias is None})'


class BatchNormalization(Module):
    """
    Applies batch normalization transformation
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        """
        :param num_features:
        :param eps:
        :param momentum:
        :param affine: whether to use trainable affine parameters
        """
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.weight = np.ones(num_features) if affine else None
        self.bias = np.zeros(num_features) if affine else None

        self.grad_weight = np.zeros_like(self.weight) if affine else None
        self.grad_bias = np.zeros_like(self.bias) if affine else None

        # store this values during forward path and re-use during backward pass
        self.mean = None
        self.input_mean = None  # input - mean
        self.var = None
        self.sqrt_var = None
        self.inv_sqrt_var = None
        self.norm_input = None

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        if self.training:
            # Training mode
            self.mean = (1 / input.shape[0]) * np.sum(input, axis=0)
            self.input_mean = input - np.full(input.shape, self.mean)
            self.var = (1 / input.shape[0]) * np.sum(self.input_mean ** 2, axis=0)
            self.sqrt_var = np.sqrt(self.var + self.eps)
            self.inv_sqrt_var = 1 / self.sqrt_var
            self.norm_input = self.input_mean * np.full(input.shape, self.inv_sqrt_var)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                               self.momentum * (input.shape[0] / (input.shape[0] - 1)) * self.var
        else:
            # Eval mode
            self.norm_input = np.divide(
                input - np.full(input.shape, self.running_mean),
                np.full(input.shape, np.sqrt(self.running_var + self.eps))
            )

        if not self.affine:
            return self.norm_input

        return self.norm_input * np.full(input.shape, self.weight) + np.full(input.shape, self.bias)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        if self.training:
            # Train mode
            grad_norm_input = None
            if self.affine:
                grad_norm_input = grad_output * self.weight
            else:
                grad_norm_input = grad_output

            grad_inv_sqrt_var = np.sum(grad_norm_input * self.input_mean, axis=0)
            grad_sqrt_var = grad_inv_sqrt_var * (-self.inv_sqrt_var ** 2)
            grad_var = 0.5 * grad_sqrt_var * self.inv_sqrt_var
            grad_sqr_input_mean = np.full(input.shape, ((1 / input.shape[0]) * grad_var))
            grad_input_mean = grad_norm_input * np.full(
                input.shape, self.inv_sqrt_var
            ) + 2 * grad_sqr_input_mean * self.input_mean
            grad_mean = np.sum(-grad_input_mean, axis=0)

            return grad_input_mean + np.full(
                input.shape,
                (1 / input.shape[0]) * grad_mean
            )

        # Eval mode
        grad_norm_input = None
        if self.affine:
            grad_norm_input = grad_output * self.weight
        else:
            grad_norm_input = grad_output
        grad_input_rmean = grad_norm_input * np.full(
            input.shape,
            1 / np.sqrt(self.running_var + self.eps)
        )
        return grad_input_rmean

    def update_grad_parameters(self, input: np.array, grad_output: np.array):
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        """
        if self.affine:
            self.grad_bias += np.sum(grad_output, axis=0)
            self.grad_weight += np.sum(grad_output * self.norm_input, axis=0)

    def zero_grad(self):
        if self.affine:
            self.grad_weight.fill(0)
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.array]:
        return [self.weight, self.bias] if self.affine else []

    def parameters_grad(self) -> List[np.array]:
        return [self.grad_weight, self.grad_bias] if self.affine else []

    def __repr__(self) -> str:
        return f'BatchNormalization(num_features={len(self.running_mean)}, ' \
               f'eps={self.eps}, momentum={self.momentum}, affine={self.affine})'


class Dropout(Module):
    """
    Applies dropout transformation
    """

    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        if self.training:
            self.mask = np.random.choice([0, 1], input.shape, p=[self.p, 1 - self.p])
            return (1 / (1 - self.p)) * self.mask * input
        return input

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        if self.training:
            return grad_output * (1 / (1 - self.p)) * self.mask
        return grad_output

    def __repr__(self) -> str:
        return f'Dropout(p={self.p})'


class Sequential(Module):
    """
    Container for consecutive application of modules
    """

    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)
        self.outputs = []

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size matching the input size of the first layer
        :return: array of size matching the output size of the last layer
        """
        self.outputs.clear()
        for module in self.modules:
            self.outputs.append(input)
            input = module.compute_output(input)

        return input


    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size matching the input size of the first layer
        :param grad_output: array of size matching the output size of the last layer
        :return: array of size matching the input size of the first layer
        """
        if self.training:
            self.modules.reverse()
            self.outputs.reverse()

            for inp, module in zip(self.outputs, self.modules):
                grad_output = module.backward(inp, grad_output)

            self.modules.reverse()
            self.outputs.reverse()

        return grad_output


    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self) -> List[np.array]:
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self) -> List[np.array]:
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self) -> str:
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str
