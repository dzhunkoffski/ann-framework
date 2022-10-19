import numpy as np
import scipy
from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(input, np.full(input.shape, 0))

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        input[np.where(input < 0)] = 0
        input[np.where(input > 0)] = 1
        return grad_output * input


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return scipy.special.expit(input)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        sigmoid_x = scipy.special.expit(input)
        return grad_output * sigmoid_x * (1 - sigmoid_x)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return scipy.special.softmax(input, axis=1)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size

        Used algorithm from here https://themaverickmeerkat.com/2019-10-23-Softmax/
        """
        prob = self.compute_output(input)
        t1 = np.einsum('ij,ik->ijk', prob, prob)
        t2 = np.einsum('ij,jk->ijk', prob, np.eye(input.shape[1], input.shape[1]))
        dSoft = t2 - t1
        return np.einsum('ijk,ik->ij', dSoft, grad_output)


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return scipy.special.log_softmax(input, axis=1)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        prob = scipy.special.softmax(input, axis=1)
        t1 = np.einsum('ij,ik->ijk', prob, prob)
        t2 = np.einsum('ij,jk->ijk', prob, np.eye(input.shape[1], input.shape[1]))
        dSoft = t2 - t1
        return np.einsum('ijk,ik->ij', dSoft, grad_output / prob)