import numpy as np


class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """

    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0  # use in __next__, reset in __iter__

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """
        if self.y.shape[0] % self.batch_size == 0:
            return self.y.shape[0] // self.batch_size
        return self.y.shape[0] // self.batch_size + 1

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        return self.y.shape[0]

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        self.batch_id = 0
        if self.shuffle:
            ixs = np.random.permutation(self.y.shape[0])
            self.X = self.X[ixs]
            self.y = self.y[ixs]
        return self

    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
        self.batch_id += 1
        if self.batch_id < self.__len__() + 1:
            if self.batch_id == self.__len__():
                return self.X[((self.batch_id - 1) * self.batch_size):], \
                       self.y[((self.batch_id - 1) * self.batch_size):]

            return self.X[((self.batch_id - 1) * self.batch_size):(self.batch_id * self.batch_size)], \
                   self.y[((self.batch_id - 1) * self.batch_size):(self.batch_id * self.batch_size)]
        raise StopIteration
