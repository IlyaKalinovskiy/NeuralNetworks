import abc


class AbstractDataSampler:
    """Base class for data loaders for pyTorch nets."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, data):
        """Everything that we need to init."""
        pass

    @abc.abstractmethod
    def sampling(self, batch_size, external_info=None):
        """Returns next batch from training data.

        Make sure your function returns Batch

        """
        pass
