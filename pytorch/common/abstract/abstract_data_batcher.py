import abc


class AbstractDataBatcher:
    """Base class for data loaders for pyTorch nets."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """Everything that we need to init."""
        pass

    @abc.abstractmethod
    def next_batch(self):
        """Returns next batch from training data.

        Make sure your function returns (batch_num, data, labels)

        """
        pass

    def start(self):
        """Starts batch generation routine."""
        pass

    def finish(self):
        """Finish batch generation routine."""
        pass
