import abc


class AbstractLRScheduler:
    """Base class for data loaders for pyTorch nets."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, optimizer):
        """Everything that we need to init."""
        pass

    @abc.abstractmethod
    def step(self):
        """Update learning rate."""
        pass
