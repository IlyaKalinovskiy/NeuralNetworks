import abc


class AbstractDataProcessor:
    """Base class for data processors."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """Everything that we need to init."""
        pass

    @abc.abstractmethod
    def process(self):
        """Returns processed data."""
        pass
