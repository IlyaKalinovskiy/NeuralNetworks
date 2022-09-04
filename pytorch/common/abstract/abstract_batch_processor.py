import abc


class AbstractBatchProcessor:
    """Base class for data processors."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """Everything that we need to init."""
        pass

    @abc.abstractmethod
    def pre_processing(self):
        """Returns processed batch."""
        pass

    @abc.abstractmethod
    def post_processing(self):
        """Returns processed batch."""
        pass
