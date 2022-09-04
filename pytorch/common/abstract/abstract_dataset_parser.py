import abc


class AbstractDatasetParser:
    """Base class for data loaders for pyTorch nets."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """Everything that we need to init."""
        pass

    @abc.abstractmethod
    def get_dataset_root(self):
        """Returns dataset root folder."""
        pass

    @abc.abstractmethod
    def get_dataset_size(self):
        """Returns number of examples in dataset."""
        pass

    @abc.abstractmethod
    def get_labels_names(self):
        """Returns next batch from validation data.

        Make sure your function returns (is_finished, data, labels)

        """
        pass

    @abc.abstractmethod
    def read_data_samples(self):
        """Read all data samples from dataset.

        Returns list of common.DataSample without loaded images. Image path and
        labels only.

        """
        pass
