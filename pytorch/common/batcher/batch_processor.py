import numpy as np

from pytorch.common.abstract.abstract_batch_processor import AbstractBatchProcessor
from pytorch.common.batcher.batch_primitives import TorchBatch


class SimpleBatchProcessor(AbstractBatchProcessor):
    """Simple data processors."""

    def __init__(self, cuda_id, use_pin_memory, use_async):
        """Everything that we need to init."""
        self.cuda_id = cuda_id
        self.use_pin_memory = use_pin_memory
        self.use_async = use_async

    def pre_processing(self, batch):
        """Returns processing batch."""
        torchBatch = TorchBatch(
            [sample.image for sample in batch.data_samples],
            [[sample.valence, sample.arousal] for sample in batch.data_samples],
            use_pin_memory=self.use_pin_memory,
        )
        data = torchBatch.data
        target = torchBatch.labels

        if self.cuda_id != -1:
            data, target = data.to(self.cuda_id), target.to(self.cuda_id)

        return data, target

    def post_processing(self, data):
        """Returns processing batch."""
        pass


class BatchProcessor4D(AbstractBatchProcessor):
    """Simple data processors."""

    def __init__(self, depth, cuda_id, use_pin_memory, use_async):
        """Everything that we need to init."""
        self.depth = depth
        self.cuda_id = cuda_id
        self.use_pin_memory = use_pin_memory
        self.use_async = use_async

    def pre_processing(self, batch):
        """Returns processing batch."""
        frames = []
        labels = []
        for i in range(len(batch.data_samples) // self.depth):
            image3D, valence, arousal = [], [], []
            for j in range(self.depth):
                image3D.append(batch.data_samples[i * self.depth + j].image)
                valence.append(batch.data_samples[i * self.depth + j].valence)
                arousal.append(batch.data_samples[i * self.depth + j].arousal)
            frame = np.stack(image3D)
            frames.append(np.transpose(frame, (1, 0, 2, 3)))
            labels.append([np.mean(valence), np.mean(arousal)])

        torchBatch = TorchBatch(frames, labels, use_pin_memory=self.use_pin_memory)
        data = torchBatch.data
        target = torchBatch.labels

        if self.cuda_id != -1:
            data, target = data.to(self.cuda_id), target.to(self.cuda_id)

        return data, target

    def post_processing(self, data):
        """Returns processing batch."""
        pass
