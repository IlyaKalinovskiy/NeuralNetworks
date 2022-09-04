import numpy as np
import torch


class DataSample:
    def __init__(
        self,
        img_rel_path=None,
        wav_rel_path=None,
        labels=None,
        valence=None,
        arousal=None,
        landmarks=None,
    ):
        self.img_rel_path = img_rel_path
        self.wav_rel_path = wav_rel_path
        self.labels = labels
        self.valence = valence
        self.arousal = arousal
        self.landmarks = landmarks
        self.image = None
        self.text_labels = ""
        self.flag = False
        self.idx = -1


class DataGroup:
    def __init__(
        self,
        folder_rel_path=None,
        wav_rel_path=None,
        data_samples=None,
        labels=None,
        idx=-1,
    ):
        self.folder_rel_path = folder_rel_path
        self.wav_rel_path = wav_rel_path
        self.data_samples = data_samples
        self.labels = labels
        self.valence = 0
        self.arousal = 0
        self.text_labels = ""
        self.variables = {}
        self.idx = idx
        for sample in data_samples:
            sample.idx = idx


class TorchBatch:
    def __init__(self, images, labels, use_pin_memory=False):
        self.data = torch.from_numpy(np.stack(images))
        self.labels = torch.from_numpy(np.stack(labels).astype(np.float32))
        if use_pin_memory:
            self.data = self.data.pin_memory()
            self.labels = self.labels.pin_memory()


class Batch:
    def __init__(self, data_samples):
        self.data_samples = data_samples
