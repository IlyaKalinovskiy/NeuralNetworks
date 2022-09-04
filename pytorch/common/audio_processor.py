import numpy as np
import torch
import librosa

from librosa.filters import mel as librosa_mel_fn
from scipy.io import wavfile

from pytorch.common.audio_preprocessing import librosa_utils


class LibrosaAudioProcessor:
    """Simple data processors."""

    def __init__(
        self,
        type,
        sampling_rate,
        max_wave_value,
        sfft={"filter_length": 1024, "hop_length": 256, "win_length": 1024},
        mel=None,
    ):
        """Everything that we need to init."""
        self.type = type
        self.sampling_rate = sampling_rate
        self.max_wave_value = max_wave_value
        self.sfft = sfft

        assert not mel is None
        mel_fmax = None if mel["mel_fmax"] == "None" else mel["mel_fmax"]
        self.mel_basis = librosa_mel_fn(
            self.sampling_rate,
            sfft["filter_length"],
            mel["n_mel_channels"],
            mel["mel_fmin"],
            mel_fmax,
        )
        self.inv_mel_basis = np.linalg.pinv(self.mel_basis)
        self.num_channels = mel["n_mel_channels"]

    def spec2wave(self, spec):
        if type(spec) is torch.Tensor:
            spec = spec.data.cpu().numpy()
        wave = librosa_utils.inv_melspectrogram(spec, self.sfft, self.inv_mel_basis)
        return wave

    def get_num_channels(self):
        return self.num_channels

    def load_wave(self, full_path):
        return librosa.load(full_path, self.sampling_rate)[0]

    def load_spec(self, spec):
        return np.load(spec)

    def save_wave(self, path, wave):
        wavfile.write(path, self.sampling_rate, self.to_int(wave))

    def to_int(self, wave):
        wave *= self.max_wave_value / max(0.01, np.max(np.abs(wave)))
        return wave.astype(np.int16)

    def get_spec(self, wave):
        """Get spectrogram of loaded audio."""
        mel = librosa_utils.melspectrogram(wave, self.sfft, self.mel_basis)
        return torch.FloatTensor(mel)

    def process(self, wav_path):
        """Returns processed data."""
        wave = self.load_wave(wav_path)
        return self.get_spec(wave)
