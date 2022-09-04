import numpy as np
import librosa
import librosa.filters

from scipy import signal
from scipy.io import wavfile


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sr, wav.astype(np.int16))


def spectrogram(y, pstft, ref_level_db=20):
    D = _stft(_preemphasis(y), pstft)
    S = _amp_to_db(np.abs(D)) - ref_level_db
    return _normalize(S)


def inv_spectrogram(spectrogram, pstft, ref_level_db=20, power=1.5):
    S = _db_to_amp(_denormalize(spectrogram) + ref_level_db)  # Convert back to linear
    return _inv_preemphasis(_griffin_lim(S**power, pstft))  # Reconstruct phase


def melspectrogram(y, pstft, mel_basis):
    D = _stft(_preemphasis(y), pstft)
    S = _amp_to_db(_linear_to_mel(np.abs(D), mel_basis))
    return _normalize(S)


def inv_melspectrogram(melspectrogram, pstft, inv_mel_basis):
    S = _mel_to_linear(
        _db_to_amp(_denormalize(melspectrogram)), inv_mel_basis
    )  # Convert back to linear
    return _inv_preemphasis(_griffin_lim(S**1.5, pstft))  # Reconstruct phase


# Based on https://github.com/librosa/librosa/issues/434
def _griffin_lim(S, pstft, griffin_lim_iters=100):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    for i in range(griffin_lim_iters):
        if i > 0:
            angles = np.exp(1j * np.angle(_stft(y, pstft)))
        y = _istft(S_complex * angles, pstft)
    return y


def _stft(y, pstft):
    return librosa.stft(
        y=y,
        n_fft=pstft["filter_length"],
        hop_length=pstft["hop_length"],
        win_length=pstft["win_length"],
    )


def _istft(y, pstft):
    return librosa.istft(
        y, hop_length=pstft["hop_length"], win_length=pstft["win_length"]
    )


# Conversions:


def _linear_to_mel(spectrogram, mel_basis):
    return np.dot(mel_basis, spectrogram)


def _mel_to_linear(mel_spectrogram, inv_mel_basis):
    return np.maximum(1e-10, np.dot(inv_mel_basis, mel_spectrogram))


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _preemphasis(x, preemphasis=0.97):
    return signal.lfilter([1, -preemphasis], [1], x)


def _inv_preemphasis(x, preemphasis=0.97):
    return signal.lfilter([1], [1, -preemphasis], x)


def _normalize(S, max_abs_value=4, min_level_db=-100):
    return np.clip(
        (2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value,
        -max_abs_value,
        max_abs_value,
    )


def _denormalize(D, max_abs_value=4, min_level_db=-100):
    return (
        (np.clip(D, -max_abs_value, max_abs_value) + max_abs_value)
        * -min_level_db
        / (2 * max_abs_value)
    ) + min_level_db


def get_hop_size(sample_rate, frame_shift_ms=12.5):
    assert frame_shift_ms is not None
    hop_length = int(frame_shift_ms / 1000 * sample_rate)
    return hop_length
