import numpy as np
import soundfile as sf
from math import gcd
from scipy.signal import resample_poly
import librosa

def load_audio(file_path, target_sample_rate, top_db=25, normalize=True, trim_silence=True):
    signal, sample_rate = sf.read(file_path, dtype="float32")

    # handle stereo audio by converting it to mono by averaging the channels
    if signal.ndim == 2:
        signal = signal.mean(axis=1)

    # handle empty or invalid audio files
    if signal.size == 0:
        raise ValueError(f"Empty audio file: {file_path}")

    if not np.all(np.isfinite(signal)):
        raise ValueError(f"Invalid audio values: {file_path}")

    # resample the audio if the sample rate is different from the target sample rate
    if sample_rate != target_sample_rate:
        common = gcd(int(sample_rate), int(target_sample_rate))
        signal = resample_poly(signal, target_sample_rate // common, sample_rate // common).astype(np.float32)
        sample_rate = target_sample_rate

    # remove dc (direct current) offset
    signal = signal - np.mean(signal)

    # handle silent audio files
    if np.max(np.abs(signal)) == 0:
        raise ValueError(f"Silent audio file: {file_path}")

    # trim leading and trailing silence from the audio signal
    if trim_silence:
        signal, _ = librosa.effects.trim(signal, top_db=top_db)
    if signal.size == 0:
        raise ValueError(f"No usable audio after trimming: {file_path}")

    # normalize the audio signal
    peak = np.max(np.abs(signal))
    if normalize and peak > 0:
        signal = signal / peak

    return signal.astype(np.float32), sample_rate