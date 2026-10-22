"""Implementasi SEMENTARA untuk bagian P1-P3.

Hanya dipakai runner dengan flag --use-stubs, supaya alur pengujian bisa dites
sebelum kode asli P1 (features.py), P2 (dtw.py, template.py), dan P3
(gaussian.py) selesai. Hapus file ini setelah kode asli masuk.

Signature di sini sekaligus menjadi kontrak yang diharapkan runner:
    extract_mfcc39(path) -> (T, 39)
    build_averaged_template(list_of_(T_i, 39)) -> (T, 39)
    dtw_distance(X, Y) -> float
    fit_gaussian(list_of_(T_i, 39)) -> model
    nll(X, model) -> float
"""

import numpy as np
import soundfile as sf
from python_speech_features import delta, mfcc
from scipy.linalg import cho_factor, cho_solve
from scipy.signal import resample_poly
from scipy.spatial.distance import cdist

SAMPLE_RATE = 16000
COVARIANCE_REG = 1e-3


def extract_mfcc39(path):
    """13 MFCC + delta + delta-delta, dinormalisasi per rekaman (CMVN)."""
    signal, sr = sf.read(path, dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if sr != SAMPLE_RATE:
        signal = resample_poly(signal, SAMPLE_RATE, sr)
    static = mfcc(signal, SAMPLE_RATE, winlen=0.025, winstep=0.01, numcep=13, nfft=512)
    d1 = delta(static, 2)
    feats = np.hstack([static, d1, delta(d1, 2)])
    return (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)


def build_averaged_template(feats_list):
    """Interpolasi linear setiap rekaman ke panjang median, lalu dirata-rata."""
    length = int(np.median([len(f) for f in feats_list]))
    grid = np.linspace(0, 1, length)
    resampled = [
        np.column_stack([np.interp(grid, np.linspace(0, 1, len(f)), f[:, d]) for d in range(f.shape[1])])
        for f in feats_list
    ]
    return np.mean(resampled, axis=0)


def dtw_distance(X, Y):
    """DTW jarak Euclidean, step (1,0)/(0,1)/(1,1), dinormalisasi panjang path.

    Diisi per anti-diagonal agar bisa divektorisasi dengan numpy.
    """
    n, m = len(X), len(Y)
    cost = cdist(X, Y)
    D = np.full((n + 1, m + 1), np.inf)
    L = np.zeros((n + 1, m + 1), dtype=np.int64)
    D[0, 0] = 0.0
    for k in range(2, n + m + 1):
        i = np.arange(max(1, k - m), min(n, k - 1) + 1)
        j = k - i
        prev_i = np.stack([i - 1, i, i - 1])
        prev_j = np.stack([j, j - 1, j - 1])
        best = np.argmin(D[prev_i, prev_j], axis=0)
        cols = np.arange(len(i))
        D[i, j] = cost[i - 1, j - 1] + D[prev_i[best, cols], prev_j[best, cols]]
        L[i, j] = L[prev_i[best, cols], prev_j[best, cols]] + 1
    return D[n, m] / L[n, m]


def fit_gaussian(feats_list):
    """Mean dan full covariance dari gabungan semua frame, plus regularisasi diagonal."""
    frames = np.vstack(feats_list)
    cov = np.cov(frames, rowvar=False) + COVARIANCE_REG * np.eye(frames.shape[1])
    return frames.mean(axis=0), cov


def nll(X, model):
    """Negative log-likelihood rata-rata per frame."""
    mean, cov = model
    chol = cho_factor(cov, lower=True)
    diff = X - mean
    mahalanobis = np.einsum("ij,ji->i", diff, cho_solve(chol, diff.T))
    log_det = 2.0 * np.log(np.diag(chol[0])).sum()
    return float(np.mean(0.5 * (X.shape[1] * np.log(2 * np.pi) + log_det + mahalanobis)))
