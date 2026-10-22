"""Recognition: menebak vowel dari satu rekaman (P4).

Semua sistem bekerja dengan pola yang sama: hitung skor ketidakmiripan input
terhadap model tiap vowel, lalu pilih skor terkecil. Perbedaannya hanya di
fungsi skornya (DTW cost atau negative log-likelihood), sehingga fungsi skor
diterima sebagai parameter dan file ini tidak bergantung langsung pada
implementasi P2/P3.
"""

from typing import Callable, Dict, Sequence, Tuple

import numpy as np

Scores = Dict[str, float]
Result = Tuple[str, Scores, float]


def recognize(X: np.ndarray, models: Dict[str, object], score_fn: Callable[[np.ndarray, object], float]) -> Result:
    """Fungsi umum: skor input ke tiap vowel, lalu pilih yang terkecil.

    Args:
        X: fitur MFCC satu rekaman, shape (T, 39).
        models: {vowel: model}; bentuk model bebas, asal dipahami score_fn.
        score_fn: score_fn(X, model) -> float, makin kecil makin mirip.

    Returns:
        (pred, scores, margin): vowel tebakan, skor per vowel, dan selisih skor
        terbaik kedua dengan terbaik (makin besar = makin yakin).
    """
    scores = {v: float(score_fn(X, m)) for v, m in models.items()}
    ranked = sorted(scores, key=scores.get)
    margin = scores[ranked[1]] - scores[ranked[0]] if len(ranked) > 1 else float("inf")
    return ranked[0], scores, margin


def recognize_dtw(X: np.ndarray, templates: Dict[str, np.ndarray], dtw_fn: Callable[[np.ndarray, np.ndarray], float]) -> Result:
    """Sistem 1: DTW terhadap averaged template tiap vowel."""
    return recognize(X, templates, dtw_fn)


def recognize_gaussian(X: np.ndarray, models: Dict[str, object], nll_fn: Callable[[np.ndarray, object], float]) -> Result:
    """Sistem 2: negative log-likelihood terhadap model Gaussian tiap vowel."""
    return recognize(X, models, nll_fn)


def recognize_dtw_nn(X: np.ndarray, templates_per_vowel: Dict[str, Sequence[np.ndarray]], dtw_fn: Callable[[np.ndarray, np.ndarray], float]) -> Result:
    """Varian pembanding: DTW ke setiap template individual (tanpa dirata-rata),
    skor vowel = cost terkecil di antara template-template vowel tersebut."""
    return recognize(X, templates_per_vowel, lambda X, ts: min(dtw_fn(X, t) for t in ts))
