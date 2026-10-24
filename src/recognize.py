from typing import Callable, Dict, Sequence, Tuple

import numpy as np

Scores = Dict[str, float]
Result = Tuple[str, Scores, float]


# Input : fitur MFCC rekaman uji, model tiap vowel, fungsi penghitung skor ketidakmiripan
# Output: vowel tebakan (skor terkecil), skor ke tiap vowel, margin keyakinan
def recognize(X: np.ndarray, models: Dict[str, object], score_fn: Callable[[np.ndarray, object], float]) -> Result:
    scores = {v: float(score_fn(X, m)) for v, m in models.items()}
    ranked = sorted(scores, key=scores.get)
    margin = scores[ranked[1]] - scores[ranked[0]] if len(ranked) > 1 else float("inf")
    return ranked[0], scores, margin


# Input : fitur MFCC rekaman uji, averaged template tiap vowel, fungsi DTW
# Output: vowel tebakan, cost DTW ke tiap vowel, margin keyakinan
def recognize_dtw(X: np.ndarray, templates: Dict[str, np.ndarray], dtw_fn: Callable[[np.ndarray, np.ndarray], float]) -> Result:
    return recognize(X, templates, dtw_fn)


# Input : fitur MFCC rekaman uji, model Gaussian (mean, covariance) tiap vowel, fungsi NLL
# Output: vowel tebakan, negative log-likelihood ke tiap vowel, margin keyakinan
def recognize_gaussian(X: np.ndarray, models: Dict[str, object], nll_fn: Callable[[np.ndarray, object], float]) -> Result:
    return recognize(X, models, nll_fn)


# Input : fitur MFCC rekaman uji, semua rekaman template tiap vowel (tanpa dirata-rata), fungsi DTW
# Output: vowel tebakan, cost DTW terkecil ke tiap vowel, margin keyakinan
def recognize_dtw_nn(X: np.ndarray, templates_per_vowel: Dict[str, Sequence[np.ndarray]], dtw_fn: Callable[[np.ndarray, np.ndarray], float]) -> Result:
    return recognize(X, templates_per_vowel, lambda X, ts: min(dtw_fn(X, t) for t in ts))
