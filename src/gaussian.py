import numpy as np
from scipy.linalg import cho_factor, cho_solve

COVARIANCE_REG = 1e-3

def _as_frames(A, name):
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] == 0:
        raise ValueError(f"{name} harus berbentuk (T, dimensi) dengan T >= 1, dapat {A.shape}")
    return A

def fit_gaussian(feats_list, reg=COVARIANCE_REG):
    """Estimasi Gaussian multivariat satu vowel dari gabungan semua frame latihnya.

    Args:
        feats_list: list fitur MFCC per rekaman, tiap elemen shape (T_i, D).
        reg: nilai yang ditambahkan ke diagonal covariance agar matriks pasti
            positive definite (frame MFCC bertetangga sangat berkorelasi).

    Returns:
        (mean, cov): mean shape (D,), full covariance shape (D, D).
    """
    if len(feats_list) == 0:
        raise ValueError("feats_list kosong, butuh minimal satu rekaman")
    frames = np.vstack([_as_frames(f, f"feats_list[{i}]") for i, f in enumerate(feats_list)])
    if len(frames) < 2:
        raise ValueError("Covariance butuh minimal 2 frame")
    cov = np.cov(frames, rowvar=False) + reg * np.eye(frames.shape[1])
    return frames.mean(axis=0), cov

def nll(X, model):
    """Negative log-likelihood rata-rata per frame (makin kecil makin cocok).

    NLL(x) = 0.5 * (D*log(2*pi) + log|cov| + (x-mean)^T cov^-1 (x-mean)),
    frame dianggap independen, lalu dirata-rata agar tidak bergantung durasi.
    """
    mean, cov = model
    X = _as_frames(X, "X")
    if X.shape[1] != len(mean):
        raise ValueError(f"Dimensi fitur berbeda: {X.shape[1]} vs {len(mean)}")
    chol = cho_factor(cov, lower=True)
    diff = X - mean
    mahalanobis = np.einsum("ij,ji->i", diff, cho_solve(chol, diff.T))
    log_det = 2.0 * np.log(np.diag(chol[0])).sum()
    return float(np.mean(0.5 * (X.shape[1] * np.log(2 * np.pi) + log_det + mahalanobis)))

if __name__ == "__main__":
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(0)
    train = [rng.normal(size=(40, 5)) @ rng.normal(size=(5, 5)) + 1.0 for _ in range(3)]
    test = rng.normal(size=(30, 5))
    model = fit_gaussian(train)
    ref = -multivariate_normal(model[0], model[1]).logpdf(test).mean()
    assert np.isclose(nll(test, model), ref), (nll(test, model), ref)
    near, far = rng.multivariate_normal(model[0], model[1], 30), test + 20
    assert nll(near, model) < nll(far, model)
    print("self-check gaussian OK")
