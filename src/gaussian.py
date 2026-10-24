import numpy as np
from scipy.linalg import cho_factor, cho_solve

COVARIANCE_REG = 1e-3
# Penyusutan covariance ke diagonalnya. Full covariance 39x39 (819 parameter) hanya
# dilatih dari 184-312 frame per vowel, sehingga korelasi antar dimensi diredam.
# 0.0 = full covariance (baseline awal), 1.0 = diagonal.
SHRINKAGE = 0.6

def _as_frames(A, name):
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] == 0:
        raise ValueError(f"{name} harus berbentuk (T, dimensi) dengan T >= 1, dapat {A.shape}")
    return A

def fit_gaussian(feats_list, reg=COVARIANCE_REG, shrinkage=SHRINKAGE):
    """Estimasi Gaussian multivariat satu vowel dari gabungan semua frame latihnya.

    Args:
        feats_list: list fitur MFCC per rekaman, tiap elemen shape (T_i, D).
        reg: nilai yang ditambahkan ke diagonal covariance agar matriks pasti
            positive definite (frame MFCC bertetangga sangat berkorelasi).
        shrinkage: 0..1, bobot penyusutan covariance ke diagonalnya:
            cov = (1 - shrinkage) * cov + shrinkage * diag(cov). 0 = full covariance,
            1 = covariance diagonal. Mengurangi jumlah parameter efektif saat frame
            latih sedikit dibanding D*(D+1)/2; varians tiap dimensi tidak berubah.

    Returns:
        (mean, cov): mean shape (D,), covariance shape (D, D).
    """
    if len(feats_list) == 0:
        raise ValueError("feats_list kosong, butuh minimal satu rekaman")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError(f"shrinkage harus di [0, 1], dapat {shrinkage}")
    frames = np.vstack([_as_frames(f, f"feats_list[{i}]") for i, f in enumerate(feats_list)])
    if len(frames) < 2:
        raise ValueError("Covariance butuh minimal 2 frame")
    cov = np.cov(frames, rowvar=False)
    cov = (1.0 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))
    return frames.mean(axis=0), cov + reg * np.eye(frames.shape[1])

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
    base = fit_gaussian(train, shrinkage=0.0)[1]
    assert np.allclose(base, np.cov(np.vstack(train), rowvar=False) + COVARIANCE_REG * np.eye(5)), "shrinkage=0 harus full covariance"
    assert np.allclose(fit_gaussian(train)[1], fit_gaussian(train, shrinkage=SHRINKAGE)[1]), "default harus SHRINKAGE"
    diag = fit_gaussian(train, shrinkage=1.0)[1]
    assert np.allclose(diag, np.diag(np.diag(diag))), "shrinkage=1 harus diagonal"
    assert np.allclose(np.diag(fit_gaussian(train, shrinkage=0.5)[1]), np.diag(base)), "shrinkage tidak mengubah varians"
    print("self-check gaussian OK")
