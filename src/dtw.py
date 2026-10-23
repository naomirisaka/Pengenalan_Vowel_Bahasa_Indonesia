import numpy as np

def _as_sequence(A, name):
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] == 0:
        raise ValueError(f"{name} harus berbentuk (T, dimensi) dengan T >= 1, dapat {A.shape}")
    return A

def local_cost(X, Y):
    X = _as_sequence(X, "X")
    Y = _as_sequence(Y, "Y")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"Dimensi fitur berbeda: {X.shape[1]} vs {Y.shape[1]}")
    squared = np.zeros((len(X), len(Y)))
    for d in range(X.shape[1]):
        diff = X[:, d, None] - Y[None, :, d]
        squared += diff * diff
    return np.sqrt(squared)

def accumulate_cost(cost):
    n, m = cost.shape
    D = np.full((n + 1, m + 1), np.inf)
    L = np.zeros((n + 1, m + 1), dtype=np.int64)
    D[0, 0] = 0.0
    for k in range(2, n + m + 1):
        i = np.arange(max(1, k - m), min(n, k - 1) + 1)
        j = k - i
        # Kandidat predecessor, berurutan: (i-1, j), (i, j-1), (i-1, j-1)
        prev_i = np.stack([i - 1, i, i - 1])
        prev_j = np.stack([j, j - 1, j - 1])
        best = np.argmin(D[prev_i, prev_j], axis=0)
        cols = np.arange(i.size)
        bi, bj = prev_i[best, cols], prev_j[best, cols]
        D[i, j] = cost[i - 1, j - 1] + D[bi, bj]
        L[i, j] = L[bi, bj] + 1
    return D, L

def backtrack(D):
    i, j = D.shape[0] - 1, D.shape[1] - 1
    path = [(i - 1, j - 1)]
    while (i, j) != (1, 1):
        candidates = [(i - 1, j), (i, j - 1), (i - 1, j - 1)]
        i, j = candidates[int(np.argmin([D[c] for c in candidates]))]
        path.append((i - 1, j - 1))
    return np.array(path[::-1])

def dtw_path(X, Y):
    D, _ = accumulate_cost(local_cost(X, Y))
    path = backtrack(D)
    return float(D[-1, -1] / len(path)), path

def dtw_distance(X, Y):
    D, L = accumulate_cost(local_cost(X, Y))
    return float(D[-1, -1] / L[-1, -1])