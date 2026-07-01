"""
noHub — Hubness Reduction via Dimensionality Reduction.

noHub learns a low-dimensional embedding that preserves local neighbourhood
structure while explicitly penalising hub formation through a combined
align + uniform loss.  A linear map is fitted on a subsample and then
applied to the full dataset for scalability.

Reference
---------
Trosten et al., "Hubs and Hyperspheres: Reducing Hubness and Improving
Transductive Few-Shot Learning with Hyperspherical Embeddings", 2023.
"""

from __future__ import annotations

import numpy as np


def nohub_embed(
    base: np.ndarray,
    queries: np.ndarray,
    malicious: np.ndarray,
    *,
    out_dims: int = 400,
    kappa: float = 0.5,
    perplexity: float = 45.0,
    n_iter: int = 50,
    learning_rate: float = 0.1,
    align_weight: float = 0.2,
    max_samples: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply noHub hubness reduction to all three vector sets.

    To keep memory and time under control, the noHub optimisation runs on
    at most ``max_samples`` vectors (subsampled uniformly).  A linear
    least-squares map is then fitted and applied to the full dataset.

    Parameters
    ----------
    base : np.ndarray  (N, d)
        Corpus vectors.
    queries : np.ndarray  (Q, d)
        Query vectors.  May be empty (0, d).
    malicious : np.ndarray  (M, d)
        Malicious vectors.  May be empty (0, d).
    out_dims : int
        Output dimensionality (≤ input dim).
    kappa : float
        Scaling factor for the von-Mises–Fisher kernel.
    perplexity : float
        Target perplexity for the neighbourhood probability matrix.
    n_iter : int
        Number of Adam optimisation steps.
    learning_rate : float
        Learning rate for Adam.
    align_weight : float
        Weight of the alignment loss (1 − align_weight for the uniform loss).
    max_samples : int
        Maximum number of vectors used for the noHub optimisation itself.
    seed : int
        Random seed for subsampling.

    Returns
    -------
    base_t, queries_t, malicious_t : np.ndarray
        Transformed vectors (L2-normalised, output dim ≤ input dim).
    """
    # ── stack all vectors, subsample if needed ──────────────────────────
    all_x = np.vstack([base, queries, malicious]).astype(np.float32)
    total = all_x.shape[0]
    max_n = int(max_samples)
    rng = np.random.default_rng(int(seed))
    if total > max_n:
        idx = rng.choice(total, size=max_n, replace=False)
    else:
        idx = np.arange(total)
    x_sub = all_x[idx].astype(np.float64)
    n_sub, in_dims = x_sub.shape
    out_dims = int(min(out_dims, in_dims))

    def _l2(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    # ── PCA initialisation ──────────────────────────────────────────────
    if out_dims < in_dims:
        x_centered = x_sub - x_sub.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
        init = x_centered @ vt[:out_dims].T
    else:
        init = x_sub
    embeddings = _l2(init)

    # ── Build P matrix (vMF kernel + perplexity calibration) ────────────
    x_norm = _l2(x_sub)
    dist = -(x_norm @ x_norm.T)
    np.fill_diagonal(dist, 1.0)

    def _ent_beta(d, beta):
        b = beta[:, None]
        p = np.exp(-d * b)
        p_sum = p.sum(axis=1, keepdims=True)
        ent = np.log(p_sum) + b * np.sum(d * p, axis=1, keepdims=True) / p_sum
        p = p / p_sum
        return ent.squeeze(), p

    log_u = np.log(float(perplexity))
    tol = 1e-2 * log_u
    beta = np.ones(n_sub)
    beta_min = np.full(n_sub, np.nan)
    beta_max = np.full(n_sub, np.nan)
    ent, p = _ent_beta(dist, beta)
    for _ in range(20):
        ent_diff = ent - log_u
        gt_mask = ent_diff > tol
        lt_mask = ent_diff < -tol
        if (not np.any(gt_mask)) and (not np.any(lt_mask)):
            break
        if np.any(gt_mask):
            beta_min = np.where(gt_mask, beta, beta_min)
            beta = np.where(
                gt_mask,
                np.where(np.isnan(beta_max), beta * 2, (beta + beta_max) / 2),
                beta,
            )
        if np.any(lt_mask):
            beta_max = np.where(lt_mask, beta, beta_max)
            beta = np.where(
                lt_mask,
                np.where(np.isnan(beta_min), beta / 2, (beta + beta_min) / 2),
                beta,
            )
        ent, p = _ent_beta(dist, beta)
    p = (p + p.T) / 2.0
    p = p / p.sum()
    p = np.maximum(p, 1e-12)

    # ── Optimise (analytic gradient + Adam, CPU/numpy only) ─────────────
    # Loss L = aw * align + (1-aw) * uniform, with
    #   sim     = E @ E.T
    #   align   = -(kappa * P * sim).sum()
    #   uniform = logsumexp(kappa * sim)            (over all entries)
    # Since L depends on E only through sim, dL/dE = (G + G.T) @ E with
    #   G = dL/dsim = kappa * ( -aw * P + (1-aw) * softmax(kappa*sim) ).
    # After each step the rows are re-projected onto the unit sphere
    # (matching the torch no_grad normalisation, which carries no gradient).
    kappa_v = float(kappa)
    aw = float(align_weight)
    lr = float(learning_rate)
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = np.zeros_like(embeddings)
    v = np.zeros_like(embeddings)
    for t in range(1, int(n_iter) + 1):
        sim = embeddings @ embeddings.T
        # softmax over all entries (numerically stable)
        z = kappa_v * sim
        z = z - z.max()
        sm = np.exp(z)
        sm = sm / sm.sum()
        g_sim = kappa_v * (-aw * p + (1.0 - aw) * sm)
        grad = (g_sim + g_sim.T) @ embeddings
        # Adam update
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * (grad * grad)
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        embeddings = embeddings - lr * m_hat / (np.sqrt(v_hat) + eps)
        embeddings = _l2(embeddings)

    y_sub = embeddings.astype(np.float32)

    # ── Fit linear map:  x_sub @ W ≈ y_sub ─────────────────────────────
    W, *_ = np.linalg.lstsq(all_x[idx].astype(np.float32), y_sub, rcond=None)

    # ── Apply to all vectors + L2-normalise output ──────────────────────
    def _map(x: np.ndarray) -> np.ndarray:
        y = x.astype(np.float32) @ W.astype(np.float32)
        norms = np.linalg.norm(y, axis=1, keepdims=True) + 1e-10
        return y / norms

    base_t = _map(base)
    queries_t = _map(queries) if queries.shape[0] > 0 else queries
    malicious_t = _map(malicious) if malicious.shape[0] > 0 else malicious

    return base_t, queries_t, malicious_t
