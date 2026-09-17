"""Spatial interpretability utilities for dense field representations.

Adapted from cybergis/rs-embed (examples/plot_utils.py). Provides a
dependency-light (SVD-based, no sklearn) PCA pseudocolor projection of a
(D, H, W) embedding grid into a 3-channel RGB image, with robust percentile
scaling and sign-stabilized components so visualizations are deterministic
and comparable across fields. Used to render predicted yield maps and latent
field representations side by side and to visually inspect whether the model
preserves within-field low/high-yield zones.
"""
import numpy as np


def _to_dhw(data):
    """Coerce an embedding array to a (D, H, W) grid."""
    data = np.asarray(data)
    if data.ndim == 3:
        return data
    if data.ndim == 2:
        return data[None, ...]
    if data.ndim == 4:
        return data[0]
    raise ValueError("Expected a (D, H, W) grid, got shape {}".format(data.shape))


def _stabilize_pca_sign(comps):
    """Make PCA component signs deterministic (largest-magnitude entry positive)."""
    comps = np.array(comps, dtype=np.float32)
    for i in range(comps.shape[0]):
        idx = int(np.argmax(np.abs(comps[i])))
        if comps[i, idx] < 0:
            comps[i] = -comps[i]
    return comps


def fit_pca_rgb(emb, *, n_samples=100_000, seed=0, center=True):
    """Fit PCA on pixels of a (D, H, W) grid and return a dict for reuse.

    No sklearn dependency (uses SVD). Returns a dict with 'mean',
    'components' and 'center' that can be passed to transform_pca_rgb.
    """
    data = getattr(emb, "data", emb)
    dhw = _to_dhw(data)
    D, H, W = dhw.shape

    X = dhw.reshape(D, H * W).T  # [N, D]
    finite_rows = np.all(np.isfinite(X), axis=1)
    X = X[finite_rows]
    N = X.shape[0]

    if N == 0:
        raise ValueError("No finite pixels available for PCA fit.")

    rng = np.random.default_rng(seed)
    if n_samples is not None and N > n_samples:
        idx = rng.choice(N, size=int(n_samples), replace=False)
        Xs = X[idx]
    else:
        Xs = X

    mean = Xs.mean(axis=0) if center else np.zeros((D,), dtype=np.float32)
    Xc = Xs - mean

    if Xc.shape[0] < 2 or np.allclose(np.nanstd(Xc, axis=0), 0.0):
        comps = np.eye(D, dtype=np.float32)[:3]
    else:
        try:
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            comps = Vt[:3].astype(np.float32)
        except np.linalg.LinAlgError:
            cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1]
            comps = vecs[:, order[:3]].T.astype(np.float32)
        comps = _stabilize_pca_sign(comps)

    return {
        "mean": mean.astype(np.float32),
        "components": comps,
        "center": bool(center),
    }


def transform_pca_rgb(emb, pca, *, pmin=2, pmax=98):
    """Project a (D, H, W) embedding grid to a 3-channel pseudocolor image.

    Uses the PCA fit from fit_pca_rgb, then applies robust percentile
    scaling to [0, 1] so the visualization is stable across fields.
    """
    dhw = _to_dhw(emb)
    D, H, W = dhw.shape
    comps = pca["components"]
    mean = pca["mean"]

    X = dhw.reshape(D, H * W).T  # [N, D]
    if pca["center"]:
        X = X - mean
    proj = X @ comps.T  # [N, 3]

    proj = np.nan_to_num(proj, nan=0.0, posinf=0.0, neginf=0.0)
    lo = np.percentile(proj, pmin, axis=0)
    hi = np.percentile(proj, pmax, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    rgb = (proj - lo) / span
    rgb = np.clip(rgb, 0.0, 1.0)

    return rgb.reshape(H, W, 3).astype(np.float32)


def plot_embedding_pseudocolor(emb, pca=None, *, ax=None, pmin=2, pmax=98, title=None):
    """Render a (D, H, W) embedding grid as a pseudocolor RGB image.

    If pca is None, a PCA fit is computed on the fly. Returns the rendered
    RGB array (H, W, 3) and optionally draws it on the given matplotlib axis.
    """
    if pca is None:
        pca = fit_pca_rgb(emb)
    rgb = transform_pca_rgb(emb, pca, pmin=pmin, pmax=pmax)

    if ax is not None:
        ax.imshow(rgb)
        ax.set_xticks([])
        ax.set_yticks([])
        if title is not None:
            ax.set_title(title)

    return rgb


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(8, 32, 32)).astype(np.float32)
    pca = fit_pca_rgb(emb)
    rgb = transform_pca_rgb(emb, pca)
    print("PCA RGB shape:", rgb.shape, "dtype:", rgb.dtype)
