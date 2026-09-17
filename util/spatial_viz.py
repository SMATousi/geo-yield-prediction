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


def modality_importance(importance_scores, top_n=None):
    """Rank modalities/features by their importance scores as a sorted DataFrame.

    Adapted from Momahmoses/agricultural-yield-forecasting's ``feature_importance``
    (src/model.py). The original hardcoded a global ``FEATURES`` list and read
    ``model.feature_importances_`` from a single tree model. Here it is generalized
    to accept a dict mapping modality/feature names to importance scores (e.g.
    aggregated attention or attribution per modality encoder), so it can serve as
    the modality-importance ranking primitive for the interpretability deliverable.
    Returns a DataFrame sorted by descending importance, optionally truncated to
    the ``top_n`` most important modalities.

    Args:
        importance_scores: dict of {name: importance} or a sequence of
            (name, importance) pairs.
        top_n: optional int; if given, keep only the top ``top_n`` rows.

    Returns:
        pandas.DataFrame with columns ``modality`` and ``importance``, sorted by
        descending importance.
    """
    import pandas as pd

    if isinstance(importance_scores, dict):
        items = list(importance_scores.items())
    else:
        items = [(str(k), float(v)) for k, v in importance_scores]

    df = pd.DataFrame(items, columns=['modality', 'importance'])
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    if top_n is not None:
        df = df.head(top_n).reset_index(drop=True)
    return df


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


# Quantile-based color classes for spatial zone-preservation inspection.
# Adapted from facebookresearch/Context-Aware-Representation-Crop-Yield-Prediction
# (data_preprocessing/plot/counties_plot.py): instead of coloring county SVG
# polygons, we color field-level yield/error rasters by quantile class so that
# low- and high-yield zones are visually separable. A companion save_colorbar
# renders the legend for the class palette.
QUANTILE_BREAKS = (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)
QUANTILE_COLORS = (
    (0.1216, 0.4667, 0.7059),  # class 0 - lowest
    (0.6824, 0.7804, 0.9098),
    (0.8588, 0.9412, 0.9725),
    (0.9922, 0.9059, 0.8549),
    (0.9843, 0.7059, 0.5412),
    (0.8706, 0.3882, 0.2784),
    (0.6471, 0.1294, 0.1490),  # class 6 - highest
)


def quantile_color_classes(values, breaks=QUANTILE_BREAKS):
    """Map a 2D array to integer color classes using quantile thresholds.

    Values above the largest break get the top class, values at or below the
    smallest break get the bottom class, and intermediate values are binned by
    the remaining quantile breaks. Returns an int array of the same shape.
    """
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.int64)
    qs = np.quantile(finite, breaks)
    classes = np.zeros(values.shape, dtype=np.int64)
    for q in qs:
        classes = classes + (values > q).astype(np.int64)
    return classes


def plot_quantile_classes(values, *, ax=None, breaks=QUANTILE_BREAKS,
                          colors=QUANTILE_COLORS, title=None, cmap_name='viridis'):
    """Render a 2D yield/error raster as a quantile-classed color map.

    This is the field-raster analogue of the county choropleth renderer: each
    pixel is assigned a color class from its position in the observed value
    distribution, so within-field low- and high-yield zones are visually
    separable. Returns the (H, W, 3) RGB array and, if ``ax`` is given, draws
    it on the axis.
    """
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[0]
    classes = quantile_color_classes(values, breaks=breaks)
    n_classes = len(colors)
    rgb = np.zeros(values.shape + (3,), dtype=np.float32)
    for c in range(n_classes):
        mask = classes == c
        rgb[mask] = colors[c]
    rgb[~np.isfinite(values)] = 0.0

    if ax is not None:
        ax.imshow(rgb)
        ax.set_xticks([])
        ax.set_yticks([])
        if title is not None:
            ax.set_title(title)

    return rgb


def save_colorbar(savepath, *, breaks=QUANTILE_BREAKS, colors=QUANTILE_COLORS,
                  label='Yield'):
    """Write a colorbar legend for the quantile color classes to an image file.

    Uses matplotlib if available; otherwise writes a plain-text legend so the
    class thresholds remain inspectable without extra dependencies.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
    except Exception:
        with open(savepath, 'w') as f:
            f.write('Quantile color classes (label={}):\n'.format(label))
            for i, q in enumerate(breaks):
                f.write('  class {}: > {:.3f}\n'.format(i, q))
            f.write('  class {}: > {:.3f}\n'.format(len(breaks), breaks[-1]))
        return savepath

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(colors) + 1) - 0.5, cmap.N)
    fig, ax = plt.subplots(figsize=(1.6, 4.0))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=np.arange(len(colors)))
    cbar.ax.set_yticklabels(['< {:.2f}'.format(breaks[0])] +
                            ['{:.2f}-{:.2f}'.format(breaks[i], breaks[i + 1])
                             for i in range(len(breaks) - 1)] +
                            ['> {:.2f}'.format(breaks[-1])])
    cbar.set_label(label)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return savepath


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(8, 32, 32)).astype(np.float32)
    pca = fit_pca_rgb(emb)
    rgb = transform_pca_rgb(emb, pca)
    print("PCA RGB shape:", rgb.shape, "dtype:", rgb.dtype)

    yield_map = rng.normal(size=(32, 32)).astype(np.float32)
    classes = quantile_color_classes(yield_map)
    print("Quantile classes shape:", classes.shape, "num classes:",
          int(classes.max()) + 1)
    rgb_q = plot_quantile_classes(yield_map)
    print("Quantile RGB shape:", rgb_q.shape, "dtype:", rgb_q.dtype)
