import os
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr


def RMSE(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def R2_Score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    R2 = corr ** 2

    return R2


def PCC(y_true, y_pred):
    corr, _ = pearsonr(y_true, y_pred)
    return corr


def SpatialCorrelation(y_true, y_pred):
    """Spatial correlation between 2D yield maps (flattened internally)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size < 2 or y_pred.size < 2:
        return float('nan')
    corr_matrix = np.corrcoef(y_true, y_pred)
    return corr_matrix[0, 1]


def ZonePreservation(y_true, y_pred, n_zones=3):
    """Fraction of pixels whose low/mid/high-yield zone label is preserved.

    Zones are defined by quantile thresholds on the observed yield map, so
    the metric measures whether within-field low- and high-yield zones are
    recovered rather than only a field-average.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0 or y_pred.size != y_true.size:
        return float('nan')
    qs = np.quantile(y_true, np.linspace(0, 1, n_zones + 1)[1:-1])
    true_zone = np.digitize(y_true, qs)
    pred_zone = np.digitize(y_pred, qs)
    return float(np.mean(true_zone == pred_zone))


def evaluate(y_true, y_pred):
    rmse = RMSE(y_true, y_pred)
    r2 = R2_Score(y_true, y_pred)
    pcc = PCC(y_true, y_pred)

    return rmse, r2, pcc


class SpatialYieldMetric:
    """Dense, spatially-aware evaluation metric for full-resolution yield maps.

    Adapted from AgriFM's CropIoUMetric: instead of per-class IoU histograms
    over segmentation labels, it accumulates per-field spatial statistics
    (spatial correlation, zone preservation, RMSE, MAE, R2) over dense yield
    maps and optionally dumps each field's predicted yield raster to an
    output_dir keyed by the sample's file_name. This provides the spatial
    yield-map evaluation suite (spatial correlation + zone preservation) that
    complements the global scalar metrics in :func:`evaluate`.
    """

    def __init__(self, output_dir=None, n_zones=3, format_only=False):
        self.output_dir = output_dir
        self.n_zones = n_zones
        self.format_only = format_only
        self.results = []
        if self.output_dir is not None:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def process(self, pred_maps, true_maps, file_names=None):
        """Accumulate per-field spatial statistics for a batch of dense maps.

        Args:
            pred_maps: (B, H, W) or (B, 1, H, W) predicted yield maps.
            true_maps: (B, H, W) or (B, 1, H, W) observed yield maps.
            file_names: optional list of per-sample file names used to name
                the dumped prediction rasters.
        """
        pred_maps = np.asarray(pred_maps)
        true_maps = np.asarray(true_maps)
        if pred_maps.ndim == 4:
            pred_maps = pred_maps[:, 0]
        if true_maps.ndim == 4:
            true_maps = true_maps[:, 0]

        for i in range(pred_maps.shape[0]):
            pred = pred_maps[i]
            true = true_maps[i]
            if not self.format_only:
                self.results.append({
                    'spatial_corr': SpatialCorrelation(true, pred),
                    'zone_preservation': ZonePreservation(true, pred, self.n_zones),
                    'rmse': RMSE(true, pred),
                    'mae': float(np.mean(np.abs(true - pred))),
                    'r2': R2_Score(true, pred),
                })
            if self.output_dir is not None:
                basename = 'field_{}'.format(i)
                if file_names is not None and i < len(file_names):
                    basename = os.path.splitext(os.path.basename(file_names[i]))[0]
                png_filename = os.path.abspath(
                    os.path.join(self.output_dir, '{}.npy'.format(basename)))
                np.save(png_filename, pred.astype(np.float32))

    def compute(self):
        """Aggregate per-field statistics into a single summary dict."""
        if not self.results:
            return {}
        keys = self.results[0].keys()
        summary = {}
        for k in keys:
            vals = [r[k] for r in self.results if np.isfinite(r[k])]
            summary[k] = float(np.mean(vals)) if vals else float('nan')
        summary['num_fields'] = len(self.results)
        return summary


if __name__ == '__main__':
    y = np.asarray([10, 20, 30, 40, 50])
    y_hat = np.asarray([11, 21, 32, 41, 51])

    print(RMSE(y, y_hat))
    print(R2_Score(y, y_hat))
    print(PCC(y, y_hat))
