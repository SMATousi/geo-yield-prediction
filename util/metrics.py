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


class ConfusionMatrix:
    """Per-pixel spatial confusion matrix over bucketed yield classes.

    Adapted from danfenghong/IEEE_TPAMI_SpectralGPT's ConfusionMatrix
    (downstream_tasks/SegMunich/train_utils/distributed_utils.py). The
    original accumulates a per-pixel confusion matrix over dense maps and
    derives global overall accuracy plus per-class precision, recall, F1 and
    IoU. Here it is reworked onto numpy (matching this repo's metric
    conventions) and used to evaluate dense yield maps spatially: continuous
    yield is bucketed into low/mid/high-yield classes before accumulation, so
    per-zone precision/recall/F1/IoU report whether within-field low- and
    high-yield zones are recovered rather than only a field-average.

    The update/compute structure is directly portable: ``update`` flattens
    predicted and ground-truth maps and increments the matrix via bincount,
    ``compute`` derives the scalar summaries, and ``reduce_from_all_processes``
    all-reduces the matrix across distributed workers.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        """Accumulate a batch of (pred, true) class maps into the matrix.

        Args:
            a: predicted class map, (B, H, W) or (H, W) int array.
            b: ground-truth class map, same shape as ``a``.
        """
        n = self.num_classes
        a = np.asarray(a).ravel()
        b = np.asarray(b).ravel()
        if self.mat is None:
            self.mat = np.zeros((n, n), dtype=np.int64)
        valid = (a >= 0) & (a < n) & (b >= 0) & (b < n)
        inds = n * a[valid].astype(np.int64) + b[valid].astype(np.int64)
        self.mat += np.bincount(inds, minlength=n * n).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.fill(0)

    def compute(self):
        """Return (global OA, per-class recall, per-class precision, per-class IoU)."""
        h = self.mat.astype(np.float64)
        total = h.sum()
        acc_global_oa = float(np.diag(h).sum() / total) if total > 0 else float('nan')
        row_sum = h.sum(1)
        col_sum = h.sum(0)
        diag = np.diag(h)
        with np.errstate(divide='ignore', invalid='ignore'):
            acc_r = np.where(row_sum > 0, diag / np.maximum(row_sum, 1), np.nan)
            acc_p = np.where(col_sum > 0, diag / np.maximum(col_sum, 1), np.nan)
            iu = np.where((row_sum + col_sum - diag) > 0,
                          diag / np.maximum(row_sum + col_sum - diag, 1), np.nan)
        return acc_global_oa, acc_r, acc_p, iu

    def reduce_from_all_processes(self):
        """All-reduce the matrix across distributed workers (no-op if not distributed)."""
        try:
            import torch.distributed as dist
        except Exception:
            return
        if not dist.is_available() or not dist.is_initialized():
            return
        dist.barrier()
        dist.all_reduce(torch.from_numpy(self.mat))

    def __str__(self):
        acc_global, acc_r, acc_p, iu = self.compute()
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = 2 * acc_p * acc_r / np.maximum(acc_p + acc_r, 1e-12)
        return ('OA: {:.1f}\nRecall: {}\nPrecision: {}\nF1_score: {}\nIoU: {}\nmean IoU: {:.1f}').format(
            acc_global * 100,
            ['{:.1f}'.format(i) for i in (acc_r * 100).tolist()],
            ['{:.1f}'.format(i) for i in (acc_p * 100).tolist()],
            ['{:.1f}'.format(i) for i in (f1 * 100).tolist()],
            ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
            np.nanmean(iu) * 100)


def _bucket_yield(values, n_classes=3):
    """Bucket continuous yield into integer classes via quantile thresholds.

    Returns (class_map, thresholds). Classes are 0..n_classes-1 with class 0
    the lowest-yield zone and class n_classes-1 the highest-yield zone.
    """
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.int64), np.zeros(n_classes - 1)
    qs = np.quantile(finite, np.linspace(0, 1, n_classes + 1)[1:-1])
    classes = np.digitize(values, qs)
    return classes.astype(np.int64), qs


class ZoneConfusionMetric:
    """Per-zone spatial evaluation of dense yield maps via a confusion matrix.

    Complements :class:`SpatialYieldMetric` (which reports scalar spatial
    correlation / zone-preservation agreement) by reporting per-class
    precision, recall, F1 and IoU over bucketed low/mid/high-yield zones.
    Continuous yield maps are bucketed into ``n_classes`` zones using quantile
    thresholds on the observed yield, then accumulated into a per-pixel
    :class:`ConfusionMatrix` so within-field zone recovery is measured
    spatially rather than only on field averages.
    """

    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.cm = ConfusionMatrix(n_classes)
        self.thresholds = None

    def process(self, pred_maps, true_maps):
        """Accumulate a batch of dense (pred, true) yield maps into the matrix.

        Args:
            pred_maps: (B, H, W) or (B, 1, H, W) predicted yield maps.
            true_maps: (B, H, W) or (B, 1, H, W) observed yield maps.
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
            true_zone, qs = _bucket_yield(true, self.n_classes)
            pred_zone = np.digitize(pred, qs).astype(np.int64)
            self.cm.update(pred_zone, true_zone)
            if self.thresholds is None:
                self.thresholds = qs

    def compute(self):
        """Return a dict of per-zone precision/recall/F1/IoU plus global OA."""
        if self.cm.mat is None:
            return {}
        oa, recall, precision, iu = self.cm.compute()
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        return {
            'overall_accuracy': oa,
            'mean_iou': float(np.nanmean(iu)),
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(),
            'per_class_iou': iu.tolist(),
            'zone_thresholds': self.thresholds.tolist() if self.thresholds is not None else None,
        }

    def reset(self):
        self.cm.reset()
        self.thresholds = None


if __name__ == '__main__':
    y = np.asarray([10, 20, 30, 40, 50])
    y_hat = np.asarray([11, 21, 32, 41, 51])

    print(RMSE(y, y_hat))
    print(R2_Score(y, y_hat))
    print(PCC(y, y_hat))

    rng = np.random.default_rng(0)
    true_map = rng.normal(size=(4, 32, 32)).astype(np.float32)
    pred_map = true_map + rng.normal(scale=0.5, size=true_map.shape).astype(np.float32)
    zcm = ZoneConfusionMetric(n_classes=3)
    zcm.process(pred_map, true_map)
    print(zcm.compute())
    print(zcm.cm)
