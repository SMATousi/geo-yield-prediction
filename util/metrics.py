from scipy.stats import pearsonr
import numpy as np


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


if __name__ == '__main__':
    y = np.asarray([10, 20, 30, 40, 50])
    y_hat = np.asarray([11, 21, 32, 41, 51])

    print(RMSE(y, y_hat))
    print(R2_Score(y, y_hat))
    print(PCC(y, y_hat))
