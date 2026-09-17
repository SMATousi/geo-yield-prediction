"""Field-level soil attribute ingestion from the ISRIC SoilGrids REST API.

Adapted from the CropWise per-property, fail-soft ingestion pattern: for each
field (keyed by its centroid lat/lon) we query the ISRIC SoilGrids v2.0
``properties/query`` endpoint one property at a time and fall back to a
scientifically sensible default whenever a property (or the whole service) is
unavailable. This gives the field-level geospatial pipeline (gap g6) a
soil-modality data source: point/tabular soil attributes that can be converted
into learned tokens, while the per-property try/except with defaults supports
the missing-modality robustness goal by never hard-failing when a data source
is absent.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import requests
except ImportError:  # pragma: no cover - requests is optional at import time
    requests = None


#: Default soil attribute values used when a property is unavailable.
DEFAULT_SOIL_PROPERTIES: Dict[str, float] = {
    "phh2o": 6.5,
    "soc": 1.5,
    "nitrogen": 0.8,
    "cec": 15.0,
    "clay": 30.0,
    "sand": 35.0,
    "silt": 35.0,
}

#: Order in which soil properties are returned as a feature vector.
SOIL_PROPERTY_ORDER: Tuple[str, ...] = (
    "phh2o",
    "soc",
    "nitrogen",
    "cec",
    "clay",
    "sand",
    "silt",
)

_SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
_SOILGRIDS_DEPTH = "0-30cm"
_SOILGRIDS_TIMEOUT = 10


def fetch_soil_properties(lat: float, lon: float) -> Dict[str, float]:
    """Fetch per-property soil attributes for a lat/lon point, fail-soft.

    Each property is queried independently against the ISRIC SoilGrids REST
    API. A property that errors or returns no usable mean value keeps its
    default, so a single missing source never hard-fails the whole field
    sample (missing-modality robustness).

    Parameters
    ----------
    lat : float
        Field centroid latitude (EPSG:4326).
    lon : float
        Field centroid longitude (EPSG:4326).

    Returns
    -------
    Dict[str, float]
        Mapping of soil property name to value (defaults where unavailable).
    """
    soil_properties = dict(DEFAULT_SOIL_PROPERTIES)
    if requests is None:
        return soil_properties

    for prop in SOIL_PROPERTY_ORDER:
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "property": prop,
                "depth": _SOILGRIDS_DEPTH,
            }
            response = requests.get(_SOILGRIDS_URL, params=params, timeout=_SOILGRIDS_TIMEOUT)
            data = response.json()
            layers = data.get("properties", {}).get("layers", [])
            if layers and layers[0].get("depths"):
                values = layers[0]["depths"][0].get("values", {})
                mean_val = values.get("mean")
                if mean_val is not None:
                    soil_properties[prop] = mean_val
        except Exception as exc:  # noqa: BLE001 - fail-soft per property
            print(f"Soil API error for {prop}: {exc}")
            continue
    return soil_properties


class SoilGrids_Dataset(Dataset):
    """Field-level soil attribute dataset keyed by field centroid lat/lon.

    Each field-year sample is identified by a ``(lat, lon)`` centroid and
    yields a fixed-length soil feature vector (one value per property in
    ``SOIL_PROPERTY_ORDER``). The vector can be projected into learned tokens
    by a downstream soil-modality encoder. Missing properties fall back to
    defaults, so the loader is robust to partial data availability.

    Parameters
    ----------
    coords : sequence of (lat, lon)
        Field centroid coordinates, one per sample.
    names : sequence of str, optional
        Optional field identifiers aligned with ``coords`` (returned as-is).
    """

    def __init__(
        self,
        coords: Sequence[Tuple[float, float]],
        names: Sequence[str] = (),
    ):
        self.coords = list(coords)
        self.names = list(names) if names else [str(i) for i in range(len(coords))]

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, index: int):
        lat, lon = self.coords[index]
        props = fetch_soil_properties(lat, lon)
        vector = torch.tensor(
            [props[prop] for prop in SOIL_PROPERTY_ORDER], dtype=torch.float32
        )
        return vector, self.names[index]


if __name__ == "__main__":
    ds = SoilGrids_Dataset([(40.0, -88.0), (41.0, -89.0)], names=["f1", "f2"])
    for vec, name in ds:
        print(name, vec.tolist())
