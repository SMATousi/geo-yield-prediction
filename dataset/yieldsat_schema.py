# --------------------------------------------------------
# YieldSAT preprocessed contract (``yieldsat_preprocessed_v1``): schema,
# canonical channel registry, per-file decoding helpers and provenance.
#
# The four country NetCDF files share the same 120 feature names but store
# them in different orders and carry country-local category dictionaries and
# time origins. Everything that indexes a file goes through the helpers here:
# channels are resolved by exact name into a canonical registry, categorical
# codes are decoded with the file's own numeric-string attribute keys, and
# ``times`` is decoded with its own ``units``/``calendar``. See
# spec/yieldsat_data_contract.md.
# --------------------------------------------------------

import datetime
import re

import numpy as np

CONTRACT_KEY = 'yieldsat_preprocessed_v1'
SOURCE_FILENAME = 'merge_s2-soil-dem-weather-coords.nc'
COUNTRIES = ('Argentina', 'Brazil', 'Germany', 'Uruguay')
NUM_TIME_SLOTS = 24

# Snapshot recorded in the spec (file bytes and the CRC32 from the original
# Preprocessed.zip central directory). A source file that differs is a
# different snapshot and must not reuse caches built from this one.
EXPECTED_SNAPSHOT = {
    'Argentina': {'bytes': 62861086332, 'crc32': 'bae1a7f0', 'rows': 5325807},
    'Brazil': {'bytes': 50280040289, 'crc32': '1c400537', 'rows': 4260262},
    'Germany': {'bytes': 7193431158, 'crc32': 'b89c8acb', 'rows': 609645},
    'Uruguay': {'bytes': 25693566077, 'crc32': '41592d6f', 'rows': 2177206},
}

# ---- canonical channel registry -------------------------------------------

OPTICAL = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A',
           'B09', 'B11', 'B12')
DEM = ('dem',)
TERRAIN = ('aspect', 'curvature', 'slope', 'twi')
SOIL_PROPERTIES = ('cec', 'cfvo', 'clay', 'nitrogen', 'phh2o', 'sand', 'silt', 'soc')
SOIL_DEPTHS = ('0-5', '5-15', '15-30', '30-60', '60-100', '100-200')
# property-major, depth-ordered (top to bottom), independent of file order
SOIL = tuple('{}_{}'.format(p, d) for p in SOIL_PROPERTIES for d in SOIL_DEPTHS)
SOIL_UNCERTAINTY = tuple(name + '_uncertainty' for name in SOIL)
WEATHER = ('temp_max', 'temp_mean', 'temp_min', 'total_prec')
COORDS = ('coord_x', 'coord_y', 'coord_z')

# Compact canonical layouts used by the cache and the adapter. Temporal
# channels vary per slot; static channels repeat through the 24 slots in the
# source and are collapsed to one verified representation per row.
TEMPORAL_CHANNELS = OPTICAL + WEATHER
STATIC_CHANNELS = DEM + TERRAIN + SOIL + SOIL_UNCERTAINTY + COORDS
ALL_CHANNELS = TEMPORAL_CHANNELS + STATIC_CHANNELS
assert len(ALL_CHANNELS) == 120 and len(set(ALL_CHANNELS)) == 120

# Model streams (spec §4). ``coord_*`` are metadata, not a stream.
STREAMS = {
    'yieldsat_s2': {'channels': OPTICAL, 'temporal': True, 'family': 'optical'},
    'yieldsat_weather': {'channels': WEATHER, 'temporal': True, 'family': 'weather'},
    'yieldsat_dem': {'channels': DEM, 'temporal': False, 'family': 'terrain'},
    'yieldsat_terrain': {'channels': TERRAIN, 'temporal': False, 'family': 'terrain'},
    'yieldsat_soil': {'channels': SOIL, 'temporal': False, 'family': 'soil'},
}
DEFAULT_STREAMS = tuple(STREAMS)
# Optional input streams, never part of STREAMS/DEFAULT_STREAMS: coordinates
# (unit-sphere x/y/z) as in the paper's neighbourhood models
# (spec/yieldsat-paper-reproduction.md D3).
OPTIONAL_STREAMS = {'yieldsat_coords': {'channels': COORDS, 'temporal': False, 'family': 'coords'}}

# Global crop vocabulary for country-qualified crop context and reporting.
CROPS = ('corn', 'rapeseed', 'soybean', 'wheat')

# ---- provenance ------------------------------------------------------------
# status: 'documented' (YieldSAT project page / release notebooks),
# 'inferred' (consistent with documentation and our probes, not stated),
# 'unresolved'. Physical constraints may only use 'documented' entries.
PROVENANCE = {
    'optical': {
        'source': 'Sentinel-2 L2A, least-cloudy acquisition per slot',
        'units': 'L2A digital numbers (reflectance x 10000); BOA offset not verified',
        'status': 'inferred',
        'notes': '20/60 m bands nearest-upsampled to 10 m upstream; SCL not in NetCDF',
    },
    'dem': {'source': 'SRTM 30 m, cubic-upsampled to 10 m', 'units': 'm (inferred)',
            'status': 'inferred'},
    'terrain': {
        'source': 'RichDEM derivatives of SRTM',
        'units': 'aspect: degrees (inferred); slope/curvature/twi: unresolved scale',
        'status': 'unresolved',
        'notes': 'slope values of 1e3-1e4 are not degrees; twi is NaN for many rows',
    },
    'soil': {
        'source': 'SoilGrids 2.0, 250 m, cubic-upsampled to 10 m',
        'units': 'SoilGrids mapped units (e.g. clay g/kg, phh2o pH*10, soc dg/kg)',
        'status': 'inferred',
    },
    'soil_uncertainty': {
        'source': 'SoilGrids 2.0 uncertainty layers',
        'units': 'SoilGrids uncertainty index (definition not verified)',
        'status': 'unresolved',
    },
    'weather': {
        'source': 'ERA5-Land daily, interpolated to field centroid',
        'units': 'sums of daily values over the inclusive interval [t_(k-1), t_k]'
                 ' (dt+1 days): temp_* in Kelvin-days, total_prec in m',
        'status': 'inferred',
        'notes': 'value/(dt+1) is flat across interval lengths (~275-300 K) while'
                 ' value/dt is not; boundary days are counted in both neighbouring'
                 ' slots; the first dated slot has an unrecorded interval start;'
                 ' spatially constant within a field; available at t_k',
    },
    'coords': {'source': 'undocumented', 'units': 'unit-norm 3-vectors, not lat/lon',
               'status': 'unresolved'},
    'target': {
        'source': 'combine yield monitor, dry-matter standard moisture, cleaned',
        'units': 't/ha (dry, standard moisture)',
        'status': 'documented',
        'notes': 'zero, infeasible and +-3 sigma outlier cells removed upstream',
    },
}


# ---- decoding helpers ------------------------------------------------------

def _to_str(value):
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, np.ndarray) and value.shape == ():
        return _to_str(value.item())
    return value


def read_band_names(h5file):
    return [_to_str(b) for b in h5file['band'][:]]


def resolve_channel_indices(band_names, channels):
    """Map canonical channel names to positions in one file's band vector.

    Fails on duplicate names in the file, on any missing channel, and (for the
    full registry) on unexpected extra names, so a schema change cannot be
    silently mis-indexed.
    """
    band_names = [_to_str(b) for b in band_names]
    if len(set(band_names)) != len(band_names):
        raise ValueError('duplicate band names in source file')
    position = {name: i for i, name in enumerate(band_names)}
    missing = [c for c in channels if c not in position]
    if missing:
        raise ValueError('source file lacks channels: {}'.format(missing))
    if tuple(channels) == ALL_CHANNELS:
        extra = sorted(set(band_names) - set(ALL_CHANNELS))
        if extra:
            raise ValueError('source file has unexpected channels: {}'.format(extra))
    return np.array([position[c] for c in channels], dtype=np.int64)


def read_dictionary(h5dataset):
    """Decode a variable's numeric-string attribute keys into {code: value}."""
    mapping = {}
    for key, value in h5dataset.attrs.items():
        if re.fullmatch(r'-?\d+', key):
            value = _to_str(value)
            if isinstance(value, np.generic):
                value = value.item()
            mapping[int(key)] = value
    return mapping


def decode_codes(codes, mapping, name='variable'):
    """Vectorised dictionary decode. Code zero is a valid category; codes
    absent from the dictionary fail instead of becoming labels."""
    codes = np.asarray(codes)
    unknown = np.setdiff1d(np.unique(codes), np.array(sorted(mapping), dtype=codes.dtype))
    if unknown.size:
        raise ValueError('{}: codes without dictionary entry: {}'.format(name, unknown[:10]))
    table = np.empty(max(mapping) + 1, dtype=object)
    for k, v in mapping.items():
        table[k] = v
    return table[codes]


def integer_lookup(mapping, name='variable'):
    """Dense lookup array for dictionaries whose values are integers (row/col,
    year). Brazil's row/col codes are *not* identity maps, so decoding is
    mandatory."""
    table = np.full(max(mapping) + 1, -1, dtype=np.int64)
    for k, v in mapping.items():
        table[k] = int(v)
    return table


_UNITS_RE = re.compile(r'days since (\d{4}-\d{2}-\d{2})(?:[ T]00:00(?::00)?)?$')


def parse_time_origin(units, calendar):
    """Return the epoch of a ``days since YYYY-MM-DD`` time variable."""
    if calendar not in ('proleptic_gregorian', 'standard', 'gregorian'):
        raise ValueError('unsupported calendar: {}'.format(calendar))
    m = _UNITS_RE.match(_to_str(units).strip())
    if not m:
        raise ValueError('unsupported time units: {}'.format(units))
    return datetime.date.fromisoformat(m.group(1))


EPOCH_1970 = datetime.date(1970, 1, 1)


def origin_offset_days(origin):
    """Days from 1970-01-01 to a file's time origin (for a shared timeline)."""
    return (origin - EPOCH_1970).days


def date_strings_to_days(values):
    """ISO date strings -> int days since 1970-01-01."""
    return np.array([(datetime.date.fromisoformat(_to_str(v)) - EPOCH_1970).days
                     for v in values], dtype=np.int64)


_NAME_RE = re.compile(
    r'^(?P<country>[A-Za-z]+)_(?P<provider>DUP\d+)_(?P<farm>farm\d+)_'
    r'(?P<field>field\d+)_(?P<crop>[a-z]+)_(?P<year>\d{4})$')


def parse_field_shared_name(name):
    """Split ``Country_DUPn_farmX_fieldY_crop_year`` into its parts."""
    m = _NAME_RE.match(_to_str(name))
    if not m:
        raise ValueError('unrecognised field_shared_name: {}'.format(name))
    parts = m.groupdict()
    parts['year'] = int(parts['year'])
    return parts
