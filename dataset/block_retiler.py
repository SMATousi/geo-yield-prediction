"""Memory-efficient block-based retiling for field-scale geospatial data.

Large multi-temporal rasters (bands, masks, SAR stacks) are memory-mapped and
only the spatial block needed for a given field is sliced out, so per-field-year
samples can be extracted from large mosaics without loading whole rasters into
RAM. Small per-acquisition arrays (day-of-year stamps) are loaded directly.

Each returned block carries its own bands / masks / DOY arrays so a field patch
can be treated as one multimodal sample while every layer is still encoded at
its native resolution before fusion.
"""

from typing import Dict, List, Tuple

import numpy as np

# Arrays that are small enough to load fully into memory.
_SMALL_ARRAYS = ("doys", "sar_ascending_doy", "sar_descending_doy")

# Arrays that are large multi-temporal rasters and must be memory-mapped.
_LARGE_ARRAYS = ("bands", "masks", "sar_ascending", "sar_descending")


def load_block_data(
    file_info: Dict[str, Dict[str, str]],
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
) -> Dict[str, np.ndarray]:
    """Load one spatial block of a multi-temporal raster mosaic.

    Parameters
    ----------
    file_info : dict
        Maps array name -> ``{"path": <npy path>}`` for every array of the
        mosaic (small DOY arrays and large band/mask/SAR arrays).
    y_start, y_end, x_start, x_end : int
        Spatial slice of the block to extract (row / column bounds).

    Returns
    -------
    block_data : dict
        ``name -> np.ndarray``. Small arrays are returned whole; large arrays
        are memory-mapped and sliced to the requested spatial block.
    """
    block_data: Dict[str, np.ndarray] = {}

    for key in _SMALL_ARRAYS:
        if key in file_info:
            block_data[key] = np.load(file_info[key]["path"])

    for key in _LARGE_ARRAYS:
        if key not in file_info:
            continue
        full_array = np.load(file_info[key]["path"], mmap_mode="r")
        if full_array.ndim == 4:  # (T, H, W, C)
            block_data[key] = np.array(
                full_array[:, y_start:y_end, x_start:x_end, :]
            )
        else:  # masks (T, H, W)
            block_data[key] = np.array(
                full_array[:, y_start:y_end, x_start:x_end]
            )
        del full_array

    return block_data


def retile_block(
    block_data: Dict[str, np.ndarray],
    patch_size: int,
    *,
    spatial_axes: Tuple[int, int] = (1, 2),
) -> Dict[str, List[np.ndarray]]:
    """Re-tile a loaded spatial block into non-overlapping patches.

    Each large array is split into ``patch_size x patch_size`` non-overlapping
    patches along its spatial axes, producing a list of per-patch arrays. Small
    DOY arrays (which have no spatial extent) are returned as a single-element
    list so every patch can be paired with the same acquisition-time stamps.

    Parameters
    ----------
    block_data : dict
        Output of :func:`load_block_data`.
    patch_size : int
        Side length of the square non-overlapping patches.
    spatial_axes : tuple of int
        Axes of the large arrays that index space (default ``(1, 2)`` for
        ``(T, H, W[, C])`` arrays).

    Returns
    -------
    retiled : dict
        ``name -> list[np.ndarray]`` of non-overlapping patches.
    """
    h_axis, w_axis = spatial_axes
    retiled: Dict[str, List[np.ndarray]] = {}

    for key, arr in block_data.items():
        if key in _SMALL_ARRAYS:
            retiled[key] = [arr]
            continue
        h, w = arr.shape[h_axis], arr.shape[w_axis]
        n_h, n_w = h // patch_size, w // patch_size
        patches = []
        for i in range(n_h):
            for j in range(n_w):
                sl = [slice(None)] * arr.ndim
                sl[h_axis] = slice(i * patch_size, (i + 1) * patch_size)
                sl[w_axis] = slice(j * patch_size, (j + 1) * patch_size)
                patches.append(np.ascontiguousarray(arr[tuple(sl)]))
        retiled[key] = patches

    return retiled


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        info = {}
        np.save(tmp / "doys.npy", np.array([10, 20, 30], dtype=np.int32))
        info["doys"] = {"path": str(tmp / "doys.npy")}
        np.save(
            tmp / "bands.npy",
            np.random.rand(3, 64, 64, 4).astype(np.float32),
        )
        info["bands"] = {"path": str(tmp / "bands.npy")}
        np.save(tmp / "masks.npy", np.random.rand(3, 64, 64).astype(np.uint8))
        info["masks"] = {"path": str(tmp / "masks.npy")}

        block = load_block_data(info, 0, 32, 0, 32)
        print("block bands:", block["bands"].shape, "masks:", block["masks"].shape)
        retiled = retile_block(block, patch_size=16)
        print("patches per array:", {k: len(v) for k, v in retiled.items()})
        print("patch shape:", retiled["bands"][0].shape)
