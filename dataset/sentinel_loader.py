import h5py
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import json
from typing import Any, Sequence

from util.sar_normalize import normalize_s1_vvvh_chw

torch.manual_seed(0)
np.random.seed(0)


def stack_binned_frames(
    frames: list,
    bins: Sequence[tuple],
    *,
    expected_channels: int | None = None,
    context: str = "",
    last_error: str | None = None,
) -> tuple:
    """Stack per-bin frames into [T,C,H,W]; empty bins become all-NaN sentinel frames.

    The NaN sentinel keeps ``T == len(bins)`` so frame/bin alignment survives
    array-only transport (e.g. through the dataloader); consumers detect empty
    frames via ``np.isnan(frame).all()`` or the returned per-bin mask. This is
    the missing-modality mechanism: a temporal bin (or, by extension, a whole
    modality) with no data is represented by a sentinel + mask instead of
    failing the sample.
    """
    shapes = {f.shape for f in frames if f is not None}
    if not shapes:
        detail = f" Last fetch error: {last_error}" if last_error else ""
        raise ValueError(f"{context}: no imagery found in any of the {len(bins)} time bins.{detail}")
    if len(shapes) > 1:
        raise ValueError(f"{context}: inconsistent frame shapes across time bins: {sorted(shapes)}.")
    shape = shapes.pop()
    if expected_channels is not None and int(shape[0]) != int(expected_channels):
        raise ValueError(f"{context}: expected C={expected_channels} per frame, got shape={shape}.")

    out = []
    frame_meta = []
    for (start, end), f in zip(bins, frames):
        empty = f is None
        out.append(np.full(shape, np.nan, dtype=np.float32) if empty else np.asarray(f, dtype=np.float32))
        frame_meta.append({"start": start, "end": end, "empty": empty})

    arr = np.stack(out, axis=0)
    mask = np.array([not m["empty"] for m in frame_meta], dtype=np.float32)
    meta = {
        "frames": frame_meta,
        "n_empty": int(sum(1 for m in frame_meta if m["empty"])),
    }
    return arr, mask, meta


class Sentinel_Dataset(Dataset):

    def __init__(self, root_dir, json_file):
        self.fips_codes = []
        self.years = []
        self.file_paths = []

        data = json.load(open(json_file))
        for obj in data:
            self.fips_codes.append(obj["FIPS"])
            self.years.append(obj["year"])

            tmp_path = []
            relative_path_list = obj["data"]["sentinel"]
            for relative_path in relative_path_list:
                tmp_path.append(os.path.join(root_dir, relative_path))
            self.file_paths.append(tmp_path)

    def __len__(self):
        return len(self.fips_codes)

    def __getitem__(self, index):
        fips_code, year = self.fips_codes[index], self.years[index]
        file_paths = self.file_paths[index]

        frames = []
        bins = []

        for file_path in file_paths:
            try:
                with h5py.File(file_path, 'r') as hf:
                    groups = hf[fips_code]
                    for i, d in enumerate(groups.keys()):
                        # only consider the 1st day of each month
                        # note that the h5 file contains the 1st and 15th of images for each month, e.g., "04-01" and "04-15"
                        if i % 2 == 0:
                            grids = np.asarray(groups[d]["data"])
                            # Sentinel-1 VV/VH backscatter -> numerically stable [0,1]
                            if grids.ndim == 3 and grids.shape[0] == 2:
                                grids = normalize_s1_vvvh_chw(grids)
                            frames.append(grids)
                            bins.append((d, d))
                hf.close()
            except (KeyError, OSError):
                # missing acquisition -> all-NaN sentinel frame + per-bin mask
                frames.append(None)
                bins.append((None, None))

        # keep T == len(bins) fixed; empty bins become NaN-sentinel frames
        x, mask, _ = stack_binned_frames(
            frames, bins, expected_channels=2, context=f"FIPS {fips_code} {year}"
        )

        # NaN-sentinel frames carry the missing-modality signal; the per-bin
        # mask is returned so the model can gate attention and substitute a
        # learned missing-modality token for absent acquisitions.
        return torch.from_numpy(x), torch.from_numpy(mask), fips_code, year


if __name__ == '__main__':
    root_dir = "/mnt/data/Tiny CropNet"
    # train = "./../data/soybean_train.json"
    train = "./../data/soybean_val.json"
    dataset = Sentinel_Dataset(root_dir, train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    max_g = 0
    for x, m, f, y in train_loader:
        print("fips: {}, year: {}, shape: {}".format(f, y, x.shape))
        max_g = max(max_g, tuple(x.shape)[2])

    print(max_g)
