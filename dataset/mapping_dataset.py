import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MappingDataset(Dataset):
    """Field-level multimodal h5 sample loader.

    Each sample is one ``.h5`` file holding multiple georegistered data layers
    (``data_keys``) plus a dense label map (``label_key``), e.g. a yield-monitor
    raster. Every layer is returned as a dict keyed by source name so that each
    modality can be encoded at its native resolution before fusion.

    Modalities are treated as optional: a ``data_key`` that is absent from the
    h5 file is skipped rather than raising, so the loader supports the
    missing-modality requirement (a field-year may lack SAR, weather, soil, ...).
    """

    def __init__(self, data_root_path, data_list_file, data_keys=('S2',), label_key='label'):
        self.data_root_path = data_root_path
        self.data_list_file = data_list_file
        self.data_keys = data_keys
        self.label_key = label_key
        data_list = np.loadtxt(data_list_file, dtype=str)
        self.data_list = [data_list] if np.ndim(data_list) == 0 else data_list.tolist()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        file_name = self.data_list[item]
        data_file = os.path.join(self.data_root_path, '%s.h5' % file_name)

        data_dict = {}
        with h5py.File(data_file, 'r') as f:
            for key in self.data_keys:
                if key in f.keys():
                    data_dict[key] = torch.from_numpy(f[key][:])
                else:
                    # missing modality -> skip the key instead of failing the sample
                    continue
            if self.label_key in f.keys():
                label = torch.from_numpy(f[self.label_key][:])
            else:
                raise KeyError(f'{self.label_key} not found in {data_file}')

        label = torch.unsqueeze(label, dim=0)
        label = label.squeeze(dim=0)
        data_dict['file_name'] = file_name
        return data_dict, label.long()


if __name__ == '__main__':
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sample = os.path.join(tmp, 'field_2021.h5')
        with h5py.File(sample, 'w') as f:
            f.create_dataset('DEM', data=np.random.rand(1, 64, 64).astype(np.float32))
            f.create_dataset('S2', data=np.random.rand(4, 32, 32).astype(np.float32))
            f.create_dataset('label', data=np.random.rand(64, 64).astype(np.float32))
        list_file = os.path.join(tmp, 'list.txt')
        with open(list_file, 'w') as f:
            f.write('field_2021\n')

        ds = MappingDataset(tmp, list_file, data_keys=('DEM', 'S2', 'SOIL'), label_key='label')
        data, label = ds[0]
        print('keys:', sorted(data.keys()))
        print('DEM:', tuple(data['DEM'].shape), 'S2:', tuple(data['S2'].shape))
        print('label:', tuple(label.shape))
