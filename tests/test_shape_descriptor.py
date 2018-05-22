import h5py
import numpy as np
from lsd import LsdExtractor

if __name__ == "__main__":

    extractor = LsdExtractor((10.0, 10.0, 10.0))

    segmentation = np.ones((100, 100, 100), dtype=np.uint64)
    segmentation[25:75,25:75,25:75] = 2

    lsds = extractor.get_descriptors(segmentation)

    with h5py.File('test.hdf', 'w') as f:
        f['volumes/lsds'] = lsds
