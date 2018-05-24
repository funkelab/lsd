import h5py
import numpy as np
import logging
from lsd import LsdExtractor

logging.basicConfig(level=logging.INFO)
logging.getLogger('lsd.local_shape_descriptor').setLevel(logging.DEBUG)

if __name__ == "__main__":

    extractor = LsdExtractor(
        sigma=(10.0, 10.0, 10.0),
        downsample=2)

    segmentation = np.ones((100, 100, 100), dtype=np.uint64)
    segmentation[25:75,25:75,25:75] = 2

    lsds = extractor.get_descriptors(
        segmentation,
        voxel_size=(2, 2, 2))

    with h5py.File('test.hdf', 'w') as f:
        f['volumes/lsds'] = lsds
