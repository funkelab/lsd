import h5py
import logging
import numpy as np
from lsd.train import LsdExtractor
from lsd.post import LsdAgglomeration

logging.basicConfig(level=logging.INFO)
logging.getLogger('lsd.post.agglomerate').setLevel(logging.DEBUG)
logging.getLogger('lsd.train.local_shape_descriptor').setLevel(logging.DEBUG)

if __name__ == "__main__":

    lsd_extractor = LsdExtractor(
        sigma=(10.0, 10.0, 10.0),
        downsample=2)

    gt = np.ones((20, 20, 20), dtype=np.uint64)
    gt[:5] = 2
    target_lsds = lsd_extractor.get_descriptors(gt)

    fragments = np.array(gt)
    fragments[5:15] = 3

    agglomeration = LsdAgglomeration(
        fragments,
        target_lsds,
        lsd_extractor,
        voxel_size=(2, 2, 2))
    agglomeration.merge_until(0)

    segmentation = agglomeration.get_segmentation()

    with h5py.File('test_agglomeration.hdf', 'w') as f:
        f['volumes/gt'] = gt
        f['volumes/target_lsds'] = target_lsds
        f['volumes/fragments'] = fragments
        f['volumes/segmentation'] = segmentation
