import h5py
import logging
import lsd
import numpy as np

logging.basicConfig(level=logging.INFO)
logging.getLogger('lsd.agglomerate').setLevel(logging.DEBUG)

if __name__ == "__main__":

    lsd_extractor = lsd.LsdExtractor(sigma=(10.0, 10.0, 10.0))

    gt = np.ones((20, 20, 20), dtype=np.uint64)
    gt[:5] = 2
    target_lsds = lsd_extractor.get_descriptors(gt)

    fragments = np.array(gt)
    fragments[5:15] = 3

    agglomeration = lsd.LsdAgglomeration(fragments, target_lsds, lsd_extractor)
    agglomeration.merge_until(0)

    segmentation = agglomeration.get_segmentation()

    with h5py.File('test_agglomeration.hdf', 'w') as f:
        f['volumes/gt'] = gt
        f['volumes/target_lsds'] = target_lsds
        f['volumes/fragments'] = fragments
        f['volumes/segmentation'] = segmentation
