import h5py
import logging
import lsd
import mahotas
import numpy as np
from scipy.ndimage.filters import gaussian_filter, maximum_filter

logging.basicConfig(level=logging.INFO)

size = (1, 100, 100)

def create_random_segmentation(seed):

    np.random.seed(seed)
    peaks = np.random.random(size).astype(np.float32)
    peaks = gaussian_filter(peaks, sigma=5.0)
    max_filtered = maximum_filter(peaks, 10)
    maxima = max_filtered==peaks
    seeds, n = mahotas.label(maxima)
    print("Creating segmentation with %d segments"%n)
    return mahotas.cwatershed(1.0 - peaks, seeds).astype(np.uint64)

if __name__ == "__main__":

    gt = create_random_segmentation(42)
    fragments = create_random_segmentation(23)
    # intersect gt and fragments to get an oversegmentation
    fragments = gt + (fragments + 1)*gt.max()

    lsd_extractor = lsd.LsdExtractor(sigma=(10.0, 10.0, 10.0))
    gt_lsds = lsd_extractor.get_descriptors(gt)

    agglomeration = lsd.LsdAgglomeration(
        fragments,
        gt_lsds,
        lsd_extractor,
        keep_lsds=True)

    # fragments and lsdss before merging
    segmentations = [np.array(agglomeration.get_segmentation()[0])]
    lsdss = [np.array(agglomeration.get_lsds()[:,0])]

    while agglomeration.merge_until(0, max_merges=1) != 0:
        segmentations.append(np.array(agglomeration.get_segmentation()[0]))
        lsdss.append(np.array(agglomeration.get_lsds()[:,0]))
    num_merges = len(segmentations)

    print("Performed %d merges"%num_merges)

    segmentations = np.array(segmentations)
    lsdss = np.array(lsdss).transpose((1, 0, 2, 3))
    gt = [gt[0]]*num_merges
    gt_lsds = np.array([gt_lsds[:,0]]*num_merges).transpose((1, 0, 2, 3))

    diffs = np.sqrt(np.sum((gt_lsds - lsdss)**2, axis=0))
    diffs /= diffs.max()

    with h5py.File('test_synthetic.hdf', 'w') as f:
        f['volumes/gt'] = gt
        f['volumes/gt_lsds'] = gt_lsds[0:3]
        f['volumes/segmentations'] = segmentations
        f['volumes/lsdss'] = lsdss[0:3]
        f['volumes/diffs'] = diffs
