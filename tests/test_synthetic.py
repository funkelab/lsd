import h5py
import logging
import mahotas
import numpy as np
import os
from lsd.train import LsdExtractor
from scipy.ndimage import gaussian_filter, maximum_filter

logging.basicConfig(level=logging.INFO)

size = (1, 100, 100)


def create_random_segmentation(seed):

    np.random.seed(seed)
    peaks = np.random.random(size).astype(np.float32)
    peaks = gaussian_filter(peaks, sigma=5.0)
    max_filtered = maximum_filter(peaks, 10)
    maxima = max_filtered == peaks
    seeds, n = mahotas.label(maxima)
    print("Creating segmentation with %d segments" % n)
    return mahotas.cwatershed(1.0 - peaks, seeds).astype(np.uint64)


def test_synthetic():

    # factor of pixel-wise uniformly sampled noise (in [-0.5, 0.5]) to add to
    # "prediction" LSDs
    for noise_factor in [0, 1, 2, 3]:

        gt = create_random_segmentation(42)
        fragments = create_random_segmentation(23)
        # intersect gt and fragments to get an oversegmentation
        fragments = gt + (fragments + 1) * gt.max()

        lsd_extractor = LsdExtractor(sigma=(10.0, 10.0, 10.0))
        predicted_lsds = lsd_extractor.get_descriptors(gt)

        if noise_factor > 0:
            noise = -0.5 + np.random.random(predicted_lsds.shape)
            predicted_lsds += noise * noise_factor
            predicted_lsds = predicted_lsds.clip(0, 1)

        name = "test_synthetic_noise=%d.hdf" % noise_factor

        with h5py.File(name, "w") as f:
            f["volumes/gt"] = gt
            f["volumes/predicted_lsds"] = predicted_lsds[0:3]

        os.remove(name)
