import logging
import mahotas
import numpy as np
from lsd.train import LsdExtractor, local_shape_descriptor
from scipy.ndimage import gaussian_filter, maximum_filter

logging.basicConfig(level=logging.INFO)


def create_random_segmentation(seed, size):

    np.random.seed(seed)
    peaks = np.random.random(size).astype(np.float32)
    peaks = gaussian_filter(peaks, sigma=5.0)
    max_filtered = maximum_filter(peaks, 10)
    maxima = max_filtered == peaks
    seeds, n = mahotas.label(maxima)
    print("Creating segmentation with %d segments" % n)
    return mahotas.cwatershed(1.0 - peaks, seeds).astype(np.uint64)


def create_lsds(size, combs):

    gt = create_random_segmentation(42, size)

    sigma = (10,) * len(size)
    voxel_size = (1,) * len(size)

    lsd_extractor = LsdExtractor(sigma=sigma)

    for comb in combs:

        # via extractor
        lsds = lsd_extractor.get_descriptors(gt, components=comb)

        assert len(comb) == lsds.shape[0]

        # without extractor
        lsds = local_shape_descriptor.get_local_shape_descriptors(
            segmentation=gt,
            components=comb,
            sigma=sigma,
            voxel_size=voxel_size,
        )

        assert len(comb) == lsds.shape[0]


def test_components_2d():

    """possible 2d component combinations

    --- base components

    mean offset: 01
    covariance: 23
    pearsons: 4
    size: 5

    --- 2 combs

    mean offset + covariance: 0123
    mean offset + pearsons: 014
    mean offset + size: 015
    covariance + pearsons: 234
    covariance + size: 235
    pearsons + size: 45

    --- 3 combs

    mean offset + covariance + pearsons: 01234
    mean offset + covariance + size: 01235
    mean offset + pearsons + size: 0145
    covariance + pearsons + size: 2345

    --- all: 012345

    """

    combs = [
        "01",
        "23",
        "4",
        "5",
        "0123",
        "014",
        "015",
        "234",
        "235",
        "45",
        "01234",
        "01235",
        "0145",
        "2345",
        "012345",
    ]

    size = (100, 100)

    create_lsds(size, combs)


def test_components_3d():

    """possible 3d component combinations

    --- base components

    mean offset: 012
    covariance: 345
    pearsons: 678
    size: 9

    --- 2 combs

    mean offset + covariance: 012345
    mean offset + pearsons: 012678
    mean offset + size: 0129
    covariance + pearsons: 345678
    covariance + size: 3459
    pearsons + size: 6789

    --- 3 combs

    mean offset + covariance + pearsons: 012345678
    mean offset + covariance + size: 0123459
    mean offset + pearsons + size: 0126789
    covariance + pearsons + size: 3456789

    --- all: 0123456789 (can ignore)

    """

    combs = [
        "012",
        "345",
        "678",
        "9",
        "012345",
        "012678",
        "0129",
        "345678",
        "3459",
        "6789",
        "012345678",
        "0123459",
        "0126789",
        "3456789",
        "0123456789",
    ]

    size = (1, 50, 50)

    create_lsds(size, combs)
