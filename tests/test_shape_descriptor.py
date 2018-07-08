from lsd import LsdExtractor
from scipy.ndimage import gaussian_filter, convolve, maximum_filter
import h5py
import logging
import mahotas
import numpy as np
import time

logging.basicConfig(level=logging.INFO)
logging.getLogger('lsd.local_shape_descriptor').setLevel(logging.DEBUG)

k_z = np.zeros((3, 1, 1), dtype=np.float32)
k_z[0, 0, 0] = 1
k_z[2, 0, 0] = -1
k_y = np.zeros((1, 3, 1), dtype=np.float32)
k_y[0, 0, 0] = 1
k_y[0, 2, 0] = -1
k_x = np.zeros((1, 1, 3), dtype=np.float32)
k_x[0, 0, 0] = 1
k_x[0, 0, 2] = -1

k_zz = np.zeros((5, 1, 1), dtype=np.float32)
k_zz[0, 0, 0] = -1
k_zz[2, 0, 0] = 2
k_zz[4, 0, 0] = -1
k_yy = np.zeros((1, 5, 1), dtype=np.float32)
k_yy[0, 0, 0] = -1
k_yy[0, 2, 0] = 2
k_yy[0, 4, 0] = -1
k_xx = np.zeros((1, 1, 5), dtype=np.float32)
k_xx[0, 0, 0] = -1
k_xx[0, 0, 2] = 2
k_xx[0, 0, 4] = -1

k_zy = np.zeros((3, 3, 1), dtype=np.float32)
k_zy[0, 0, 0] = 1
k_zy[0, 2, 0] = -1
k_zy[2, 0, 0] = -1
k_zy[2, 2, 0] = 1
k_zx = np.zeros((3, 1, 3), dtype=np.float32)
k_zx[0, 0, 0] = 1
k_zx[0, 0, 2] = -1
k_zx[2, 0, 0] = -1
k_zx[2, 0, 2] = 1
k_yx = np.zeros((1, 3, 3), dtype=np.float32)
k_yx[0, 0, 0] = 1
k_yx[0, 0, 2] = -1
k_yx[0, 2, 0] = -1
k_yx[0, 2, 2] = 1

def create_random_segmentation(size, seed):

    np.random.seed(seed)
    peaks = np.random.random(size).astype(np.float32)
    peaks = gaussian_filter(peaks, sigma=5.0)
    max_filtered = maximum_filter(peaks, 10)
    maxima = max_filtered==peaks
    seeds, n = mahotas.label(maxima)
    print("Creating segmentation with %d segments"%n)
    return mahotas.cwatershed(1.0 - peaks, seeds).astype(np.uint64)

def get_descriptors(mask, sigma, fast=True):

    count = gaussian_filter(mask, sigma, mode='constant', output=np.float32)

    if fast:
        d_z = convolve(count, k_z, mode='constant')
        d_y = convolve(count, k_y, mode='constant')
        d_x = convolve(count, k_x, mode='constant')
        d_zz = convolve(count, k_zz, mode='constant')
        d_yy = convolve(count, k_yy, mode='constant')
        d_xx = convolve(count, k_xx, mode='constant')
        d_zy = convolve(count, k_zy, mode='constant')
        d_zx = convolve(count, k_zx, mode='constant')
        d_yx = convolve(count, k_yx, mode='constant')
    else:
        d_z = gaussian_filter(mask, sigma, order=(1, 0, 0), mode='constant', output=np.float32)
        d_y = gaussian_filter(mask, sigma, order=(0, 1, 0), mode='constant', output=np.float32)
        d_x = gaussian_filter(mask, sigma, order=(0, 0, 1), mode='constant', output=np.float32)
        d_zz = gaussian_filter(mask, sigma, order=(2, 0, 0), mode='constant', output=np.float32)
        d_yy = gaussian_filter(mask, sigma, order=(0, 2, 0), mode='constant', output=np.float32)
        d_xx = gaussian_filter(mask, sigma, order=(0, 0, 2), mode='constant', output=np.float32)
        d_zy = gaussian_filter(mask, sigma, order=(1, 1, 0), mode='constant', output=np.float32)
        d_zx = gaussian_filter(mask, sigma, order=(1, 0, 1), mode='constant', output=np.float32)
        d_yx = gaussian_filter(mask, sigma, order=(0, 1, 1), mode='constant', output=np.float32)

    d = np.stack([
        d_z, d_y, d_x,
        d_zz, d_yy, d_xx,
        d_zy, d_zx, d_yx])

    # normalize, move to [0, 1]
    count[count==0] = 1
    d = d/count + 0.5

    lsds = np.concatenate([d, count[None,:]])

    lsds[:,mask==0] = 0

    return lsds

if __name__ == "__main__":

    extractor = LsdExtractor(sigma=(5.0, 5.0, 5.0))

    segmentation = create_random_segmentation((100, 100, 100), seed=42)
    ids = np.unique(segmentation)

    start = time.time()
    lsds = extractor.get_descriptors(segmentation)
    print("Computed original LSDs in %fs"%(time.time() - start))

    start = time.time()
    lsds_compare_fast = np.zeros_like(lsds)
    for i in ids:
        lsds_compare_fast += get_descriptors(segmentation==i, (5.0, 5.0, 5.0))
    print("Computed alternative LSDs in %fs"%(time.time() - start))

    start = time.time()
    lsds_compare = np.zeros_like(lsds)
    for i in ids:
        lsds_compare += get_descriptors(segmentation==i, (5.0, 5.0, 5.0), fast=False)
    print("Computed alternative LSDs in %fs"%(time.time() - start))

    with h5py.File('test_shape_descriptor.hdf', 'w') as f:
        f['volumes/segmentation'] = segmentation
        f['volumes/lsds'] = lsds
        f['volumes/lsds_compare_fast'] = lsds_compare_fast
        f['volumes/lsds_compare'] = lsds_compare
