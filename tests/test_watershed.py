import h5py
import lsd
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger('lsd.fragments').setLevel(logging.DEBUG)

if __name__ == "__main__":

    with h5py.File('batch_44001.hdf', 'r') as f:
        lsds = f['volumes/embedding'][:]

    centers = lsds[0:3] - 0.5
    fragments, distances, seeds = lsd.fragments.watershed(
        centers,
        sigma=1.0,
        return_distances=True,
        return_seeds=True)

    with h5py.File('test_watershed.hdf', 'w') as f:
        f['volumes/lsds'] = lsds
        f['volumes/fragments'] = fragments
        f['volumes/distances'] = distances
        f['volumes/seeds'] = seeds
