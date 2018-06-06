import h5py
import logging
import lsd
import mahotas
import numpy as np
from scipy.ndimage.filters import gaussian_filter, maximum_filter

# hack to avoid duplicate luigi logs
from luigi.interface import setup_interface_logging
setup_interface_logging.has_run = True

logging.basicConfig(level=logging.INFO)
# logging.getLogger('lsd.persistence.sqlite_rag_provider').setLevel(logging.DEBUG)

size = (1, 200, 200)

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

    lsd_extractor = lsd.LsdExtractor(sigma=(0.0, 10.0, 10.0))
    predicted_lsds = lsd_extractor.get_descriptors(gt)

    rag_provider = lsd.persistence.SqliteRagProvider.from_fragments(
        fragments,
        'test_parallel_rag.db')

    def block_done(block_roi):

        rag = rag_provider[block_roi.to_slices()]
        done = [ d['agglomerated'] for _u, _v, d in rag.edges(data=True) ]

        return all(done)

    agglomeration = lsd.ParallelLsdAgglomeration(
        rag_provider,
        fragments,
        predicted_lsds,
        lsd_extractor,
        block_write_size=(1, 10, 10),
        block_done_function=block_done,
        num_workers=1)

    agglomeration.merge_until(0)
