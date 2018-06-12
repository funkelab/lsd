from scipy.ndimage.filters import gaussian_filter, maximum_filter
import gunpowder as gp
import h5py
import logging
import lsd
import mahotas
import numpy as np

# hack to avoid duplicate luigi logs
from luigi.interface import setup_interface_logging
setup_interface_logging.has_run = True

logging.basicConfig(level=logging.INFO)
# logging.getLogger('lsd.persistence.sqlite_rag_provider').setLevel(logging.DEBUG)
# logging.getLogger('lsd.agglomerate').setLevel(logging.DEBUG)

# context is 30 in xy, 0 in z; net size to work on is 10x500x500
size = (10, 560, 560)

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

    initial_fragments = np.array(fragments)

    print("Computing target LSDs...")
    lsd_extractor = lsd.LsdExtractor(sigma=(0.1, 10.0, 10.0), downsample=2)
    target_lsds = lsd_extractor.get_descriptors(gt)

    rag_provider = lsd.persistence.SqliteRagProvider.from_fragments(
        fragments,
        'test_parallel_rag.db')

    # PARALLEL AGGLOMERATION

    def block_done(block_roi):

        rag = rag_provider[block_roi.to_slices()]
        done = [ d['agglomerated'] for _u, _v, d in rag.edges(data=True) ]

        return all(done)

    print("Starting parallel agglomeration...")
    agglomeration = lsd.ParallelLsdAgglomeration(
        rag_provider,
        fragments,
        target_lsds,
        lsd_extractor,
        block_write_size=(10, 50, 50),
        block_done_function=block_done,
        num_workers=10)

    try:
        agglomeration.merge_until(0)
    except:
        pass

    # READ RESULT

    segmentation = np.array(fragments)

    rag = rag_provider[gp.Roi((0, 0, 0), size).to_slices()]
    rag.contract_merged_nodes(segmentation)

    with h5py.File('test_parallel.hdf', 'w') as f:

        f['initial_fragments'] = initial_fragments
        f['fragments'] = fragments
        f['segmentation'] = segmentation
        f['target_lsds'] = target_lsds
