from .merge_tree import MergeTree
from funlib.segment.arrays import relabel
import daisy
import logging
import numpy as np
import waterz

logger = logging.getLogger(__name__)

def parallel_aff_agglomerate(
        affs,
        fragments,
        rag_provider,
        block_size,
        context,
        merge_function,
        threshold,
        num_workers):
    '''Agglomerate fragments in parallel using ``waterz``.

    Args:

        affs (`class:daisy.Array`):

            An array containing affinities.

        fragments (`class:daisy.Array`):

            An array containing fragments.

        rag_provider (`class:SharedRagProvider`):

            A RAG provider to read nodes from and write found edges to.

        block_size (``tuple`` of ``int``):

            The size of the blocks to process in parallel, in world units.

        context (``tuple`` of ``int``):

            The context to consider for agglomeration, in world units.

        merge_function (``string``):

            The merge function to use for ``waterz``.

        threshold (``float``):

            Until which threshold to agglomerate.

        num_workers (``int``):

            The number of parallel workers.

    Returns:

        True, if all tasks succeeded.
    '''

    assert fragments.data.dtype == np.uint64

    shape = affs.shape[1:]
    context = daisy.Coordinate(context)

    total_roi = affs.roi.grow(context, context)
    read_roi = daisy.Roi((0,)*affs.roi.dims(), block_size).grow(context, context)
    write_roi = daisy.Roi((0,)*affs.roi.dims(), block_size)

    return daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        lambda b: agglomerate_in_block(
            affs,
            fragments,
            rag_provider,
            b,
            merge_function,
            threshold),
        lambda b: block_done(b, rag_provider),
        num_workers=num_workers,
        read_write_conflict=False,
        fit='shrink')

def block_done(block, rag_provider):

    return (
        rag_provider.has_edges(block.write_roi) or
        rag_provider.num_nodes(block.write_roi) == 0)

def agglomerate_in_block(
        affs,
        fragments,
        rag_provider,
        block,
        merge_function,
        threshold):

    logger.info(
        "Agglomerating in block %s with context of %s",
        block.write_roi, block.read_roi)

    # get the sub-{affs, fragments, graph} to work on
    affs = affs.intersect(block.read_roi)
    fragments = fragments.to_ndarray(affs.roi, fill_value=0)
    rag = rag_provider[affs.roi]

    # waterz uses memory proportional to the max label in fragments, therefore
    # we relabel them here and use those
    fragments_relabelled, n, fragment_relabel_map = relabel(
        fragments,
        return_backwards_map=True)

    logger.debug("affs shape: %s", affs.shape)
    logger.debug("fragments shape: %s", fragments.shape)
    logger.debug("fragments num: %d", n)

    # convert affs to float32 ndarray with values between 0 and 1
    affs = affs.to_ndarray()[0:3]
    if affs.dtype == np.uint8:
        affs = affs.astype(np.float32)/255.0

    # So far, 'rag' does not contain any edges belonging to write_roi (there
    # might be a few edges from neighboring blocks, though). Run waterz until
    # threshold 0 to get the waterz RAG, which tells us which nodes are
    # neighboring. Use this to populate 'rag' with edges. Then run waterz for
    # the given threshold.

    # for efficiency, we create one waterz call with both thresholds
    generator = waterz.agglomerate(
            affs=affs,
            thresholds=[0, threshold],
            fragments=fragments_relabelled,
            scoring_function=merge_function,
            discretize_queue=256,
            return_merge_history=True,
            return_region_graph=True)

    # add edges to RAG
    _, _, initial_rag = next(generator)
    for edge in initial_rag:
        u, v = fragment_relabel_map[edge['u']], fragment_relabel_map[edge['v']]
        # this might overwrite already existing edges from neighboring blocks,
        # but that's fine, we only write attributes for edges within write_roi
        rag.add_edge(u, v, merge_score=None, agglomerated=True)

    # agglomerate fragments using affs
    _, merge_history, _ = next(generator)

    # cleanup generator
    for _, _, _ in generator:
        pass

    # create a merge tree from the merge history
    merge_tree = MergeTree(fragment_relabel_map)
    for merge in merge_history:

        a, b, c, score = merge['a'], merge['b'], merge['c'], merge['score']
        merge_tree.merge(
            fragment_relabel_map[a],
            fragment_relabel_map[b],
            fragment_relabel_map[c],
            score)

    # mark edges in original RAG with score at time of merging
    logger.debug("marking merged edges...")
    num_merged = 0
    for u, v, data in rag.edges(data=True):
        merge_score = merge_tree.find_merge(u, v)
        data['merge_score'] = merge_score
        if merge_score is not None:
            num_merged += 1

    logger.info("merged %d edges", num_merged)

    # write back results (only within write_roi)
    logger.debug("writing to DB...")
    rag.write_edges(block.write_roi)
