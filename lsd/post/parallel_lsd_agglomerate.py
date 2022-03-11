from __future__ import absolute_import
from .agglomerate import LsdAgglomeration
from .merge_tree import MergeTree
import daisy
import logging
import numpy as np
import skimage.future

logger = logging.getLogger(__name__)

def parallel_lsd_agglomerate(
        lsds,
        fragments,
        rag_provider,
        lsd_extractor,
        block_size,
        context,
        num_workers):
    '''Agglomerate fragments in parallel using only the shape descriptors.

    Args:

        lsds (`class:daisy.Array`):

            An array containing the LSDs.

        fragments (`class:daisy.Array`):

            An array containing fragments.

        rag_provider (`class:SharedRagProvider`):

            A RAG provider to read nodes from and write found edges to.

        lsd_extractor (``LsdExtractor``):

            The local shape descriptor object used to compute the difference
            between the segmentation and the target LSDs.

        block_size (``tuple`` of ``int``):

            The size of the blocks to process in parallel, in world units.

        context (``tuple`` of ``int``):

            The context to consider for agglomeration, in world units.

        num_workers (``int``):

            The number of parallel workers.

    Returns:

        True, if all tasks succeeded.
    '''

    assert fragments.data.dtype == np.uint64

    shape = lsds.shape[1:]
    context = daisy.Coordinate(context)

    total_roi = lsds.roi.grow(context, context)
    read_roi = daisy.Roi((0,)*lsds.roi.dims(), block_size).grow(context, context)
    write_roi = daisy.Roi((0,)*lsds.roi.dims(), block_size)

    return daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        lambda b: agglomerate_in_block(
            lsds,
            fragments,
            rag_provider,
            lsd_extractor,
            b),
        lambda b: block_done(b, rag_provider),
        num_workers=num_workers,
        read_write_conflict=False,
        fit='shrink')

def block_done(block, rag_provider):

    rag = rag_provider[block.write_roi]
    return rag.number_of_edges() > 0 or rag.number_of_nodes() <= 1

def agglomerate_in_block(
        lsds,
        fragments,
        rag_provider,
        lsd_extractor,
        block):

    logger.info(
        "Agglomerating in block %s with context of %s",
        block.write_roi, block.read_roi)

    # get the sub-{lsds, fragments, graph} to work on
    lsds = lsds.intersect(block.read_roi)
    fragments = fragments.to_ndarray(lsds.roi, fill_value=0)
    rag = rag_provider[lsds.roi]
    voxel_size = lsds.voxel_size
    lsds = lsds.to_ndarray()

    # So far, 'rag' does not contain any edges belonging to write_roi (there
    # might be a few edges from neighboring blocks, though). Use the fragments
    # to get an initial RAG (merge_rag) which we also use for agglomeration.
    merge_rag = skimage.future.graph.RAG(fragments)

    # Keep the original RAG edges
    for (u, v) in merge_rag.edges():
        # this might overwrite already existing edges from neighboring blocks,
        # but that's fine, we only write attributes for edges within write_roi
        rag.add_edge(u, v, {'merge_score': None, 'agglomerated': True})

    # agglomerate to match target LSDs
    agglomeration = LsdAgglomeration(
        fragments,
        lsds,
        lsd_extractor,
        voxel_size=voxel_size,
        rag=merge_rag,
        log_prefix='%s: '%block.write_roi)
    merge_history = agglomeration.merge_until(0)

    # create a merge tree from the merge history
    merge_tree = MergeTree(np.unique(fragments))
    for merge in merge_history:
        a, b, c, score = merge['a'], merge['b'], merge['c'], merge['score']
        merge_tree.merge(a, b, c, float(score))

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
    rag.sync_edges(block.write_roi)
