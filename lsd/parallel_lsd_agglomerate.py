from agglomerate import LsdAgglomeration
from peach import Coordinate, Roi, run_with_dask
import logging

logger = logging.getLogger(__name__)

class ParallelLsdAgglomeration(object):
    '''Create a local shape descriptor agglomerator for parallel agglomeration.

    For large volumes, ``rag``, ``fragments``, and ``target_lsds`` should be
    held in an out-of-memory datastructure that supports slicing to read
    subgraphs/subarrays.

    Args:

        rag_provider (`class:SharedRagProvider`):

            A RAG provider to query for subgraphs. Results will be written back
            to this provider in mutually exclusive writes.

        fragments (array-like):

            Label array of the nodes in the RAG provided by ``rag_provider``.

        target_lsds (array-like):

            The local shape descriptors to match.

        lsd_extractor (``LsdExtractor``):

            The local shape descriptor object used to compute the difference
            between the segmentation and the target LSDs.

        block_write_size (``tuple`` of ``int``):

            The write size of the blocks processed in parallel in world units.

            Note that due to context needed to compute the LSDs the actual read
            size per block is larger. The necessary context is automatically
            determined from the given ``lsd_extractor``. Results will only be
            computed and written back for blocks that fit entirely into the
            total ROI.

        block_done_function (function):

            A function that is passed a `class:Roi` and should return ``True``
            if the block represented by this ROI was already processed. Only
            blocks for which this function returns ``False`` are processed.

        num_workers (``int``):

            The number of blocks to process in parallel.

        voxel_size (``tuple`` of ``int``, optional):

            The voxel size of ``fragments``. Defaults to 1.
    '''

    def __init__(
            self,
            rag_provider,
            fragments,
            target_lsds,
            lsd_extractor,
            block_write_size,
            block_done_function,
            num_workers,
            voxel_size=None):

        self.rag_provider = rag_provider
        self.fragments = fragments
        self.target_lsds = target_lsds
        self.lsd_extractor = lsd_extractor
        self.block_write_size = block_write_size
        self.block_done_function = block_done_function
        self.num_workers = num_workers
        self.voxel_size = voxel_size

    def merge_until(self, threshold, max_merges=-1):
        '''Merge until the given threshold. Since edges are scored by how much
        they decrease the distance to ``target_lsds``, a threshold of 0 should
        be optimal. Returns True if all tasks succeeded.'''

        logger.info("Merging until %f...", threshold)

        dims = len(self.fragments.shape)
        if not self.voxel_size:
            self.voxel_size = Coordinate((1,)*dims)

        voxel_size = self.voxel_size

        context = Coordinate(self.lsd_extractor.get_context())

        # assure that context is a multiple of voxel size
        one = Coordinate((1,)*len(context))
        context = ((context - one)/voxel_size + one)*voxel_size

        total_roi = Roi((0,)*dims, voxel_size*self.fragments.shape)
        write_roi = Roi((0,)*dims, self.block_write_size)
        read_roi = write_roi.grow(context, context)

        assert (write_roi/voxel_size)*voxel_size == write_roi, (
            "block_write_size needs to be a multiple of voxel_size")
        assert (write_roi/voxel_size)*voxel_size == write_roi, (
            "read_roi needs to be a multiple of voxel_size")

        return run_with_dask(
            total_roi,
            read_roi,
            write_roi,
            lambda r, w: self.__agglomerate_block(r, w),
            self.block_done_function,
            self.num_workers)

    def __agglomerate_block(self, read_roi, write_roi):

        logger.info(
            "Agglomerating in block %s with context of %s",
            write_roi, read_roi)

        read_roi_voxels = read_roi/self.voxel_size
        write_roi_voxels = write_roi/self.voxel_size

        # get the sub-{graph, fragments, LSDs} to work on
        rag = self.rag_provider[read_roi.to_slices()]
        fragments = self.fragments[read_roi_voxels.to_slices()]
        target_lsds = self.target_lsds[(slice(None),) + read_roi_voxels.to_slices()]

        # agglomerate on a copy of the original RAG
        # (agglomeration changes the RAG)
        merge_rag = rag.copy()

        # contract previously merged nodes and fragments
        merge_rag.contract_merged_nodes(self.fragments)

        # agglomerate to match target LSDs
        agglomeration = LsdAgglomeration(
            fragments,
            target_lsds,
            self.lsd_extractor,
            voxel_size=self.voxel_size,
            rag=merge_rag,
            log_prefix='%s: '%write_roi)
        num_merged = agglomeration.merge_until(0)

        # mark edges in original RAG as 'merged'
        rag.label_merged_edges(merge_rag)

        # mark the block as 'agglomerated'
        rag.set_edge_attributes('agglomerated', 1)

        logger.info("merged %d edges", num_merged)

        # write back results (only within write_roi)
        rag.sync_edge_attributes(write_roi)
