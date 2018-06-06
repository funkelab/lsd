from peach import Coordinate, Roi, BlockTask, ProcessBlocks, BlockDoneTarget
import logging
import luigi

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
        be optimal.'''

        logger.info("Merging until %f...", threshold)

        block_done_function = self.block_done_function

        class AgglomerateTask(BlockTask):

            # class attributes that do not change between AgglomerateTasks
            rag_provider = self.rag_provider
            fragments = self.fragments
            target_lsds = self.target_lsds
            lsd_extractor = self.lsd_extractor

            def run(self):

                logger.info(
                    "Agglomerating in block %s with context of %s",
                    self.write_roi, self.read_roi)

                # get the subgraph to work on
                rag = self.rag_provider[self.read_roi.to_slices()]

                # TODO: perform agglomeration
                #
                # for now, just mark the block as processed
                rag.set_edge_attributes('agglomerated', 1)

                # write back results
                rag.sync()

            def output(self):

                return BlockDoneTarget(
                    self.write_roi,
                    block_done_function)

        dims = len(self.fragments.shape)
        if not self.voxel_size:
            self.voxel_size = Coordinate((1,)*dims)

        context = Coordinate(self.lsd_extractor.get_context())

        total_roi = Roi((0,)*dims, self.voxel_size*self.fragments.shape)
        write_roi = Roi((0,)*dims, self.block_write_size)
        read_roi = write_roi.grow(context, context)

        process_blocks = ProcessBlocks(
            total_roi,
            read_roi,
            write_roi,
            AgglomerateTask)

        luigi.build([process_blocks], log_level='INFO',
                workers=self.num_workers)
