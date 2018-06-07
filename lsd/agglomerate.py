from scipy.ndimage.measurements import find_objects
from skimage.future.graph import RAG
from graph_merge import merge_hierarchical
import gunpowder as gp
import numpy as np
import logging
import math

logger = logging.getLogger(__name__)

class LsdAgglomeration(object):
    '''Create a local shape descriptor agglomerator.

    Args:

        fragments (``np.ndarray``):

            Initial fragments.

        target_lsds (``np.ndarray``):

            The local shape descriptors to match.

        lsd_extractor (``LsdExtractor``):

            The local shape descriptor object used to compute the difference
            between the segmentation and the target LSDs.

        voxel_size (``tuple`` of ``int``, optional):

            The voxel size of ``fragments``. Defaults to 1.

        keep_lsds (``bool``, optional):

            If ``True``, keeps the reconstructed local shape descriptors, which
            can then be queried with `func:get_lsds`.
    '''

    def __init__(
            self,
            fragments,
            target_lsds,
            lsd_extractor,
            voxel_size=None,
            keep_lsds=False):

        self.segmentation = np.array(fragments)
        if keep_lsds:
            self.lsds = np.zeros_like(target_lsds)
        else:
            self.lsds = None
        self.fragments = fragments
        self.target_lsds = target_lsds
        self.lsd_extractor = lsd_extractor
        self.context = lsd_extractor.get_context()

        if voxel_size is None:
            self.voxel_size = (1,)*len(fragments.shape)
        else:
            self.voxel_size = voxel_size

        self.__initialize_rag()

    def merge_until(self, threshold, max_merges=-1):
        '''Merge until the given threshold. Since edges are scored by how much
        they decrease the distance to ``target_lsds``, a threshold of 0 should
        be optimal.'''

        logger.info("Merging until %f...", threshold)

        merge_func = lambda _, src, dst: self.__update_lsds(src, dst)
        weight_func = lambda _g, _s, u, v: self.__score_merge(u, v)
        num_merges = merge_hierarchical(
            self.fragments,
            self.rag,
            thresh=threshold,
            rag_copy=False,
            in_place_merge=True,
            merge_func=merge_func,
            weight_func=weight_func,
            max_merges=max_merges,
            return_segmenation=False)

        logger.info("Finished merging")

        return num_merges

    def get_segmentation(self):
        '''Return the segmentation obtained so far by calls to
        ``merge_until``.'''

        return self.segmentation

    def get_lsds(self):
        '''Return the local shape descriptors corresponding to the current
        segmentation. ``keep_lsds`` has to be set in the constructor.'''
        return self.lsds

    def __initialize_rag(self):

        self.rag = RAG(self.fragments, connectivity=2)
        logger.info(
            "RAG contains %d nodes and %d edges",
            len(self.rag.nodes()),
            len(self.rag.edges()))

        logger.info("Computing LSDs for initial fragments...")

        for u in self.rag.nodes():

            logger.debug("Initializing node %d", u)

            bb = find_objects(self.fragments==u)[0]
            self.rag.node[u]['roi'] = self.__slice_to_roi(bb)
            self.rag.node[u]['score'] = self.__compute_score(
                u,
                update_lsds=True)
            self.rag.node[u]['labels'] = [u] # needed by scikit

            logger.debug("Node %d: %s", u, self.rag.node[u])

        logger.info("Scoring initial edges...")

        for (u, v) in self.rag.edges():
            logger.debug("Initializing edge (%d, %d)", u, v)
            score = self.__score_merge(u, v)
            self.rag[u][v]['weight'] = score['weight']

    def __update_lsds(self, src, dst):
        '''Callback for merge_hierarchical, called before src and dst are
        merged.'''

        # src and dst get merged into dst, update the stats for dst

        self.rag.node[dst]['score'] = self.__compute_score(
            src, dst,
            update_rois=True,
            update_segmentation=True,
            update_lsds=True)

        logger.info(
            "Merged %d into %d with score %f",
            src,
            dst,
            self.rag[src][dst]['weight'])
        logger.debug(
            "Updated score of %d (merged with %d) to %f",
            dst, src, self.rag.node[dst]['score'])

    def __score_merge(self, u, v):
        '''Callback for merge_hierarchical, called to get the weight of a new
        edge.'''

        score_u = self.rag.node[u]['score']
        score_v = self.rag.node[v]['score']

        score_uv = self.__compute_score(u, v)

        weight = score_uv - (score_u + score_v)

        logger.debug("Scoring merge between %d and %d with %f", u, v, weight)

        return {'weight': weight}

    def __compute_score(
            self,
            u,
            v=None,
            update_rois=False,
            update_segmentation=False,
            update_lsds=False):
        '''Compute the LSDs score for either one (v=None) or two fragments
        together.

        If ``update_rois``, the ROI for v will be replaced with the ROI for
        u+v.

        If ``update_segmentation``, also replaces all occurences of u with v in
        ``self.segmentation``.'''

        # get ROI
        roi_u = self.rag.node[u]['roi']

        if v is not None:

            # u and v are given, we compute the score for an edge -> update
            # LSDs only on edge bounding box
            roi_v = self.rag.node[v]['roi']

            segmentation_roi = gp.Roi(
                (0,)*len(self.segmentation.shape),
                self.segmentation.shape)

            # print("roi u: %s"%roi_u)
            # print("roi v: %s"%roi_v)
            context = tuple(int(math.ceil(c/vs)) for c, vs in zip(self.context, self.voxel_size))
            roi_u_grown = roi_u.grow(context, context)
            roi_v_grown = roi_v.grow(context, context)
            roi_u_grown = roi_u_grown.intersect(segmentation_roi)
            roi_v_grown = roi_v_grown.intersect(segmentation_roi)
            # print("grown roi u: %s"%roi_u)
            # print("grown roi v: %s"%roi_v)

            compute_roi = roi_u_grown.intersect(roi_v_grown)
            compute_roi = compute_roi.intersect(roi_u.union(roi_v))
            total_roi = compute_roi.grow(context, context)
            total_roi = total_roi.intersect(segmentation_roi)
            # print("compute roi v: %s"%compute_roi)
            # print("total roi v: %s"%total_roi)

            if update_rois:
                # set the ROI of v to the union of u and v
                self.rag.node[v]['roi'] = roi_u.union(roi_v)

        else:

            total_roi = roi_u
            compute_roi = total_roi

        # get slice of segmentation for dst roi
        segmentation = self.segmentation[total_roi.to_slices()]
        if not update_segmentation:
            segmentation = np.array(segmentation)

        if v is not None:
            # mark u as v
            segmentation[segmentation==u] = v
        else:
            v = u

        # get LSDs for u(+v)
        compute_in_total_roi = compute_roi - total_roi.get_begin()
        lsds = self.lsd_extractor.get_descriptors(
            segmentation,
            roi=compute_in_total_roi,
            labels=[v],
            voxel_size=self.voxel_size)

        # subtract from target LSDs
        v_mask = segmentation[compute_in_total_roi.to_slices()] == v
        lsds_slice = (slice(None),) + compute_roi.to_slices()
        diff = self.target_lsds[lsds_slice] - lsds
        diff[:,v_mask==0] = 0

        if update_lsds and self.lsds is not None:
            self.lsds[lsds_slice][:,v_mask] = lsds[:,v_mask]

        return np.sum(diff**2)

    def __slice_to_roi(self, slices):

        offset = tuple(s.start for s in slices)
        shape = tuple(s.stop - s.start for s in slices)

        roi = gp.Roi(offset, shape)
        roi = roi.snap_to_grid((self.lsd_extractor.downsample,)*roi.dims())

        return roi
