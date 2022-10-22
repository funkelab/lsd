from __future__ import absolute_import
from lsd.train import LsdExtractor
from gunpowder import BatchFilter, Array, BatchRequest, Batch
import logging
import numpy as np
import time

logger = logging.getLogger(__name__)


class AddLocalShapeDescriptor(BatchFilter):

    """Create a local segmentation shape discriptor to each voxel.

    Args:

        segmentation (:class:`ArrayKey`): The array storing the segmentation
            to use.

        descriptor (:class:`ArrayKey`): The array of the shape descriptor to
            generate.

        lsds_mask (:class:`ArrayKey`, optional): The array to store a binary mask
            the size of the descriptors. Background voxels, which do not have a
            descriptor, will be set to 0. This can be used as a loss scale
            during training, such that background is ignored.

        labels_mask (:class:`ArrayKey`, optional): The array to use as a mask
            for labels. Lsds connecting at least one masked out label will be
            masked out in lsds_mask.

        unlabelled (:class:`ArrayKey`, optional): A binary array to indicate
            unlabelled areas with 0. Lsds from labelled to unlabelled voxels are set
            to 0, lsds between unlabelled voxels are masked out (they will not be
            used for training).

        sigma (float or tuple of float): The context to consider to compute
            the shape descriptor in world units. This will be the standard
            deviation of a Gaussian kernel or the radius of the sphere.

        mode (string): Either ``gaussian`` or ``sphere``. Specifies how to
            accumulate local statistics: ``gaussian`` uses Gaussian convolution
            to compute a weighed average of statistics inside an object.
            ``sphere`` accumulates values in a sphere.

        downsample (int, optional): Downsample the segmentation mask to extract
            the statistics with the given factore. Default is 1 (no
            downsampling).
    """

    def __init__(
        self,
        segmentation,
        descriptor,
        lsds_mask=None,
        labels_mask=None,
        unlabelled=None,
        sigma=5.0,
        mode="gaussian",
        downsample=1,
    ):

        self.segmentation = segmentation
        self.descriptor = descriptor
        self.lsds_mask = lsds_mask
        self.labels_mask = labels_mask
        self.unlabelled = unlabelled

        try:
            self.sigma = tuple(sigma)
        except:
            self.sigma = (sigma,) * 3

        self.mode = mode
        self.downsample = downsample
        self.voxel_size = None
        self.context = None
        self.skip = False

        self.extractor = LsdExtractor(self.sigma, self.mode, self.downsample)

    def setup(self):

        spec = self.spec[self.segmentation].copy()
        spec.dtype = np.float32

        self.voxel_size = spec.voxel_size
        self.provides(self.descriptor, spec)

        if self.lsds_mask:
            self.provides(self.lsds_mask, spec.copy())

        if self.mode == "gaussian":
            self.context = tuple(s * 3 for s in self.sigma)
        elif self.mode == "sphere":
            self.context = tuple(self.sigma)
        else:
            raise RuntimeError("Unkown mode %s" % mode)

    def prepare(self, request):
        deps = BatchRequest()
        if self.descriptor in request:

            dims = len(request[self.descriptor].roi.get_shape())

            if dims == 2:
                self.context = self.context[0:2]

            # increase segmentation ROI to fit Gaussian
            context_roi = request[self.descriptor].roi.grow(self.context, self.context)

            # ensure context roi is multiple of voxel size
            context_roi = context_roi.snap_to_grid(self.voxel_size, mode="shrink")

            grown_roi = request[self.segmentation].roi.union(context_roi)

            deps[self.segmentation] = request[self.descriptor].copy()
            deps[self.segmentation].roi = grown_roi

        else:
            self.skip = True

        if self.unlabelled:
            deps[self.unlabelled] = deps[self.segmentation].copy()

        if self.labels_mask:
            deps[self.labels_mask] = deps[self.segmentation].copy()

        return deps

    def process(self, batch, request):
        if self.skip:
            return

        dims = len(self.voxel_size)

        segmentation_array = batch[self.segmentation]

        # get voxel roi of requested descriptors
        # this is the only region in
        # which we have to compute the descriptors
        seg_roi = segmentation_array.spec.roi
        descriptor_roi = request[self.descriptor].roi
        voxel_roi_in_seg = (
            seg_roi.intersect(descriptor_roi) - seg_roi.get_offset()
        ) / self.voxel_size

        crop = voxel_roi_in_seg.get_bounding_box()

        descriptor = self.extractor.get_descriptors(
            segmentation=segmentation_array.data,
            voxel_size=self.voxel_size,
            roi=voxel_roi_in_seg,
        )

        # create descriptor array
        descriptor_spec = self.spec[self.descriptor].copy()
        descriptor_spec.roi = request[self.descriptor].roi.copy()
        descriptor_array = Array(descriptor, descriptor_spec)

        old_batch = batch

        # Create new batch for descriptor:
        batch = Batch()

        # create lsds mask array
        if self.lsds_mask and self.lsds_mask in request:

            if self.labels_mask:

                mask = self._create_mask(old_batch, self.labels_mask, descriptor, crop)

            else:

                mask = (segmentation_array.crop(descriptor_roi).data != 0).astype(
                    np.float32
                )

                mask_shape = len(mask.shape)

                assert mask.shape[-mask_shape:] == descriptor.shape[-mask_shape:]

                mask = np.array([mask] * descriptor.shape[0])

            if self.unlabelled:

                unlabelled_mask = self._create_mask(
                    old_batch, self.unlabelled, descriptor, crop
                )

                mask = mask * unlabelled_mask

            batch[self.lsds_mask] = Array(
                mask.astype(descriptor.dtype), descriptor_spec.copy()
            )

        batch[self.descriptor] = descriptor_array

        return batch

    def _create_mask(self, batch, mask, lsds, crop):

        mask = batch.arrays[mask].data

        mask = np.array([mask] * lsds.shape[0])

        mask = mask[(slice(None),) + crop]

        return mask
