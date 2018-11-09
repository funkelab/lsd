import gunpowder as gp
import numpy as np
import time
import logging
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import convolve
from numpy.lib.stride_tricks import as_strided

logger = logging.getLogger(__name__)

def get_local_shape_descriptors(
        segmentation,
        sigma,
        voxel_size=None,
        roi=None,
        labels=None,
        mode='gaussian',
        downsample=1):
    '''
    Compute local shape descriptors for the given segmentation.

    Args:

        segmentation (``np.array`` of ``int``):

            A label array to compute the local shape descriptors for.

        sigma (``tuple`` of ``float``):

            The radius to consider for the local shape descriptor.

        voxel_size (``tuple`` of ``int``, optional):

            The voxel size of ``segmentation``. Defaults to 1.

        roi (``gunpowder.Roi``, optional):

            Restrict the computation to the given ROI.

        labels (array-like of ``int``, optional):

            Restrict the computation to the given labels. Defaults to all
            labels inside the ``roi`` of ``segmentation``.

        mode (``string``, optional):

            Only gaussian mode supported. Keep parameter for 
            compatibility reasons with alternative shape 
            descriptors. An error is thrown if anything else
            than gaussian is provided.

        downsample (``int``, optional):

            Compute the local shape descriptor on a downsampled volume for
            faster processing. Defaults to 1 (no downsampling).
    '''
    return LsdExtractor(sigma, mode, downsample).get_descriptors(
        segmentation,
        voxel_size,
        roi,
        labels)

class LsdExtractor(object):

    def __init__(self, sigma, mode='gaussian', downsample=1):
        '''
        Create an extractor for local shape descriptors. The extractor caches
        the data repeatedly needed for segmentations of the same size. If this
        is not desired, `func:get_local_shape_descriptors` should be used
        instead.

        Args:

            sigma (``tuple`` of ``float``):

                The radius to consider for the local shape descriptor.

            mode (``string``, optional):

                Only gaussian mode supported. Determines over what region
                the local shape descriptor is computed. For ``gaussian``, a
                Gaussian with the given ``sigma`` is used.

            downsample (``int``, optional):

                Compute the local shape descriptor on a downsampled volume for
                faster processing. Defaults to 1 (no downsampling).
        '''
        self.sigma = sigma
        self.mode = mode
        self.downsample = downsample

    def get_descriptors(
            self,
            segmentation,
            voxel_size=None,
            roi=None,
            labels=None):
        '''Compute local shape descriptors for a given segmentation.

        Args:

            segmentation (``np.array`` of ``int``):

                A label array to compute the local shape descriptors for.

            voxel_size (``tuple`` of ``int``, optional):

                The voxel size of ``segmentation``. Defaults to 1.

            roi (``gunpowder.Roi``, optional):

                Restrict the computation to the given ROI in voxels.

            labels (array-like of ``int``, optional):

                Restrict the computation to the given labels. Defaults to all
                labels inside the ``roi`` of ``segmentation``.
        '''

        dims = len(segmentation.shape)
        if voxel_size is None:
            voxel_size = gp.Coordinate((1,)*dims)
        else:
            voxel_size = gp.Coordinate(voxel_size)

        if roi is None:
            roi = gp.Roi((0,)*dims, segmentation.shape)

        roi_slices = roi.to_slices()

        if labels is None:
            labels = np.unique(segmentation[roi_slices])

        # prepare full-res descriptor volumes for roi
        descriptors = np.zeros((10,) + roi.get_shape(), dtype=np.float32)

        # get sub-sampled shape, roi, voxel size and sigma
        df = self.downsample
        logger.debug(
            "Downsampling segmentation %s with factor %f",
            segmentation.shape, df)
        sub_shape = tuple(s/df for s in segmentation.shape)
        sub_roi = roi/df
        assert sub_roi*df == roi, (
            "Segmentation shape %s is not a multiple of downsampling factor "
            "%d (sub_roi=%s, roi=%s)."%(
                segmentation.shape, self.downsample,
                sub_roi, roi))
        sub_voxel_size = tuple(v*df for v in voxel_size)
        sub_sigma_voxel = tuple(s/v for s, v in zip(self.sigma, sub_voxel_size))
        logger.debug("Downsampled shape: %s", sub_shape)
        logger.debug("Downsampled voxel size: %s", sub_voxel_size)
        logger.debug("Sigma in voxels: %s", sub_sigma_voxel)

        # for all labels
        for label in labels:

            if label == 0:
                continue

            logger.debug("Creating shape descriptors for label %d", label)

            mask = (segmentation==label).astype(np.float32)
            logger.debug("Label mask %s", mask.shape)
            sub_mask = mask[::df, ::df, ::df]
            logger.debug("Downsampled label mask %s", sub_mask.shape)

            sub_descriptor = self.__get_descriptor(sub_mask, 
                                                   sub_sigma_voxel, 
                                                   sub_voxel_size, 
                                                   sub_roi)

            logger.debug("Upscaling descriptors...")
            start = time.time()
            descriptor = self.__upsample(sub_descriptor, df)
            logger.debug("%f seconds", time.time() - start)

            logger.debug("Accumulating descriptors...")
            start = time.time()
            descriptors += descriptor
            logger.debug("%f seconds", time.time() - start)

        # clip outliers
        # np.clip(descriptors, 0.0, 1.0, out=descriptors)

        return descriptors

    def __get_descriptor(self, mask, sigma_voxel, voxel_size, roi):
        # number of inside voxels
        logger.debug("Calculate soft mask...")
        start = time.time()
        soft_mask = self.__aggregate(mask, sigma_voxel, self.mode, roi)
        logger.debug("%f seconds", time.time() - start)

        # kernels
        logger.debug("Computing derivatives...")
        k_z, k_y, k_x = self.__generate_kernels(voxel_size)

        # first derivatives
        d_z = convolve(soft_mask, k_z, mode='constant')
        d_y = convolve(soft_mask, k_y, mode='constant')
        d_x = convolve(soft_mask, k_x, mode='constant')

        # second derivatives
        d_zz = convolve(d_z, k_z, mode='constant')
        d_yy = convolve(d_y, k_y, mode='constant')
        d_xx = convolve(d_x, k_x, mode='constant')
        d_zy = convolve(d_z, k_y, mode='constant')
        d_zx = convolve(d_z, k_x, mode='constant')
        d_yx = convolve(d_y, k_x, mode='constant')

        # normalize s.t. peak value of a single gaussian is 1. Note that this
        # implies a value > 1 for addition of multiple gaussians. 
        # Change to max normalization if [0,1] range required.
        norm_fac_0 = np.sqrt((2*np.pi)**3 * np.prod([s**2 for s in sigma_voxel]))
        soft_mask *= norm_fac_0

        """
        TODO: Normalize derivatives according to max derivative of 
        single gaussian or alternative method. First derived at max 
        point scales as 1/sigma (x_max = +- sigma) while second derivative scales as
        1/sigma**2 (x_max = 0).
        """

        d = np.stack([d_z, d_y, d_x,
                      d_zz, d_yy, d_xx,
                      d_zy, d_zx, d_yx])
        
        lsd = np.concatenate([d, soft_mask[None, :]])
        lsd[:, soft_mask==0] = 0
        logger.debug("%f seconds", time.time() - start)

        return lsd

    def __generate_kernels(self, voxel_size):
        """
        For downsampled version this should
        be the subsampled voxel_size.
        """

        k_z = np.zeros((3, 1, 1), dtype=np.float32)
        k_z[0, 0, 0] = voxel_size[2]
        k_z[2, 0, 0] = -voxel_size[2]

        k_y = np.zeros((1, 3, 1), dtype=np.float32)
        k_y[0, 0, 0] = voxel_size[1]
        k_y[0, 2, 0] = -voxel_size[1]

        k_x = np.zeros((1, 1, 3), dtype=np.float32)
        k_x[0, 0, 0] = voxel_size[0]
        k_x[0, 0, 2] = -voxel_size[0]

        return k_z, k_y, k_x

    def __aggregate(self, array, sigma, mode='gaussian', roi=None):

        if roi is None:
            roi_slices = (slice(None),)
        else:
            roi_slices = roi.to_slices()

        if mode == 'gaussian':

            return gaussian_filter(
                array,
                sigma=sigma,
                mode='constant',
                cval=0.0,
                truncate=3.0)[roi_slices]

        elif mode == 'sphere':
            raise NotImplementedError("Sphere mode not supported with derivative based lsds.")

        else:
            raise RuntimeError("Unknown mode %s"%mode)

    def get_context(self):
        '''Return the context needed to compute the LSDs.'''
        if self.mode == 'gaussian':
            return tuple((3.0*s for s in self.sigma))
        elif self.mode == 'sphere':
            raise NotImplementedError("Sphere mode not supported with derivative based lsds.")
            return self.sigma

    def __upsample(self, array, f):

        shape = array.shape
        stride = array.strides

        view = as_strided(
            array,
            (shape[0], shape[1], f, shape[2], f, shape[3], f),
            (stride[0], stride[1], 0, stride[2], 0, stride[3], 0))

        return view.reshape(shape[0], shape[1]*f, shape[2]*f, shape[3]*f)
