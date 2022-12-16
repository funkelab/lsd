import h5py
import lsd
import logging
import gunpowder as gp
from lsd.train import LsdExtractor

logging.basicConfig(level=logging.INFO)
# logging.getLogger('lsd.agglomerate').setLevel(logging.DEBUG)

if __name__ == "__main__":

    with h5py.File('batch_44001.hdf', 'r') as f:

        raw = f['volumes/raw'][:]
        raw_offset = f['volumes/raw'].attrs['offset']

        lsds = f['volumes/embedding'][:]
        lsds_offset = f['volumes/embedding'].attrs['offset']
        voxel_size = f['volumes/embedding'].attrs['resolution']

        affs = f['volumes/labels/pred_affinities'][:]
        affs_offset = f['volumes/labels/pred_affinities'].attrs['offset']

        gt = f['volumes/labels/neuron_ids'][:]
        gt_offset = f['volumes/labels/neuron_ids'].attrs['offset']

    lsds_offset = gp.Coordinate(lsds_offset)
    affs_offset = gp.Coordinate(affs_offset)
    voxel_size = gp.Coordinate(voxel_size)

    raw_roi = gp.Roi(raw_offset, voxel_size*raw.shape)
    lsds_roi = gp.Roi(affs_offset, voxel_size*lsds.shape[1:])
    affs_roi = gp.Roi(affs_offset, voxel_size*affs.shape[1:])

    # crop lsds to affinities
    lsds = lsds[
        (slice(None),) +
        ((affs_roi - lsds_offset)/voxel_size).to_slices()]

    lsd_extractor = LsdExtractor(
        sigma=(80.0, 80.0, 80.0),
        downsample=2)
    voxel_size = (8, 8, 8)

    with h5py.File('test_real.hdf', 'w') as f:
        f['volumes/lsds'] = lsds[0:3]
        f['volumes/lsds'].attrs['resolution'] = voxel_size
        f['volumes/lsds'].attrs['offset'] = lsds_roi.get_offset()
        f['volumes/raw'] = raw
        f['volumes/raw'].attrs['resolution'] = voxel_size
        f['volumes/raw'].attrs['offset'] = raw_roi.get_offset()
        f['volumes/affs'] = affs
        f['volumes/affs'].attrs['resolution'] = voxel_size
        f['volumes/affs'].attrs['offset'] = affs_offset
        f['volumes/neuron_ids'] = gt
        f['volumes/neuron_ids'].attrs['resolution'] = voxel_size
        f['volumes/neuron_ids'].attrs['offset'] = gt_offset
