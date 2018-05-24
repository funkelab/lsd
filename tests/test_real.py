import h5py
import lsd
import logging
import gunpowder as gp

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

    lsds_offset = gp.Coordinate(lsds_offset)
    affs_offset = gp.Coordinate(affs_offset)
    voxel_size = gp.Coordinate(voxel_size)

    raw_roi = gp.Roi(raw_offset, voxel_size*raw.shape)
    lsds_roi = gp.Roi(affs_offset, voxel_size*lsds.shape[1:])
    affs_roi = gp.Roi(affs_offset, voxel_size*affs.shape[1:])

    # crop lsds to affinities
    lsds = lsds[
        (slice(None),) +
        ((affs_roi - lsds_offset)/voxel_size).get_bounding_box()]

    fragments, distances, seeds = lsd.fragments.watershed_from_affinities(
        affs,
        return_distances=True,
        return_seeds=True)

    lsd_extractor = lsd.LsdExtractor(
        sigma=(80.0, 80.0, 80.0),
        downsample=1)
    voxel_size = (8, 8, 8)

    with h5py.File('test_real.hdf', 'w') as f:
        f['volumes/lsds'] = lsds
        f['volumes/lsds'].attrs['resolution'] = voxel_size
        f['volumes/lsds'].attrs['offset'] = lsds_roi.get_offset()
        f['volumes/fragments'] = fragments
        f['volumes/fragments'].attrs['resolution'] = voxel_size
        f['volumes/fragments'].attrs['offset'] = lsds_roi.get_offset()
        f['volumes/distances'] = distances
        f['volumes/distances'].attrs['resolution'] = voxel_size
        f['volumes/distances'].attrs['offset'] = lsds_roi.get_offset()
        f['volumes/seeds'] = seeds
        f['volumes/seeds'].attrs['resolution'] = voxel_size
        f['volumes/seeds'].attrs['offset'] = lsds_roi.get_offset()
        f['volumes/raw'] = raw
        f['volumes/raw'].attrs['resolution'] = voxel_size
        f['volumes/raw'].attrs['offset'] = raw_roi.get_offset()

    agglomeration = lsd.LsdAgglomeration(
        fragments,
        lsds,
        lsd_extractor,
        voxel_size=voxel_size)

    for threshold in [-1000,-100,-10,-1,0,1]:

        agglomeration.merge_until(threshold)
        segmentation = agglomeration.get_segmentation()

        ds_name = 'volumes/segmentation_%d'%threshold
        with h5py.File('test_real.hdf', 'r+') as f:
            f[ds_name] = segmentation
            f[ds_name].attrs['resolution'] = voxel_size
            f[ds_name].attrs['offset'] = lsds_roi.get_offset()
