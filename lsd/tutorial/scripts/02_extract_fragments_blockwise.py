import daisy
import hashlib
import json
import logging
import lsd
import numpy as np
import os
import pymongo
import sys
import time

logging.basicConfig(level=logging.INFO)

def extract_fragments(
        experiment,
        setup,
        iteration,
        affs_file,
        affs_dataset,
        fragments_file,
        fragments_dataset,
        block_size,
        context,
        db_host,
        db_name,
        num_workers,
        fragments_in_xy,
        queue,
        epsilon_agglomerate=0,
        mask_file=None,
        mask_dataset=None,
        filter_fragments=0,
        replace_sections=None,
        **kwargs):

    '''

    Extract fragments in parallel blocks. Requires that affinities have been
    predicted before.

    When running parallel inference, the worker files are located in the setup
    directory of each experiment since that is where the training was done and
    checkpoints are located. When running watershed (and agglomeration) in
    parallel, we call a worker file which can be located anywhere. By default,
    we assume there is a workers directory inside the current directory that
    contains worker scripts (e.g `workers/extract_fragments_worker.py`).

    Args:

        * following three params just used to build out file directory *

        experiment (``string``):

            Name of the experiment (fib25, hemi, zfinch, ...).

        setup (``string``):

            Name of the setup to predict (setup01, setup02, ...).

        iteration (``int``):

            Training iteration.

        affs_file (``string``):

            Path to file (zarr/n5) where predictions are stored.

        affs_dataset (``string``):

            Predictions dataset to use (e.g 'volumes/affs'). If using a scale pyramid,
            will try scale zero assuming stored in directory `s0` (e.g
            'volumes/affs/s0').

        fragments_file (``string``):

            Path to file (zarr/n5) to store fragments (supervoxels) - generally
            a good idea to store in the same place as affs.

        fragments_dataset (``string``):

            Name of dataset to write fragments (supervoxels) to (e.g
            'volumes/fragments').

        block_size (``tuple`` of ``int``):

            The size of one block in world units (must be multiple of voxel
            size).

        context (``tuple`` of ``int``):

            The context to consider for fragment extraction in world units.

        db_host (``string``):

            Name of MongoDB client.

        db_name (``string``):

            Name of MongoDB database to use (for logging successful blocks in
            check function and writing nodes to the region adjacency graph).

        num_workers (``int``):

            How many blocks to run in parallel.

        fragments_in_xy (``bool``):

            Whether to extract fragments for each xy-section separately.

        queue (``string``):

            Name of cpu queue to use (e.g local)

        epsilon_agglomerate (``float``, optional):

            Perform an initial waterz agglomeration on the extracted fragments
            to this threshold. Skip if 0 (default).

        mask_file (``string``, optional):

            Path to file (zarr/n5) containing mask.

        mask_dataset (``string``, optional):

            Name of mask dataset. Data should be uint8 where 1 == masked in, 0
            == masked out.

        filter_fragments (``float``, optional):

            Filter fragments that have an average affinity lower than this
            value.

        replace_sections (``list`` of ``int``, optional):

            Replace fragments data with zero in given sections (useful if large
            artifacts are causing issues). List of section numbers (in voxels).

    '''

    logging.info(f"Reading affs from {affs_file}")

    try:
        affs = daisy.open_ds(affs_file, affs_dataset)
    except:
        affs_dataset = affs_dataset + '/s0'
        source = daisy.open_ds(affs_file, affs_dataset)

    network_dir = os.path.join(experiment, setup, str(iteration))

    client = pymongo.MongoClient(db_host)
    db = client[db_name]

    if 'blocks_extracted' not in db.list_collection_names():
            blocks_extracted = db['blocks_extracted']
            blocks_extracted.create_index(
                [('block_id', pymongo.ASCENDING)],
                name='block_id')
    else:
        blocks_extracted = db['blocks_extracted']

    # prepare fragments dataset. By default use same roi as affinities, change
    # roi if extracting fragments in cropped region
    fragments = daisy.prepare_ds(
        fragments_file,
        fragments_dataset,
        affs.roi,
        affs.voxel_size,
        np.uint64,
        daisy.Roi((0,0,0), block_size),
        compressor={'id': 'zlib', 'level':5})

    context = daisy.Coordinate(context)
    total_roi = affs.roi.grow(context, context)

    read_roi = daisy.Roi((0,)*affs.roi.dims(), block_size).grow(context, context)
    write_roi = daisy.Roi((0,)*affs.roi.dims(), block_size)

    #get number of voxels in block
    num_voxels_in_block = (write_roi/affs.voxel_size).size()

    #blockwise watershed
    daisy.run_blockwise(
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=lambda: start_worker(
            affs_file,
            affs_dataset,
            fragments_file,
            fragments_dataset,
            db_host,
            db_name,
            context,
            fragments_in_xy,
            queue,
            network_dir,
            epsilon_agglomerate,
            mask_file,
            mask_dataset,
            filter_fragments,
            replace_sections,
            num_voxels_in_block),
        check_function=lambda b: check_block(
            blocks_extracted,
            b),
        num_workers=num_workers,
        read_write_conflict=False,
        fit='shrink')

def start_worker(
        affs_file,
        affs_dataset,
        fragments_file,
        fragments_dataset,
        db_host,
        db_name,
        context,
        fragments_in_xy,
        queue,
        network_dir,
        epsilon_agglomerate,
        mask_file,
        mask_dataset,
        filter_fragments,
        replace_sections,
        num_voxels_in_block,
        **kwargs):

    worker_id = daisy.Context.from_env().worker_id

    logging.info(f"worker {worker_id} started...")

    output_dir = os.path.join('.extract_fragments_blockwise', network_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_out = os.path.join(output_dir, 'extract_fragments_blockwise_%d.out' %worker_id)
    log_err = os.path.join(output_dir, 'extract_fragments_blockwise_%d.err' %worker_id)

    config = {
            'affs_file': affs_file,
            'affs_dataset': affs_dataset,
            'fragments_file': fragments_file,
            'fragments_dataset': fragments_dataset,
            'db_host': db_host,
            'db_name': db_name,
            'context': context,
            'fragments_in_xy': fragments_in_xy,
            'queue': queue,
            'epsilon_agglomerate': epsilon_agglomerate,
            'mask_file': mask_file,
            'mask_dataset': mask_dataset,
            'filter_fragments': filter_fragments,
            'replace_sections': replace_sections,
            'num_voxels_in_block': num_voxels_in_block
        }

    config_str = ''.join(['%s'%(v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    config_file = os.path.join(output_dir, '%d.config'%config_hash)

    with open(config_file, 'w') as f:
        json.dump(config, f)

    logging.info('Running block with config %s...'%config_file)

    worker = 'workers/extract_fragments_worker.py'

    worker_command = os.path.join('.', worker)

    base_command = [
        'bsub',
        '-n', '1',
        '-o', f'{log_out}',
        f'python {worker_command} {config_file} > {log_out}'
    ]

    logging.info(f'Base command: {base_command}')

    daisy.call(base_command, log_out=log_out, log_err=log_err)

def check_block(blocks_extracted, block):

    done = blocks_extracted.count({'block_id': block.block_id}) >= 1

    return done

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()

    extract_fragments(**config)

    end = time.time()

    seconds = end - start
    logging.info(f'Total time to extract fragments: {seconds}')
