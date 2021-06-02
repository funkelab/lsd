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

def agglomerate(
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
        queue,
        merge_function,
        **kwargs):

    '''

    Agglomerate in parallel blocks. Requires that affinities and supervoxels
    have been generated.

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

            Predictions dataset to use (e.g 'volumes/affs').

        fragments_file (``string``):

            Path to file (zarr/n5) where fragments (supervoxels) are stored.

        fragments_dataset (``string``):

            Name of fragments (supervoxels) dataset (e.g 'volumes/fragments').

        block_size (``tuple`` of ``int``):

            The size of one block in world units (must be multiple of voxel
            size).

        context (``tuple`` of ``int``):

            The context to consider for fragment extraction in world units.

        db_host (``string``):

            Name of MongoDB client.

        db_name (``string``):

            Name of MongoDB database to use (for logging successful blocks in
            check function and reading nodes from + writing edges to the region
            adjacency graph).

        num_workers (``int``):

            How many blocks to run in parallel.

        merge_function (``string``):

            Symbolic name of a merge function. See dictionary in worker script
            (workers/agglomerate_worker.py).

    '''

    logging.info(f"Reading affs from {affs_file}")
    affs = daisy.open_ds(affs_file, affs_dataset, mode='r')

    network_dir = os.path.join(experiment, setup, str(iteration), merge_function)

    logging.info(f"Reading fragments from {fragments_file}")
    fragments = daisy.open_ds(fragments_file, fragments_dataset, mode='r')

    client = pymongo.MongoClient(db_host)
    db = client[db_name]

    blocks_agglomerated = 'blocks_agglomerated_' + merge_function

    if blocks_agglomerated not in db.list_collection_names():
        blocks_agglomerated = db[blocks_agglomerated]
        blocks_agglomerated.create_index(
                [('block_id', pymongo.ASCENDING)],
                name='block_id')
    else:
        blocks_agglomerated = db[blocks_agglomerated]

    context = daisy.Coordinate(context)
    total_roi = affs.roi.grow(context, context)

    read_roi = daisy.Roi((0,)*affs.roi.dims(), block_size).grow(context, context)
    write_roi = daisy.Roi((0,)*affs.roi.dims(), block_size)

    daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        process_function=lambda: start_worker(
            affs_file,
            affs_dataset,
            fragments_file,
            fragments_dataset,
            db_host,
            db_name,
            queue,
            merge_function,
            network_dir),
        check_function=lambda b: check_block(
            blocks_agglomerated,
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
        queue,
        merge_function,
        network_dir,
        **kwargs):

    worker_id = daisy.Context.from_env().worker_id

    logging.info(f"worker {worker_id} started...")

    output_dir = os.path.join('.agglomerate_blockwise', network_dir)

    os.makedirs(output_dir, exist_ok=True)

    log_out = os.path.join(output_dir, 'agglomerate_blockwise_%d.out' %worker_id)
    log_err = os.path.join(output_dir, 'agglomerate_blockwise_%d.err' %worker_id)

    config = {
            'affs_file': affs_file,
            'affs_dataset': affs_dataset,
            'fragments_file': fragments_file,
            'fragments_dataset': fragments_dataset,
            'db_host': db_host,
            'db_name': db_name,
            'queue': queue,
            'merge_function': merge_function
        }

    config_str = ''.join(['%s'%(v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    config_file = os.path.join(output_dir, '%d.config'%config_hash)

    with open(config_file, 'w') as f:
        json.dump(config, f)

    logging.info('Running block with config %s...'%config_file)

    worker = 'workers/agglomerate_worker.py'

    worker_command = os.path.join('.', worker)

    base_command = [
        'bsub',
        '-n', '1',
        '-o', f'{log_out}',
        f'python {worker_command} {config_file} > {log_out}'
    ]

    logging.info(f'Base command: {base_command}')

    daisy.call(base_command, log_out=log_out, log_err=log_err)

def check_block(blocks_agglomerated, block):

    done = blocks_agglomerated.count({'block_id': block.block_id}) >= 1

    return done

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()

    agglomerate(**config)

    end = time.time()

    seconds = end - start
    logging.info(f'Total time to agglomerate: {seconds}')
