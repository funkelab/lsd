Local Shape Descriptors (for Neuron Segmentation)
---

![](https://github.com/LocalShapeDescriptors/LocalShapeDescriptors.github.io/blob/master/assets/gifs/lsd_particles.gif)

This repository contains code to compute Local Shape Descriptors (LSDs) from an instance segmentation. LSDs can then be used during training as an auxiliary target, which we found to improve boundary prediction and therefore segmentation quality. Read more about it in our [paper](https://www.biorxiv.org/content/10.1101/2021.01.18.427039v2) and/or [blog post](https://localshapedescriptors.github.io/).

| Paper             |  Blog Post |
:-------------------------:|:-------------------------:
[![Paper](https://github.com/LocalShapeDescriptors/LocalShapeDescriptors.github.io/blob/master/assets/img/paper_image_resized.png)](https://www.biorxiv.org/content/10.1101/2021.01.18.427039v2)  |  [![Blog post](https://github.com/LocalShapeDescriptors/LocalShapeDescriptors.github.io/blob/master/assets/img/medium/lsds_header.jpeg)](https://localshapedescriptors.github.io/)

---

[Quick 2d Examples](#example)

[Notebooks](#nbook)

[Example networks & pipelines](#networks)

[Parallel processing](#parallel)

---

**Cite:**

```bibtex
@article{sheridan_local_2021,
	title = {Local Shape Descriptors for Neuron Segmentation},
	url = {https://www.biorxiv.org/content/10.1101/2021.01.18.427039v1},
	urldate = {2021-01-20},
	journal = {bioRxiv},
	author = {Sheridan, Arlo and Nguyen, Tri and Deb, Diptodip and Lee, Wei-Chung Allen and Saalfeld, Stephan and Turaga, Srinivas and Manor, Uri and Funke, Jan},
	year = {2021}
}
```

**Notes:**

* Tested on Ubuntu 18.04 with Python 3. 

* This is not production level software and was developed in a pure research environment. Therefore some scripts may not work out of the box. For example, all paper networks were originally written using now deprecated tensorflow/cudnn versions and rely on an outdated singularity container. Because of this, the singularity image will not build from the current recipe - if replicating with the current implementations, please reach out for the singularity container (it is too large to upload here). Alternatively, consider reimplementing networks in pytorch (recommended - see [Training](https://github.com/funkelab/lsd/edit/master/README.md#training)). 

* Post-proccesing steps were designed for use with a specific cluster and will need to be tweaked for individual use cases. If the need / use increases then we will look into refactoring, packaging and distributing.

* Currently, several post-processing scripts (e.g [watershed](https://github.com/funkelab/lsd/blob/master/lsd/post/fragments.py)) are located inside this repo which creates more dependencies than needed for using the lsds. One forseeable issue is that agglomeration requires networkx==2.2 for the MergeTree and boost is required for `funlib.segment`. We have restructured the repo to use `lsd.train` and `lsd.post` submodules. For just calculating the lsds, it is sufficient to use `lsd.train`, e.g:

```py
from lsd.train import local_shape_descriptor
```

---

<a name="example"></a>

## Quick 2d Examples

The following tutorial allows you to run in the browser using google colab. In order to replicate the tutorial locally, create a conda environment and install the relevant packages. E.g:

1) `conda create -n lsd_test python=3`
2) `conda activate lsd_test`
3) `pip install matplotlib scikit-image gunpowder`
4) `pip install git+https://github.com/funkelab/lsd.git`

tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/funkelab/lsd/blob/master/lsd/tutorial/notebooks/quick_tutorial.ipynb)

---

<a name="nbook"></a>

## Notebooks
 
* Examble colab notebooks are located [here](https://github.com/funkelab/lsd/tree/tutorial/lsd/tutorial/notebooks). You can download or run below (control + click open in colab). When running a notebook, you will probably get the message: "Warning: This notebook was not authored by Google". This can be ignored, you can run anyway.
 
* We uploaded ~1.7 tb of data (raw/labels/masks/rags etc.) to an s3 bucket. The following tutorial shows some examples for accessing and visualizing the data.
  
    * Data download: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/funkelab/lsd/blob/master/lsd/tutorial/notebooks/lsd_data_download.ipynb)
 
* If implementing the LSDs in your own training pipeline (i.e pure pytorch/tensorflow), calculate the LSDs on a label array of unique objects and use them as the target for your network (see quick 2d examples above for calculating). 

* The following tutorials show how to set up 2D training/prediction pipelines using [Gunpowder](http://funkey.science/gunpowder/). It is recommended to follow them in order (skip the basic tutorial if familiar with gunpowder). **Note:** Google Colab can sometimes be slow especially due to data I/O. These notebooks will run much faster in a jupyter notebook on a local gpu, but the Colab versions should provide a starting point.
 
    * Basic Gunpowder tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/funkelab/lsd/blob/master/lsd/tutorial/notebooks/basic_gp_tutorial.ipynb)

    * Train Affinities: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/funkelab/lsd/blob/master/lsd/tutorial/notebooks/train_affinities.ipynb)

    * Train LSDs: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/funkelab/lsd/blob/master/lsd/tutorial/notebooks/train_lsds.ipynb)

    * Train MTLSD: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/funkelab/lsd/blob/master/lsd/tutorial/notebooks/train_mtlsd.ipynb)

    * Inference (using pretrained MTLSD checkpoint): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/funkelab/lsd/blob/master/lsd/tutorial/notebooks/inference.ipynb)

    * Watershed, agglomeration, segmentation: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/funkelab/lsd/blob/master/lsd/tutorial/notebooks/segment.ipynb)

* Bonus notebooks:

    * Training using sparse ground truth (useful if you only have a subset of training data but still want dense predictions): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/funkelab/lsd/blob/master/lsd/tutorial/notebooks/train_lsds_sparse_labels.ipynb)

    * Ignore regions during training (useful if you want the network to learn to predict zeros in certain regions, eg glia ids): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/funkelab/lsd/blob/master/lsd/tutorial/notebooks/train_lsds_ignore_glia.ipynb)
    
    * Train lsds on non-em data with pytorch: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/funkelab/lsd/blob/master/lsd/tutorial/notebooks/train_lsds_non_em_pytorch.ipynb)

---

<a name="networks"></a>

## Example networks & pipelines
 
* There are some example networks and training/prediction pipelines from the fib25 dataset [here](https://github.com/funkelab/lsd/tree/tutorial/lsd/tutorial/example_nets/fib25).
 
### Training
 
* Since networks in this paper were implemented in Tensorflow, there was a two step process for training. First the networks were created using the `mknet.py` files. This saved tensor placeholders and meta data in config files that were then used for both training and prediction. The mknet files used the now deprecated mala repository to create the networks. If reimplementing in Tensorflow, consider migrating to [funlib.learn.tensorflow](https://github.com/funkelab/funlib.learn.tensorflow). 
 
* If using Pytorch, the networks can just be created directly inside the train scripts since placeholders aren't required. For example, the logic from this tensorflow [mknet script](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/example_nets/fib25/vanilla/mknet.py) and this tensorflow [train script](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/example_nets/fib25/vanilla/train.py) can be condensed to this pytorch [train script](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/example_nets/fib25/vanilla/train_pytorch.py).
 
* For training an autocontext network (e.g `acrlsd`), the current implementation learns the LSDs in a [first pass](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/example_nets/fib25/lsd/train.py). A saved checkpoint is then used when creating the [second pass](https://github.com/funkelab/lsd/blob/4397779ea4702eb3d593898d6240819e761fd41a/lsd/tutorial/example_nets/fib25/acrlsd/mknet.py#L122) in order to [predict LSDs](https://github.com/funkelab/lsd/blob/4397779ea4702eb3d593898d6240819e761fd41a/lsd/tutorial/example_nets/fib25/acrlsd/train.py#L158) prior to learning the Affinities. One could modify this to use a single setup and remove the need for writing the LSDs to disk.
  
### Inference
 
* By default, the predict scripts ([example](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/example_nets/fib25/mtlsd/predict.py)) contain the worker logic to be distributed by the scheduler during parallel processing (see below).
 
* If you just need to process a relatively small volume, it is sometimes not necessary to use blockwise processing. In this case, it is recommended to use a [scan node](http://funkey.science/gunpowder/api.html#scan), and specify input/output shapes + context. An example can be found in the inference colab notebook above.
 
* Similar to training, the current autocontext implementations assume the predicted LSDs are written to a zarr/n5 container and then used as input to the second pass to predict affinities. This can also be changed to predict on the fly if needed.
 
<details>
  <summary>Visualizations of example training/prediction pipelines</summary>
<br/><br/>
 
Vanilla affinities [training](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/example_nets/fib25/vanilla/train.py):
 
![](https://github.com/LocalShapeDescriptors/LocalShapeDescriptors.github.io/blob/master/assets/img/train_nodes.svg)
<br/><br/>
Autocontext [LSD](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/example_nets/fib25/lsd/predict.py) and [affinities](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/example_nets/fib25/acrlsd/predict.py) prediction: 

![](https://github.com/LocalShapeDescriptors/LocalShapeDescriptors.github.io/blob/master/assets/img/predict_nodes.svg)
 
</details>

---

<a name="parallel"></a>

## Parallel processing
  
* If you are running on small data then this section may be irrelevant. See the `Watershed, agglomeration, segmentation` notebook above if you just want to get a sense of obtaining a segmentation from affinities. 
 
* Example processing scripts can be found [here](https://github.com/funkelab/lsd/tree/tutorial/lsd/tutorial/scripts)

* We create segmentations following the approach in [this paper](https://ieeexplore.ieee.org/document/8364622). Generally speaking, after training a network there are five steps to obtain a segmentation:
 
1) Predict boundaries (this can involve the use of LSDs as an auxiliary task)
2) Generate supervoxels (fragments) using seeded watershed. The fragment centers of mass are stored as region adjacency graph nodes.
3) Generate edges between nodes using hierarchical agglomeration. The edges are weighted by the underlying affinities. Edges with lower scores are merged earlier.
4) Cut the graph at a predefined threshold and relabel connected components. Store the node - component lookup tables.
5) Use the lookup tables to relabel supervoxels and generate a segmentation.
 
![](https://github.com/LocalShapeDescriptors/LocalShapeDescriptors.github.io/blob/master/assets/img/pipeline.jpeg)
 
* Everything was done in parallel using daisy ([github](https://github.com/funkelab/daisy), [docs](https://daisy-docs.readthedocs.io/en/latest/index.html)), but one could use multiprocessing or dask instead.
 
* For our experiments we used [MongoDB](https://www.mongodb.com/) for all storage (block checks, rags, scores, etc) due to the size of the data. Depending on use case, it might be better to read/write to file rather than mongo. See watershed for further info.
 
* The following examples were written for use with the Janelia LSF cluster and are just meant to be used as a guide. Users will likely need to customize for their own specs (for example if using a SLURM cluster). 
 
* Need to install [funlib.segment](https://github.com/funkelab/funlib.segment) and [funlib.evaluate](https://github.com/funkelab/funlib.evaluate) if using/adapting segmentation/evaluation scripts.
  
### Inference
 
The worker logic is located in individual `predict.py` scripts ([example](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/example_nets/fib25/vanilla/predict.py)). The [master script](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/scripts/01_predict_blockwise.py) distributes using `daisy.run_blockwise`. The only need for MongoDb here is for the block check function (to check which blocks have successfully completed). To remove the need for mongo, one could remove the check function (remember to also remove `block_done_callback` in `predict.py`) or replace with custom function (e.g check chunk completion directly in output container).

<details>
 <summary>Example roi config</summary>

```json
{
  "container": "hemi_roi_1.zarr",
  "offset": [140800, 205120, 198400],
  "size": [3000, 3000, 3000]
}
```
</details>

<details>
 <summary>Example predict config</summary>

```json
 {
  "base_dir": "/path/to/base/directory",
  "experiment": "hemi",
  "setup": "setup01",
  "iteration": 400000,
  "raw_file": "predict_roi.json",
  "raw_dataset" : "volumes/raw",
  "out_base" : "output",
  "file_name": "foo.zarr",
  "num_workers": 5,
  "db_host": "mongodb client",
  "db_name": "foo",
  "queue": "gpu_rtx",
  "singularity_image": "/path/to/singularity/image"
}
```
</details>
 
### Watershed
 
The worker logic is located in a single [script](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/scripts/workers/extract_fragments_worker.py) which is then distributed by the [master script](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/scripts/02_extract_fragments_blockwise.py). By default the nodes are stored in mongo using a [MongoDbGraphProvider](https://github.com/funkelab/daisy/blob/master/daisy/persistence/mongodb_graph_provider.py). To write to file (i.e compressed numpy arrays), you can use the [FileGraphProvider](https://github.com/funkelab/daisy/blob/master/daisy/persistence/file_graph_provider.py) instead (inside the worker script).

<details>
 <summary>Example watershed config</summary>

```json
{
  "experiment": "hemi",
  "setup": "setup01",
  "iteration": 400000,
  "affs_file": "foo.zarr",
  "affs_dataset": "/volumes/affs",
  "fragments_file": "foo.zarr",
  "fragments_dataset": "/volumes/fragments",
  "block_size": [1000, 1000, 1000],
  "context": [248, 248, 248],
  "db_host": "mongodb client",
  "db_name": "foo",
  "num_workers": 6,
  "fragments_in_xy": false,
  "epsilon_agglomerate": 0,
  "queue": "local"
}
```
</details>
 
### Agglomerate
 
Same as watershed. [Worker script](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/scripts/workers/agglomerate_worker.py), [master script](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/scripts/03_agglomerate_blockwise.py). Change to FileGraphProvider if needed.
 
<details>
 <summary>Example agglomerate config</summary>

```json
{
  "experiment": "hemi",
  "setup": "setup01",
  "iteration": 400000,
  "affs_file": "foo.zarr",
  "affs_dataset": "/volumes/affs",
  "fragments_file": "foo.zarr",
  "fragments_dataset": "/volumes/fragments",
  "block_size": [1000, 1000, 1000],
  "context": [248, 248, 248],
  "db_host": "mongodb client",
  "db_name": "foo",
  "num_workers": 4,
  "queue": "local",
  "merge_function": "hist_quant_75"
}
```
</details>
 
### Find segments
 
In contrast to the above three methods, when [creating LUTs](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/scripts/04_find_segments.py) there just needs to be enough RAM to hold the RAG in memory. The only thing done in parallel is reading the graph (`graph_provider.read_blockwise()`). It could be adapted to use multiprocessing/dask for distributing the connected components for each threshold, but if the rag is too large there will be pickling errors when passing the nodes/edges. Daisy doesn't need to be used for scheduling here since nothing is written to containers. 

<details>
 <summary>Example find segments config</summary>
 
```json
{
  "db_host": "mongodb client",
  "db_name": "foo",
  "fragments_file": "foo.zarr",
  "edges_collection": "edges_hist_quant_75",
  "thresholds_minmax": [0, 1],
  "thresholds_step": 0.02,
  "block_size": [1000, 1000, 1000],
  "num_workers": 5,
  "fragments_dataset": "/volumes/fragments",
  "run_type": "test"
}
```
</details>
 
### Extract segmentation
 
This [script](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/scripts/05_extract_segmentation_from_lut.py) does use daisy to write the segmentation to file, but doesn't necessarily require bsub/sbatch to distribute (you can run locally). 

<details>
 <summary>Example extract segmentation config</summary> 
 
```json
{
  "fragments_file": "foo.zarr",
  "fragments_dataset": "/volumes/fragments",
  "edges_collection": "edges_hist_quant_75",
  "threshold": 0.4,
  "block_size": [1000, 1000, 1000],
  "out_file": "foo.zarr",
  "out_dataset": "volumes/segmentation_40",
  "num_workers": 3,
  "run_type": "test"
}
```
</details>
 
### Evaluate volumes
 
[Evaluate](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/scripts/05_evaluate_volumes.py) Voi scores. Assumes dense voxel ground truth (not skeletons). This also assumes the ground truth (and segmentation) can fit into memory, which was fine for hemi and fib25 volumes assuming ~750 GB of RAM. The script should probably be refactored to run blockwise.
 
<details>
 <summary>Example evaluate volumes config</summary>

```json
{
  "experiment": "hemi",
  "setup": "setup01",
  "iteration": 400000,
  "gt_file": "hemi_roi_1.zarr",
  "gt_dataset": "volumes/labels/neuron_ids",
  "fragments_file": "foo.zarr",
  "fragments_dataset": "/volumes/fragments",
  "db_host": "mongodb client",
  "rag_db_name": "foo",
  "edges_collection": "edges_hist_quant_75",
  "scores_db_name": "scores",
  "thresholds_minmax": [0, 1],
  "thresholds_step": 0.02,
  "num_workers": 4,
  "method": "vanilla",
  "run_type": "test"
}
```
</details>
 
### Evaluate annotations

For the zebrafinch, ground truth skeletons were used due to the size of the dataset. These skeletons were cropped, masked, and relabelled for the sub Rois that were tested in the paper. We [evaluated](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/scripts/05_evaluate_annotations.py) voi, erl, and the mincut metric on the consolidated skeletons. The current implementation could be refactored / made more modular. It also uses `node_collections` which are now deprecated in daisy. To use with the current implementation, you should checkout daisy commit `39723ca`.

 
<details>
 <summary>Example evaluate annotations config</summary>
 
```json
{
  "experiment": "zebrafinch",
  "setup": "setup01",
  "iteration": 400000,
  "config_slab": "mtlsd",
  "fragments_file": "foo.zarr",
  "fragments_dataset": "/volumes/fragments",
  "edges_db_host": "mongodb client",
  "edges_db_name": "foo",
  "edges_collection": "edges_hist_quant_75",
  "scores_db_name": "scores",
  "annotations_db_host": "mongo client",
  "annotations_db_name": "foo",
  "annotations_skeletons_collection_name": "zebrafinch",
  "node_components": "zebrafinch_components",
  "node_mask": "zebrafinch_mask",
  "roi_offset": [50800, 43200, 44100],
  "roi_shape": [10800, 10800, 10800],
  "thresholds_minmax": [0.5, 1],
  "thresholds_step": 1,
  "run_type": "11_micron_roi_masked"
}
```
 </details>
