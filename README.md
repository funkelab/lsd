Local Shape Descriptors (for Neuron Segmentation)
=================================================

![](https://localshapedescriptors.github.io/assets/img/3d_mesh_vect.jpeg)

This repository contains code to compute Local Shape Descriptors (LSDs) from an instance segmentation. Those LSDs can then be used during training as an auxiliary target, which we found to improve boundary prediction and therefore segmentation quality. Read more about it in our [paper](https://www.biorxiv.org/content/10.1101/2021.01.18.427039v1).

Here you find:
  * LSD calculation from `numpy` arrays: `lsd/local_shape_descriptor.py`
  * LSD [`gunpowder`](http://funkey.science/gunpowder) node: `lsd/gp/add_local_shape_descriptor.py`

Soon (ETA February 1st), you will also find here:

* links to all training and testing data, as well as results
* scripts, examples, and tutorials for parallel inference and post-processing
