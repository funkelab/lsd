import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import *

def _2d(array, f):

    shape = array.shape
    stride = array.strides

    view = as_strided(
            array,
            (shape[0], shape[1], f, shape[2], f),
            (stride[0], stride[1], 0, stride[2], 0))

    return view.reshape(shape[0], shape[1]*f, shape[2]*f)

def _3d(array, f):

    shape = array.shape
    stride = array.strides

    view = as_strided(
            array,
            (shape[0], shape[1], f, shape[2], f, shape[3], f),
            (stride[0], stride[1], 0, stride[2], 0, stride[3], 0))

    return view.reshape(shape[0], shape[1]*f, shape[2]*f, shape[3]*f)

def _2d_3d(array, f):

    shape = array.shape
    stride = array.strides

    if len(array.shape) == 4:
        sh = (shape[0], shape[1], f, shape[2], f, shape[3], f)
        st = (stride[0], stride[1], 0, stride[2], 0, stride[3], 0)
    else:
        sh = (shape[0], shape[1], f, shape[2], f)
        st = (stride[0], stride[1], 0, stride[2], 0)

    view = as_strided(array,sh,st)

    l = [shape[0]]
    [l.append(shape[i+1]*f) for i,j in enumerate(shape[1:])]

    return view.reshape(l)

def test_strided():

    a_2d = np.array([[[1,1,1],[1,1,1],[1,1,1]]])
    a_3d = np.array([[[[1,1,1],[1,1,1],[1,1,1]]]])

    for f in range(10):

        assert_array_equal(_2d_3d(a_2d, f), _2d(a_2d, f))
        assert_array_equal(_2d_3d(a_3d, f), _3d(a_3d, f))
