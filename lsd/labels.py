from __future__ import absolute_import
from .replace_values import replace_values as cpp_replace_values
import numpy as np

def replace_values(array, old_values, new_values):

    max_old_value = old_values.max()

    if max_old_value < 1024**3:

        values_map = np.arange(old_values.max() + 1, dtype=new_values.dtype)
        values_map[old_values] = new_values

        return values_map[array]

    else:

        values_map = {
            old_value: new_value
            for old_value, new_value in zip(old_values, new_values)
        }

        return cpp_replace_values(array, values_map)

def relabel(array, return_backwards_map=False):
    '''Relabel array, such that IDs are consecutive. Excludes 0.'''

    # get all labels except 0
    old_labels = np.unique(array)
    old_labels = old_labels[old_labels != 0]

    if old_labels.size == 0:

        if return_backwards_map:
            return array, 1, [0]
        else:
            return array, 1

    # shift previous labels to save some memory in replace_values
    min_label = old_labels.min()
    assert min_label > 0
    offset = old_labels.dtype.type(min_label - 1)
    shifted_array = array.copy()
    shifted_array[shifted_array != 0] -= offset
    shifted_old_labels = old_labels - offset

    n = len(old_labels) + 1
    new_labels = np.arange(1, n, dtype=array.dtype)

    replaced = replace_values(shifted_array, shifted_old_labels, new_labels)

    if return_backwards_map:

        backwards_map = np.insert(old_labels, 0, 0)
        return replaced, n, backwards_map

    return replaced, n
