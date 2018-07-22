import numpy as np

def replace_values(array, old_values, new_values):

    values_map = np.arange(old_values.max() + 1, dtype=new_values.dtype)
    values_map[old_values] = new_values

    return values_map[array]

def relabel(array, return_backwards_map=False):
    '''Relabel array, such that IDs are consecutive. Excludes 0.'''

    # get all labels except 0
    old_labels = np.unique(array)
    old_labels = old_labels[old_labels != 0]

    # shift previous labels to save some memory in replace_values
    offset = old_labels.min()
    shifted_array = array - offset
    shifted_old_labels = old_labels - offset

    n = len(old_labels) + 1
    new_labels = np.arange(1, n, dtype=array.dtype)

    replaced = replace_values(shifted_array, shifted_old_labels, new_labels)

    if return_backwards_map:

        backwards_map = np.insert(old_labels, 0, 0)
        return replaced, n, backwards_map

    return replaced, n
