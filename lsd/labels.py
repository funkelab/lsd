import numpy as np

def replace_values(array, old_values, new_values):

    values_map = np.arange(old_values.max() + 1, dtype=new_values.dtype)
    values_map[old_values] = new_values

    return values_map[array]

def relabel(array):
    '''Relabel array, such that IDs are consecutive.'''
    old_labels = np.unique(array)
    n = len(old_labels) + 1
    new_labels = np.arange(1, n, dtype=array.dtype)

    return replace_values(array, old_labels, new_labels), n
