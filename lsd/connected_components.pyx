from libc.stdint cimport uint64_t
from libcpp.map cimport map
import numpy as np
cimport numpy as np

def connected_components(
        np.ndarray[uint64_t, ndim=1] nodes,
        np.ndarray[uint64_t, ndim=2] edges,
        np.ndarray[float, ndim=1] scores,
        float threshold):

    cdef np.ndarray[uint64_t, ndim=1] components
    components = np.empty_like(nodes)

    num_nodes = nodes.shape[0]
    num_edges = edges.shape[0]
    assert components.shape[0] == nodes.shape[0], (
        "components array has different shape than nodes array")
    assert edges.shape[1] == 2, (
        "edges array does not have two columns")
    assert num_edges == len(scores), (
        "number of edges does not match number of scores")

    # the C++ part assumes contiguous memory, make sure we have it (and do 
    # nothing, if we do)
    if not nodes.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous nodes arrray (avoid this by passing C_CONTIGUOUS arrays)")
        nodes = np.ascontiguousarray(nodes)
    if not edges.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous edges arrray (avoid this by passing C_CONTIGUOUS arrays)")
        edges = np.ascontiguousarray(edges)
    if not scores.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous scores arrray (avoid this by passing C_CONTIGUOUS arrays)")
        scores = np.ascontiguousarray(scores)

    cdef uint64_t* nodes_data
    cdef uint64_t* edges_data
    cdef uint64_t* components_data
    cdef float* scores_data

    nodes_data = &nodes[0]
    edges_data = &edges[0, 0]
    scores_data = &scores[0]
    components_data = &components[0]

    find_connected_components(
        num_nodes,
        nodes_data,
        num_edges,
        edges_data,
        scores_data,
        threshold,
        components_data)

    return components

cdef extern from "find_connected_components.h":

    void find_connected_components(
            const uint64_t num_nodes,
            const uint64_t* nodes_data,
            const uint64_t num_edges,
            const uint64_t* edges_data,
            const float* scores_data,
            const float threshold,
            uint64_t* components)

