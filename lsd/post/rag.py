from __future__ import absolute_import
from funlib.segment.arrays import replace_values
from networkx import Graph, connected_components
from scipy.ndimage.measurements import center_of_mass
import copy
import numpy as np
import skimage.future

class Rag(skimage.future.graph.RAG):
    '''A region adjacency graph (RAG) with the following attributes:

    Edge attributes:

        merge_score (float or None):

            The score at which a given edge was merged.

        agglomerated (int):

            Either 0 or 1, indicates whether an edge was processed by an
            agglomeration algorithm.

        center_{z,y,x} (float):

            A representative location of an edge. This is used (at least) by
            `class:SharedRagProvider<SharedRagProviders>` to query and write
            edges within a certain region of interest.

    Node attributes:

        labels (list of nodes):

            Stores for each node all nodes that were merged into this node.
            Agglomeration algorithms are expected to update this list as they
            modify the RAG (as scikit's ``merge_hierarchical`` does).

    Args:

        fragments (``ndarray``, optional):

            Creates a RAG from a label image. If not given, an empty RAG is
            created.

        connectivity (int, optional):

            The connectivity to consider for the RAG extraction from
            ``fragments``.
    '''

    def __init__(self, fragments=None, connectivity=2):

        super(Rag, self).__init__(fragments, connectivity)

        if fragments is not None:
            self.__find_edge_centers(fragments)
            self.__add_esential_edge_attributes()
            self.__add_esential_node_attributes()

    def set_edge_attributes(self, key, value):
        '''Set all the attribute of all edges to the given value.'''

        for _u, _v, data in self.edges(data=True):
            data[key] = value

    def get_connected_components(self, threshold):
        '''Get all connected components in the RAG, as indicated by the
        'merge_score' attribute of edges.'''

        merge_graph = Graph()
        merge_graph.add_nodes_from(self.nodes())

        for u, v, data in self.edges(data=True):
            if data['merge_score'] is not None and data['merge_score'] <= threshold:
                merge_graph.add_edge(u, v)

        components = connected_components(merge_graph)

        return [ list(component) for component in components ]

    def contract_merged_nodes(self, threshold, fragments=None):
        '''Contract this RAG by merging all edges under the given threshold.

        This will create new edges that will have only ``merge_score`` and
        ``agglomerated`` attributes, set to 0. Other edge attributes will be
        lost.

        Args:

            threshold (``float``):

                The threshold under which to consider edges as merged.

            fragments (``ndarray``, optional):

                If given, also updates the labels in ``fragments`` according to
                the merges performed.
        '''

        # get currently connected componets
        components = self.get_connected_components(threshold)

        # replace each connected component by a single node
        component_nodes = self.__contract_nodes(components)

        if fragments is not None:

            # relabel fragments of the same connected components to match merged RAG
            self.__relabel(fragments, components, component_nodes)

    def get_segmentation(self, threshold, fragments):
        '''Get the segmentation of this RAG by merging all edges under the
        given threshold.

        Args:

            threshold (``float``):

                The threshold under which to consider edges as merged.

            fragments (``ndarray``):

                If given, also updates the labels in ``fragments`` according to
                the merges performed.
        '''

        # get currently connected componets
        components = self.get_connected_components(threshold)

        segments = list(range(1, len(components) + 1))

        # relabel fragments of the same connected components to match merged RAG
        self.__relabel(fragments, components, segments)

    def __find_edge_centers(self, fragments):
        '''Get the center of an edge as the mean of the fragment centroids.'''

        print(self.nodes())
        fragment_centers = {
            fragment: center
            for fragment, center in zip(
                self.nodes(),
                center_of_mass(fragments, fragments, self.nodes()))
        }

        for u, v, data in self.edges(data=True):

            center_u = fragment_centers[u]
            center_v = fragment_centers[v]

            center_edge = tuple(
                (cu + cv)/2
                for cu, cv in zip(center_u, center_v)
            )

            data['center_z'] = center_edge[0]
            data['center_y'] = center_edge[1]
            data['center_x'] = center_edge[2]

    def __add_esential_edge_attributes(self):

        for u, v, data in self.edges(data=True):

            if 'merge_score' not in data:
                data['merge_score'] = None

            if 'agglomeration' not in data:
                data['agglomerated'] = 0

    def __add_esential_node_attributes(self):

        for node, data in self.nodes(data=True):

            if 'labels' not in data:
                data['labels'] = [node]

    def __contract_nodes(self, components):
        '''Contract all nodes of one component into a single node, return the
        single node for each component.

        This will create new edges that will have only ``merge_score`` and
        ``agglomerated`` attributes, set to 0.
        '''

        self.__add_esential_node_attributes()

        component_nodes = []

        for component in components:

            for i in range(1, len(component)):
                self.merge_nodes(
                    component[i - 1],
                    component[i],
                    # set default attributes for new edges
                    weight_func=lambda _, _src, _dst, _n: {
                        'merge_score': None,
                        'agglomerated': 0
                    })

            component_nodes.append(component[-1])

        return component_nodes

    def __relabel(self, array, components, component_labels):

        old_values = []
        new_values = []

        for component, label in zip(components, component_labels):
            for c in component:
                old_values.append(c)
                new_values.append(label)

        array[:] = replace_values(array, old_values, new_values)
