from scipy.ndimage.measurements import center_of_mass
from networkx import Graph, connected_components
import copy
import skimage

class Rag(skimage.future.graph.RAG):

    def __init__(self, fragments=None, voxel_size=None, connectivity=2):

        super(Rag, self).__init__(fragments, connectivity)

        if fragments is not None:
            self.__find_edge_centers(fragments)
            self.__add_esential_edge_attributes()

    def set_edge_attributes(self, key, value):
        '''Set all the attribute of all edges to the given value.'''

        for _u, _v, data in self.edges_iter(data=True):
            data[key] = value

    def get_connected_components(self):

        merge_graph = Graph()
        merge_graph.add_nodes_from(self.nodes())

        for u, v, data in self.edges_iter(data=True):
            if data['merged']:
                merge_graph.add_edge(u, v)

        components = connected_components(merge_graph)

        return [ list(component) for component in components ]

    def label_merged_edges(self, merged_rag):
        '''Set 'merged' to 1 for all edges that got merged in ``merged_rag``.

        ``merged_rag`` should be a RAG obtained from agglomeration of this RAG,
        where each node has an attribute 'labels' that stores a list of the
        original nodes that make up the merged node.'''

        for merged_node, data in merged_rag.nodes_iter(data=True):
            for node in data['labels']:
                self.node[node]['merged_node'] = merged_node

        for u, v, data in self.edges_iter(data=True):
            if self.node[u]['merged_node'] == self.node[v]['merged_node']:
                data['merged'] = 1

    def copy(self):
        '''Return a deep copy of this RAG.'''

        return copy.deepcopy(self)

    def __find_edge_centers(self, fragments):
        '''Get the center of an edge as the mean of the fragment centroids.'''

        print(self.nodes())
        fragment_centers = {
            fragment: center
            for fragment, center in zip(
                self.nodes(),
                center_of_mass(fragments, fragments, self.nodes()))
        }

        for u, v, data in self.edges_iter(data=True):

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

        for u, v, data in self.edges_iter(data=True):

            data['merged'] = 0
            data['agglomerated'] = 0
