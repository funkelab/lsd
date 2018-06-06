from scipy.ndimage.measurements import center_of_mass
from networkx import Graph, connected_components
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

        return connected_components(merge_graph)

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
