import skimage

class Rag(skimage.future.graph.RAG):

    def __init__(self, fragments=None, voxel_size=None, connectivity=2):

        super(Rag, self).__init__(fragments, connectivity)
        self.__find_edge_centers(fragments)
        self.__add_esential_edge_attributes()

    def __find_edge_centers(self, fragments):

        for u, v, data in self.edges_iter(data=True):
            data['center_x'] = 0
            data['center_y'] = 0
            data['center_z'] = 0
            # TODO: find the real edge center

    def __add_esential_edge_attributes(self):

        for u, v, data in self.edges_iter(data=True):
            data['merged'] = 0
            data['agglomerated'] = 0
