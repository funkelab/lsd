import skimage

class Rag(skimage.future.graph.RAG):

    def __init__(self, fragments=None, voxel_size=None, connectivity=2):

        super(Rag, self).__init__(fragments, connectivity)
        self.__find_edge_centers(fragments)

    def __find_edge_centers(self, fragments):

        for u, v, data in self.edges_iter(data=True):
            data['location'] = (0, 0, 0)
            # TODO: find the real edge center
