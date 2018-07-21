from rag import Rag

class SharedRagProvider(object):
    '''Interface for shared region adjacency graph providers that supports
    slicing to retrieve sub-RAGs.

    Implementations should support the following interactions::

        # provider is a SharedRagProvider

        # slicing notation to extract a sub-RAG
        sub_rag = provider[0:10, 0:10, 0:10]

        # sub_rag should inherit from SubRag

        # write changes made edge attributes
        sub_rag.sync_edge_attributes()
    '''

    def __getitem__(self, slices):
        raise RuntimeError("not implemented in %s"%self.name())

    def name(self):
        return type(self).__name__

class SubRag(Rag):

    def sync_edge_attributes(self, roi):
        '''Write back modifications made to edge attributes with the
        persistency backend associated to this sub-RAG. Restrict the sync to
        the given ROI.'''
        raise RuntimeError("not implemented in %s"%self.name())

    def name(self):
        return type(self).__name__
