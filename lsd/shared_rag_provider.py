from __future__ import absolute_import
from .rag import Rag

class SharedRagProvider(object):
    '''Interface for shared region adjacency graph providers that supports
    slicing to retrieve sub-RAGs.

    Implementations should support the following interactions::

        # provider is a SharedRagProvider

        # slicing notation to extract a sub-RAG
        sub_rag = provider[0:10, 0:10, 0:10]

        # sub_rag should inherit from SubRag

        # write nodes
        sub_rag.sync_nodes()

        # write edges
        sub_rag.sync_edges()
    '''

    def __getitem__(self, roi):
        raise RuntimeError("not implemented in %s"%self.name())

    def name(self):
        return type(self).__name__

class SubRag(Rag):

    def sync_edges(self, roi):
        '''Write edges and their attributes. Restrict the sync to the given
        ROI.'''
        raise RuntimeError("not implemented in %s"%self.name())

    def sync_nodes(self):
        '''Write nodes and their attributes.'''
        raise RuntimeError("not implemented in %s"%self.name())

    def name(self):
        return type(self).__name__
