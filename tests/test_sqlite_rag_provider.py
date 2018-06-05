from lsd.persistence import SqliteRagProvider
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logging.getLogger('lsd.persistence.sqlite_rag_provider').setLevel(logging.DEBUG)

if __name__ == "__main__":

    fragments = np.ones((10, 10, 10), dtype=np.uint64)
    fragments[0:5] = 2

    rag_provider = SqliteRagProvider.from_fragments(fragments, 'test_sqlite_rag_provider.db')

    sub_rag = rag_provider[:,5:8,5:]
    print(sub_rag.edges())

    sub_rag = rag_provider[:,:,:]

    print(sub_rag.edges(data=True)[0])
    sub_rag.edges(data=True)[0][2]['merged'] = 1
    sub_rag.sync()

    sub_rag = rag_provider[:,:,:]
    print(sub_rag.edges(data=True)[0])
