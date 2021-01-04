from networkx import DiGraph

cimport cython
from libc.stdint cimport uint16_t, uint32_t
from libc.string cimport memset
from cpython.mem cimport PyMem_Malloc, PyMem_Free


cdef packed struct CythonNode:
    uint16_t level
    uint32_t next
assert sizeof(CythonNode) == 6
# 6 bytes per node, so a processor with 30MB of L3 cache will run
# efficiently for graphs up to 5 million nodes

cdef class MergeTreeDbCython():
    '''Class for merge tree database
    We need this class to wrap around the memory allocation/dealloc, note
    the use of __cinit__ and __dealloc__ that tells Python how to 
    deallocate memory when it garbage clean the object
    '''

    cdef CythonNode* root

    def __cinit__(self, size_t n_nodes):
        self.root = <CythonNode *> PyMem_Malloc(n_nodes * sizeof(CythonNode))
        memset(self.root, 0, n_nodes * sizeof(CythonNode))

    def __dealloc__(self):
        PyMem_Free(self.root)

    def add_node(self, uint32_t node, uint32_t next, uint32_t level):
        self.root[node].next = next
        assert level <= 65535  # max value for uint16_t
        self.root[node].level = level

    def find_merge(self, uint32_t u, uint32_t v):

        while True:

            if u == v:
                break

            if self.root[u].level > self.root[v].level:
                u, v = v, u

            if self.root[u].next == 0:
                return None

            u = self.root[u].next

        return u

cdef packed struct CythonNodeLong:
    uint32_t level
    uint32_t next
assert sizeof(CythonNodeLong) == 8
# 6 bytes per node, so a processor with 30MB of L3 cache will run
# efficiently for graphs up to 5 million nodes

cdef class MergeTreeDbCythonLong():
    '''Class for merge tree database
    We need this class to wrap around the memory allocation/dealloc, note
    the use of __cinit__ and __dealloc__ that tells Python how to 
    deallocate memory when it garbage clean the object
    '''

    cdef CythonNodeLong* root

    def __cinit__(self, size_t n_nodes):
        self.root = <CythonNodeLong *> PyMem_Malloc(n_nodes * sizeof(CythonNodeLong))
        memset(self.root, 0, n_nodes * sizeof(CythonNodeLong))

    def __dealloc__(self):
        PyMem_Free(self.root)

    def add_node(self, uint32_t node, uint32_t next, uint32_t level):
        self.root[node].next = next
        self.root[node].level = level

    def find_merge(self, uint32_t u, uint32_t v):

        while True:

            if u == v:
                break

            if self.root[u].level > self.root[v].level:
                u, v = v, u

            if self.root[u].next == 0:
                return None

            u = self.root[u].next

        return u


class MergeTree(DiGraph):

    def __init__(self, leaf_nodes=None):

        super(MergeTree, self).__init__()
        self.from_cython_db_id = {}
        self.to_cython_db_id = {}
        self.next_cython_id = 1

        if leaf_nodes is not None:
            for n in leaf_nodes:
                self.add_node(n, score=0, level=0)
                self.to_cython_db_id[n] = self.next_cython_id
                self.from_cython_db_id[self.next_cython_id] = n
                self.next_cython_id += 1

            self.next_id = max(leaf_nodes) + 1
            self.id_to_node = {n: n for n in leaf_nodes}

        self.max_level = 0
        self.cython_db = None

    def merge(self, u, v, target, score):

        t = self.next_id
        self.next_id += 1

        level = max(
            self.nodes[self.id_to_node[u]]['level'],
            self.nodes[self.id_to_node[v]]['level']) + 1
        self.max_level = max(self.max_level, level)
        self.add_node(t, score=score, level=level)

        self.add_edge(self.id_to_node[u], t)
        self.add_edge(self.id_to_node[v], t)

        self.id_to_node[target] = t

        self.to_cython_db_id[t] = self.next_cython_id
        self.from_cython_db_id[self.next_cython_id] = t
        self.next_cython_id += 1

    def malloc_cython_db(self):
        '''Construct optimized DB by only parsing node ID and node level'''

        n_nodes = self.next_cython_id
        self.n_cython_nodes = self.next_cython_id
        n_nodes = self.n_cython_nodes
        print("Num nodes in merge tree: %d" % n_nodes)
        print("max_level: %d" % self.max_level)

        if self.max_level < 65536:
            self.cython_db = MergeTreeDbCython(n_nodes)
        else:
            self.cython_db = MergeTreeDbCythonLong(n_nodes)

        for node, data in self.nodes(data=True):

            parents = list(self.successors(node))
            if len(parents) == 0:
                next_node = 0
            else:
                assert len(parents) == 1
                next_node = parents[0]
                assert next_node in self.to_cython_db_id
                next_node = self.to_cython_db_id[next_node]

            level = data['level']

            cython_nodeid = self.to_cython_db_id[node]
            assert cython_nodeid < self.n_cython_nodes
            assert next_node < self.n_cython_nodes
            self.cython_db.add_node(<int>cython_nodeid, next_node, level)

        print("Finished allocating db")

    def find_merge(self, u_p, v_p, check=False):

        if self.cython_db is None:
            # allocate cython db on first run of `find_merge`
            self.malloc_cython_db()
            assert self.cython_db is not None

        if not self.has_node(u_p) or not self.has_node(v_p):
            return None

        u = self.cython_db.find_merge(
                self.to_cython_db_id[u_p],
                self.to_cython_db_id[v_p])

        if u is None:
            return None

        u = self.from_cython_db_id[u]
        if check:
            assert u == self.find_merge_check(u_p, v_p)
        return self.nodes[u]['score']

    def find_merge_check(self, u, v):
        '''Check results using old algorithm without Cython'''

        if not self.has_node(u) or not self.has_node(v):
            return None

        while True:

            if u == v:
                return self.nodes[u]['score']

            if self.nodes[u]['level'] > self.nodes[v]['level']:
                u, v = v, u

            parents = list(self.successors(u))

            if len(parents) == 0:
                return None
            assert len(parents) == 1

            u = parents[0]
