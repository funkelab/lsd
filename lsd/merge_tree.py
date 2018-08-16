from networkx import DiGraph

class MergeTree(DiGraph):

    def __init__(self, leaf_nodes):

        super(MergeTree, self).__init__()

        for n in leaf_nodes:
            self.add_node(n, score=0, level=0)

        self.next_id = max(leaf_nodes) + 1
        self.id_to_node = { n: n for n in leaf_nodes }

    def merge(self, u, v, target, score):

        t = self.next_id
        self.next_id += 1

        level = max(
            self.node[self.id_to_node[u]]['level'],
            self.node[self.id_to_node[v]]['level']) + 1
        self.add_node(t, score=score, level=level)

        self.add_edge(self.id_to_node[u], t)
        self.add_edge(self.id_to_node[v], t)

        self.id_to_node[target] = t

    def find_merge(self, u, v):

        if not self.has_node(u) or not self.has_node(v):
            return None

        while True:

            if u == v:
                return self.node[u]['score']

            if self.node[u]['level'] > self.node[v]['level']:
                u, v = v, u

            parents = list(self.successors(u))

            if len(parents) == 0:
                return None
            assert len(parents) == 1

            u = parents[0]
