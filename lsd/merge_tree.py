from networkx import DiGraph

class MergeTree(DiGraph):

    def __init__(self, leaf_nodes):

        super(MergeTree, self).__init__()

        for n in leaf_nodes:
            self.add_node(n, score=0)

        self.next_id = max(leaf_nodes) + 1
        self.id_to_node = { n: n for n in leaf_nodes }

    def merge(self, u, v, target, score):

        t = self.next_id
        self.next_id += 1

        self.add_node(t, score=score)

        self.add_edge(self.id_to_node[u], t)
        self.add_edge(self.id_to_node[v], t)

        self.id_to_node[target] = t

    def find_merge(self, u, v):

        if not self.has_node(u) or not self.has_node(v):
            return None

        while True:

            if u == v:
                return self.node[u]['score']

            if self.node[u]['score'] > self.node[v]['score']:
                u, v = v, u

            parents = list(self.successors(u))

            if len(parents) == 0:
                return None

            u = parents[0]
