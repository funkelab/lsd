from __future__ import absolute_import
from ..shared_rag_provider import SharedRagProvider, SubRag
from networkx.convert import to_dict_of_dicts
from daisy import Coordinate
import sqlite3
import logging

logger = logging.getLogger(__name__)

class SqliteSubRag(SubRag):

    def __init__(self, filename, read_only, sync_edge_attributes):

        super(SubRag, self).__init__()

        self.filename = filename
        self.read_only = read_only
        self._sync_edge_attributes = sync_edge_attributes

    def _contains(self, roi, edge):

        u, v = edge
        min_node = self.node[u]

        # Some nodes are outside of the originally requested ROI (they have
        # been pulled in by edges leaving the ROI). These nodes have no
        # attributes, so we can't perform an inclusion test. However, we
        # know they are outside of the sub-RAG ROI, and therefore also
        # outside of 'roi', whatever it is.
        if 'center_z' not in min_node:
            return False

        min_node_center = Coordinate((
            min_node['center_z'],
            min_node['center_y'],
            min_node['center_x']))

        return roi.contains(min_node_center)

    def sync_edges(self, roi):

        if self.read_only:
            raise RuntimeError("Trying to write to read-only DB")

        logger.debug("Writing edges in %s", roi)

        connection = sqlite3.connect(self.filename, timeout=300.0)
        c = connection.cursor()

        for u, v, data in self.edges(data=True):

            u, v = min(u, v), max(u, v)
            if not self._contains(roi, (u, v)):
                continue

            values = {
                'u': u,
                'v': v,
            }
            values.update(data)

            names = ', '.join(SqliteRagProvider.edge_attributes)
            values = ', '.join([
                str(values[key]) if values[key] is not None else 'null'
                for key in SqliteRagProvider.edge_attributes
            ])

            query = 'INSERT INTO edges (%s) VALUES (%s)'%(names, values)
            logger.debug(query)
            c.execute(query)

        connection.commit()
        connection.close()

    def sync_nodes(self):

        if self.read_only:
            raise RuntimeError("Trying to write to read-only DB")

        logger.debug("Writing all nodes")

        connection = sqlite3.connect(self.filename, timeout=300.0)
        c = connection.cursor()

        for node, data in self.nodes(data=True):

            values = {
                'id': node
            }
            values.update(data)

            names = ', '.join(SqliteRagProvider.node_attributes)
            values = ', '.join([
                str(values[key]) if values[key] is not None else 'null'
                for key in SqliteRagProvider.node_attributes
            ])

            query = 'INSERT INTO nodes (%s) VALUES (%s)'%(names, values)
            logger.debug(query)
            c.execute(query)

        connection.commit()
        connection.close()

class SqliteRagProvider(SharedRagProvider):
    '''A shared region adjacency graph stored in an SQLite file.
    '''

    # all node_attributes
    node_attributes = [
        'id',
        'center_x', 'center_y', 'center_z'
    ]

    # all edge attributes
    edge_attributes = [
        'u', 'v',
        'merge_score',
        'agglomerated'
    ]

    # edge atttributes that should be written back by
    # SubRag.sync_edge_attributes()
    sync_edge_attributes = ['merge_score', 'agglomerated']

    # SQL datatypes for each node attribute
    node_attribute_dtypes = {
        'id': 'bigint',
        'center_x': 'real',
        'center_y': 'real',
        'center_z': 'real'
    }

    # SQL datatypes for each edge attribute
    edge_attribute_dtypes = {
        'u': 'bigint',
        'v': 'bigint',
        'merge_score': 'real',
        'agglomerated': 'boolean'
    }

    @staticmethod
    def from_fragments(fragments, filename, connectivity=2):
        '''Create a SqliteRagProvider, populated with a RAG extracted from a
        label image.'''

        rag = SqliteRagProvider(filename, 'w')
        rag.__write_rag(SubRag(fragments, connectivity=connectivity))

        return rag

    def __init__(self, filename, mode):

        self.filename = filename
        self.read_only = mode == 'r'

        connection = sqlite3.connect(self.filename)
        c = connection.cursor()

        if mode == 'w':

            # start with a fresh DB
            try:
                c.execute('''
                    DROP TABLE edges
                ''')
            except sqlite3.OperationalError:
                # edges did not exist
                pass

            try:
                c.execute('''
                    DROP TABLE nodes
                ''')
            except sqlite3.OperationalError:
                # nodes did not exist
                pass

        # make sure required tables are present
        try:

            attributes = ', '.join([
                '%s %s'%(name, self.node_attribute_dtypes[name])
                for name in self.node_attributes
            ])
            c.execute('CREATE TABLE nodes (%s)'%attributes)

        except sqlite3.OperationalError:
            # table did already exist
            pass

        try:

            attributes = ', '.join([
                '%s %s'%(name, self.edge_attribute_dtypes[name])
                for name in self.edge_attributes
            ])
            c.execute('CREATE TABLE edges (%s)'%attributes)

        except sqlite3.OperationalError:
            # table did already exist
            pass

        connection.commit()
        connection.close()

    def __slice_condition(self, value, start, stop):

        if start is None and stop is None:
            return None

        if start is None:
            return '%s < %d'%(value, stop)

        if stop is None:
            return '%s >= %d'%(value, start)

        return '%s >= %d AND %s < %d'%(value, start, value, stop)

    def __getitem__(self, roi):

        assert roi.dims() == 3, "Sorry, SQLite backend does only 3D"

        slice_conditions = [
            self.__slice_condition('center_%s'%dim, s.start, s.stop)
            for dim, s in zip(['z', 'y', 'x'], roi.to_slices())
            if s.start is not None or s.stop is not None
        ]
        if len(slice_conditions) > 0:
            contains_condition = 'WHERE ' + ' AND '.join(slice_conditions)
        else:
            contains_condition = ''

        connection = sqlite3.connect(self.filename)
        c = connection.cursor()

        graph = SqliteSubRag(self.filename, self.read_only, self.sync_edge_attributes)

        node_query = '''
            SELECT * FROM nodes %s
        '''%contains_condition
        logger.debug(node_query)
        rows = c.execute(node_query)

        # convert rows into dictionary
        rows = [
            {
                name: row[i]
                for i, name in enumerate(self.node_attributes)
            }
            for row in rows
        ]

        logger.debug("found %d nodes", len(rows))

        # create a list of nodes and their attributes
        node_list = [
            (row['id'], self.__remove_keys(row, ['id']))
            for row in rows
        ]
        logger.debug("read nodes: %s", node_list)

        graph.add_nodes_from(node_list)

        nodes_condition = 'WHERE u in (%s)'%', '.join(
            '%d'%u for u in graph.nodes())
        edge_query = '''
            SELECT * FROM edges %s
        '''%nodes_condition
        logger.debug(edge_query)
        rows = c.execute(edge_query)

        # convert rows into dictionary
        rows = [
            {
                name: row[i]
                for i, name in enumerate(self.edge_attributes)
            }
            for row in rows
        ]
        connection.close()

        # create a list of edges and their attributes
        edge_list = [
            (row['u'], row['v'], self.__remove_keys(row, ['u', 'v']))
            for row in rows
        ]
        logger.debug("read edges: %s", edge_list)

        graph.add_edges_from(edge_list)

        return graph

    def __remove_keys(self, dictionary, keys):

        for key in keys:
            del dictionary[key]
        return dictionary

    def __write_rag(self, rag):
        '''Write a complete RAG. This replaces whatever was stored in the DB
        before.'''

        if self.read_only:
            raise RuntimeError("Trying to write to read-only DB")

        connection = sqlite3.connect(self.filename)
        c = connection.cursor()
        c.execute('DELETE FROM edges')

        for u, v, data in rag.edges(data=True):

            u, v = min(u, v), max(u, v)

            values = {
                'u': u,
                'v': v
            }
            values.update(data)

            names = ', '.join(self.edge_attributes)
            values = ', '.join([
                str(values[key]) if values[key] is not None else 'null'
                for key in self.edge_attributes
            ])
            query = 'INSERT INTO edges (%s) VALUES (%s)'%(names, values)
            logger.debug(query)
            c.execute(query)

        connection.commit()
        connection.close()
