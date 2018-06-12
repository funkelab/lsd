from lsd import SharedRagProvider, SubRag
from networkx.convert import to_dict_of_dicts
import sqlite3
import logging

logger = logging.getLogger(__name__)

class SqliteSubRag(SubRag):

    def __init__(self, filename, sync_edge_attributes):

        super(SubRag, self).__init__()

        self.filename = filename
        self.sync_edge_attributes = sync_edge_attributes

    def sync(self, roi):

        logger.info("Writing back edge attributes in %s", roi)

        connection = sqlite3.connect(self.filename, timeout=30.0)
        c = connection.cursor()

        for u, v, data in self.edges_iter(data=True):

            u, v = min(u, v), max(u, v)

            update = ', '.join([
                key + ' = ' + str(data[key])
                for key in self.sync_edge_attributes
            ])

            query = '''
                UPDATE edges
                SET %s
                WHERE u == %d AND v == %d
                AND center_z >= %d AND center_z < %d
                AND center_y >= %d AND center_y < %d
                AND center_x >= %d AND center_x < %d
            '''%(
                update,
                u, v,
                roi.get_begin()[0],
                roi.get_end()[0],
                roi.get_begin()[1],
                roi.get_end()[1],
                roi.get_begin()[2],
                roi.get_end()[2]
            )
            logger.debug(query)
            c.execute(query)

        connection.commit()
        connection.close()

class SqliteRagProvider(SharedRagProvider):
    '''A shared region adjacency graph stored in an SQLite file.
    '''

    # all edge attributes
    edge_attributes = [
        'u', 'v',
        'center_x', 'center_y', 'center_z',
        'merged',
        'agglomerated'
    ]

    # edge atttributes that should be written back by SubRag.sync()
    sync_edge_attributes = ['merged', 'agglomerated']

    # SQL datatypes for each edge attribute
    edge_attribute_dtypes = {
        'u': 'bigint',
        'v': 'bigint',
        'center_x': 'real',
        'center_y': 'real',
        'center_z': 'real',
        'merged': 'boolean',
        'agglomerated': 'boolean'
    }

    @staticmethod
    def from_fragments(fragments, filename, connectivity=2):
        '''Create a SqliteRagProvider, populated with a RAG extracted from a
        label image.'''

        rag = SqliteRagProvider(filename)
        rag.__write_rag(SubRag(fragments, connectivity=connectivity))

        return rag

    def __init__(self, filename):

        self.filename = filename

        connection = sqlite3.connect(self.filename)
        c = connection.cursor()
        try:
            c.execute('''
                DROP TABLE edges
            ''')
        except sqlite3.OperationalError:
            pass

        attributes = ', '.join([
            '%s %s'%(name, self.edge_attribute_dtypes[name])
            for name in self.edge_attributes
        ])
        c.execute('CREATE TABLE edges (%s)'%attributes)

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

    def __getitem__(self, slices):

        assert len(slices) == 3, "Sorry, SQLite backend does only 3D"

        slice_conditions = [
            self.__slice_condition('center_%s'%dim, s.start, s.stop)
            for dim, s in zip(['z', 'y', 'x'], slices)
            if s.start is not None or s.stop is not None
        ]
        if len(slice_conditions) > 0:
            contains_condition = 'WHERE ' + ' AND '.join(slice_conditions)
        else:
            contains_condition = ''

        connection = sqlite3.connect(self.filename)
        c = connection.cursor()
        edge_query = '''
            SELECT * FROM edges %s
        '''%contains_condition
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

        graph = SqliteSubRag(self.filename, self.sync_edge_attributes)
        graph.add_edges_from(edge_list)

        return graph

    def __remove_keys(self, dictionary, keys):

        for key in keys:
            del dictionary[key]
        return dictionary

    def __write_rag(self, rag):
        '''Write a complete RAG. This replaces whatever was stored in the DB
        before.'''

        connection = sqlite3.connect(self.filename)
        c = connection.cursor()
        c.execute('DELETE FROM edges')

        for u, v, data in rag.edges_iter(data=True):

            u, v = min(u, v), max(u, v)

            values = {
                'u': u,
                'v': v
            }
            values.update(data)

            names = ', '.join(self.edge_attributes)
            values = ', '.join([
                str(values[key])
                for key in self.edge_attributes
            ])
            query = 'INSERT INTO edges (%s) VALUES (%s)'%(names, values)
            logger.debug(query)
            c.execute(query)

        connection.commit()
        connection.close()
