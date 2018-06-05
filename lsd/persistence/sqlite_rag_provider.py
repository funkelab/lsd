from lsd import SharedRagProvider, SubRag
from networkx.convert import to_dict_of_dicts
import sqlite3
import logging

logger = logging.getLogger(__name__)

class SqliteSubRag(SubRag):

    def __init__(self, filename):

        super(SubRag, self).__init__()

        self.filename = filename

    def sync(self):

        connection = sqlite3.connect(self.filename)
        c = connection.cursor()

        for u, v, data in self.edges_iter(data=True):

            merged = data['merged']

            c.execute('''
                UPDATE edges
                SET merged = %d
                WHERE u == %d and v == %d
            '''%(merged, u, v))

        connection.commit()
        connection.close()

class SqliteRagProvider(SharedRagProvider):
    '''A shared region adjacency graph stored in an SQLite file.
    '''

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
        c.execute('''
            DROP TABLE edges
        ''')
        c.execute('''
            CREATE TABLE edges (
                u bigint, v bigint,
                center_z real, center_y real, center_x real,
                merged boolean)
        ''')

        connection.commit()
        connection.close()

    def __slice_condition(self, value, start, stop):

        if start is None and stop is None:
            return None

        if start is None:
            return '%s < %f'%(value, stop)

        if stop is None:
            return '%s >= %f'%(value, start)

        return '%s BETWEEN %f AND %f'%(value, start, stop)

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

        edge_query = '''
            SELECT * FROM edges %s
        '''%contains_condition
        logger.debug(edge_query)

        connection = sqlite3.connect(self.filename)
        c = connection.cursor()
        rows = c.execute(edge_query)
        edge_list = [
            (r[0], r[1], {
                'location': (r[2], r[3], r[4]),
                'merged': r[5]
            })
            for r in rows
        ]
        connection.close()

        graph = SqliteSubRag(self.filename)
        graph.add_edges_from(edge_list)

        # TODO: add edge attributes

        return graph

    def __write_rag(self, rag):

        connection = sqlite3.connect(self.filename)
        c = connection.cursor()
        c.execute('''
            DELETE FROM edges
        ''')

        for u, v, data in rag.edges_iter(data=True):

            location = data['location']

            c.execute('''
                INSERT INTO
                    edges (u, v, center_z, center_y, center_x, merged)
                VALUES
                    (%d, %d, %f, %f, %f, %d)
            '''%(
                u, v,
                location[0], location[1], location[2],
                False
            ))

        connection.commit()
        connection.close()
