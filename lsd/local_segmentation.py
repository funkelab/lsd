import daisy
from funlib.segment.arrays import replace_values

import numpy as np

from .persistence import MongoDbRagProvider


class LocalSegmentationExtractor:
    def __init__(
        self,
        fragments_host: str,
        fragments_db: str,
        edges_collection: str,
        fragments_file: str,
        fragments_dataset: str,
    ):
        self.fragments_host = fragments_host
        self.fragments_db = fragments_db
        self.edges_collection = edges_collection
        self.fragments_file = fragments_file
        self.fragments_dataset = fragments_dataset

    def get_local_segmentation(self, roi: daisy.Roi, threshold: float):
        # open fragments
        fragments = daisy.open_ds(self.fragments_file, self.fragments_dataset)

        # open RAG DB
        rag_provider = MongoDbRagProvider(
            self.fragments_db,
            host=self.fragments_host,
            mode="r",
            edges_collection=self.edges_collection,
        )

        segmentation = fragments[roi]
        segmentation.materialize()
        ids = [int(id) for id in list(np.unique(segmentation.data))]
        rag = rag_provider.read_rag(ids)

        if len(rag.nodes()) == 0:
            raise Exception('RAG is empty')

        components = rag.get_connected_components(threshold)

        values_map = np.array(
            [
                [fragment, i]
                for i in range(1, len(components)+1)
                for fragment in components[i-1]
            ],
            dtype=np.uint64,
        )
        old_values = values_map[:, 0]
        new_values = values_map[:, 1]
        replace_values(
            segmentation.data, old_values, new_values, inplace=True
        )

        return segmentation

