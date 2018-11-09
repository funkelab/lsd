import lsd
import numpy as np

if __name__ == "__main__":

    labels = np.array([0, 1, 10, 1, 0])
    print(labels)
    print(lsd.region_growing.get_rois(labels))

    labels = np.array([
        [0, 1, 10, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 2, 100, 3, 0],
    ])
    print(labels)
    print(lsd.region_growing.get_rois(labels))
