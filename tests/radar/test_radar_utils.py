import numpy as np
from adapt.radar.radar_utils import compute_all_cell_centroids, compute_cell_centroid

def test_compute_cell_centroid_simple():
    labels = np.array([
        [0, 1, 1],
        [0, 1, 1],
        [0, 0, 0],
    ])

    centroid = compute_cell_centroid(labels, 1)
    assert centroid == (0.5, 1.5)


def test_compute_all_cell_centroids():
    labels = np.array([
        [1, 1, 0],
        [0, 2, 2],
        [0, 2, 2],
    ])

    centroids = compute_all_cell_centroids(labels)

    assert set(centroids.keys()) == {1, 2}

def test_compute_cell_centroid_missing_label():
    labels = np.zeros((4, 4), dtype=int)
    assert compute_cell_centroid(labels, 1) is None
