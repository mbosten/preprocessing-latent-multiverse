import gudhi as gd
import numpy as np


def mask_infinities(array):
        return array[array[:, 1] < np.inf]


def compute_alpha_complex_persistence(data, homology_dimensions: list = [0, 1, 2]):
    """
    Computes the alpha complex persistence for the given data points.

    Parameters:
    data (array-like): A set of points in a metric space.
    homology_dimensions (list): List of homology dimensions to compute persistence for.

    Returns:
    persistence_per_dimension (dict): A dictionary where keys are homology dimensions and values are lists of tuples representing the persistence pairs.
    """

    # Create an Alpha Complex from the data points
    alpha_complex = gd.AlphaComplex(points=data, precision="fast")
    simplex_tree = alpha_complex.create_simplex_tree()

    persistence_per_dimension = {}

    # Compute the persistence
    for dim in homology_dimensions:
          persistence_pairs = mask_infinities(simplex_tree.persistence_intervals_in_dimension(dim))
          persistence_per_dimension[dim] = persistence_pairs


    return persistence_per_dimension