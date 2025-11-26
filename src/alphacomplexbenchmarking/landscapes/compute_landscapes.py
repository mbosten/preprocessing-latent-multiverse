from gudhi.representations import Landscape

def compute_landscapes(persistence_per_dimension, num_landscapes: int = 5, resolution: int = 1000, homology_dimensions: list = [0, 1, 2]):
    LS = Landscape(resolution=resolution, keep_endpoints=False)

    landscapes_per_dimension = {}

    for dim in homology_dimensions:
        persistence_pairs = persistence_per_dimension.get(dim, [])
        if len(persistence_pairs) == 0:
            landscapes_per_dimension[dim] = None
            continue
        
        landscapes_per_dimension[dim] = LS.fit_transform([persistence_pairs])

    return landscapes_per_dimension