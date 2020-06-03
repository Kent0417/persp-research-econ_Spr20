import numpy as np

#resample(normalized_weights/n_parts, method = resampling_method)
def resample(weights, method="systematic"):

    n_parts = len(weights)

    if method == "multinomial":
        indx = np.empty(n_parts)

        cumulative_weights = np.cumsum(weights/sum(weights))
        offset = np.random.rand(n_parts)

        for i in range(n_parts):
            indx[i] = min(*np.where([offset[i] < x for x in cumulative_weights]))

        return indx

    elif method == "systematic":
        raise Exception("Underconstructing....")

    elif method == "polyalgo":
        weights = weights / sum(weights)
        weights = weights.flatten()
        return np.random.choice(np.arange(len(weights)),n_parts, replace=True, p=weights)
