import numpy as np


def get_count_of_unique_elements_in_y(y):
    """
    input: numpy array
    output: dict with frequencies of unique elements
    """
    counts = {}
    unique_elements, counts_elements = np.unique(y, return_counts=True)
    for x in zip(unique_elements, counts_elements):
        counts[x[0]] = x[1]
    return counts


def get_distribution_from_y(y):
    """
    input: numpy array
    output: dict with distribution of different items as percentage
    """
    counts = get_count_of_unique_elements_in_y(y)
    total_num_elements = y.size
    distr = {}
    for key in counts:
        distr[key] = counts[key]/total_num_elements * 100
    return distr