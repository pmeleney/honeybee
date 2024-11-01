import itertools
import numpy as np

def _flatten_list(nested_list):
    flat_list = []
    for sublist in nested_list:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def _fill_rect(xl, yt, size):
    xr = range(xl,xl+size[0])
    yr = range(yt,yt+size[1])
    rect = list(itertools.product(xr, yr))
    return rect

def _distance(position1, position2):
        a = position1[0] - position2[0]
        b = position1[1] - position2[1]
        return np.sqrt((a**2 + b**2))