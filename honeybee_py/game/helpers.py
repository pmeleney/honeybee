import itertools
import numpy as np

def _flatten_list(nested_list):
    """
    Flatten a nested list into a single-level list.
    
    Args:
        nested_list (list): A list that may contain nested sublists
        
    Returns:
        list: A flattened list containing all elements from the nested structure
    """
    flat_list = []
    for sublist in nested_list:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def _fill_rect(xl, yt, size):
    """
    Generate a list of coordinate tuples for a rectangle.
    
    Args:
        xl (int): Left x-coordinate of the rectangle
        yt (int): Top y-coordinate of the rectangle
        size (tuple): Size of the rectangle as (width, height)
        
    Returns:
        list: List of (x, y) coordinate tuples representing the rectangle
    """
    xr = range(xl,xl+size[0])
    yr = range(yt,yt+size[1])
    rect = list(itertools.product(xr, yr))
    return rect

def _distance(position1, position2):
    """
    Calculate the Euclidean distance between two positions.
    
    Args:
        position1 (tuple or list): First position as (x, y) coordinates
        position2 (tuple or list): Second position as (x, y) coordinates
        
    Returns:
        float: Euclidean distance between the two positions
    """
    a = position1[0] - position2[0]
    b = position1[1] - position2[1]
    return np.sqrt((a**2 + b**2))