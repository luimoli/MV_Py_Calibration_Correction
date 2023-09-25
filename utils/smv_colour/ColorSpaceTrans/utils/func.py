import torch
import numpy as np

def stack(arr):
    """[Stacks arrays in sequence along the last axis (tail).]
    Args:
        arr ([array_like]): [arrays to perform the stacking.]
    Returns:
        [type]: [description]
    """
    return torch.cat([x[..., None] for x in arr], axis=-1)


def split(arr):
    """[Splits arrays in sequence along the last axis (tail).]
    Args:
        arr ([array_like]): [Array to perform the splitting.]
    Returns:
        [type]: [description]
    """
    return [arr[..., x] for x in range(arr.shape[-1])]

def dot_vector(matrix, vector):
    # print(type(vector))
    # exit()
    # print(len(vector.shape))
    try:
        if len(matrix.shape) == 3:
            return torch.einsum('ij,hwj->hwi', matrix, vector)
        else:
            return torch.einsum('ij,...j->...i', matrix, vector)
    except TypeError:
        if len(matrix.shape) == 3:
            return np.einsum('ij,hwj->hwi', np.array(matrix), vector)
        else:
            return np.einsum('ij,...j->...i', np.array(matrix), vector)