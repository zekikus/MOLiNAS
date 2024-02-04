import numpy as np
from cell_module.ops import OPS as ops_dict

INPUT = 'input'
OUTPUT = 'output'
OPS = list(ops_dict.keys())
OPS.remove('bottleneck')

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


def encode_caz(matrix, ops):
    encoding = {f"{op}-{in_out}-{i}":0 for in_out in ["in","out"] for op in OPS for i in range(1, 7)}
    encoding.update({f"in-out-{i}":0 for i in range(1, 7)})
    encoding.update({f"out-in-{i}":0 for i in range(1, 7)})

    for i in range(7):
        op = ops[i].split("-")[0]
        out_edges = int(matrix[i,:].sum())
        in_edges = int(matrix[:,i].sum())
        
        if ops[i] == INPUT and out_edges != 0:
            encoding[f"in-out-{out_edges}"] = 1
        elif ops[i] == OUTPUT and in_edges != 0:
            encoding[f"out-in-{in_edges}"] = 1
        else:
            if in_edges !=  0:
                encoding[f"{op}-in-{in_edges}"] = 1
            if out_edges != 0:
                encoding[f"{op}-out-{out_edges}"] = 1

    return np.array(list(encoding.values()))


def encode_paths(path_indices):
    """
    The function encodes path indices into a one-hot encoding format.
    
    :param path_indices: A list of integers representing the indices of the paths to be encoded
    :return: The function `encode_paths` returns a one-hot encoding of paths. It takes in a list of path
    indices and returns a numpy array of zeros with length equal to the total number of possible paths.
    The function then sets the values at the indices in the input list to 1 and returns the resulting
    array.
    """
    num_paths = sum([len(OPS) ** i for i in range(OP_SPOTS + 1)])
    encoding = np.zeros(num_paths)
    for index in path_indices:
        encoding[index] = 1
    return encoding