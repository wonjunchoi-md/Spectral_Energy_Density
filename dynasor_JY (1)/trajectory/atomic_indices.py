import os
import re
import numpy as np
from dynasor.logging_tools import logger


def parse_gromacs_index_file(fname):
    """
    Parses a gromacs style index (ndx) file.
    Returns a dict with key values as
    group-name: [1, 3, 8]

    Note that atomic indices in gromacs-ndx file starts with 1, but the returned dict starts with 0
    """

    if not os.path.isfile(fname):
        raise ValueError('Index file not found')

    atomic_indices = dict()
    header_re = re.compile(r'^ *\[ *([a-zA-Z0-9_.-]+) *\] *$')
    with open(fname, 'r') as fobj:
        for line in fobj.readlines():
            match = header_re.match(line)
            if match is not None:  # get name of group
                name = match.group(1)
                if name in atomic_indices.keys():
                    logger.warning(f'Group name {name} appears twice in index file, only one used.')
                atomic_indices[name] = []
            else:  # get indices for group
                indices = [int(i) for i in line.split()]
                atomic_indices[name].extend(indices)

    # cast to indices to numpy arrays and shift so indices start with 0
    for name, indices in atomic_indices.items():
        atomic_indices[name] = np.array(indices) - 1

    return atomic_indices
