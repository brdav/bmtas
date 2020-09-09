import json
from pathlib import Path
from collections import OrderedDict
import torch
import numpy as np


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def normalize_tensor(normals, dim=1):
    norm = torch.norm(normals, p='fro', dim=dim, keepdim=True) + 1e-12
    return normals.div(norm)


def normalize_array(normals, dim=1):
    norm = np.linalg.norm(normals, ord=2, axis=dim, keepdims=True) + 1e-12
    return np.divide(normals, norm)


def progress(current, total):
    base = '{}/{} ({:.0f}%)'
    return base.format(current, total, 100.0 * current / total)


def distributed_print(rank):
    def func(*args):
        if rank == 0:
            print(*args)
    return func
