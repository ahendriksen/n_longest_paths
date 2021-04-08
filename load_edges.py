#!/usr/bin/env python3

import numpy as np
from pathlib import Path

# Parameters
n = 64
edges_path = 'examples/edges64.npy'
in_longest_path = 'examples/out64.npy'


edges = np.load(edges_path)
in_path = np.load(in_longest_path)
