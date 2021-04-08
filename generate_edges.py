#!/usr/bin/env python3

import numpy as np
from pathlib import Path

# Parameters
n = 64
output_path = 'examples/edges64.npy'

# Data format
edge_dtype = [('from', '<i8'), ('to', '<i8'), ('norm', '<f4')]

# Compute edges
edges = [(i, j, 0.0 + i + j) for i in range(n) for j in range(i+1, n)]
edges_np = np.array(edges, dtype=edge_dtype)

# Save
Path(output_path).parent.mkdir(exist_ok=True)
np.save(output_path, edges_np)
