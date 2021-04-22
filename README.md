# n_longest_paths

This is a preliminary investigation to see if the "repeated longest path
problem" can be solved efficiently using Rust. Initial results indicate that
this is indeed the case.


Python programs to generate sample problems and load them back in have been
included in `generate_edges.py` `load_edges.py`.

To run this program, execute

``` sh
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Clone repo
git clone https://github.com/ahendriksen/n_longest_paths.git
cd n_longest_paths
# Build and run
cargo run -- --faster --multiplicative examples/edges64.npy examples/out64.npy
```
For larger problems, use the `--faster` flag. We have two implementations, one
is quadratic in the input length, and the other one (the "faster" one) includes
some mitigations to be reasonably fast on larger input sizes.

To run benchmarks, execute:
``` sh
# Run benchmarks
cargo bench
# Open report in browser: (linux-specific)
xdg-open target/criterion/report/index.html
```
## Using as a python library 

First follow steps above. Then in an environment where numpy is already installed, use: 

``` sh
pip install maturin
maturin develop
```

Then start a python session and run: 

``` python
import numpy as np
from n_longest_paths import mark_longest_paths

def wrap_mark_longest_paths(src, dest, length, num_to_mark):
    return mark_longest_paths(src.astype(np.uint64), dest.astype(np.uint64), length.astype(np.float32), num_to_mark)

src = np.array([0, 0, 0])
dest = np.array([1, 1, 1])
length = np.array([1.1, 1.2, 1.3])  
wrap_mark_longest_paths(src, dest, length, 1)
```

The result should be `array([False, False, True])`: the longest path from node
`0` to node `1` was marked as the longest path.

