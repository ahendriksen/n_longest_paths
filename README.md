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
