# pi-snow

A Rust snowpack model, built to reproduce SNOWPACK's outputs and then improve on them,
with a CUDA batch engine for running many slopes at once.

- `crates/pi-snow-core`, `pi-snow-io`, `pi-snow-physics`: the model, its SNOWPACK-format
  inputs and outputs, and the layer physics.
- `crates/pi-snow-cli`: the `pi-snow` binary.
- `crates/pi-snow-gpu`: the `pi-snow-gpu` batch engine and its kernel
  (`kernels/pi_snow.cu`).
- `experimental/`: code from finished GPU experiments, kept buildable.

The toolchain is pinned to 1.93.1 in `rust-toolchain.toml`. Build with
`cargo build --release`; output goes to this checkout's own `target/`.

This history was split out of pi-pod, the avalanche research repository whose experiments
drive the model. Records there made before the split name paths under `rust/`; the same
files sit at this repository's root.

## License

Copyright (c) 2026 The Planetary Intelligence Network contributors.

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option. Use it for anything; keep the copyright
notice and license text with copies or substantial portions of it.

Unless you explicitly state otherwise, any contribution you intentionally submit for
inclusion in this work, as defined in the Apache-2.0 license, shall be dual licensed as
above, without any additional terms or conditions.
