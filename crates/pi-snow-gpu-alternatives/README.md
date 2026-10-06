# GPU alternatives v1 executables

The fork's control driver is named `pi-snow-gpu-alt-control`:

```sh
cd rust
cargo build --release --locked -p pi-snow-gpu-alternatives --bin pi-snow-gpu-alt-control
```

Its subcommands are documented in `src/bin/pi-snow-gpu.rs`. Cargo explicitly
assigns that source file to the new executable name. Runtime code and kernels
are unchanged by the rename.

`target/release/pi-snow-gpu` belongs to the production `pi-snow-gpu` crate.
The experiment's current G2 and watch scripts build that production package
explicitly for their controls; those invocations keep their production name.
Historical handovers that built the fork's former `pi-snow-gpu` executable
retain their recorded commands and source pins. They are historical evidence;
a new fork-control invocation uses `pi-snow-gpu-alt-control`.
