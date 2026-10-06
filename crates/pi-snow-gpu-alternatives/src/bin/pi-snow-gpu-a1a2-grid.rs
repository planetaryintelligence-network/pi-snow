//! Full-grid G2 driver using the unchanged A1+A2 candidate kernel.
#[path = "../a1a2_gpu.rs"] mod a1a2_gpu;
#[path = "../a1a2_runner.rs"] mod a1a2_runner;
#[path = "../a1a2_grid.rs"] mod a1a2_grid;
pub use pi_snow_gpu_alternatives::{forcing, sha256, stepper};
use anyhow::{Context, Result, bail};
use std::{collections::HashMap, path::PathBuf};
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() != Some("grid") { bail!("expected grid"); }
    let mut values = HashMap::new();
    while let Some(k) = args.next() {
        if !["--forcing", "--ini", "--pixels", "--batch", "--nmax", "--threads", "--out"].contains(&k.as_str()) { bail!("unknown option {k}"); }
        let v = args.next().with_context(|| format!("{k} requires value"))?;
        if values.insert(k.clone(), v).is_some() { bail!("duplicate {k}"); }
    }
    let get = |k: &str| values.get(k).with_context(|| format!("missing {k}"));
    let indices = std::fs::read_to_string(get("--pixels")?)?.lines().map(str::parse).collect::<std::result::Result<Vec<usize>, _>>()?;
    let args = a1a2_grid::GridArgs { forcing_dir: PathBuf::from(get("--forcing")?), ini_template: PathBuf::from(get("--ini")?), pixels: indices,
        batch: get("--batch")?.parse()?, nmax: get("--nmax")?.parse()?, threads: get("--threads")?.parse()?, out_dir: PathBuf::from(get("--out")?) };
    if args.batch == 0 || args.nmax == 0 || args.threads == 0 { bail!("positive batch/nmax/threads required"); }
    let gpu = a1a2_gpu::Gpu::new(a1a2_gpu::Precision::F32)?;
    eprintln!("{}", gpu.build.to_json());
    a1a2_grid::run(&gpu, &args)
}
