//! Registered A1+A2 compact FP32 arm, isolated from the default GPU modules.

#[path = "../a1a2_gpu.rs"]
mod a1a2_gpu;
#[path = "../a1a2_runner.rs"]
mod a1a2_runner;
#[path = "../a1a2_lockstep.rs"]
mod a1a2_lockstep;

pub use pi_snow_gpu_alternatives::{sha256, stepper};

use anyhow::{Context, Result, bail};
use pi_snow_core::PhysicsOptions;
use pi_snow_gpu_alternatives::batch;
use a1a2_gpu::{Gpu, Precision};
use a1a2_runner::ColumnInput;
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("nvrtc") => {
            let source = include_str!("../../kernels/pi_snow_a1a2.cu");
            let options = CompileOptions {
                arch: Some("sm_120"),
                options: vec!["--fmad=true".into(), "--prec-div=true".into(), "--prec-sqrt=true".into(),
                    "--ftz=false".into(), "-DPI_F32".into(), "-DPI_COMPACT".into()],
                ..Default::default()
            };
            let ptx = compile_ptx_with_opts(source, options).map_err(|e| anyhow::anyhow!("NVRTC: {e:?}"))?;
            let ptx = ptx.to_src();
            println!("{{\"kernel_source_sha256\":\"{}\",\"ptx_sha256\":\"{}\"}}",
                pi_snow_gpu_alternatives::sha256::sha256_hex(source.as_bytes()),
                pi_snow_gpu_alternatives::sha256::sha256_hex(ptx.as_bytes()));
            Ok(())
        }
        Some("identity") => {
            let gpu = Gpu::new(Precision::F32)?;
            println!("{}", gpu.build.to_json());
            Ok(())
        }
        Some("lockstep") => {
            let all: Vec<String> = args.collect();
            let (parsed, extras) = batch::parse_args_with(&all, &["--from", "--to", "--out"])?;
            let get = |key: &str| extras.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
                .with_context(|| format!("{key} is required"));
            let from: usize = get("--from")?.parse()?;
            let to: usize = get("--to")?.parse()?;
            let out = PathBuf::from(get("--out")?);
            let loaded = parsed.jobs.iter().map(batch::load).collect::<Result<Vec<_>>>()?;
            let inputs: Vec<ColumnInput> = loaded.iter().map(|(u, rows)| ColumnInput {
                cfg: &u.cfg, site: &u.site, start: u.start.clone(), rows: rows.clone(), profile_times: None,
            }).collect();
            let gpu = Gpu::new(Precision::F32)?;
            let tally = a1a2_lockstep::run_development(&gpu, &inputs, parsed.nmax, from, to)?;
            let text = format!("{{\"from\":{from},\"to\":{to},\"units\":{},\"engine\":{},\"tally\":{}}}\n",
                inputs.len(), gpu.build.to_json(), tally.to_json());
            std::fs::write(out, text)?;
            Ok(())
        }
        Some("season") => {
            let all: Vec<String> = args.collect();
            let parsed = batch::parse_args(&all)?;
            let loaded = parsed.jobs.iter().map(batch::load).collect::<Result<Vec<_>>>()?;
            let stems: Vec<PathBuf> = loaded.iter().map(|(u, _)| pi_snow_gpu_alternatives::unit::output_stem(u)).collect();
            let inputs: Vec<ColumnInput> = loaded.iter().map(|(u, rows)| ColumnInput {
                cfg: &u.cfg, site: &u.site, start: u.start.clone(), rows: rows.clone(), profile_times: None,
            }).collect();
            let gpu = Gpu::new(Precision::F32)?;
            let options = PhysicsOptions::default();
            let results = a1a2_runner::run_hourly(&gpu, &inputs, parsed.nmax, &options)?;
            for (result, stem) in results.iter().zip(stems) {
                std::fs::write(stem.with_extension("smet"), &result.output.smet)?;
                std::fs::write(stem.with_extension("pro"), &result.output.pro)?;
            }
            Ok(())
        }
        _ => bail!("usage: pi-snow-gpu-a1a2 nvrtc | identity | lockstep [batch args] --from STEP --to STEP --out JSON | season [batch args]"),
    }
}
