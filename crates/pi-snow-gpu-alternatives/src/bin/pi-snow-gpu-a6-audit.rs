//! A6 deep-element cap, isolated kernel and host backend.

#[path = "../a6_audit_gpu.rs"]
mod a6_gpu;
#[path = "../a6_audit_runner.rs"]
mod a6_runner;

pub use pi_snow_gpu_alternatives::{sha256, stepper};

use anyhow::{Context, Result, bail};
use pi_snow_core::{Column, PhysicsOptions};
use pi_snow_gpu_alternatives::forcing;
use pi_snow_io::{ini::{Ini, RunConfig}, smet::Smet};
use a6_gpu::{Gpu, Precision};
use a6_runner::ColumnInput;
use std::{fs, path::PathBuf};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("nvrtc") => {
            let source = include_str!("../../kernels/pi_snow_a6_audit.cu");
            let options = cudarc::nvrtc::CompileOptions {
                arch: Some("sm_120"),
                options: vec!["--fmad=false".into(), "--prec-div=true".into(), "--prec-sqrt=true".into(), "--ftz=false".into()],
                ..Default::default()
            };
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(source, options)
                .map_err(|e| anyhow::anyhow!("NVRTC: {e:?}"))?.to_src();
            println!("{{\"kernel_source_sha256\":\"{}\",\"ptx_sha256\":\"{}\"}}",
                pi_snow_gpu_alternatives::sha256::sha256_hex(source.as_bytes()),
                pi_snow_gpu_alternatives::sha256::sha256_hex(ptx.as_bytes()));
            Ok(())
        }
        Some("season") => {
            let forcing_dir = PathBuf::from(args.next().context("FORCING_DIR")?);
            let template_path = PathBuf::from(args.next().context("INI_TEMPLATE")?);
            let pixels_path = PathBuf::from(args.next().context("PIXELS_TSV")?);
            let keys_path = PathBuf::from(args.next().context("KEYS_FILE")?);
            let out = PathBuf::from(args.next().context("OUT_DIR")?);
            if args.next().is_some() { bail!("too many arguments"); }
            if out.exists() {
                if !out.is_dir() || fs::read_dir(&out)?.next().is_some() {
                    bail!("refusing to overwrite non-empty output {}", out.display());
                }
            } else {
                fs::create_dir_all(&out)?;
            }
            let template = fs::read_to_string(template_path)?;
            let pixels = forcing::read_pixels(&pixels_path)?;
            let keys: Vec<_> = fs::read_to_string(keys_path)?.lines().map(str::trim)
                .filter(|s| !s.is_empty()).map(str::to_owned).collect();
            if keys.len() != 68 { bail!("expected 68 frozen A6 pixels, found {}", keys.len()); }
            let mut columns = Vec::with_capacity(keys.len());
            for key in &keys {
                let pixel = pixels.iter().find(|p| p.key == *key).with_context(|| format!("missing pixel {key}"))?;
                let mut raw = fs::File::open(forcing_dir.join("forcing.f32"))?;
                let hourly = forcing::hourly_forcing(&forcing::read_hourly(&mut raw, pixel.index)?)?;
                let rows = forcing::expand(&hourly, 1, forcing::HOURS - 1);
                let cfg = RunConfig::from_ini(&Ini::parse(&forcing::ini_text(&template, key)?)?)?;
                let site = pixel.site()?;
                let start = Column::bare_ground(forcing::GROUND_TEMPERATURE_K, site.slope_angle, site.slope_azimuth);
                let meteo = Smet::parse(&forcing::smet_text(pixel, &rows))?;
                let index = stepper::ForcingIndex::of(&meteo)?;
                let rows: Vec<stepper::ForcingRow> = meteo.rows.iter().map(|r| index.row(r)).collect();
                columns.push((cfg, site, start, rows, key.clone()));
            }
            let inputs: Vec<_> = columns.iter().map(|(cfg, site, start, rows, _)| ColumnInput {
                cfg, site, start: start.clone(), rows: rows.clone(), profile_times: Some(rows.iter().filter(|r| r.t.timestamp() % 86400 == 43200).map(|r| r.t.timestamp()).collect()),
            }).collect();
            let gpu = Gpu::new(Precision::F64)?;
            eprintln!("{}", gpu.build.to_json());
            let results = a6_runner::run_hourly(&gpu, &inputs, 2048, &PhysicsOptions::default())?;
            let daily_dir = out.join("daily"); fs::create_dir(&daily_dir)?;
            let mut audit = String::from("key\tmax_mass_abs\tmax_energy_abs\tmax_mass_rel\tmax_energy_rel\tmerges\tviolations\n");
            for (result, (_, _, _, _, key)) in results.iter().zip(&columns) {
                if let Some(failure) = &result.failure { bail!("A6 GPU column {key} failed: {failure}"); }
                fs::write(out.join(format!("{key}_grid.smet")), &result.output.smet)?;
                fs::write(out.join(format!("{key}_grid.pro")), &result.output.pro)?;
                let mut daily = String::from("timestamp\tHS\tSWE\tTSS\twater\tn\n");
                for (t, v) in &result.daily { daily.push_str(&format!("{}\t{}\t{}\t{}\t{}\t{}\n", t,v[0],v[1],v[2],v[3],v[4])); }
                fs::write(daily_dir.join(format!("{key}.daily.tsv")), daily)?;
                audit.push_str(&format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\n", key, result.audit[0], result.audit[1], result.audit[2], result.audit[3], result.audit[4], result.audit[5]));
                println!("column={} steps={} max_elements={}", key, result.steps, result.max_elements);
            }
            fs::write(out.join("conservation.tsv"), audit)?;
            Ok(())
        }
        _ => bail!("usage: pi-snow-gpu-a6 nvrtc | season FORCING_DIR INI_TEMPLATE PIXELS_TSV KEYS_FILE OUT_DIR"),
    }
}
