//! A5 table candidate driver. It reuses the registered engine runners.

use std::path::PathBuf;

use anyhow::{Context, Result};
use pi_snow_gpu_alternatives::{batch, lockstep, runner::ColumnInput};

#[path = "../a5_gpu.rs"]
mod a5_gpu;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("identity") => candidate_identity(&args[1..]),
        Some("lockstep") => lockstep_run(&args[1..]),
        Some("grid") => grid_run(&args[1..]),
        _ => anyhow::bail!("usage: pi-snow-gpu-a5 identity|lockstep|grid ..."),
    }
}

fn candidate_identity(args: &[String]) -> Result<()> {
    let mut values = std::collections::HashMap::new();
    let mut it = args.iter();
    while let Some(key) = it.next() {
        let value = it.next().with_context(|| format!("{key} needs a value"))?;
        if !matches!(key.as_str(), "--table" | "--axes") {
            anyhow::bail!("unknown option {key}");
        }
        if values.insert(key.as_str(), value.as_str()).is_some() {
            anyhow::bail!("duplicate option {key}");
        }
    }
    let get = |name: &str| {
        values
            .get(name)
            .copied()
            .map(PathBuf::from)
            .with_context(|| format!("{name} is required"))
    };
    let gpu = a5_gpu::new_f64(&get("--table")?, &get("--axes")?)?;
    println!("{}", gpu.build.to_json());
    Ok(())
}

fn lockstep_run(args: &[String]) -> Result<()> {
    let (parsed, extras) = batch::parse_args_with(args, &["--table", "--axes", "--from", "--to"])?;
    let get = |name: &str| -> Result<String> {
        extras
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.clone())
            .with_context(|| format!("{name} is required"))
    };
    let table = PathBuf::from(get("--table")?);
    let axes = PathBuf::from(get("--axes")?);
    let from: usize = get("--from")?.parse()?;
    let to: usize = get("--to")?.parse()?;
    let mut loaded = Vec::new();
    for job in &parsed.jobs {
        loaded.push(batch::load(job)?);
    }
    let inputs: Vec<ColumnInput> = loaded
        .iter()
        .map(|(unit, rows)| ColumnInput {
            cfg: &unit.cfg,
            site: &unit.site,
            start: unit.start.clone(),
            rows: rows.clone(),
            profile_times: None,
        })
        .collect();
    let gpu = a5_gpu::new_f64(&table, &axes)?;
    eprintln!("{}", gpu.build.to_json());
    let tally = lockstep::run_development(&gpu, &inputs, parsed.nmax, from, to)?;
    println!("{}", tally.to_json());
    Ok(())
}

fn grid_run(args: &[String]) -> Result<()> {
    let mut values = std::collections::HashMap::new();
    let mut it = args.iter();
    while let Some(key) = it.next() {
        if !key.starts_with("--") {
            anyhow::bail!("unexpected grid argument {key}");
        }
        let value = it.next().with_context(|| format!("{key} needs a value"))?;
        if values.insert(key.as_str(), value.as_str()).is_some() {
            anyhow::bail!("duplicate option {key}");
        }
    }
    let get = |key: &str| {
        values
            .get(key)
            .copied()
            .with_context(|| format!("{key} is required"))
    };
    let table = PathBuf::from(get("--table")?);
    let axes = PathBuf::from(get("--axes")?);
    let index_file = PathBuf::from(get("--pixels")?);
    let pixels = std::fs::read_to_string(&index_file)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.trim().parse::<usize>().map_err(anyhow::Error::from))
        .collect::<Result<Vec<_>>>()?;
    let grid_args = pi_snow_gpu_alternatives::grid::GridArgs {
        forcing_dir: PathBuf::from(get("--forcing")?),
        ini_template: PathBuf::from(get("--ini")?),
        pixels,
        batch: get("--batch")?.parse()?,
        nmax: values
            .get("--nmax")
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(2048),
        out_dir: PathBuf::from(get("--out")?),
        threads: values
            .get("--threads")
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(1),
    };
    let gpu = a5_gpu::new_f64(&table, &axes)?;
    eprintln!("{}", gpu.build.to_json());
    pi_snow_gpu_alternatives::grid::run(&gpu, &grid_args)
}
