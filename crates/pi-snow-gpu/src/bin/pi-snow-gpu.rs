//! `pi-snow-gpu`: the engine-gpu-v1 driver. Subcommands:
//!
//! ```text
//! pi-snow-gpu batch ...                                    pi-snow batch's interface (src/batch.rs)
//! pi-snow-gpu lockstep --list FILE --from R --to R --out JSON   G1a on unit directories
//! pi-snow-gpu cpu-unit <ini> <end>                         CPU stepper on one unit (G0)
//! pi-snow-gpu dev-lockstep <columns> <days> <from> <to> [variant]  lockstep, synthetic
//! pi-snow-gpu dev-compare <columns> <days> <dir> [mix]     GPU vs CPU files, synthetic
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use pi_snow_core::PhysicsOptions;
use pi_snow_gpu::gpu::{Gpu, Precision};
use pi_snow_gpu::runner::{self, ColumnInput};
use pi_snow_gpu::{lockstep, stepper, synth, unit};

const NMAX: usize = 2048;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("batch") => pi_snow_gpu::batch::main(&args[2..]),
        Some("cpu-unit") => cpu_unit(&args[2..]),
        Some("lockstep") => units_lockstep(&args[2..]),
        Some("dev-lockstep") => dev_lockstep(&args[2..]),
        Some("dev-compare") => dev_compare(&args[2..]),
        _ => bail!("usage: see the module documentation"),
    }
}

fn cpu_unit(args: &[String]) -> Result<()> {
    let ini = PathBuf::from(args.first().context("ini")?);
    let end = unit::parse_end(args.get(1).context("end")?)?;
    let u = unit::load(&ini)?;
    let out = stepper::run_season(
        &u.cfg,
        &PhysicsOptions::default(),
        &u.site,
        &u.forcing,
        u.start.clone(),
        end,
    )?;
    let stem = unit::output_stem(&u);
    std::fs::create_dir_all(stem.parent().unwrap())?;
    std::fs::write(stem.with_extension("smet"), out.smet)?;
    std::fs::write(stem.with_extension("pro"), out.pro)?;
    eprintln!("cpu stepper: {} steps for {}", out.steps, u.site.station_id);
    Ok(())
}

fn num(args: &[String], i: usize) -> Result<usize> {
    Ok(args.get(i).context("missing argument")?.parse()?)
}

fn dev_lockstep(args: &[String]) -> Result<()> {
    let (columns, days, from, to) = (num(args, 0)?, num(args, 1)?, num(args, 2)?, num(args, 3)?);
    let cfg = if args.get(4).map(String::as_str) == Some("variant") {
        synth::config_variant()
    } else {
        synth::config()
    };
    let s = synth::generate(columns, days);
    let inputs: Vec<ColumnInput> = s
        .sites
        .iter()
        .zip(s.rows)
        .map(|(site, rows)| ColumnInput { cfg: &cfg, site, start: synth::bare(site), rows })
        .collect();
    let gpu = Gpu::new(Precision::F64)?;
    eprintln!("{}", gpu.build.to_json());
    let started = std::time::Instant::now();
    let tally = lockstep::run(&gpu, &inputs, NMAX, from, to)?;
    eprintln!("lockstep {:.1} s", started.elapsed().as_secs_f64());
    println!("{}", tally.to_json());
    Ok(())
}

fn dev_compare(args: &[String]) -> Result<()> {
    let (columns, days) = (num(args, 0)?, num(args, 1)?);
    let dir = PathBuf::from(args.get(2).context("dir")?);
    let precision = if args.get(3).map(String::as_str) == Some("mix") { Precision::Mix } else { Precision::F64 };
    std::fs::create_dir_all(dir.join("cpu"))?;
    std::fs::create_dir_all(dir.join("gpu"))?;
    let cfg = synth::config();
    let s = synth::generate(columns, days);
    let options = PhysicsOptions::default();
    let started = std::time::Instant::now();
    for (site, rows) in s.sites.iter().zip(&s.rows) {
        let mut st = stepper::Stepper::new(&cfg, &options, site, synth::bare(site))?;
        let mut w = stepper::SeasonWriter::new(site);
        for r in rows {
            if let stepper::Outcome::Advanced { output: true } = st.step(r)? {
                w.push(&mut st, r.t);
            }
        }
        let out = w.finish(site, st.steps);
        std::fs::write(dir.join("cpu").join(format!("{}.smet", site.station_id)), out.smet)?;
        std::fs::write(dir.join("cpu").join(format!("{}.pro", site.station_id)), out.pro)?;
    }
    eprintln!("cpu {:.2} s", started.elapsed().as_secs_f64());
    let gpu = Gpu::new(precision)?;
    eprintln!("{}", gpu.build.to_json());
    let inputs: Vec<ColumnInput> = s
        .sites
        .iter()
        .zip(s.rows)
        .map(|(site, rows)| ColumnInput { cfg: &cfg, site, start: synth::bare(site), rows })
        .collect();
    let started = std::time::Instant::now();
    let results = runner::run_hourly(&gpu, &inputs, NMAX, &options)?;
    eprintln!("gpu {:.2} s", started.elapsed().as_secs_f64());
    for (r, i) in results.iter().zip(&inputs) {
        let id = &i.site.station_id;
        std::fs::write(dir.join("gpu").join(format!("{id}.smet")), &r.output.smet)?;
        std::fs::write(dir.join("gpu").join(format!("{id}.pro")), &r.output.pro)?;
        eprintln!("{id}: steps {} max elements {} failure {:?}", r.steps, r.max_elements, r.failure);
    }
    Ok(())
}

fn units_lockstep(args: &[String]) -> Result<()> {
    let (parsed, extras) = pi_snow_gpu::batch::parse_args_with(args, &["--from", "--to", "--out"])?;
    let get = |k: &str| extras.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone()).with_context(|| format!("{k} is required"));
    let from: usize = get("--from")?.parse()?;
    let to: usize = get("--to")?.parse()?;
    let out = PathBuf::from(get("--out")?);
    let mut loaded = Vec::new();
    for job in &parsed.jobs {
        loaded.push(pi_snow_gpu::batch::load(job)?);
    }
    let inputs: Vec<ColumnInput> = loaded
        .iter()
        .map(|(u, rows)| ColumnInput { cfg: &u.cfg, site: &u.site, start: u.start.clone(), rows: rows.clone() })
        .collect();
    let gpu = Gpu::new(parsed.precision)?;
    let started = std::time::Instant::now();
    let tally = lockstep::run(&gpu, &inputs, parsed.nmax, from, to)?;
    let text = format!(
        "{{\"from\": {from}, \"to\": {to}, \"units\": {}, \"seconds\": {:.1}, \"engine\": {}, \"tally\": {}}}\n",
        inputs.len(),
        started.elapsed().as_secs_f64(),
        gpu.build.to_json(),
        tally.to_json()
    );
    let tmp = out.with_extension("partial");
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, &out)?;
    Ok(())
}
