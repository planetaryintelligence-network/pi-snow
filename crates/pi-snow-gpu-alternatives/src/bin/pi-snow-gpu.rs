//! `pi-snow-gpu-alt-control`: the isolated gpu-alternatives-v1 control driver. Subcommands:
//!
//! ```text
//! pi-snow-gpu-alt-control batch ...                                    pi-snow batch's interface (src/batch.rs)
//! pi-snow-gpu-alt-control lockstep --list FILE --from R --to R --out JSON [--precision F64|F32|F32K]
//!                                                        G1a or precision development lockstep
//! pi-snow-gpu-alt-control grid --forcing DIR --ini TEMPLATE --pixels FILE --batch B --out DIR
//!                [--precision F64|MIX|F32|F32K] [--nmax N] [--threads T]   grid season, daily fields (G2)
//! pi-snow-gpu-alt-control write-units --forcing DIR --ini TEMPLATE --pixels FILE --out DIR
//!                unit directories (INI, 10-minute SMET, SNO) for pi-snow and batch (G1c, G2)
//! pi-snow-gpu-alt-control cpu-grid --forcing DIR --ini TEMPLATE --pixels FILE --threads T --out JSON
//!                the CPU stepper on grid pixels, no output: physics-only CPU cost (G2)
//! pi-snow-gpu-alt-control cpu-unit <ini> <end>                         CPU stepper on one unit (G0)
//! pi-snow-gpu-alt-control dev-lockstep <columns> <days> <from> <to> [variant]  lockstep, synthetic
//! pi-snow-gpu-alt-control dev-compare <columns> <days> <dir> [mix]     GPU vs CPU files, synthetic
//! pi-snow-gpu-alt-control identity [F64|F32|F32K]                      compile and report default PTX
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use pi_snow_core::PhysicsOptions;
use pi_snow_gpu_alternatives::gpu::{Gpu, Precision};
use pi_snow_gpu_alternatives::runner::{self, ColumnInput};
use pi_snow_gpu_alternatives::{lockstep, stepper, synth, unit};

const NMAX: usize = 2048;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("batch") => pi_snow_gpu_alternatives::batch::main(&args[2..]),
        Some("cpu-marker-control") => cpu_marker_control(&args[2..]),
        Some("cpu-unit") => cpu_unit(&args[2..]),
        Some("lockstep") => units_lockstep(&args[2..]),
        Some("grid") => grid(&args[2..]),
        Some("write-units") => write_units(&args[2..]),
        Some("cpu-grid") => cpu_grid(&args[2..]),
        Some("dev-lockstep") => dev_lockstep(&args[2..]),
        Some("dev-compare") => dev_compare(&args[2..]),
        Some("identity") => identity(&args[2..]),
        _ => bail!("usage: see the module documentation"),
    }
}

fn identity(args: &[String]) -> Result<()> {
    let precision = match args.first().map(String::as_str).unwrap_or("F64") {
        "F64" => Precision::F64,
        "F32" => Precision::F32,
        "F32K" => Precision::F32K,
        other => bail!("unsupported identity precision {other}"),
    };
    let gpu = Gpu::new(precision)?;
    println!("{}", gpu.build.to_json());
    Ok(())
}

fn cpu_marker_control(args: &[String]) -> Result<()> {
    let (parsed, extras) = pi_snow_gpu_alternatives::batch::parse_args_with(args, &["--out"])?;
    let out = extras.iter().find(|(k, _)| k == "--out").map(|(_, v)| PathBuf::from(v))
        .context("--out is required")?;
    let options = PhysicsOptions::default();
    let mut steps = 0u64;
    let mut count_change_steps = 0u64;
    let mut marker_difference_steps = 0u64;
    let mut marker_differences = 0u64;
    let mut grain_flips = 0u64;
    let mut grain_flip_steps = 0u64;
    let mut hardness_flips = 0u64;
    let mut hardness_flip_steps = 0u64;
    let mut first_count_change = None;
    for job in &parsed.jobs {
        let (u, rows) = pi_snow_gpu_alternatives::batch::load(job)?;
        let mut base = stepper::Stepper::new(&u.cfg, &options, &u.site, u.start.clone())?;
        let mut control = stepper::Stepper::new_marker_control(&u.cfg, &options, &u.site, u.start.clone())?;
        for row in &rows {
            let _ = base.step(row)?;
            let _ = control.step(row)?;
            steps += 1;
            if base.column.elements.len() != control.column.elements.len() {
                count_change_steps += 1;
                if first_count_change.is_none() {
                    first_count_change = Some(format!("{} {} baseline={} control={}", u.site.station_id,
                        row.t.format("%Y-%m-%dT%H:%M:%S"), base.column.elements.len(), control.column.elements.len()));
                }
            }
            let mut marker_bad = false;
            let mut grain_bad = false;
            let mut hardness_bad = false;
            for (a, b) in base.column.elements.iter().zip(&control.column.elements) {
                if a.micro.marker != b.micro.marker {
                    marker_bad = true;
                    marker_differences += 1;
                }
            }
            let ag = stepper::grain_hardness_values(&base.column, &options);
            let bg = stepper::grain_hardness_values(&control.column, &options);
            for (a, b) in ag.iter().zip(&bg) {
                if a.0 != b.0 { grain_bad = true; grain_flips += 1; }
                if a.1 != b.1 { hardness_bad = true; hardness_flips += 1; }
            }
            marker_difference_steps += marker_bad as u64;
            grain_flip_steps += grain_bad as u64;
            hardness_flip_steps += hardness_bad as u64;
        }
    }
    let first = first_count_change.map(|v| format!("\"{}\"", v.replace('\\', "\\\\").replace('"', "\\\"")))
        .unwrap_or_else(|| "null".into());
    std::fs::write(out, format!(
        "{{\"units\": {}, \"steps\": {steps}, \"count_change_steps\": {count_change_steps}, \"first_count_change\": {first}, \"marker_difference_steps\": {marker_difference_steps}, \"marker_element_differences\": {marker_differences}, \"grain_0513_flips\": {grain_flips}, \"grain_0513_flip_steps\": {grain_flip_steps}, \"hardness_0534_flips\": {hardness_flips}, \"hardness_0534_flip_steps\": {hardness_flip_steps}, \"control\": {{\"melted_out_ice\": 1e-4, \"refrozen_mass_fraction\": 1e-3}}}}\n",
        parsed.jobs.len()))?;
    Ok(())
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
        .map(|(site, rows)| ColumnInput { cfg: &cfg, site, start: synth::bare(site), rows, profile_times: None })
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
        .map(|(site, rows)| ColumnInput { cfg: &cfg, site, start: synth::bare(site), rows, profile_times: None })
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
    let (parsed, extras) = pi_snow_gpu_alternatives::batch::parse_args_with(args, &["--from", "--to", "--out"])?;
    let get = |k: &str| extras.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone()).with_context(|| format!("{k} is required"));
    let from: usize = get("--from")?.parse()?;
    let to: usize = get("--to")?.parse()?;
    let out = PathBuf::from(get("--out")?);
    let mut loaded = Vec::new();
    for job in &parsed.jobs {
        loaded.push(pi_snow_gpu_alternatives::batch::load(job)?);
    }
    let inputs: Vec<ColumnInput> = loaded
        .iter()
        .map(|(u, rows)| ColumnInput { cfg: &u.cfg, site: &u.site, start: u.start.clone(), rows: rows.clone(), profile_times: None })
        .collect();
    let gpu = Gpu::new(parsed.precision)?;
    let started = std::time::Instant::now();
    let tally = if matches!(parsed.precision, Precision::F32 | Precision::F32K) {
        lockstep::run_development(&gpu, &inputs, parsed.nmax, from, to)?
    } else {
        lockstep::run(&gpu, &inputs, parsed.nmax, from, to)?
    };
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

/// `--name value` options, each required once.
fn options(args: &[String], names: &[&str]) -> Result<std::collections::HashMap<String, String>> {
    let mut out = std::collections::HashMap::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if !names.contains(&a.as_str()) {
            bail!("unknown option {a:?}");
        }
        let v = it.next().with_context(|| format!("{a} needs a value"))?;
        if out.insert(a.clone(), v.clone()).is_some() {
            bail!("{a} given twice");
        }
    }
    Ok(out)
}

/// A file of pixel indices, one per line (positions in pixels.tsv).
fn read_indices(path: &str) -> Result<Vec<usize>> {
    std::fs::read_to_string(path)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Ok(l.trim().parse::<usize>()?))
        .collect()
}

fn grid(args: &[String]) -> Result<()> {
    let o = options(args, &["--forcing", "--ini", "--pixels", "--batch", "--out", "--precision", "--nmax", "--threads"])?;
    let get = |k: &str| o.get(k).cloned().with_context(|| format!("{k} is required"));
    let precision = match o.get("--precision").map(String::as_str).unwrap_or("F64") {
        "F64" => Precision::F64,
        "MIX" => Precision::Mix,
        "F32" => Precision::F32,
        "F32K" => Precision::F32K,
        other => bail!("bad precision {other}"),
    };
    let args = pi_snow_gpu_alternatives::grid::GridArgs {
        forcing_dir: PathBuf::from(get("--forcing")?),
        ini_template: PathBuf::from(get("--ini")?),
        pixels: read_indices(&get("--pixels")?)?,
        batch: get("--batch")?.parse()?,
        nmax: o.get("--nmax").map(|s| s.parse()).transpose()?.unwrap_or(NMAX),
        out_dir: PathBuf::from(get("--out")?),
        threads: o.get("--threads").map(|s| s.parse()).transpose()?.unwrap_or(1),
    };
    let gpu = Gpu::new(precision)?;
    pi_snow_gpu_alternatives::grid::run(&gpu, &args)
}

fn write_units(args: &[String]) -> Result<()> {
    use pi_snow_gpu_alternatives::forcing;
    let o = options(args, &["--forcing", "--ini", "--pixels", "--out"])?;
    let get = |k: &str| o.get(k).cloned().with_context(|| format!("{k} is required"));
    let dir = PathBuf::from(get("--forcing")?);
    let pixels = forcing::read_pixels(&dir.join("pixels.tsv"))?;
    let template = std::fs::read_to_string(get("--ini")?)?;
    let out = PathBuf::from(get("--out")?);
    let mut f = std::fs::File::open(dir.join("forcing.f32"))?;
    let mut list = String::new();
    for i in read_indices(&get("--pixels")?)? {
        let p = &pixels[i];
        let h = forcing::hourly_forcing(&forcing::read_hourly(&mut f, i)?)?;
        let rows = forcing::expand(&h, 1, forcing::HOURS - 1);
        let unit = out.join(&p.key);
        std::fs::create_dir_all(unit.join("input"))?;
        std::fs::write(unit.join("snowpack.ini"), forcing::ini_text(&template, &p.key)?)?;
        std::fs::write(unit.join("input").join(format!("{}.smet", p.key)), forcing::smet_text(p, &rows))?;
        std::fs::write(unit.join("input").join(format!("{}.sno", p.key)), forcing::sno_text(p))?;
        list.push_str(&format!("{}\t{}\n", unit.display(), rows.last().unwrap().t.format("%Y-%m-%dT%H:%M")));
    }
    std::fs::write(out.join("units.list"), list)?;
    Ok(())
}

fn cpu_grid(args: &[String]) -> Result<()> {
    use pi_snow_gpu_alternatives::forcing;
    use pi_snow_io::ini::{Ini, RunConfig};
    let o = options(args, &["--forcing", "--ini", "--pixels", "--threads", "--out"])?;
    let get = |k: &str| o.get(k).cloned().with_context(|| format!("{k} is required"));
    let dir = PathBuf::from(get("--forcing")?);
    let pixels = forcing::read_pixels(&dir.join("pixels.tsv"))?;
    let cfg = RunConfig::from_ini(&Ini::parse(&forcing::ini_text(&std::fs::read_to_string(get("--ini")?)?, "px")?)?)?;
    let indices = read_indices(&get("--pixels")?)?;
    let threads: usize = get("--threads")?.parse()?;
    let options = PhysicsOptions::default();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let started = std::time::Instant::now();
    let per: Vec<(usize, f64, usize)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads.max(1))
            .map(|_| {
                s.spawn(|| -> Result<Vec<(usize, f64, usize)>> {
                    let mut f = std::fs::File::open(dir.join("forcing.f32"))?;
                    let mut mine = Vec::new();
                    loop {
                        let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&i) = indices.get(k) else { break };
                        let p = &pixels[i];
                        let site = p.site()?;
                        let rows = forcing::expand(&forcing::hourly_forcing(&forcing::read_hourly(&mut f, i)?)?, 1, forcing::HOURS - 1);
                        let t = std::time::Instant::now();
                        let mut st = stepper::Stepper::new(&cfg, &options, &site, pi_snow_core::Column::bare_ground(pi_snow_core::constants::T_MELT, site.slope_angle, site.slope_azimuth))?;
                        for r in &rows {
                            if let stepper::Outcome::Advanced { output: true } = st.step(r)? {
                                st.acc = stepper::Accum::default();
                            }
                        }
                        mine.push((i, t.elapsed().as_secs_f64(), st.column.elements.len()));
                    }
                    Ok(mine)
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap().unwrap()).collect()
    });
    let wall = started.elapsed().as_secs_f64();
    let body: Vec<String> = per.iter().map(|(i, s, n)| format!("[{i}, {s:.6}, {n}]")).collect();
    std::fs::write(
        get("--out")?,
        format!("{{\"threads\": {threads}, \"wall_seconds\": {wall:.3}, \"columns\": {}, \"per_column\": [{}]}}\n", per.len(), body.join(", ")),
    )?;
    Ok(())
}
