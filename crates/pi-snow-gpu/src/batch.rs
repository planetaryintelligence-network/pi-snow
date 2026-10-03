//! `pi-snow-gpu batch`: `pi-snow batch`'s interface on the GPU.
//!
//! ```text
//! pi-snow-gpu batch [--jobs N] [-e END] [--list FILE] [--report FILE]
//!                   [--precision F64|MIX|F32|F32K] [--nmax N] [UNIT_DIR]...
//! ```
//!
//! Same unit directories (`snowpack.ini`, `input/<id>.smet`, `input/<id>.sno`), same
//! `--list` lines (`UNIT_DIR<TAB>END`), same SMET and PRO files written where each INI
//! says, so the engine-benchmark referee can swap the binary unchanged
//! (experiments/engine-gpu-v1/plan.md section 8). `--jobs` is accepted and ignored: all
//! units of one run configuration go to the GPU as one batch. Default physics only:
//! `--physics` is refused, and so is any INI the kernel does not honour as verified
//! (beyond what `pi-snow` itself refuses): `ATMOSPHERIC_STABILITY` other than
//! `MO_HOLTSLAG`, or a new-snow density model other than `LEHNING_NEW`.
//! A unit that fails is reported and writes nothing, as with `pi-snow`; the others run.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use pi_snow_core::PhysicsOptions;
use pi_snow_io::ini::{Ini, NewSnowDensity, RunConfig, Stability};

use crate::gpu::{self, Gpu, Precision};
use crate::runner::{self, ColumnInput};
use crate::stepper::{Constants, ForcingIndex};
use crate::unit::{self, Unit, parse_end};

pub struct Job {
    pub unit_dir: PathBuf,
    pub end: DateTime<Utc>,
}

pub struct Args {
    pub report: Option<PathBuf>,
    /// Also write `<station>.daily.tsv` (the 12:00 UTC fields, exact) into this directory.
    pub daily_out: Option<PathBuf>,
    /// `STATION<TAB>YYYY-MM-DDTHH:MM` lines: write PRO profiles only at those times for
    /// the stations listed (others keep every profile). Not a `pi-snow` option.
    pub profile_times: Option<PathBuf>,
    pub precision: Precision,
    pub nmax: usize,
    pub jobs: Vec<Job>,
}

pub fn parse_args(args: &[String]) -> Result<Args> {
    parse_args_with(args, &[]).map(|(a, _)| a)
}

/// As [`parse_args`], also collecting the values of `extra` options (each takes one value).
pub fn parse_args_with(args: &[String], extra: &[&str]) -> Result<(Args, Vec<(String, String)>)> {
    let mut extras = Vec::new();
    let mut end: Option<DateTime<Utc>> = None;
    let (mut report, mut list, mut daily_out, mut profile_times) = (None, None, None, None);
    let mut precision = Precision::F64;
    let mut nmax = 2048usize;
    let mut dirs = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |what: &str| it.next().with_context(|| format!("{what} needs a value"));
        match a.as_str() {
            "--jobs" => {
                value("--jobs")?.parse::<usize>().context("--jobs needs a number")?;
            }
            "-e" => end = Some(parse_end(value("-e")?)?),
            "--report" => report = Some(PathBuf::from(value("--report")?)),
            "--list" => list = Some(PathBuf::from(value("--list")?)),
            "--daily-out" => daily_out = Some(PathBuf::from(value("--daily-out")?)),
            "--profile-times" => profile_times = Some(PathBuf::from(value("--profile-times")?)),
            "--precision" => {
                precision = match value("--precision")?.as_str() {
                    "F64" => Precision::F64,
                    "MIX" => Precision::Mix,
                    "F32" => Precision::F32,
                    "F32K" => Precision::F32K,
                    other => bail!("--precision is F64, MIX, F32 or F32K, not {other:?}"),
                }
            }
            "--nmax" => nmax = value("--nmax")?.parse().context("--nmax needs a number")?,
            e if extra.contains(&e) => {
                let v = value(e)?.clone();
                extras.push((e.to_string(), v));
            }
            "--physics" => bail!("pi-snow-gpu runs the default physics only; --physics is refused"),
            a if a.starts_with("--physics") => bail!("--physics is refused"),
            a if a.starts_with('-') => bail!("unknown option {a:?} for `pi-snow-gpu batch`"),
            dir => dirs.push(PathBuf::from(dir)),
        }
    }
    let mut jobs = Vec::new();
    if let Some(list) = list {
        let text = std::fs::read_to_string(&list).with_context(|| list.display().to_string())?;
        for (n, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let (dir, e) = line
                .split_once('\t')
                .with_context(|| format!("{}:{}: expected UNIT_DIR<TAB>END", list.display(), n + 1))?;
            jobs.push(Job {
                unit_dir: PathBuf::from(dir),
                end: parse_end(e.trim()).with_context(|| format!("{}:{}", list.display(), n + 1))?,
            });
        }
    }
    if !dirs.is_empty() {
        let end = end.context("unit directories need -e END")?;
        jobs.extend(dirs.into_iter().map(|unit_dir| Job { unit_dir, end }));
    }
    let mut seen = std::collections::HashSet::new();
    for job in &jobs {
        let key = std::fs::canonicalize(&job.unit_dir).unwrap_or_else(|_| job.unit_dir.clone());
        if !seen.insert(key) {
            bail!("unit directory {} is listed twice", job.unit_dir.display());
        }
    }
    if jobs.is_empty() {
        bail!("usage: pi-snow-gpu batch [-e END] [--list FILE] [--report FILE] [UNIT_DIR]...");
    }
    Ok((Args { report, daily_out, profile_times, precision, nmax, jobs }, extras))
}

/// Refusals beyond `pi-snow`'s own (plan.md section 8).
pub fn check_unit(u: &Unit) -> Result<()> {
    check_config(&u.cfg)
}

fn check_config(cfg: &RunConfig) -> Result<()> {
    if cfg.stability != Stability::MoHoltslag {
        bail!("ATMOSPHERIC_STABILITY other than MO_HOLTSLAG is not verified on the GPU");
    }
    if cfg.new_snow_density != NewSnowDensity::LehningNew {
        bail!("new-snow density other than LEHNING_NEW is not verified on the GPU");
    }
    Ok(())
}

fn validate_ini(job: &Job) -> Result<()> {
    let path = job.unit_dir.join("snowpack.ini");
    let source = std::fs::read_to_string(&path).with_context(|| path.display().to_string())?;
    let cfg = RunConfig::from_ini(&Ini::parse(&source)?)?;
    check_config(&cfg)
}

pub struct UnitReport {
    pub unit_dir: PathBuf,
    pub station_id: Option<String>,
    pub steps: i64,
    pub max_elements: i32,
    pub error: Option<String>,
}

fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

pub fn report_json(reports: &[UnitReport], build: &gpu::Build, wall: f64) -> String {
    let units: Vec<String> = reports
        .iter()
        .map(|r| {
            format!(
                "{{\"unit\":{},\"station\":{},\"status\":{},\"steps\":{},\"max_elements\":{},\"error\":{}}}",
                json_string(&r.unit_dir.display().to_string()),
                r.station_id.as_deref().map_or("null".into(), json_string),
                json_string(if r.error.is_none() { "ok" } else { "error" }),
                r.steps,
                r.max_elements,
                r.error.as_deref().map_or("null".into(), json_string),
            )
        })
        .collect();
    format!(
        "{{\"version\":{},\"engine\":{},\"wall_seconds\":{:.6},\"units\":[{}]}}\n",
        json_string(env!("CARGO_PKG_VERSION")),
        build.to_json(),
        wall,
        units.join(",")
    )
}

pub fn load(job: &Job) -> Result<(Unit, Vec<crate::stepper::ForcingRow>)> {
    let u = unit::load(&job.unit_dir.join("snowpack.ini"))?;
    check_unit(&u)?;
    let index = ForcingIndex::of(&u.forcing)?;
    let rows = u
        .forcing
        .rows
        .iter()
        .take_while(|r| r.timestamp <= job.end)
        .map(|r| index.row(r))
        .collect();
    Ok((u, rows))
}

/// Run every job; returns one report per job, in job order.
fn read_profile_times(path: &Path) -> Result<std::collections::HashMap<String, std::collections::HashSet<i64>>> {
    let mut out: std::collections::HashMap<String, std::collections::HashSet<i64>> = Default::default();
    for (n, line) in std::fs::read_to_string(path)?.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let (station, t) = line
            .split_once('\t')
            .with_context(|| format!("{}:{}: expected STATION<TAB>TIME", path.display(), n + 1))?;
        out.entry(station.to_string()).or_default().insert(parse_end(t.trim())?.timestamp());
    }
    Ok(out)
}

pub fn run(
    gpu: &Gpu,
    jobs: &[Job],
    nmax: usize,
    daily_out: Option<&Path>,
    profile_times: Option<&Path>,
) -> Result<Vec<UnitReport>> {
    let times = profile_times.map(read_profile_times).transpose()?;
    let options = PhysicsOptions::default();
    let mut reports: Vec<UnitReport> = jobs
        .iter()
        .map(|j| UnitReport {
            unit_dir: j.unit_dir.clone(),
            station_id: None,
            steps: 0,
            max_elements: 0,
            error: None,
        })
        .collect();
    let mut loaded: Vec<(usize, Unit, Vec<crate::stepper::ForcingRow>)> = Vec::new();
    for (i, job) in jobs.iter().enumerate() {
        match load(job) {
            Ok((u, rows)) => {
                reports[i].station_id = Some(u.site.station_id.clone());
                loaded.push((i, u, rows));
            }
            Err(e) => reports[i].error = Some(format!("{e:#}")),
        }
    }
    // One GPU batch per distinct set of run constants.
    let mut groups: Vec<([u64; gpu::K_COUNT], Vec<usize>)> = Vec::new();
    for (pos, (i, u, _)) in loaded.iter().enumerate() {
        let k = gpu::run_constants(&u.cfg, &Constants::new(&u.cfg, &options, &u.site)).map(f64::to_bits);
        match groups.iter_mut().find(|(g, _)| *g == k) {
            Some((_, members)) => members.push(pos),
            None => groups.push((k, vec![pos])),
        }
        let _ = i;
    }
    for (_, members) in groups {
        let inputs: Vec<ColumnInput> = members
            .iter()
            .map(|&p| {
                let (_, u, rows) = &loaded[p];
                ColumnInput {
                    cfg: &u.cfg,
                    site: &u.site,
                    start: u.start.clone(),
                    rows: rows.clone(),
                    profile_times: times.as_ref().and_then(|m| m.get(&u.site.station_id).cloned()),
                }
            })
            .collect();
        let results = match runner::run_hourly(gpu, &inputs, nmax, &options) {
            Ok(r) => r,
            Err(e) => {
                for &p in &members {
                    reports[loaded[p].0].error = Some(format!("{e:#}"));
                }
                continue;
            }
        };
        for (&p, r) in members.iter().zip(results) {
            let (i, u, _) = &loaded[p];
            let rep = &mut reports[*i];
            rep.steps = r.steps;
            rep.max_elements = r.max_elements;
            if let Some(f) = r.failure {
                rep.error = Some(f);
                continue;
            }
            if let Err(e) = write_outputs(u, &r.output.smet, &r.output.pro) {
                rep.error = Some(format!("{e:#}"));
            }
            if let Some(dir) = daily_out {
                let mut text = String::from("t\tHS\tSWE\tTSS\twater\tn\n");
                for (t, f) in &r.daily {
                    text.push_str(&format!("{t}\t{:?}\t{:?}\t{:?}\t{:?}\t{:?}\n", f[0], f[1], f[2], f[3], f[4]));
                }
                std::fs::create_dir_all(dir)?;
                std::fs::write(dir.join(format!("{}.daily.tsv", u.site.station_id)), text)?;
            }
        }
    }
    Ok(reports)
}

fn write_outputs(u: &Unit, smet: &str, pro: &str) -> Result<()> {
    let stem = unit::output_stem(u);
    std::fs::create_dir_all(stem.parent().unwrap_or(Path::new(".")))?;
    std::fs::write(stem.with_extension("smet"), smet)?;
    std::fs::write(stem.with_extension("pro"), pro)?;
    Ok(())
}

/// Entry point; `args` excludes the program and `batch`.
pub fn main(args: &[String]) -> Result<()> {
    let parsed = parse_args(args)?;
    // Refuse unsupported configuration before CUDA initialization, so an invalid INI
    // produces the same actionable error on hosts without a usable GPU.
    for job in &parsed.jobs {
        validate_ini(job)?;
    }
    let gpu = Gpu::new(parsed.precision)?;
    let started = std::time::Instant::now();
    let reports = run(
        &gpu,
        &parsed.jobs,
        parsed.nmax,
        parsed.daily_out.as_deref(),
        parsed.profile_times.as_deref(),
    )?;
    let wall = started.elapsed().as_secs_f64();
    if let Some(path) = &parsed.report {
        std::fs::write(path, report_json(&reports, &gpu.build, wall))
            .with_context(|| path.display().to_string())?;
    }
    let failed = reports.iter().filter(|r| r.error.is_some()).count();
    eprintln!(
        "pi-snow-gpu batch {} ({}): {} units in {:.3} s, {} failed",
        env!("CARGO_PKG_VERSION"),
        gpu.build.precision.label(),
        reports.len(),
        wall,
        failed
    );
    for r in reports.iter().filter(|r| r.error.is_some()) {
        eprintln!("  {}: {}", r.unit_dir.display(), r.error.as_deref().unwrap_or(""));
    }
    if failed > 0 {
        bail!("{failed} of {} units failed", reports.len());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn physics_flags_are_refused() {
        assert!(parse_args(&args(&["-e", "2026-01-01T00:00", "--physics", "S5=NONE", "u"])).is_err());
        assert!(parse_args(&args(&["-e", "2026-01-01T00:00", "--physics=S5=NONE", "u"])).is_err());
    }

    #[test]
    fn jobs_is_accepted_and_precision_parsed() {
        let a = parse_args(&args(&["--jobs", "8", "--precision", "MIX", "-e", "2026-01-01T00:00", "u"])).unwrap();
        assert_eq!(a.precision, Precision::Mix);
        assert_eq!(a.jobs.len(), 1);
    }

    #[test]
    fn unverified_ini_modes_are_refused_before_the_gpu_runs() {
        // Inside the workspace's own target/, never the system temporary directory.
        let dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/test-scratch"))
            .join(format!("batch-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("input")).unwrap();
        std::fs::write(
            dir.join("input/SYN.smet"),
            "SMET 1.1 ASCII\n[HEADER]\nstation_id = SYN\nlatitude = 40.6\nlongitude = -111.6\naltitude = 2600\nnodata = -999\nfields = timestamp TA RH VW DW ISWR ILWR PSUM\n[DATA]\n2025-11-01T00:00 270 0.8 2 270 0 250 0\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("input/SYN.sno"),
            "SMET 1.1 ASCII\n[HEADER]\nstation_id = SYN\nProfileDate = 2025-11-01T00:00\nHS_Last = 0\nSlopeAngle = 0\nSlopeAzi = 0\nnSoilLayerData = 0\nnSnowLayerData = 0\nfields = timestamp Layer_Thick T Vol_Frac_I Vol_Frac_W Vol_Frac_V Vol_Frac_S Rho_S Conduc_S HeatCapac_S rg rb dd sp mk mass_hoar ne CDot metamo\n[DATA]\n",
        )
        .unwrap();

        let cases = [
            (
                crate::synth::INI.replace("MO_HOLTSLAG", "RICHARDSON"),
                "MO_HOLTSLAG",
            ),
            (
                format!(
                    "{}\n[SnowpackAdvanced]\nHN_DENSITY_MODEL = ZWART\n",
                    crate::synth::INI
                ),
                "LEHNING_NEW",
            ),
            (
                format!("{}\n[Snowpack]\nSW_MODE = REFLECTED\n", crate::synth::INI),
                "SW_MODE",
            ),
            (
                format!(
                    "{}\n[SnowpackAdvanced]\nWATERTRANSPORTMODEL_SNOW = RICHARDSEQUATION\n",
                    crate::synth::INI
                ),
                "WATERTRANSPORTMODEL_SNOW",
            ),
        ];
        let job = Job {
            unit_dir: dir.clone(),
            end: parse_end("2026-01-01T00:00").unwrap(),
        };
        for (ini, key) in cases {
            std::fs::write(dir.join("snowpack.ini"), ini).unwrap();
            let validation = validate_ini(&job).unwrap_err();
            assert!(validation.to_string().contains(key), "{key}: {validation:#}");
            let error = match load(&job) {
                Ok(_) => panic!("batch accepted an unverified INI value for {key}"),
                Err(error) => error,
            };
            assert!(error.to_string().contains(key), "{key}: {error:#}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
