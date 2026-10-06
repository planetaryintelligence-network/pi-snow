//! `pi-snow batch`: many columns in one process, spread over threads.
//!
//! ```text
//! pi-snow batch --jobs N [-e END] [--list FILE] [--report FILE]
//!               [--physics KEY=VALUE]... [--merra2-deposition CSV] [UNIT_DIR]...
//! ```
//!
//! A unit directory is laid out as `rust_engine_arm_season` stages it:
//! `snowpack.ini` and `input/`; output goes where the INI says, relative to the
//! unit. `UNIT_DIR` arguments take the end time from `-e`; `--list FILE` has one
//! `UNIT_DIR<TAB>END` per line, for sets whose stations end at different times.
//! Units run in the order given, taken from one shared queue, so the same
//! `--physics` flags and the same code as `pi-snow -c ... -e ...` apply to
//! each. A column shares nothing with another: its state is local to
//! `engine::run_with_options`. A unit that fails, or panics, is recorded and the
//! others still run and write; the exit status is non-zero if any failed.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use pi_snow_core::PhysicsOptions;

use crate::unit::{parse_end, parse_physics, run_unit_with_deposition};
use pi_snow_physics::chan_albedo::DepositionSeries;

/// The number of worker threads `run_batch` will start.
pub fn effective_threads(jobs: usize, units: usize) -> usize {
    jobs.clamp(1, units.max(1))
}

/// Progress line on stderr. A closed stderr must not kill a worker and with it the batch.
fn log(line: std::fmt::Arguments) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr().lock(), "{line}");
}

/// Most threads `--jobs` may ask for; a typo should not exhaust the process's threads.
const MAX_JOBS: usize = 1024;

/// Stack of each worker thread.
const WORKER_STACK_BYTES: usize = 8 << 20;

pub struct Job {
    pub unit_dir: PathBuf,
    pub end: DateTime<Utc>,
}

pub struct UnitReport {
    pub unit_dir: PathBuf,
    pub station_id: Option<String>,
    pub steps: usize,
    pub seconds: f64,
    /// `None` when the unit ran to the end.
    pub error: Option<String>,
}

/// Run `jobs` on `threads` threads, results in the order of `jobs`.
pub fn run_batch(jobs: &[Job], physics: &PhysicsOptions, threads: usize) -> Vec<UnitReport> {
    run_batch_with_deposition(jobs, physics, threads, None)
}

pub fn run_batch_with_deposition(
    jobs: &[Job],
    physics: &PhysicsOptions,
    threads: usize,
    merra2: Option<&DepositionSeries>,
) -> Vec<UnitReport> {
    let next = AtomicUsize::new(0);
    let threads = effective_threads(threads, jobs.len());
    let mut done: Vec<(usize, UnitReport)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                // 8 MiB, the main thread's default, not std's 2 MiB for spawned threads:
                // a column must not fail on a thread that would have run as a process.
                std::thread::Builder::new()
                    .stack_size(WORKER_STACK_BYTES)
                    .spawn_scoped(scope, || {
                        let mut mine = Vec::new();
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            let Some(job) = jobs.get(i) else { break };
                            let report = run_job(job, physics, merra2);
                            match &report.error {
                                None => log(format_args!(
                                    "pi-snow batch: {} ok, {} steps in {:.3} s",
                                    job.unit_dir.display(),
                                    report.steps,
                                    report.seconds
                                )),
                                Some(e) => log(format_args!(
                                    "pi-snow batch: {} FAILED: {e}",
                                    job.unit_dir.display()
                                )),
                            }
                            mine.push((i, report));
                        }
                        mine
                    })
                    .expect("cannot spawn a batch worker thread")
            })
            .collect();
        workers
            .into_iter()
            // `run_job` catches unit panics, so a worker panicking is a bug here.
            .flat_map(|w| w.join().expect("batch worker panicked outside a unit"))
            .collect()
    });
    done.sort_by_key(|(i, _)| *i);
    done.into_iter().map(|(_, r)| r).collect()
}

fn run_job(job: &Job, physics: &PhysicsOptions, merra2: Option<&DepositionSeries>) -> UnitReport {
    let ini = job.unit_dir.join("snowpack.ini");
    let started = std::time::Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        run_unit_with_deposition(&ini, job.end, physics, merra2)
    }));
    let failed = |error: String| UnitReport {
        unit_dir: job.unit_dir.clone(),
        station_id: None,
        steps: 0,
        seconds: started.elapsed().as_secs_f64(),
        error: Some(error),
    };
    match outcome {
        Ok(Ok(run)) => UnitReport {
            unit_dir: job.unit_dir.clone(),
            station_id: Some(run.station_id),
            steps: run.steps,
            seconds: run.seconds,
            error: None,
        },
        Ok(Err(e)) => failed(format!("{e:#}")),
        Err(panic) => failed(match panic.downcast_ref::<String>() {
            Some(s) => format!("panic: {s}"),
            None => match panic.downcast_ref::<&str>() {
                Some(s) => format!("panic: {s}"),
                None => "panic".to_string(),
            },
        }),
    }
}

fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The per-unit report as JSON, in job order.
pub fn report_json(reports: &[UnitReport], threads: usize, wall_seconds: f64) -> String {
    let units: Vec<String> = reports
        .iter()
        .map(|r| {
            format!(
                "{{\"unit\":{},\"station\":{},\"status\":{},\"steps\":{},\"seconds\":{:.6},\"error\":{}}}",
                json_string(&r.unit_dir.display().to_string()),
                r.station_id
                    .as_deref()
                    .map_or("null".to_string(), json_string),
                json_string(if r.error.is_none() { "ok" } else { "error" }),
                r.steps,
                r.seconds,
                r.error.as_deref().map_or("null".to_string(), json_string),
            )
        })
        .collect();
    format!(
        "{{\"version\":{},\"threads\":{},\"wall_seconds\":{:.6},\"units\":[{}]}}\n",
        json_string(env!("CARGO_PKG_VERSION")),
        threads,
        wall_seconds,
        units.join(",")
    )
}

struct Args {
    jobs: usize,
    report: Option<PathBuf>,
    merra2_deposition: Option<PathBuf>,
    units: Vec<Job>,
}

fn parse_args(args: &[String]) -> Result<Args> {
    let mut jobs = 1usize;
    let mut seen_jobs = false;
    let mut end: Option<DateTime<Utc>> = None;
    let mut report = None;
    let mut merra2_deposition = None;
    let mut list = None;
    let mut dirs: Vec<PathBuf> = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |what: &str| it.next().with_context(|| format!("{what} needs a value"));
        match a.as_str() {
            "--jobs" => {
                if seen_jobs {
                    bail!("--jobs given twice");
                }
                seen_jobs = true;
                jobs = value("--jobs")?.parse().context("--jobs needs a number")?;
                if !(1..=MAX_JOBS).contains(&jobs) {
                    bail!("--jobs must be between 1 and {MAX_JOBS}");
                }
            }
            "-e" => {
                if end.is_some() {
                    bail!("-e given twice");
                }
                end = Some(parse_end(value("-e")?)?);
            }
            "--report" => {
                if report.is_some() {
                    bail!("--report given twice");
                }
                report = Some(path_value(value("--report")?, "--report")?);
            }
            "--list" => {
                if list.is_some() {
                    bail!("--list given twice");
                }
                list = Some(path_value(value("--list")?, "--list")?);
            }
            "--merra2-deposition" => {
                if merra2_deposition.is_some() {
                    bail!("--merra2-deposition given twice");
                }
                merra2_deposition = Some(path_value(
                    value("--merra2-deposition")?,
                    "--merra2-deposition",
                )?);
            }
            // `parse_physics` reads these from the whole argument list.
            "--physics" => {
                value("--physics")?;
            }
            a if a.starts_with('-') => bail!("unknown option {a:?} for `pi-snow batch`"),
            dir => dirs.push(PathBuf::from(dir)),
        }
    }
    let mut units = Vec::new();
    if let Some(list) = list {
        units.extend(read_list(&list)?);
    }
    if !dirs.is_empty() {
        let end = end.context("unit directories need -e END")?;
        units.extend(dirs.into_iter().map(|unit_dir| Job { unit_dir, end }));
    }
    // Two units writing one directory would truncate each other's files and
    // both report success.
    let mut seen = std::collections::HashSet::new();
    for job in &units {
        let key = std::fs::canonicalize(&job.unit_dir).unwrap_or_else(|_| job.unit_dir.clone());
        if !seen.insert(key) {
            bail!("unit directory {} is listed twice", job.unit_dir.display());
        }
    }
    if units.is_empty() {
        bail!(
            "usage: pi-snow batch --jobs N [-e END] [--list FILE] [--physics K=V]... [UNIT_DIR]..."
        );
    }
    Ok(Args {
        jobs,
        report,
        merra2_deposition,
        units,
    })
}

fn path_value(value: &str, option: &str) -> Result<PathBuf> {
    if value.starts_with('-') {
        bail!("{option} needs a path, got {value:?}");
    }
    Ok(PathBuf::from(value))
}

fn read_list(path: &Path) -> Result<Vec<Job>> {
    let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
    parse_list(&text, &path.display().to_string())
}

fn parse_list(text: &str, source: &str) -> Result<Vec<Job>> {
    let mut jobs = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let (dir, end) = line
            .split_once('\t')
            .with_context(|| format!("{source}:{}: expected UNIT_DIR<TAB>END", n + 1))?;
        jobs.push(Job {
            unit_dir: PathBuf::from(dir),
            end: parse_end(end.trim()).with_context(|| format!("{source}:{}", n + 1))?,
        });
    }
    Ok(jobs)
}

/// Entry point for `pi-snow batch`; `args` excludes the program and `batch`.
pub fn main(args: &[String]) -> Result<()> {
    let physics = parse_physics(args)?;
    let parsed = parse_args(args)?;
    let started = std::time::Instant::now();
    let merra2 = parsed
        .merra2_deposition
        .as_ref()
        .map(|path| {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading MERRA-2 deposition CSV {}", path.display()))?;
            DepositionSeries::from_csv(&text).map_err(anyhow::Error::msg)
        })
        .transpose()?;
    let reports = run_batch_with_deposition(&parsed.units, &physics, parsed.jobs, merra2.as_ref());
    let wall = started.elapsed().as_secs_f64();
    if let Some(path) = &parsed.report {
        std::fs::write(
            path,
            report_json(
                &reports,
                effective_threads(parsed.jobs, reports.len()),
                wall,
            ),
        )
        .with_context(|| path.display().to_string())?;
    }
    let failed = reports.iter().filter(|r| r.error.is_some()).count();
    log(format_args!(
        "pi-snow batch {}: {} units on {} threads in {:.3} s, {} failed",
        env!("CARGO_PKG_VERSION"),
        reports.len(),
        effective_threads(parsed.jobs, reports.len()),
        wall,
        failed
    ));
    if failed > 0 {
        bail!("{failed} of {} units failed", reports.len());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_escapes_quotes_and_control_characters() {
        assert_eq!(json_string("a\"b\\c\nd\u{1}"), "\"a\\\"b\\\\c\\nd\\u0001\"");
    }

    #[test]
    fn options_are_refused_when_unknown_or_incomplete() {
        let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(parse_args(&a(&["--jobs", "0", "-e", "2026-01-01T00:00", "u"])).is_err());
        assert!(parse_args(&a(&["--bogus", "u"])).is_err());
        assert!(parse_args(&a(&["u"])).is_err(), "a unit needs an end time");
        assert!(parse_args(&a(&["--jobs", "2"])).is_err(), "no units");
        let ok = parse_args(&a(&[
            "--jobs",
            "3",
            "-e",
            "2026-01-01T00:00",
            "--physics",
            "S5=NONE",
            "u1",
            "u2",
        ]))
        .unwrap();
        assert_eq!((ok.jobs, ok.units.len()), (3, 2));
    }

    #[test]
    fn repeated_units_and_options_are_refused() {
        let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let e = "2026-01-01T00:00";
        assert!(
            parse_args(&a(&["-e", e, "u1", "u1"])).is_err(),
            "same unit twice"
        );
        assert!(parse_args(&a(&["-e", e, "-e", e, "u1"])).is_err());
        assert!(parse_args(&a(&["--jobs", "2", "--jobs", "3", "-e", e, "u1"])).is_err());
        assert!(parse_args(&a(&["-e", e, "--report", "-e", "u1"])).is_err());
        assert!(parse_args(&a(&["--jobs", "100000", "-e", e, "u1"])).is_err());
    }

    #[test]
    fn list_files_tolerate_blank_lines_crlf_and_spaces() {
        let text = "/data/unit one\t2026-01-01T00:00\r\n\r\n\n/data/u2\t 2026-02-01T12:30 \n";
        let jobs = parse_list(text, "list").unwrap();
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0].unit_dir, PathBuf::from("/data/unit one"));
        assert_eq!(jobs[1].end, parse_end("2026-02-01T12:30").unwrap());
        assert!(parse_list("/data/u1 2026-01-01T00:00\n", "list").is_err());
    }

    #[test]
    fn threads_are_clamped_to_the_units() {
        assert_eq!(effective_threads(9, 1), 1);
        assert_eq!(effective_threads(2, 5), 2);
        assert_eq!(effective_threads(4, 0), 1);
    }

    #[test]
    fn a_missing_unit_is_reported_and_does_not_stop_the_queue() {
        let end = parse_end("2026-01-01T00:00").unwrap();
        let jobs: Vec<Job> = (0..3)
            .map(|i| Job {
                unit_dir: PathBuf::from(format!("/nonexistent/unit{i}")),
                end,
            })
            .collect();
        let reports = run_batch(&jobs, &PhysicsOptions::default(), 2);
        assert_eq!(reports.len(), 3);
        for (i, r) in reports.iter().enumerate() {
            assert_eq!(r.unit_dir, jobs[i].unit_dir, "reports keep job order");
            assert!(r.error.is_some());
        }
    }
}
