//! The grid season on the GPU (G2): many columns per batch, one day of forcing per
//! launch, daily fields recorded on the device at 12:00 UTC (plan.md sections 3.4, 5).
//! Each batch's outputs are written atomically, so a stopped run resumes at the first
//! unfinished batch.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Result, bail};
use pi_snow_core::PhysicsOptions;
use pi_snow_core::constants::T_MELT;
use pi_snow_io::ini::{Ini, RunConfig};

use crate::forcing::{self, HOURS, Hourly, Pixel};
use crate::gpu::{self, Batch, D_COUNT, F_COUNT, Gpu, HostState, ROW_ABSENT};
use crate::runner::pack_rows;
use crate::stepper::{Constants, ForcingRow};

pub const DAYS: usize = 181;

pub struct GridArgs {
    pub forcing_dir: PathBuf,
    pub ini_template: PathBuf,
    pub pixels: Vec<usize>,
    pub batch: usize,
    pub nmax: usize,
    pub out_dir: PathBuf,
    pub threads: usize,
    /// The `nonfinite_scan` block (A2.2) and, when `--batch auto` chose the width, the `batch_auto` block (A2.3),
    /// as JSON object text; recorded in every batch's JSON.
    pub nonfinite_scan_json: String,
    pub batch_auto_json: Option<String>,
}

/// Rows of season day `day` (0-based): the hourly intervals ending in that day, and the
/// boundary row on day 0.
pub fn day_hours(day: usize) -> (usize, usize) {
    let first = day * 24 + 1;
    let last = (day * 24 + 24).min(HOURS - 1);
    (first, last)
}

fn par_map<T: Send, R: Send>(items: Vec<T>, threads: usize, f: impl Fn(T) -> R + Sync) -> Vec<R> {
    let threads = threads.max(1);
    let n = items.len();
    let mut slots: Vec<Option<R>> = (0..n).map(|_| None).collect();
    let items: Vec<(usize, T)> = items.into_iter().enumerate().collect();
    let chunk = n.div_ceil(threads).max(1);
    let mut groups: Vec<Vec<(usize, T)>> = Vec::new();
    let mut it = items.into_iter();
    loop {
        let g: Vec<(usize, T)> = it.by_ref().take(chunk).collect();
        if g.is_empty() {
            break;
        }
        groups.push(g);
    }
    let f = &f;
    let results: Vec<Vec<(usize, R)>> = std::thread::scope(|s| {
        let handles: Vec<_> = groups
            .into_iter()
            .map(|g| s.spawn(move || g.into_iter().map(|(i, x)| (i, f(x))).collect::<Vec<_>>()))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (i, r) in results.into_iter().flatten() {
        slots[i] = Some(r);
    }
    slots.into_iter().map(|x| x.unwrap()).collect()
}

/// Write `bytes` to `path` through a temporary name and a rename.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("partial");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn after<'a>(text: &'a str, marker: &str) -> Option<&'a str> {
    text.find(marker).map(|i| &text[i + marker.len()..])
}

/// The string value of `"key": "..."` first found after `marker` in a hand-built JSON text.
fn string_after(text: &str, marker: &str, key: &str) -> Option<String> {
    let rest = after(after(text, marker)?, &format!("\"{key}\": \""))?;
    Some(rest[..rest.find('"')?].to_string())
}

fn columns_of(text: &str) -> Option<usize> {
    let rest = after(text, "\"columns\": ")?;
    rest[..rest.find(|c: char| !c.is_ascii_digit())?].parse().ok()
}

/// The setting, kernel and precision a batch JSON (or this run) records. A directory written before the settings
/// existed has no `nonfinite_scan` block: it ran the pinned kernel.
fn identity_of(batch_json: &str, engine_marker: &str) -> (String, String, String) {
    let setting = string_after(batch_json, "\"nonfinite_scan\": {", "setting").unwrap_or_else(|| "(absent)".into());
    let kernel = string_after(batch_json, "\"nonfinite_scan\": {", "kernel").unwrap_or_else(|| "pinned".into());
    let precision = string_after(batch_json, engine_marker, "precision").unwrap_or_else(|| "?".into());
    (setting, kernel, precision)
}

/// Refuse to resume an output directory whose finished batches were written at another width or with another
/// setting: a restart skips every batch whose JSON exists, so a different width (the `auto` budget moves) would drop
/// columns without an error, and a different kernel or precision would mix engines in one directory.
pub fn check_resume(out_dir: &Path, n_pixels: usize, batch: usize, nonfinite_scan_json: &str, engine_json: &str) -> Result<()> {
    if !out_dir.is_dir() {
        return Ok(());
    }
    let mine = identity_of(&format!("\"nonfinite_scan\": {nonfinite_scan_json}, \"engine\": {engine_json}"), "\"engine\": {");
    let mut names: Vec<_> = std::fs::read_dir(out_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("batch-") && n.ends_with(".json"))
        .collect();
    names.sort();
    for name in names {
        let Ok(bi) = name["batch-".len()..name.len() - ".json".len()].parse::<usize>() else { continue };
        let text = std::fs::read_to_string(out_dir.join(&name))?;
        let (columns, theirs) = (columns_of(&text), identity_of(&text, "\"engine\": {"));
        let expected = n_pixels.saturating_sub(bi * batch).min(batch);
        let width_ok = columns == Some(expected) && expected > 0;
        // A directory from before the settings existed has no setting to compare: it ran the pinned kernel, so only
        // the kernel and precision must agree.
        let same_engine = theirs.1 == mine.1 && theirs.2 == mine.2 && (theirs.0 == "(absent)" || theirs.0 == mine.0);
        if !width_ok || !same_engine {
            bail!(
                "{} was written by a run with {} columns in {name} and setting {}, kernel {}, precision {}; this run has width {batch}, \
                 setting {}, kernel {}, precision {} and would expect {expected} columns there. Resuming would skip batches and drop or mix \
                 columns: pass --batch {} (and the same --nonfinite-scan and --precision) to resume, or use a new --out",
                out_dir.display(),
                columns.map_or("?".to_string(), |c| c.to_string()),
                theirs.0, theirs.1, theirs.2,
                mine.0, mine.1, mine.2,
                columns.map_or("N".to_string(), |c| c.to_string())
            );
        }
    }
    Ok(())
}

pub fn run(gpu: &Gpu, args: &GridArgs) -> Result<()> {
    let options = PhysicsOptions::default();
    let pixels = forcing::read_pixels(&args.forcing_dir.join("pixels.tsv"))?;
    let template = std::fs::read_to_string(&args.ini_template)?;
    let cfg = RunConfig::from_ini(&Ini::parse(&forcing::ini_text(&template, "px")?)?)?;
    check_resume(&args.out_dir, args.pixels.len(), args.batch, &args.nonfinite_scan_json, &gpu.build.to_json())?;
    std::fs::create_dir_all(&args.out_dir)?;
    let forcing_path = args.forcing_dir.join("forcing.f32");
    for (bi, chunk) in args.pixels.chunks(args.batch).enumerate() {
        let stem = args.out_dir.join(format!("batch-{bi:04}"));
        if stem.with_extension("json").exists() {
            eprintln!("batch {bi}: done, skipped");
            continue;
        }
        let t0 = Instant::now();
        let chosen: Vec<&Pixel> = chunk.iter().map(|&i| &pixels[i]).collect();
        let sites = chosen.iter().map(|p| p.site()).collect::<Result<Vec<_>>>()?;
        let b = chosen.len();
        let constants: Vec<Constants> = sites.iter().map(|s| Constants::new(&cfg, &options, s)).collect();
        let k = gpu::run_constants(&cfg, &constants[0])?;
        let mut par = vec![0.0; gpu::P_COUNT * b];
        for (c, (s, cs)) in sites.iter().zip(&constants).enumerate() {
            for (p, x) in gpu::column_params(s, cs).iter().enumerate() {
                par[p * b + c] = *x;
            }
        }
        // Hourly series, gap-filled, read and built in parallel.
        let hourly: Vec<Result<Hourly>> = par_map(chunk.to_vec(), args.threads, |i| {
            let mut f = File::open(&forcing_path)?;
            forcing::hourly_forcing(&forcing::read_hourly(&mut f, i)?)
        });
        let hourly = hourly.into_iter().collect::<Result<Vec<_>>>()?;
        let t_read = t0.elapsed().as_secs_f64();

        let (free_before, total) = gpu.mem_info()?;
        let mut batch = Batch::new(gpu, b, args.nmax, 145, DAYS, &k, &par)?;
        let (free_after, _) = gpu.mem_info()?;
        let mut host = HostState::new(b, args.nmax);
        for (c, s) in sites.iter().enumerate() {
            let col = pi_snow_core::Column::bare_ground(T_MELT, s.slope_angle, s.slope_azimuth);
            host.set_column(c, &col)?;
        }
        batch.upload(gpu, &host)?;
        let day0 = forcing::season_start().timestamp();
        let (mut t_prep, mut t_gpu) = (0.0, 0.0);
        for day in 0..DAYS {
            let tp = Instant::now();
            let (first, last) = day_hours(day);
            let rows: Vec<Vec<ForcingRow>> =
                par_map(hourly.iter().collect(), args.threads, |h| forcing::expand(h, first, last));
            let refs: Vec<&[ForcingRow]> = rows.iter().map(|r| r.as_slice()).collect();
            let (ft, fv, n_rows) = pack_rows(&refs);
            debug_assert_eq!(fv.len(), n_rows * F_COUNT * b);
            if ft.iter().any(|&t| t == ROW_ABSENT) {
                bail!("day {day}: a column has fewer rows than the others");
            }
            t_prep += tp.elapsed().as_secs_f64();
            let tg = Instant::now();
            batch.upload_forcing(gpu, &ft, &fv)?;
            batch.reset_cursors(gpu)?;
            batch.advance(gpu, n_rows, n_rows, 1, day0)?;
            batch.synchronize(gpu)?;
            t_gpu += tg.elapsed().as_secs_f64();
        }
        let daily = batch.daily(gpu)?;
        batch.download(gpu, &mut host, 0)?;
        let mut bytes = Vec::with_capacity(daily.len() * 8);
        for x in &daily {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        let mut tsv = String::from("c\tindex\tkey\tmax_elements\tfail\tfail_t\tsteps\tfinal_elements\n");
        for (c, p) in chosen.iter().enumerate() {
            tsv.push_str(&format!(
                "{c}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                p.index, p.key, host.maxn[c], host.fail[c], host.fail_t[c], host.steps[c], host.n[c]
            ));
        }
        write_atomic(&stem.with_extension("daily.f64"), &bytes)?;
        write_atomic(&stem.with_extension("tsv"), tsv.as_bytes())?;
        let wall = t0.elapsed().as_secs_f64();
        let json = format!(
            "{{\"batch\": {bi}, \"columns\": {b}, \"nmax\": {}, \"days\": {DAYS}, \"daily_layout\": \"little-endian f64 [day][HS, SWE, TSS, water, n, fail][column]\", \"seconds\": {{\"wall\": {wall:.3}, \"read_and_fill\": {t_read:.3}, \"host_rows\": {t_prep:.3}, \"gpu_upload_and_kernel\": {t_gpu:.3}}}, \"device_bytes_batch\": {}, \"device_free_before\": {free_before}, \"device_free_after_alloc\": {free_after}, \"device_total\": {total}, \"engine\": {}, \"nonfinite_scan\": {}, \"batch_auto\": {}}}\n",
            args.nmax,
            batch.device_bytes(),
            gpu.build.to_json(),
            args.nonfinite_scan_json,
            args.batch_auto_json.as_deref().unwrap_or("null")
        );
        write_atomic(&stem.with_extension("json"), json.as_bytes())?;
        eprintln!(
            "batch {bi}: {b} columns, wall {wall:.1} s (gpu {t_gpu:.1}, host rows {t_prep:.1}, read {t_read:.1})"
        );
        let _ = D_COUNT;
    }
    Ok(())
}
