//! Columns with hourly output on the GPU: each column's whole forcing season is resident,
//! the kernel advances every column to its next output step, and the host formats the
//! output from the downloaded state with the CPU's own output code (`stepper`).

use anyhow::{Result, bail};
use chrono::{DateTime, Utc};
use pi_snow_core::{Column, PhysicsOptions};
use pi_snow_io::ini::RunConfig;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::a6_gpu::{self, Batch, F_COUNT, Gpu, HostState, ROW_ABSENT};
use crate::stepper::{self, Constants, ForcingRow, SeasonOutput, SeasonWriter, Site};

/// One column to run: its configuration, site, start state and forcing rows (already
/// cut at the end time).
pub struct ColumnInput<'a> {
    pub cfg: &'a RunConfig,
    pub site: &'a Site,
    pub start: Column,
    pub rows: Vec<ForcingRow>,
    /// Write a PRO profile only at these output times (Unix seconds); `None` writes one
    /// at every output step, as `pi-snow` does.
    pub profile_times: Option<std::collections::HashSet<i64>>,
}

/// What became of one column.
pub struct ColumnResult {
    pub output: SeasonOutput,
    pub steps: i64,
    pub max_elements: i32,
    /// `None` when the column ran to its end.
    pub failure: Option<String>,
    /// At every 12:00 UTC output: time and HS, SWE, TSS, liquid water, element count,
    /// summed as the kernel's grid mode sums them (the G2 determinism check).
    pub daily: Vec<(i64, [f64; 5])>,
}

pub struct ColumnSummary {
    pub steps: i64,
    pub max_elements: i32,
    pub failure: Option<String>,
    pub daily: Vec<(i64, [f64; 5])>,
}

pub struct OutputTargets {
    pub smet: PathBuf,
    pub pro: PathBuf,
}

/// The grid mode's daily fields from a rebuilt column (same summation order).
pub fn daily_fields(c: &Column) -> [f64; 5] {
    [
        c.snow_height(),
        c.swe(),
        *c.node_temperature.last().unwrap(),
        c.liquid_water(),
        c.elements.len() as f64,
    ]
}

fn nan_or(x: Option<f64>) -> f64 {
    x.unwrap_or(f64::NAN)
}

/// Forcing rows in the kernel layout, padded with absent rows to the longest column.
pub fn pack_rows(rows: &[&[ForcingRow]]) -> (Vec<i64>, Vec<f64>, usize) {
    let b = rows.len();
    let r_max = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut ft = vec![ROW_ABSENT; r_max.max(1) * b];
    let mut fv = vec![f64::NAN; r_max.max(1) * F_COUNT * b];
    for (c, col) in rows.iter().enumerate() {
        for (r, row) in col.iter().enumerate() {
            ft[r * b + c] = row.t.timestamp();
            let v = [
                nan_or(row.ta),
                nan_or(row.rh),
                nan_or(row.vw),
                nan_or(row.dw),
                nan_or(row.iswr),
                nan_or(row.ilwr),
                nan_or(row.psum),
                nan_or(row.tsg),
            ];
            for (f, x) in v.iter().enumerate() {
                fv[(r * F_COUNT + f) * b + c] = *x;
            }
        }
    }
    (ft, fv, r_max)
}

/// The engine refuses forcing values that are present but NaN in a way the kernel's
/// NaN-for-missing encoding could not represent; SMET has no NaN, so this never fires
/// on parsed files.
fn check_rows(rows: &[ForcingRow]) -> Result<()> {
    for r in rows {
        for x in [r.ta, r.rh, r.vw, r.dw, r.iswr, r.ilwr, r.psum, r.tsg].into_iter().flatten() {
            if x.is_nan() {
                bail!("a NaN forcing value at {} cannot be told from a missing one", r.t);
            }
        }
    }
    Ok(())
}

/// Run every column to its end with hourly output, all columns in one batch. All
/// columns must share one run configuration (the kernel's run constants).
pub fn run_hourly(
    gpu: &Gpu,
    inputs: &[ColumnInput],
    nmax: usize,
    options: &PhysicsOptions,
) -> Result<Vec<ColumnResult>> {
    run_hourly_inner(gpu, inputs, nmax, options, None)
}

pub fn run_hourly_streaming(
    gpu: &Gpu,
    inputs: &[ColumnInput],
    nmax: usize,
    options: &PhysicsOptions,
    targets: &[OutputTargets],
) -> Result<Vec<ColumnSummary>> {
    if targets.len() != inputs.len() {
        bail!("one output target is required per input column");
    }
    Ok(run_hourly_inner(gpu, inputs, nmax, options, Some(targets))?
        .into_iter()
        .map(|r| ColumnSummary {
            steps: r.steps,
            max_elements: r.max_elements,
            failure: r.failure,
            daily: r.daily,
        })
        .collect())
}

fn run_hourly_inner(
    gpu: &Gpu,
    inputs: &[ColumnInput],
    nmax: usize,
    options: &PhysicsOptions,
    output_targets: Option<&[OutputTargets]>,
) -> Result<Vec<ColumnResult>> {
    if *options != PhysicsOptions::default() {
        bail!("the GPU engine runs the default physics only");
    }
    let b = inputs.len();
    if b == 0 {
        return Ok(Vec::new());
    }
    let constants: Vec<Constants> = inputs
        .iter()
        .map(|i| Constants::new(i.cfg, options, i.site))
        .collect();
    let k = a6_gpu::run_constants(inputs[0].cfg, &constants[0])?;
    for (i, c) in inputs.iter().zip(&constants) {
        if a6_gpu::run_constants(i.cfg, c)?.map(f64::to_bits) != k.map(f64::to_bits) {
            bail!("columns of one batch must share run constants ({})", i.site.station_id);
        }
        a6_gpu::check_supported(i.cfg, c, &i.start)?;
        check_rows(&i.rows)?;
    }
    let mut par = vec![0.0; a6_gpu::P_COUNT * b];
    for (c, (i, cs)) in inputs.iter().zip(&constants).enumerate() {
        for (p, x) in a6_gpu::column_params(i.site, cs).iter().enumerate() {
            par[p * b + c] = *x;
        }
    }
    let row_refs: Vec<&[ForcingRow]> = inputs.iter().map(|i| i.rows.as_slice()).collect();
    let (ft, fv, rows) = pack_rows(&row_refs);
    let mut batch = Batch::new(gpu, b, nmax, rows, 0, &k, &par)?;
    let mut host = HostState::new(b, nmax);
    for (c, i) in inputs.iter().enumerate() {
        host.set_column(c, &i.start)?;
    }
    batch.upload(gpu, &host)?;
    batch.upload_forcing(gpu, &ft, &fv)?;

    let mut writers: Vec<SeasonWriter> = inputs.iter().map(|i| SeasonWriter::new(i.site)).collect();
    let mut pro_files = output_targets
        .map(|targets| {
            targets
                .iter()
                .zip(inputs)
                .map(|(target, input)| {
                    let partial = partial_path(&target.pro);
                    if let Some(parent) = partial.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    let mut file = std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&partial)?;
                    let site = input.site;
                    let header = pi_snow_io::pro::write_header(&pi_snow_io::pro::ProHeader {
                        station_name: site.station_name.clone(),
                        latitude: site.latitude,
                        longitude: site.longitude,
                        altitude: site.altitude,
                        slope_angle_deg: site.slope_angle,
                        slope_azimuth_deg: site.slope_azimuth,
                    });
                    file.write_all(header.as_bytes())?;
                    Ok(Some((partial, file)))
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let mut daily: Vec<Vec<(i64, [f64; 5])>> = vec![Vec::new(); b];
    let mut upto = inputs.iter().map(|i| i.start.elements.len()).max().unwrap_or(0);
    loop {
        batch.advance(gpu, rows, usize::MAX, 0, 0)?;
        let (cursor, ready, fail) = batch.flags(gpu)?;
        let finished = (0..b).all(|c| ready[c] == 0 && (fail[c] != 0 || cursor[c] as usize >= rows));
        if ready.iter().any(|&r| r != 0) {
            let counts = batch.counts(gpu)?;
            upto = upto.max(counts.iter().copied().max().unwrap_or(0) as usize);
            batch.download(gpu, &mut host, upto)?;
            for c in 0..b {
                if ready[c] == 0 {
                    continue;
                }
                let t: DateTime<Utc> = inputs[c].rows[cursor[c] as usize - 1].t;
                let site = inputs[c].site;
                let column = host.column(c, site.slope_angle, site.slope_azimuth);
                let l = host.last(c);
                let row = stepper::output_row(
                    t,
                    &host.accum(c),
                    &column,
                    l.ta,
                    l.rh,
                    l.vw,
                    l.dw,
                    l.tsg,
                    options,
                );
                if t.timestamp() % 86_400 == 43_200 {
                    daily[c].push((t.timestamp(), daily_fields(&column)));
                }
                let with_profile =
                    inputs[c].profile_times.as_ref().is_none_or(|k| k.contains(&t.timestamp()));
                if let Some(files) = &mut pro_files {
                    if with_profile {
                        let profile = stepper::profile(t, &column, options);
                        files[c]
                            .as_mut()
                            .unwrap()
                            .1
                            .write_all(pi_snow_io::pro::write_timestep(&profile).as_bytes())?;
                    }
                    writers[c].push_parts(row, t, &column, options, false);
                } else {
                    writers[c].push_parts(row, t, &column, options, with_profile);
                }
            }
        }
        if finished {
            break;
        }
    }
    batch.download(gpu, &mut host, upto)?;
    let mut out = Vec::with_capacity(b);
    for (c, ((w, i), d)) in writers.into_iter().zip(inputs).zip(daily).enumerate() {
        let failure = match host.fail[c] {
            0 => None,
            a6_gpu::FAIL_CRAZY => Some(format!(
                "T_CRAZY at {}",
                DateTime::from_timestamp(host.fail_t[c], 0).unwrap_or_default()
            )),
            a6_gpu::FAIL_OVERFLOW => Some(format!(
                "OVERFLOW (N_MAX {nmax}) at {}",
                DateTime::from_timestamp(host.fail_t[c], 0).unwrap_or_default()
            )),
            other => Some(format!("failure code {other}")),
        };
        let output = w.finish(i.site, host.steps[c] as usize);
        if let Some(targets) = output_targets {
            let (pro_partial, mut pro_file) = pro_files.as_mut().unwrap()[c].take().unwrap();
            if failure.is_some() {
                drop(pro_file);
                std::fs::remove_file(pro_partial)?;
            } else {
                pro_file.flush()?;
                drop(pro_file);
                let smet_partial = partial_path(&targets[c].smet);
                if let Some(parent) = smet_partial.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let mut smet_file = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&smet_partial)?;
                smet_file.write_all(output.smet.as_bytes())?;
                smet_file.flush()?;
                drop(smet_file);
                std::fs::rename(smet_partial, &targets[c].smet)?;
                std::fs::rename(pro_partial, &targets[c].pro)?;
            }
        }
        out.push(ColumnResult {
            output: if output_targets.is_some() {
                SeasonOutput {
                    smet: String::new(),
                    pro: String::new(),
                    steps: output.steps,
                }
            } else {
                output
            },
            steps: host.steps[c],
            max_elements: host.maxn[c],
            failure,
            daily: d,
        });
    }
    Ok(out)
}

fn partial_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("output"))
        .to_os_string();
    name.push(format!(".partial.{}", std::process::id()));
    path.with_file_name(name)
}
