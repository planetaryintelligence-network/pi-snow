//! G1a: the CPU stepper and the kernel advance one step from the same state, and the
//! two results are compared; the GPU is then reset to the CPU's state so differences
//! cannot accumulate (plan.md section 5).

use std::fmt::Write as _;

use anyhow::{Result, bail};
use pi_snow_core::PhysicsOptions;

use crate::gpu::{self, Batch, ELEMENT_FIELDS, F_COUNT, Gpu, HostState};
use crate::runner::{ColumnInput, pack_rows};
use crate::stepper::{Outcome, Stepper};

/// Agreement scale per element field, as registered in plan.md G1a.
pub const ELEMENT_SCALE: [f64; 11] = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1.0, 1.0, 1.0];
pub const NODE_SCALE: f64 = 1.0;
pub const BUFFER_SCALE: f64 = 1e-3;
pub const ACC_SCALE: f64 = 1.0;
pub const REL_TOL: f64 = 1e-9;

pub fn agrees(g: f64, c: f64, scale: f64) -> bool {
    if g.to_bits() == c.to_bits() || (g.is_nan() && c.is_nan()) {
        return true;
    }
    (g - c).abs() <= REL_TOL * c.abs().max(scale)
}

#[derive(Default)]
pub struct Tally {
    pub steps: u64,
    pub comparisons: u64,
    pub agreeing: u64,
    pub bitwise: u64,
    pub count_mismatch_steps: u64,
    pub marker_mismatch_steps: u64,
    pub outcome_mismatch_steps: u64,
    pub mo_pass_mismatch_steps: u64,
    /// Steps whose final surface budget took more than one Monin-Obukhov pass, per side.
    pub mo_multi_pass_steps_gpu: u64,
    pub mo_multi_pass_steps_cpu: u64,
    pub max_rel: f64,
    pub listed: Vec<String>,
}

const LIST_LIMIT: usize = 2000;

impl Tally {
    fn compare(&mut self, what: &str, g: f64, c: f64, scale: f64, ctx: &str) {
        self.comparisons += 1;
        if g.to_bits() == c.to_bits() {
            self.bitwise += 1;
        }
        if agrees(g, c, scale) {
            self.agreeing += 1;
            if !(g.is_nan() || c.is_nan()) {
                let rel = (g - c).abs() / c.abs().max(scale);
                self.max_rel = self.max_rel.max(rel);
            }
        } else if self.listed.len() < LIST_LIMIT {
            self.listed.push(format!("{ctx}\t{what}\tgpu={g:e}\tcpu={c:e}"));
        }
    }

    pub fn to_json(&self) -> String {
        let mut s = String::new();
        let _ = write!(
            s,
            "{{\"steps\": {}, \"comparisons\": {}, \"agreeing\": {}, \"bitwise\": {}, \"count_mismatch_steps\": {}, \"marker_mismatch_steps\": {}, \"outcome_mismatch_steps\": {}, \"mo_pass_mismatch_steps\": {}, \"mo_multi_pass_steps_gpu\": {}, \"mo_multi_pass_steps_cpu\": {}, \"max_rel_among_agreeing\": {:e}, \"listed\": [{}]}}",
            self.steps,
            self.comparisons,
            self.agreeing,
            self.bitwise,
            self.count_mismatch_steps,
            self.marker_mismatch_steps,
            self.outcome_mismatch_steps,
            self.mo_pass_mismatch_steps,
            self.mo_multi_pass_steps_gpu,
            self.mo_multi_pass_steps_cpu,
            self.max_rel,
            self.listed
                .iter()
                .map(|l| format!("\"{}\"", l.replace('\\', "\\\\").replace('"', "'")))
                .collect::<Vec<_>>()
                .join(", ")
        );
        s
    }
}

/// Run steps `[from, to)` of every column in lockstep. The CPU is advanced from the
/// season start to `from` first (no comparison there).
pub fn run(
    gpu: &Gpu,
    inputs: &[ColumnInput],
    nmax: usize,
    from: usize,
    to: usize,
) -> Result<Tally> {
    let options = PhysicsOptions::default();
    let b = inputs.len();
    let mut steppers = inputs
        .iter()
        .map(|i| Stepper::new(i.cfg, &options, i.site, i.start.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let k = gpu::run_constants(inputs[0].cfg, &steppers[0].constants);
    let mut par = vec![0.0; gpu::P_COUNT * b];
    for (c, (i, s)) in inputs.iter().zip(&steppers).enumerate() {
        if gpu::run_constants(i.cfg, &s.constants).map(f64::to_bits) != k.map(f64::to_bits) {
            bail!("columns must share run constants");
        }
        gpu::check_supported(i.cfg, &s.constants, &i.start)?;
        for (p, x) in gpu::column_params(i.site, &s.constants).iter().enumerate() {
            par[p * b + c] = *x;
        }
    }
    // Fast-forward the CPU.
    for (s, i) in steppers.iter_mut().zip(inputs) {
        for row in i.rows.iter().take(from) {
            if let Outcome::Advanced { output: true } = s.step(row)? {
                let _ = s.take_output_row(row.t);
            }
        }
    }
    let mut batch = Batch::new(gpu, b, nmax, 1, 0, &k, &par)?;
    let mut host = HostState::new(b, nmax);
    let mut tally = Tally::default();
    let end = to.min(inputs.iter().map(|i| i.rows.len()).max().unwrap_or(0));
    for r in from..end {
        // CPU state into the host buffer.
        for (c, s) in steppers.iter().enumerate() {
            host.set_column(c, &s.column)?;
            host.buffer[c] = s.deposition.buffered();
            host.set_accum(c, &s.acc);
            host.set_last(c, &s.last);
            host.has_prev[c] = s.previous.is_some() as i32;
            host.prev_t[c] = s.previous.map(|t| t.timestamp()).unwrap_or(0);
            host.first_out[c] = s.first_output as i32;
            host.steps[c] = s.steps as i64;
            host.fail[c] = 0;
            host.out_ready[c] = 0;
            host.cursor[c] = 0;
        }
        batch.upload(gpu, &host)?;
        let slices: Vec<&[crate::stepper::ForcingRow]> = inputs
            .iter()
            .map(|i| if r < i.rows.len() { &i.rows[r..r + 1] } else { &i.rows[0..0] })
            .collect();
        let (ft, fv, rows) = pack_rows(&slices);
        debug_assert_eq!(fv.len(), rows.max(1) * F_COUNT * b);
        batch.upload_forcing(gpu, &ft, &fv)?;
        batch.advance(gpu, rows, 1, 0, 0)?;
        let upto = batch.counts(gpu)?.into_iter().max().unwrap_or(0).max(0) as usize;
        batch.download(gpu, &mut host, upto.min(nmax))?;
        for (c, s) in steppers.iter_mut().enumerate() {
            if r >= inputs[c].rows.len() {
                continue;
            }
            let row = &inputs[c].rows[r];
            let cpu = s.step(row);
            let ctx = format!(
                "{}\t{}\tmo_passes gpu={} cpu={}",
                inputs[c].site.station_id,
                row.t.format("%Y-%m-%dT%H:%M"),
                host.mo_passes[c],
                s.mo_passes
            );
            tally.steps += 1;
            if host.mo_passes[c] as u32 != s.mo_passes {
                tally.mo_pass_mismatch_steps += 1;
            }
            tally.mo_multi_pass_steps_gpu += (host.mo_passes[c] > 1) as u64;
            tally.mo_multi_pass_steps_cpu += (s.mo_passes > 1) as u64;
            let gpu_failed = host.fail[c] != 0;
            let (cpu_output, cpu_failed) = match &cpu {
                Ok(Outcome::Advanced { output }) => (*output, false),
                Ok(_) => (false, false),
                Err(_) => (false, true),
            };
            if gpu_failed != cpu_failed || (host.out_ready[c] != 0) != cpu_output {
                tally.outcome_mismatch_steps += 1;
                if tally.listed.len() < LIST_LIMIT {
                    tally.listed.push(format!(
                        "{ctx}\toutcome\tgpu fail={} out={}\tcpu {:?}",
                        host.fail[c], host.out_ready[c], cpu.as_ref().map_err(|e| e.to_string())
                    ));
                }
            }
            if cpu_failed {
                bail!("CPU column {} failed at {}: {:?}", inputs[c].site.station_id, row.t, cpu.err());
            }
            let n = s.column.elements.len();
            if host.n[c] as usize != n {
                tally.count_mismatch_steps += 1;
                if tally.listed.len() < LIST_LIMIT {
                    tally.listed.push(format!("{ctx}\tcount\tgpu={}\tcpu={n}", host.n[c]));
                }
            } else {
                let mut marker_bad = false;
                for (i, e) in s.column.elements.iter().enumerate() {
                    let v = gpu::element_values(e);
                    for (f, x) in v.iter().enumerate() {
                        let g = host.elem[f][i * b + c];
                        tally.compare(
                            &format!("{}[{i}]", ELEMENT_FIELDS[f]),
                            g,
                            *x,
                            ELEMENT_SCALE[f],
                            &ctx,
                        );
                    }
                    if host.marker[i * b + c] != e.micro.marker as u32 {
                        marker_bad = true;
                    }
                }
                if marker_bad {
                    tally.marker_mismatch_steps += 1;
                }
                for (j, t) in s.column.node_temperature.iter().enumerate() {
                    tally.compare(&format!("T[{j}]"), host.t[j * b + c], *t, NODE_SCALE, &ctx);
                }
            }
            tally.compare("buffer", host.buffer[c], s.deposition.buffered(), BUFFER_SCALE, &ctx);
            let ga = host.accum(c).to_array();
            for (q, x) in s.acc.to_array().iter().enumerate() {
                tally.compare(&format!("acc[{q}]"), ga[q], *x, ACC_SCALE, &ctx);
            }
            if cpu_output {
                let _ = s.take_output_row(row.t);
            }
        }
    }
    Ok(tally)
}
