//! G1a: the CPU stepper and the kernel advance one step from the same state, and the
//! two results are compared; the GPU is then reset to the CPU's state so differences
//! cannot accumulate (plan.md section 5).

use std::fmt::Write as _;

use anyhow::{bail, Result};
use pi_snow_core::PhysicsOptions;

use crate::gpu::{self, Batch, Gpu, HostState, ELEMENT_FIELDS, F_COUNT};
use crate::runner::{pack_rows, ColumnInput};
use crate::stepper::{Outcome, Stepper};

/// Agreement scale per element field, as registered in plan.md G1a.
pub const ELEMENT_SCALE: [f64; 11] = [
    1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1.0, 1.0, 1.0,
];
pub const NODE_SCALE: f64 = 1.0;
pub const BUFFER_SCALE: f64 = 1e-3;
pub const ACC_SCALE: f64 = 1.0;
pub const REL_TOL: f64 = 1e-9;

pub fn agrees(g: f64, c: f64, scale: f64) -> bool {
    agrees_with(g, c, scale, REL_TOL)
}

fn agrees_with(g: f64, c: f64, scale: f64, rel_tol: f64) -> bool {
    if g.to_bits() == c.to_bits() || (g.is_nan() && c.is_nan()) {
        return true;
    }
    (g - c).abs() <= rel_tol * c.abs().max(scale)
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
    pub marker_activated_gpu: u64,
    pub marker_activated_cpu: u64,
    pub marker_activation_disagreements: u64,
    pub grain_0513_flips: u64,
    pub grain_0513_flip_steps: u64,
    pub hardness_0534_flips: u64,
    pub hardness_0534_flip_steps: u64,
    pub tolerance_site_steps: [u64; 6],
    pub count_change_causes: [u64; 3],
    pub count_change_events: Vec<String>,
    pub mo_pass_histogram_gpu: Vec<u64>,
    pub mo_pass_histogram_cpu: Vec<u64>,
    pub rel_tol: f64,
    pub scale_factor: f64,
    pub max_rel: f64,
    pub listed: Vec<String>,
    pub table_flux_samples: u64,
    pub loop_flux_samples: u64,
    pub flux_gate_failures: u64,
    pub candidate_refusals: u64,
    pub max_abs_qs_w_m2: f64,
    pub max_abs_ql_w_m2: f64,
}

const LIST_LIMIT: usize = 2000;

fn json_quote(value: &str) -> String {
    let mut out = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl Tally {
    fn compare(&mut self, what: &str, g: f64, c: f64, scale: f64, ctx: &str) {
        self.comparisons += 1;
        if g.to_bits() == c.to_bits() {
            self.bitwise += 1;
        }
        if agrees_with(g, c, scale * self.scale_factor, self.rel_tol) {
            self.agreeing += 1;
            if !(g.is_nan() || c.is_nan()) {
                let rel = (g - c).abs() / c.abs().max(scale * self.scale_factor);
                self.max_rel = self.max_rel.max(rel);
            }
        } else if self.listed.len() < LIST_LIMIT {
            self.listed
                .push(format!("{ctx}\t{what}\tgpu={g:e}\tcpu={c:e}"));
        }
    }

    pub fn to_json(&self) -> String {
        let mut s = String::new();
        let _ = write!(
            s,
            "{{\"steps\": {}, \"comparisons\": {}, \"agreeing\": {}, \"bitwise\": {}, \"count_mismatch_steps\": {}, \"count_change_causes\": {{\"deposition\": {}, \"merge\": {}, \"melt_out\": {}}}, \"count_change_events\": [{}], \"marker_mismatch_steps\": {}, \"outcome_mismatch_steps\": {}, \"mo_pass_mismatch_steps\": {}, \"mo_multi_pass_steps_gpu\": {}, \"mo_multi_pass_steps_cpu\": {}, \"mo_pass_histogram_gpu\": {:?}, \"mo_pass_histogram_cpu\": {:?}, \"marker_activated_gpu\": {}, \"marker_activated_cpu\": {}, \"marker_activation_disagreements\": {}, \"grain_0513_flips\": {}, \"grain_0513_flip_steps\": {}, \"hardness_0534_flips\": {}, \"hardness_0534_flip_steps\": {}, \"tolerance_site_steps\": {{\"deposition_epsilon\": {}, \"melted_out_ice\": {}, \"refrozen_marker\": {}, \"gradient_thickness_floor\": {}, \"heat_linearisation_delta\": {}}}, \"rel_tol\": {:e}, \"scale_factor\": {:e}, \"max_rel_among_agreeing\": {:e}, \"a5_table_flux_samples\": {}, \"a5_loop_flux_samples\": {}, \"a5_flux_gate_failures\": {}, \"candidate_refusals\": {}, \"cpu_refusals\": {}, \"max_abs_qs_w_m2\": {:e}, \"max_abs_ql_w_m2\": {:e}, \"listed\": [{}]}}",
            self.steps,
            self.comparisons,
            self.agreeing,
            self.bitwise,
            self.count_mismatch_steps,
            self.count_change_causes[0], self.count_change_causes[1], self.count_change_causes[2],
            self.count_change_events.iter().map(|s| json_quote(s)).collect::<Vec<_>>().join(", "),
            self.marker_mismatch_steps,
            self.outcome_mismatch_steps,
            self.mo_pass_mismatch_steps,
            self.mo_multi_pass_steps_gpu,
            self.mo_multi_pass_steps_cpu,
            self.mo_pass_histogram_gpu,
            self.mo_pass_histogram_cpu,
            self.marker_activated_gpu,
            self.marker_activated_cpu,
            self.marker_activation_disagreements,
            self.grain_0513_flips,
            self.grain_0513_flip_steps,
            self.hardness_0534_flips,
            self.hardness_0534_flip_steps,
            self.tolerance_site_steps[0], self.tolerance_site_steps[1], self.tolerance_site_steps[2],
            self.tolerance_site_steps[3], self.tolerance_site_steps[4], self.tolerance_site_steps[5],
            self.rel_tol,
            self.scale_factor,
            self.max_rel,
            self.table_flux_samples,
            self.loop_flux_samples,
            self.flux_gate_failures,
            self.candidate_refusals,
            self.max_abs_qs_w_m2,
            self.max_abs_ql_w_m2,
            self.listed
                .iter()
                .map(|l| json_quote(l))
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
    run_with_tolerance(gpu, inputs, nmax, from, to, REL_TOL, 1.0)
}

/// Registered A1 development comparison. G1a continues to call `run` at its fixed tolerance.
pub fn run_development(
    gpu: &Gpu,
    inputs: &[ColumnInput],
    nmax: usize,
    from: usize,
    to: usize,
) -> Result<Tally> {
    run_with_tolerance(gpu, inputs, nmax, from, to, 1e-4, 1000.0)
}

fn run_with_tolerance(
    gpu: &Gpu,
    inputs: &[ColumnInput],
    nmax: usize,
    from: usize,
    to: usize,
    rel_tol: f64,
    scale_factor: f64,
) -> Result<Tally> {
    let options = PhysicsOptions::default();
    let b = inputs.len();
    let mut steppers = inputs
        .iter()
        .map(|i| Stepper::new(i.cfg, &options, i.site, i.start.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let k = gpu::run_constants(inputs[0].cfg, &steppers[0].constants)?;
    let mut par = vec![0.0; gpu::P_COUNT * b];
    for (c, (i, s)) in inputs.iter().zip(&steppers).enumerate() {
        if gpu::run_constants(i.cfg, &s.constants)?.map(f64::to_bits) != k.map(f64::to_bits) {
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
    let mut tally = Tally {
        rel_tol,
        scale_factor,
        mo_pass_histogram_gpu: vec![0; 201],
        mo_pass_histogram_cpu: vec![0; 201],
        ..Tally::default()
    };
    let end = to.min(inputs.iter().map(|i| i.rows.len()).max().unwrap_or(0));
    let mut active = vec![true; b];
    for r in from..end {
        let previous_acc = steppers.iter().map(|s| s.acc).collect::<Vec<_>>();
        let pre_markers = steppers
            .iter()
            .map(|s| {
                s.column
                    .elements
                    .iter()
                    .map(|e| e.micro.marker as u32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let pre_counts = steppers
            .iter()
            .map(|s| s.column.elements.len())
            .collect::<Vec<_>>();
        let pre_thin = steppers
            .iter()
            .map(|s| {
                s.column
                    .elements
                    .iter()
                    .any(|e| e.thickness < s.cfg.minimum_element_length)
            })
            .collect::<Vec<_>>();
        // CPU state into the host buffer.
        for (c, s) in steppers.iter().enumerate() {
            host.set_column(c, &s.column)?;
            host.buffer[c] = s.deposition.buffered();
            // Compare each hourly flux with a one-step accumulator reset.
            host.set_accum(c, &crate::stepper::Accum::default());
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
            .enumerate()
            .map(|(c, i)| {
                if active[c] && r < i.rows.len() {
                    &i.rows[r..r + 1]
                } else {
                    &i.rows[0..0]
                }
            })
            .collect();
        let (ft, fv, rows) = pack_rows(&slices);
        debug_assert_eq!(fv.len(), rows.max(1) * F_COUNT * b);
        batch.upload_forcing(gpu, &ft, &fv)?;
        batch.advance(gpu, rows, 1, 0, 0)?;
        let upto = batch.counts(gpu)?.into_iter().max().unwrap_or(0).max(0) as usize;
        batch.download(gpu, &mut host, upto.min(nmax))?;
        for (c, s) in steppers.iter_mut().enumerate() {
            if !active[c] || r >= inputs[c].rows.len() {
                continue;
            }
            let row = &inputs[c].rows[r];
            s.acc = crate::stepper::Accum::default();
            let cpu = s.step(row);
            if let Err(error) = &cpu {
                tally.count_change_events.push(format!(
                    "{}\t{}\tCPU_REFUSAL",
                    inputs[c].site.station_id,
                    row.t.format("%Y-%m-%dT%H:%M")
                ));
                active[c] = false;
                if tally.listed.len() < LIST_LIMIT {
                    tally.listed.push(format!(
                        "{}\t{}\tCPU refusal: {error}",
                        inputs[c].site.station_id,
                        row.t.format("%Y-%m-%dT%H:%M")
                    ));
                }
                continue;
            }
            let cpu_step_acc = s.acc;
            let mut accumulated = previous_acc[c].to_array();
            for (total, delta) in accumulated.iter_mut().zip(cpu_step_acc.to_array()) {
                *total += delta;
            }
            s.acc = crate::stepper::Accum::from_array(&accumulated);
            let ctx = format!(
                "{}\t{}\tmo_passes gpu={} cpu={}",
                inputs[c].site.station_id,
                row.t.format("%Y-%m-%dT%H:%M"),
                host.mo_passes[c],
                s.mo_passes
            );
            tally.steps += 1;
            let gpu_n_now = host.n[c].max(0) as usize;
            let cpu_n_now = s.column.elements.len();
            if matches!(
                gpu.build.precision,
                gpu::Precision::F32 | gpu::Precision::F32K
            ) && (gpu_n_now != pre_counts[c] || cpu_n_now != pre_counts[c])
            {
                let raw_cause = host.count_change_cause[c] as usize;
                let cause = if (1..=3).contains(&raw_cause) {
                    raw_cause
                } else if gpu_n_now > pre_counts[c] || cpu_n_now > pre_counts[c] {
                    1
                } else if pre_thin[c] {
                    2
                } else {
                    3
                };
                tally.count_change_causes[cause - 1] += 1;
                let cause_name = ["deposition", "merge", "melt_out"][cause - 1];
                tally.count_change_events.push(format!(
                    "{ctx}\tcause={cause_name}\tpre={}\tgpu={gpu_n_now}\tcpu={cpu_n_now}",
                    pre_counts[c]
                ));
            }
            if host.mo_passes[c] != -1 && host.mo_passes[c] as u32 != s.mo_passes {
                tally.mo_pass_mismatch_steps += 1;
            }
            tally.mo_pass_histogram_gpu[(host.mo_passes[c].max(0) as usize).min(200)] += 1;
            tally.mo_pass_histogram_cpu[(s.mo_passes as usize).min(200)] += 1;
            tally.mo_multi_pass_steps_gpu += (host.mo_passes[c] > 1) as u64;
            tally.mo_multi_pass_steps_cpu += (s.mo_passes > 1) as u64;
            if matches!(
                gpu.build.precision,
                gpu::Precision::F32 | gpu::Precision::F32K
            ) {
                for bit in 0..6 {
                    tally.tolerance_site_steps[bit] +=
                        ((host.tolerance_flags[c] >> bit) & 1) as u64;
                }
            }
            let gpu_failed = host.fail[c] != 0;
            let (cpu_output, cpu_failed, cpu_advanced) = match &cpu {
                Ok(Outcome::Advanced { output }) => (*output, false, true),
                Ok(_) => (false, false, false),
                Err(_) => (false, true, false),
            };
            if gpu_failed != cpu_failed || (host.out_ready[c] != 0) != cpu_output {
                tally.outcome_mismatch_steps += 1;
                if tally.listed.len() < LIST_LIMIT {
                    tally.listed.push(format!(
                        "{ctx}\toutcome\tgpu fail={} out={}\tcpu {:?}",
                        host.fail[c],
                        host.out_ready[c],
                        cpu.as_ref().map_err(|e| e.to_string())
                    ));
                }
            }
            if cpu_failed {
                bail!(
                    "CPU column {} failed at {}: {:?}",
                    inputs[c].site.station_id,
                    row.t,
                    cpu.err()
                );
            }
            let n = s.column.elements.len();
            let gpu_n = host.n[c].max(0) as usize;
            if gpu_n != n {
                tally.count_mismatch_steps += 1;
                if tally.listed.len() < LIST_LIMIT {
                    tally
                        .listed
                        .push(format!("{ctx}\tcount\tgpu={}\tcpu={n}", host.n[c]));
                }
            }
            let common_n = gpu_n.min(n);
            let diagnostic_precision = matches!(
                gpu.build.precision,
                gpu::Precision::F32 | gpu::Precision::F32K
            );
            let classes = if diagnostic_precision {
                let gpu_column =
                    host.column(c, inputs[c].site.slope_angle, inputs[c].site.slope_azimuth);
                Some((
                    crate::stepper::grain_hardness_values(&gpu_column, &options),
                    crate::stepper::grain_hardness_values(&s.column, &options),
                ))
            } else {
                None
            };
            let mut marker_bad = false;
            let mut grain_bad = false;
            let mut hardness_bad = false;
            for i in 0..common_n {
                let e = &s.column.elements[i];
                if host.marker[i * b + c] != e.micro.marker as u32 {
                    marker_bad = true;
                }
                if diagnostic_precision {
                    if let Some(&before) = pre_markers[c].get(i) {
                        let gpu_active = before & 1 == 0 && host.marker[i * b + c] & 1 != 0;
                        let cpu_active = before & 1 == 0 && e.micro.marker as u32 & 1 != 0;
                        tally.marker_activated_gpu += gpu_active as u64;
                        tally.marker_activated_cpu += cpu_active as u64;
                        tally.marker_activation_disagreements += (gpu_active != cpu_active) as u64;
                    }
                    if let Some((gpu_classes, cpu_classes)) = &classes {
                        if gpu_classes[i].0 != cpu_classes[i].0 {
                            grain_bad = true;
                            tally.grain_0513_flips += 1;
                        }
                        if gpu_classes[i].1 != cpu_classes[i].1 {
                            hardness_bad = true;
                            tally.hardness_0534_flips += 1;
                        }
                    }
                }
            }
            tally.grain_0513_flip_steps += grain_bad as u64;
            tally.hardness_0534_flip_steps += hardness_bad as u64;
            if marker_bad {
                tally.marker_mismatch_steps += 1;
            }
            if gpu_n == n {
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
                }
                for (j, t) in s.column.node_temperature.iter().enumerate() {
                    tally.compare(&format!("T[{j}]"), host.t[j * b + c], *t, NODE_SCALE, &ctx);
                }
            }
            tally.compare(
                "buffer",
                host.buffer[c],
                s.deposition.buffered(),
                BUFFER_SCALE,
                &ctx,
            );
            let gpu_acc = host.accum(c);
            let cpu_state_finite = s
                .column
                .elements
                .iter()
                .all(|e| crate::gpu::element_values(e).iter().all(|v| v.is_finite()))
                && s.column.node_temperature.iter().all(|v| v.is_finite());
            let gpu_n_within_capacity = gpu_n <= nmax;
            let gpu_n_finite = (0..gpu_n.min(nmax)).all(|i| {
                (0..gpu::ELEMENT_FIELDS.len()).all(|f| host.elem[f][i * b + c].is_finite())
            }) && (0..=gpu_n.min(nmax)).all(|i| host.t[i * b + c].is_finite());
            let accum_finite = gpu_acc.to_array().iter().all(|v| v.is_finite());
            if !cpu_state_finite
                || !gpu_n_finite
                || !gpu_n_within_capacity
                || !accum_finite
                || gpu_failed
            {
                tally.candidate_refusals += 1;
                if tally.listed.len() < LIST_LIMIT {
                    tally.listed.push(format!("{ctx}\trefusal nonfinite_or_gpu_failure={} gpu_state_finite={} cpu_state_finite={} gpu_n_within_capacity={} accum_finite={}", gpu_failed, gpu_n_finite, cpu_state_finite, gpu_n_within_capacity, accum_finite));
                }
                tally.flux_gate_failures += 1;
            } else if cpu_advanced && host.mo_passes[c] == -1 {
                tally.table_flux_samples += 1;
                let qs = (gpu_acc.qs - cpu_step_acc.qs).abs();
                let ql = (gpu_acc.ql - cpu_step_acc.ql).abs();
                tally.max_abs_qs_w_m2 = tally.max_abs_qs_w_m2.max(qs);
                tally.max_abs_ql_w_m2 = tally.max_abs_ql_w_m2.max(ql);
                if !qs.is_finite() || !ql.is_finite() || qs > 0.01 || ql > 0.01 {
                    tally.flux_gate_failures += 1;
                    if tally.listed.len() < LIST_LIMIT {
                        tally.listed.push(format!(
                            "{ctx}\ttable flux qs_abs={qs:e} ql_abs={ql:e} limit=1e-2 W m^-2"
                        ));
                    }
                }
            } else if cpu_advanced {
                tally.loop_flux_samples += 1;
                if gpu_acc.qs.to_bits() != cpu_step_acc.qs.to_bits()
                    || gpu_acc.ql.to_bits() != cpu_step_acc.ql.to_bits()
                {
                    tally.flux_gate_failures += 1;
                    if tally.listed.len() < LIST_LIMIT {
                        tally.listed.push(format!("{ctx}\tloop flux bit mismatch gpu_qs={:016x} cpu_qs={:016x} gpu_ql={:016x} cpu_ql={:016x}", gpu_acc.qs.to_bits(), cpu_step_acc.qs.to_bits(), gpu_acc.ql.to_bits(), cpu_step_acc.ql.to_bits()));
                    }
                }
            }
            if cpu_output {
                let _ = s.take_output_row(row.t);
            }
        }
    }
    Ok(tally)
}
