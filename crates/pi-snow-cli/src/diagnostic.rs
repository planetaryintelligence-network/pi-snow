//! Optional report-only temperature and cold-step records.

use chrono::{DateTime, Utc};
use pi_snow_physics::surface_diagnostic::SurfaceIteration;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
pub enum DiagnosticMode {
    Minimum,
    Trace,
}

#[derive(Debug, Clone, Copy)]
pub struct Minimum {
    pub kelvin: f64,
    pub time: DateTime<Utc>,
    pub step: usize,
    pub node: usize,
}

#[derive(Debug, Clone)]
pub struct Refusal {
    pub kind: &'static str,
    pub kelvin: f64,
    pub time: DateTime<Utc>,
    pub step: usize,
    pub node: usize,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct StepTrace {
    pub step: usize,
    pub time: DateTime<Utc>,
    pub dt_seconds: f64,
    pub air_temperature_k: f64,
    pub relative_humidity: f64,
    pub wind_m_s: f64,
    pub incoming_shortwave_horizontal_w_m2: f64,
    pub incoming_longwave_w_m2: f64,
    pub pressure_pa: f64,
    pub ground_temperature_k: f64,
    pub z_wind_m: f64,
    pub z_scalar_m: f64,
    pub roughness_length_m: f64,
    pub min_node: usize,
    pub min_node_k: f64,
    pub snowfall_m: f64,
    pub rain_m: f64,
    pub deposited_kg_m2: f64,
    pub deposition_buffer_kg_m2: f64,
    pub shortwave_absorbed_w_m2: f64,
    pub internal_sources_w_m2: Vec<f64>,
    pub budget: [f64; 7],
    pub inferred_ground_flux_w_m2: f64,
    pub pre_solve_element_thickness_m: Vec<f64>,
    pub pre_solve_grain_type: Vec<u16>,
    pub pre_solve_density_kg_m3: Vec<f64>,
    pub pre_solve_node_temperature_k: Vec<f64>,
    pub post_solve_node_temperature_k: Vec<f64>,
    pub post_step_element_thickness_m: Vec<f64>,
    pub post_step_node_temperature_k: Vec<f64>,
    pub surface_iterations: Vec<SurfaceIteration>,
}

#[derive(Debug, Clone, Default)]
pub struct RunDiagnostics {
    pub minimum: Option<Minimum>,
    pub refusal: Option<Refusal>,
    pub trace: Vec<StepTrace>,
}

pub struct Collector {
    pub mode: DiagnosticMode,
    pub report: RunDiagnostics,
    preceding: VecDeque<StepTrace>,
    first_cold_step: Option<usize>,
}

impl Collector {
    pub fn new(mode: DiagnosticMode, initial: &[f64], time: DateTime<Utc>) -> Self {
        let mut report = RunDiagnostics::default();
        if let Some((node, &kelvin)) = initial
            .iter()
            .enumerate()
            .filter(|(_, t)| t.is_finite())
            .min_by(|a, b| a.1.total_cmp(b.1))
        {
            report.minimum = Some(Minimum {
                kelvin,
                time,
                step: 0,
                node,
            });
        }
        Self {
            mode,
            report,
            preceding: VecDeque::new(),
            first_cold_step: None,
        }
    }

    pub fn observe_minimum(
        &mut self,
        time: DateTime<Utc>,
        step: usize,
        nodes: &[f64],
    ) -> Option<(usize, f64)> {
        let (node, &kelvin) = nodes
            .iter()
            .enumerate()
            .filter(|(_, t)| t.is_finite())
            .min_by(|a, b| a.1.total_cmp(b.1))?;
        if self.report.minimum.map_or(true, |old| kelvin < old.kelvin) {
            self.report.minimum = Some(Minimum {
                kelvin,
                time,
                step,
                node,
            });
        }
        Some((node, kelvin))
    }

    pub fn observe_trace(&mut self, row: StepTrace) {
        if !matches!(self.mode, DiagnosticMode::Trace) {
            return;
        }
        if self.first_cold_step.is_none() && row.min_node_k < 220.0 {
            self.first_cold_step = Some(row.step);
            self.report.trace.extend(self.preceding.drain(..));
        }
        if self
            .first_cold_step
            .is_some_and(|first| row.step <= first + 8)
        {
            self.report.trace.push(row.clone());
        }
        self.preceding.push_back(row);
        while self.preceding.len() > 12 {
            self.preceding.pop_front();
        }
    }
}

pub fn minimum_report_json(
    units: &[crate::batch::UnitReport],
    threads: usize,
    wall: f64,
) -> String {
    let rows = units.iter().map(|u| {
        let minimum = u.minimum.map_or_else(|| "null".to_string(), |m| format!(
            "{{\"temperature_K\":{},\"time\":{},\"step\":{},\"node\":{}}}",
            number(m.kelvin), quote(&m.time.to_rfc3339()), m.step, m.node));
        let refusal = u.refusal.as_ref().map_or_else(|| "null".to_string(), |r| format!(
            "{{\"kind\":{},\"temperature_K\":{},\"time\":{},\"step\":{},\"node\":{},\"detail\":{}}}",
            quote(r.kind), number(r.kelvin), quote(&r.time.to_rfc3339()), r.step, r.node, quote(&r.detail)));
        format!("{{\"unit\":{},\"status\":{},\"minimum\":{},\"refusal\":{}}}",
            quote(&u.unit_dir.display().to_string()), quote(if u.error.is_none() { "complete" } else { "refused" }), minimum, refusal)
    }).collect::<Vec<_>>();
    format!(
        "{{\"version\":{},\"threads\":{},\"wall_seconds\":{},\"units\":[{}]}}\n",
        quote(env!("CARGO_PKG_VERSION")),
        threads,
        number(wall),
        rows.join(",")
    )
}

pub fn trace_jsonl(station: &str, report: &RunDiagnostics) -> String {
    let minimum = report.minimum.map_or_else(
        || "null".to_string(),
        |m| {
            format!(
                "{{\"temperature_K\":{},\"time\":{},\"step\":{},\"node\":{}}}",
                number(m.kelvin),
                quote(&m.time.to_rfc3339()),
                m.step,
                m.node
            )
        },
    );
    let refusal = report.refusal.as_ref().map_or_else(|| "null".to_string(), |r| format!(
        "{{\"kind\":{},\"temperature_K\":{},\"time\":{},\"step\":{},\"node\":{},\"detail\":{}}}",
        quote(r.kind), number(r.kelvin), quote(&r.time.to_rfc3339()), r.step, r.node, quote(&r.detail)));
    let mut out = format!(
        "{{\"record\":\"summary\",\"station\":{},\"minimum\":{},\"refusal\":{}}}\n",
        quote(station),
        minimum,
        refusal
    );
    for row in &report.trace {
        out.push_str(&step_json(row));
        out.push('\n');
    }
    out
}

fn step_json(r: &StepTrace) -> String {
    let iterations = r.surface_iterations.iter().map(|i| format!(
        "{{\"iteration\":{},\"linearization_input_K\":{},\"linearization_K\":{},\"a_W_m2\":{},\"b_W_m2_K\":{},\"finite_difference_budget_f0_W_m2\":{},\"finite_difference_budget_f1_W_m2\":{},\"raw_finite_difference_slope_W_m2_K\":{},\"radiative_slope_W_m2_K\":{},\"solved_surface_K\":{},\"nonlinear_budget_W_m2\":{},\"surface_linearization_residual_W_m2\":{},\"conduction\":{{\"layers\":{},\"min_thickness_m\":{},\"max_thickness_m\":{},\"min_conductance_W_m2_K\":{},\"max_conductance_W_m2_K\":{},\"min_half_capacity_W_m2_K\":{},\"max_half_capacity_W_m2_K\":{},\"min_diagonal\":{},\"max_diagonal\":{},\"min_abs_off_diagonal\":{},\"max_abs_off_diagonal\":{},\"residual_linf\":{}}}}}",
        i.iteration, number(i.linearization_input_k), number(i.linearization_temperature_k),
        number(i.a_w_m2), number(i.b_w_m2_k), number(i.finite_difference_budget_f0_w_m2),
        number(i.finite_difference_budget_f1_w_m2), number(i.raw_finite_difference_slope_w_m2_k),
        number(i.radiative_slope_w_m2_k),
        number(i.solved_surface_k), number(i.nonlinear_budget_w_m2),
        number(i.surface_linearization_residual_w_m2), i.conduction.layers, number(i.conduction.min_thickness_m),
        number(i.conduction.max_thickness_m), number(i.conduction.min_conductance_w_m2k),
        number(i.conduction.max_conductance_w_m2k), number(i.conduction.min_half_capacity_w_m2k),
        number(i.conduction.max_half_capacity_w_m2k), number(i.conduction.min_diagonal),
        number(i.conduction.max_diagonal), number(i.conduction.min_abs_off_diagonal),
        number(i.conduction.max_abs_off_diagonal), number(i.conduction.residual_linf)
    )).collect::<Vec<_>>().join(",");
    format!(
        "{{\"record\":\"step\",\"step\":{},\"time\":{},\"dt_seconds\":{},\"air_temperature_K\":{},\"relative_humidity\":{},\"wind_m_s\":{},\"incoming_shortwave_horizontal_W_m2\":{},\"incoming_longwave_W_m2\":{},\"pressure_Pa\":{},\"ground_temperature_K\":{},\"z_wind_m\":{},\"z_scalar_m\":{},\"roughness_length_m\":{},\"min_node\":{},\"min_node_temperature_K\":{},\"snowfall_m\":{},\"rain_m\":{},\"deposited_kg_m2\":{},\"deposition_buffer_kg_m2\":{},\"shortwave_absorbed_W_m2\":{},\"internal_sources_W_m2\":[{}],\"budget\":{{\"olwr\":{},\"lwr_net\":{},\"sensible\":{},\"latent\":{},\"rain\":{},\"vapour_mass\":{},\"total\":{}}},\"inferred_ground_flux_W_m2\":{},\"pre_solve_element_thickness_m\":[{}],\"pre_solve_grain_type\":[{}],\"pre_solve_density_kg_m3\":[{}],\"pre_solve_node_temperature_K\":[{}],\"post_solve_node_temperature_K\":[{}],\"post_step_element_thickness_m\":[{}],\"post_step_node_temperature_K\":[{}],\"surface_iterations\":[{}]}}",
        r.step,
        quote(&r.time.to_rfc3339()),
        number(r.dt_seconds),
        number(r.air_temperature_k),
        number(r.relative_humidity),
        number(r.wind_m_s),
        number(r.incoming_shortwave_horizontal_w_m2),
        number(r.incoming_longwave_w_m2),
        number(r.pressure_pa),
        number(r.ground_temperature_k),
        number(r.z_wind_m),
        number(r.z_scalar_m),
        number(r.roughness_length_m),
        r.min_node,
        number(r.min_node_k),
        number(r.snowfall_m),
        number(r.rain_m),
        number(r.deposited_kg_m2),
        number(r.deposition_buffer_kg_m2),
        number(r.shortwave_absorbed_w_m2),
        numbers(&r.internal_sources_w_m2),
        number(r.budget[0]),
        number(r.budget[1]),
        number(r.budget[2]),
        number(r.budget[3]),
        number(r.budget[4]),
        number(r.budget[5]),
        number(r.budget[6]),
        number(r.inferred_ground_flux_w_m2),
        numbers(&r.pre_solve_element_thickness_m),
        r.pre_solve_grain_type
            .iter()
            .map(u16::to_string)
            .collect::<Vec<_>>()
            .join(","),
        numbers(&r.pre_solve_density_kg_m3),
        numbers(&r.pre_solve_node_temperature_k),
        numbers(&r.post_solve_node_temperature_k),
        numbers(&r.post_step_element_thickness_m),
        numbers(&r.post_step_node_temperature_k),
        iterations
    )
}

fn number(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.12}")
    } else {
        "null".to_string()
    }
}

fn numbers(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| number(*v))
        .collect::<Vec<_>>()
        .join(",")
}

fn quote(s: &str) -> String {
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
