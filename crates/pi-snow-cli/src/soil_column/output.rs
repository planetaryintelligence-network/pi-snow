//! Keyed SMET extension; native columns and formatting remain byte-for-byte.
use super::SoilColumn;
use chrono::{DateTime, Utc};

pub const FIELDS: [&str; 7] = [
    "TSOIL_005",
    "TSOIL_010",
    "TSOIL_020",
    "TSOIL_050",
    "SOIL_H",
    "G_SNOW_SOIL",
    "G_BOT",
];

#[derive(Debug, Clone)]
pub struct Row {
    pub timestamp: DateTime<Utc>,
    /// Temperatures in kelvin, enthalpy in J m-2, downward fluxes in W m-2.
    pub values: [f64; 7],
    pub temperature_k: [f64; 10],
    pub enthalpy_j_m2: [f64; 10],
    pub liquid_fraction: [f64; 10],
}

#[derive(Debug, Default, Clone)]
pub struct Rows {
    pub rows: Vec<Row>,
    interval_seconds: f64,
    interval_top_j_m2: f64,
    interval_bottom_j_m2: f64,
}

impl Rows {
    pub fn accumulate(&mut self, dt: f64, top_j_m2: f64, bottom_j_m2: f64) {
        self.interval_seconds += dt;
        self.interval_top_j_m2 += top_j_m2;
        self.interval_bottom_j_m2 += bottom_j_m2;
    }

    pub fn push(&mut self, timestamp: DateTime<Utc>, soil: &SoilColumn) {
        assert!(self.interval_seconds > 0.0);
        self.rows.push(Row {
            timestamp,
            values: [
                soil.temperature_at_depth(0.05),
                soil.temperature_at_depth(0.10),
                soil.temperature_at_depth(0.20),
                soil.temperature_at_depth(0.50),
                soil.energy_j_m2(),
                self.interval_top_j_m2 / self.interval_seconds,
                self.interval_bottom_j_m2 / self.interval_seconds,
            ],
            temperature_k: soil.temperatures(),
            enthalpy_j_m2: soil.enthalpy,
            liquid_fraction: std::array::from_fn(|i| soil.phase(i).liquid_fraction),
        });
        self.interval_seconds = 0.0;
        self.interval_top_j_m2 = 0.0;
        self.interval_bottom_j_m2 = 0.0;
    }

    pub fn extend_smet(&self, native: &str) -> String {
        let mut result = String::with_capacity(native.len() + self.rows.len() * 150);
        let mut data = false;
        let mut index = 0;
        for line in native.lines() {
            if data {
                let row = self.rows.get(index).expect("missing soil output row");
                let timestamp = row.timestamp.format("%Y-%m-%dT%H:%M:%S").to_string();
                assert_eq!(
                    line.split_whitespace().next(),
                    Some(timestamp.as_str()),
                    "soil/native output timestamps differ"
                );
                result.push_str(line);
                for value in row.values {
                    result.push_str(&format!(" {value:.9}"));
                }
                result.push('\n');
                index += 1;
            } else {
                result.push_str(line);
                if line
                    .split_once('=')
                    .is_some_and(|(k, _)| k.trim() == "fields")
                {
                    result.push(' ');
                    result.push_str(&FIELDS.join(" "));
                } else if line
                    .split_once('=')
                    .is_some_and(|(k, _)| k.trim() == "units_offset")
                {
                    result.push_str(" 0 0 0 0 0 0 0");
                } else if line
                    .split_once('=')
                    .is_some_and(|(k, _)| k.trim() == "units_multiplier")
                {
                    result.push_str(" 1 1 1 1 1 1 1");
                }
                result.push('\n');
                data = line == "[DATA]";
            }
        }
        assert_eq!(index, self.rows.len(), "extra soil output rows");
        result
    }
}

/// Hourly physical state and aggregate diagnostics, never station substep rows.
#[derive(Debug, Clone)]
pub struct Data {
    pub runtime: super::runtime::Runtime,
    pub output: Rows,
    pub maximum_row_ledger_residual_j_m2: f64,
}

impl Data {
    pub fn new(runtime: super::runtime::Runtime) -> Self {
        Self {
            runtime,
            output: Rows::default(),
            maximum_row_ledger_residual_j_m2: 0.0,
        }
    }

    pub fn snapshot(&self) -> [f64; 3] {
        [
            self.runtime.column.energy_j_m2(),
            self.runtime.top_energy_j_m2,
            self.runtime.bottom_energy_j_m2,
        ]
    }

    pub fn finish_step(&mut self, dt: f64, before: [f64; 3]) {
        let after = self.snapshot();
        let top = after[1] - before[1];
        let bottom = after[2] - before[2];
        let residual = (after[0] - before[0] - top + bottom).abs();
        self.maximum_row_ledger_residual_j_m2 = self.maximum_row_ledger_residual_j_m2.max(residual);
        self.output.accumulate(dt, top, bottom);
    }

    /// Offset IDs distinguish soil nodes from the native snow node indices.
    pub fn crazy_temperature(&self, min: f64, max: f64) -> Option<(usize, f64)> {
        self.runtime
            .column
            .temperatures()
            .into_iter()
            .enumerate()
            .find(|(_, t)| *t < min || *t > max)
            .map(|(i, t)| (usize::MAX - i, t))
    }

    pub fn nonfinite(&self) -> Option<String> {
        self.runtime
            .column
            .enthalpy
            .iter()
            .enumerate()
            .find(|(_, h)| !h.is_finite())
            .map(|(i, _)| format!("soil.enthalpy[{i}]"))
            .or_else(|| {
                self.runtime
                    .column
                    .temperatures()
                    .iter()
                    .enumerate()
                    .find(|(_, t)| !t.is_finite())
                    .map(|(i, _)| format!("soil.temperature[{i}]"))
            })
    }

    pub fn extend_smet(&self, native: &str) -> String {
        let extended = self.output.extend_smet(native);
        let c = &self.runtime.counters;
        let metadata = format!(
            "soil_bare_rows = {}\nsoil_snow_rows = {}\nsoil_certified_rows = {}\nsoil_uncertified_rows = {}\nsoil_predicate_not_computed_rows = {}\nsoil_multiple_roots = {}\nsoil_nonconvergences = {}\nsoil_newton_iterations = {}\nsoil_column_solves = {}\nsoil_budget_evaluations = {}\nsoil_bare_turbulence_calls = {}\nsoil_scan_resolve_solves = {}\nsoil_bare_seconds = {:.9}\nsoil_snow_seconds = {:.9}\nsoil_initial_graft_j_m2 = {:.9}\nsoil_max_row_ledger_residual_j_m2 = {:.9}\nsoil_max_bare_consistency_w_m2 = {:.9}\nsoil_bare_consistency_failures = {}\nsoil_max_coupled_ledger_residual_j_m2 = {:.9}\nsoil_max_phase_ledger_residual_j_m2 = {:.9}\n",
            c.bare_rows,
            c.snow_rows,
            c.certified_rows,
            c.uncertified_rows,
            c.predicate_not_computed_rows,
            c.multiple_roots,
            c.nonconvergences,
            c.newton_iterations,
            c.column_solves,
            c.budget_evaluations,
            c.bare_turbulence_calls,
            c.scan_resolve_solves,
            c.bare_seconds,
            c.snow_seconds,
            self.runtime.initial_graft_j_m2,
            self.maximum_row_ledger_residual_j_m2,
            c.maximum_bare_consistency_w_m2,
            c.bare_consistency_failures,
            c.maximum_coupled_ledger_residual_j_m2,
            c.maximum_phase_ledger_residual_j_m2
        );
        extended.replacen("[DATA]\n", &(metadata + "[DATA]\n"), 1)
    }
}
