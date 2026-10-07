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
