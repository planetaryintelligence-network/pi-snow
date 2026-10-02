//! SMET: the format MeteoIO uses for forcing series, model time series and, in a
//! derived shape, for snow profiles.
//!
//! Round-tripping has to be byte identical, because a golden test that compares
//! our output to a reference run must not be comparing formatting. The parser
//! therefore keeps the original lines and reconstructs from them; parsed values
//! are a view over that text, not a replacement for it. Files we generate from
//! scratch go through [`SmetBuilder`], which controls formatting deliberately.

use std::collections::BTreeMap;
use std::fmt;

use chrono::{DateTime, NaiveDateTime, Utc};

use crate::error::IoFormatError;

const FORMAT: &str = "SMET";

/// One data row: its timestamp, and the numeric values after it.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub timestamp: DateTime<Utc>,
    /// Values in `fields` order, excluding the timestamp. `None` where the value
    /// equals the file's `nodata` sentinel.
    pub values: Vec<Option<f64>>,
    raw: String,
}

impl Row {
    pub fn raw(&self) -> &str {
        &self.raw
    }
}

/// A parsed SMET file.
#[derive(Debug, Clone, PartialEq)]
pub struct Smet {
    /// Header keys in parse order, values as written.
    pub header: BTreeMap<String, String>,
    /// `fields`, including the leading `timestamp`.
    pub fields: Vec<String>,
    /// The `nodata` sentinel. Defaults to -999 when the header omits it.
    pub nodata: f64,
    pub rows: Vec<Row>,
    /// Everything before `[DATA]`, verbatim, so the file can be rebuilt.
    header_lines: Vec<String>,
    trailing_newline: bool,
}

impl Smet {
    pub fn parse(text: &str) -> Result<Self, IoFormatError> {
        let trailing_newline = text.ends_with('\n');
        let mut header_lines = Vec::new();
        let mut header = BTreeMap::new();
        let mut fields: Vec<String> = Vec::new();
        let mut nodata = -999.0_f64;
        let mut lines = text.lines().enumerate();
        let mut saw_data = false;

        for (_, line) in lines.by_ref() {
            header_lines.push(line.to_string());
            if line.trim() == "[DATA]" {
                saw_data = true;
                break;
            }
            if let Some((key, value)) = line.split_once('=') {
                let (key, value) = (key.trim().to_string(), value.trim().to_string());
                if key == "fields" {
                    fields = value.split_whitespace().map(str::to_string).collect();
                } else if key == "nodata" {
                    nodata = value.parse().unwrap_or(-999.0);
                }
                header.insert(key, value);
            }
        }
        if !saw_data {
            return Err(IoFormatError::NoDataSection { format: FORMAT });
        }
        if fields.is_empty() {
            return Err(IoFormatError::NoFields { format: FORMAT });
        }

        let expected = fields.len() - 1;
        let mut rows = Vec::new();
        for (index, line) in lines {
            if line.trim().is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let stamp = parts.next().ok_or(IoFormatError::WrongArity {
                format: FORMAT,
                line: index + 1,
                expected: expected + 1,
                found: 0,
            })?;
            let timestamp = parse_timestamp(stamp).ok_or_else(|| IoFormatError::BadValue {
                format: FORMAT,
                line: index + 1,
                what: "timestamp",
                text: stamp.to_string(),
            })?;
            let mut values = Vec::with_capacity(expected);
            for part in parts {
                let value: f64 = part.parse().map_err(|_| IoFormatError::BadValue {
                    format: FORMAT,
                    line: index + 1,
                    what: "value",
                    text: part.to_string(),
                })?;
                values.push(if value == nodata { None } else { Some(value) });
            }
            if values.len() != expected {
                return Err(IoFormatError::WrongArity {
                    format: FORMAT,
                    line: index + 1,
                    expected,
                    found: values.len(),
                });
            }
            rows.push(Row {
                timestamp,
                values,
                raw: line.to_string(),
            });
        }

        Ok(Self {
            header,
            fields,
            nodata,
            rows,
            header_lines,
            trailing_newline,
        })
    }

    /// Index of a named field among `values` (so, excluding the timestamp).
    pub fn field_index(&self, name: &str) -> Option<usize> {
        self.fields.iter().skip(1).position(|f| f == name)
    }

    /// Value of `name` at `row`, or `None` if the field is absent or nodata.
    pub fn value(&self, row: usize, name: &str) -> Option<f64> {
        self.rows
            .get(row)?
            .values
            .get(self.field_index(name)?)
            .copied()
            .flatten()
    }

    /// The whole column for `name`, nodata as `None`.
    pub fn column(&self, name: &str) -> Option<Vec<Option<f64>>> {
        let index = self.field_index(name)?;
        Some(self.rows.iter().map(|r| r.values[index]).collect())
    }
}

impl fmt::Display for Smet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for line in &self.header_lines {
            writeln!(f, "{line}")?;
        }
        for (index, row) in self.rows.iter().enumerate() {
            if index + 1 == self.rows.len() && !self.trailing_newline {
                write!(f, "{}", row.raw)?;
            } else {
                writeln!(f, "{}", row.raw)?;
            }
        }
        Ok(())
    }
}

/// Parse the SMET timestamp forms we encounter: `YYYY-MM-DDTHH:MM` and
/// `YYYY-MM-DDTHH:MM:SS`. Both are UTC, because every file in this chain sets
/// `tz = 0`.
pub fn parse_timestamp(text: &str) -> Option<DateTime<Utc>> {
    for format in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(text, format) {
            return Some(naive.and_utc());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Writing the output time series from scratch
// ---------------------------------------------------------------------------

/// The 68 data fields of the reference output time series, in the reference's
/// order. The feature extractor reads by name, but the order is kept so a file
/// diff against the reference lines up column for column.
pub const OUTPUT_FIELDS: [&str; 68] = [
    "Qs",
    "Ql",
    "Qg",
    "TSG",
    "Qg0",
    "Qr",
    "dIntEnergySnow",
    "meltFreezeEnergySnow",
    "ColdContentSnow",
    "OLWR",
    "ILWR",
    "LWR_net",
    "OSWR",
    "ISWR",
    "Qw",
    "pAlbedo",
    "mAlbedo",
    "ISWR_h",
    "ISWR_dir",
    "ISWR_diff",
    "TA",
    "TSS_mod",
    "TSS_meas",
    "T_bottom",
    "RH",
    "VW",
    "VW_drift",
    "DW",
    "MS_Snow",
    "HS_mod",
    "HS_meas",
    "hoar_size",
    "wind_trans24",
    "HN3",
    "HN6",
    "HN12",
    "HN24",
    "HN72_24",
    "PSUM24",
    "ski_pen",
    "SWE",
    "MS_Water",
    "MS_Water_Soil",
    "MS_Ice_Soil",
    "MS_Wind",
    "MS_Rain",
    "MS_SN_Runoff",
    "MS_Surface_Mass_Flux",
    "MS_Soil_Runoff",
    "MS_Sublimation",
    "MS_Evap",
    "TS0",
    "TS1",
    "TS2",
    "TS3",
    "TS4",
    "Sclass1",
    "Sclass2",
    "zSd",
    "Sd",
    "zSn",
    "Sn",
    "zSs",
    "Ss",
    "zS4",
    "S4",
    "zS5",
    "S5",
];

/// The reference file's unit transforms, one per field including the timestamp.
/// They state that temperatures are written in degrees Celsius (offset 273.15
/// back to kelvin) and heights in centimetres (multiplier 0.01 back to metres).
/// The feature extractor ignores them and reads the display values as written,
/// which is why the engine must write display units, not SI.
const UNITS_OFFSET: &str = "0 0 0 0 273.15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 273.15 273.15 273.15 273.15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 273.15 273.15 273.15 273.15 273.15 0 0 0 0 0 0 0 0 0 0 0 0";
const UNITS_MULTIPLIER: &str = "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.01 1 1 1 1 0.01 0.01 0.001 0.01 0.01 0.01 0.01 0.01 0.01 0.001 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1";

/// One output row: a timestamp and the 68 values, nodata where unset.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputRow {
    pub timestamp: DateTime<Utc>,
    values: [Option<f64>; 68],
}

impl OutputRow {
    pub fn nodata(timestamp: DateTime<Utc>) -> Self {
        Self {
            timestamp,
            values: [None; 68],
        }
    }

    /// Set a field by name. Panics on an unknown name, deliberately: a typo in a
    /// field name would otherwise write a silent column of nodata.
    pub fn set(&mut self, name: &str, value: f64) {
        let index = OUTPUT_FIELDS
            .iter()
            .position(|f| *f == name)
            .unwrap_or_else(|| panic!("unknown output field {name:?}"));
        self.values[index] = Some(value);
    }

    /// The first field holding a NaN or an infinity, with its value.
    pub fn first_non_finite(&self) -> Option<(&'static str, f64)> {
        OUTPUT_FIELDS
            .iter()
            .zip(self.values.iter())
            .find_map(|(name, v)| v.filter(|x| !x.is_finite()).map(|x| (*name, x)))
    }
}

/// Write a complete output time series file.
#[allow(clippy::too_many_arguments)]
pub fn write_output_smet(
    station_id: &str,
    station_name: &str,
    latitude: f64,
    longitude: f64,
    altitude: f64,
    slope_angle_deg: f64,
    slope_azimuth_deg: f64,
    rows: &[OutputRow],
) -> String {
    let mut out = String::from("SMET 1.1 ASCII\n[HEADER]\n");
    let mut kv = |k: &str, v: String| out.push_str(&format!("{k:<16} = {v}\n"));
    kv("station_id", station_id.to_string());
    kv("station_name", station_name.to_string());
    kv("latitude", format!("{latitude:.6}"));
    kv("longitude", format!("{longitude:.6}"));
    kv("altitude", format!("{altitude:.1}"));
    kv("nodata", "-999".to_string());
    kv("tz", "0".to_string());
    kv("units_offset", UNITS_OFFSET.to_string());
    kv("units_multiplier", UNITS_MULTIPLIER.to_string());
    kv("slope_angle", format!("{slope_angle_deg}"));
    kv("slope_azi", format!("{slope_azimuth_deg}"));
    kv("fields", format!("timestamp {}", OUTPUT_FIELDS.join(" ")));
    kv(
        "creator_name",
        format!("pi-snow {}", env!("CARGO_PKG_VERSION")),
    );
    out.push_str("[DATA]\n");
    for row in rows {
        out.push_str(&row.timestamp.format("%Y-%m-%dT%H:%M:%S").to_string());
        for v in &row.values {
            match v {
                Some(x) => out.push_str(&format!(" {x:>8.3}")),
                None => out.push_str("     -999"),
            }
        }
        out.push('\n');
    }
    out
}
