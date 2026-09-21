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
            rows.push(Row { timestamp, values, raw: line.to_string() });
        }

        Ok(Self { header, fields, nodata, rows, header_lines, trailing_newline })
    }

    /// Index of a named field among `values` (so, excluding the timestamp).
    pub fn field_index(&self, name: &str) -> Option<usize> {
        self.fields.iter().skip(1).position(|f| f == name)
    }

    /// Value of `name` at `row`, or `None` if the field is absent or nodata.
    pub fn value(&self, row: usize, name: &str) -> Option<f64> {
        self.rows.get(row)?.values.get(self.field_index(name)?).copied().flatten()
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
