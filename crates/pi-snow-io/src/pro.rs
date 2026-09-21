//! PRO: the layered profile time series.
//!
//! Structure: a `[STATION_PARAMETERS]` block, a `[HEADER]` block declaring which
//! four-digit field codes the file carries, then `[DATA]` holding, per timestep,
//! a `0500,<dd.mm.yyyy HH:MM:SS>` line followed by one line per code. Each of
//! those is `<code>,<count>,<value>,<value>,...` with the values running from the
//! bottom element to the top.
//!
//! Two conventions in this format cost time if you meet them by surprise, so they
//! are asserted in the tests rather than left as folklore:
//!
//! * A timestep with no snow writes exactly `0501,1,0` and nothing else. The
//!   Python feature extractor detects that case by length and skips it.
//! * Which codes appear depends on the run's configuration, not on the format.
//!   The v3b runs write `0505` (age) and not `0540` (date of birth) because
//!   `PROF_AGE_OR_DATE = AGE`, and do not write `0607` at all.

use std::collections::BTreeMap;
use std::fmt;

use chrono::{DateTime, NaiveDateTime, Utc};

use crate::error::IoFormatError;

const FORMAT: &str = "PRO";

/// One timestep's profile.
#[derive(Debug, Clone, PartialEq)]
pub struct Timestep {
    pub timestamp: DateTime<Utc>,
    /// Field code to its values, bottom element first.
    fields: BTreeMap<String, Vec<f64>>,
    /// Raw lines, in file order, so the file can be rebuilt exactly.
    raw: Vec<String>,
}

impl Timestep {
    pub fn field(&self, code: &str) -> Option<&[f64]> {
        self.fields.get(code).map(Vec::as_slice)
    }

    /// The per-element values of `code`, with the grain-type row's trailing
    /// extra value removed.
    ///
    /// `0513` is the one per-element code that writes `nElems + 1` values: 24
    /// values for a 23-element profile. Everything else writes exactly `nElems`.
    /// The Python feature extractor handles this by slicing `0513` as `[2..-1]`
    /// where every other code is `[2..]`, and a reader that misses it ends up
    /// with one array longer than the rest, which surfaces much later as an
    /// "inconsistent profile arrays" error rather than as a parse failure.
    pub fn element_field(&self, code: &str) -> Option<&[f64]> {
        let values = self.fields.get(code)?;
        Some(if code == "0513" && values.len() == self.element_count() + 1 {
            &values[..values.len() - 1]
        } else {
            values.as_slice()
        })
    }

    /// Number of elements, taken from `0501`. Zero for a bare-ground timestep,
    /// which the format writes as the single value `0`.
    pub fn element_count(&self) -> usize {
        match self.fields.get("0501") {
            Some(heights) if heights.len() == 1 && heights[0] == 0.0 => 0,
            Some(heights) => heights.len(),
            None => 0,
        }
    }

    pub fn is_bare_ground(&self) -> bool {
        self.element_count() == 0
    }
}

/// A parsed PRO file.
#[derive(Debug, Clone, PartialEq)]
pub struct Profiles {
    /// `[STATION_PARAMETERS]` entries, as written.
    pub station: BTreeMap<String, String>,
    /// Field codes declared in `[HEADER]`, in declaration order, with their
    /// description text.
    pub declared_codes: Vec<(String, String)>,
    pub timesteps: Vec<Timestep>,
    header_lines: Vec<String>,
    trailing_newline: bool,
}

impl Profiles {
    pub fn parse(text: &str) -> Result<Self, IoFormatError> {
        let trailing_newline = text.ends_with('\n');
        let mut header_lines = Vec::new();
        let mut station = BTreeMap::new();
        let mut declared_codes = Vec::new();
        let mut lines = text.lines().enumerate();
        let mut saw_data = false;
        let mut in_header_block = false;

        for (_, line) in lines.by_ref() {
            header_lines.push(line.to_string());
            let trimmed = line.trim();
            if trimmed == "[DATA]" {
                saw_data = true;
                break;
            }
            if trimmed == "[HEADER]" {
                in_header_block = true;
                continue;
            }
            if in_header_block {
                if let Some((code, rest)) = trimmed.split_once(',')
                    && code.len() == 4
                    && code.chars().all(|c| c.is_ascii_digit())
                {
                    declared_codes.push((code.to_string(), rest.to_string()));
                }
            } else if let Some((key, value)) = trimmed.split_once('=') {
                station.insert(key.trim().to_string(), value.trim().to_string());
            }
        }
        if !saw_data {
            return Err(IoFormatError::NoDataSection { format: FORMAT });
        }

        let mut timesteps: Vec<Timestep> = Vec::new();
        for (index, line) in lines {
            if line.trim().is_empty() {
                continue;
            }
            let code = line.get(..4).unwrap_or_default();
            if code == "0500" {
                let stamp = line.split(',').nth(1).unwrap_or_default().trim();
                let timestamp = parse_pro_timestamp(stamp).ok_or_else(|| IoFormatError::BadValue {
                    format: FORMAT,
                    line: index + 1,
                    what: "timestamp",
                    text: stamp.to_string(),
                })?;
                if timesteps.iter().any(|t| t.timestamp == timestamp) {
                    return Err(IoFormatError::Duplicate {
                        what: "PRO timestamp",
                        value: stamp.to_string(),
                    });
                }
                timesteps.push(Timestep {
                    timestamp,
                    fields: BTreeMap::new(),
                    raw: vec![line.to_string()],
                });
                continue;
            }
            let step = timesteps
                .last_mut()
                .ok_or(IoFormatError::ValueBeforeTimestamp { line: index + 1 })?;
            step.raw.push(line.to_string());

            // 0540 is a date field and 0514/0530 are fixed-length summary rows;
            // all are stored as text-free numbers where they parse and skipped
            // where they do not, because the engine reads none of them as floats.
            let mut parts = line.split(',');
            let _ = parts.next();
            let _count = parts.next();
            let values: Vec<f64> = parts.filter_map(|p| p.trim().parse::<f64>().ok()).collect();
            if !values.is_empty() && step.fields.insert(code.to_string(), values).is_some() {
                return Err(IoFormatError::Duplicate {
                    what: "PRO field within one timestep",
                    value: code.to_string(),
                });
            }
        }

        Ok(Self { station, declared_codes, timesteps, header_lines, trailing_newline })
    }
}

impl fmt::Display for Profiles {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for line in &self.header_lines {
            writeln!(f, "{line}")?;
        }
        let total: usize = self.timesteps.iter().map(|t| t.raw.len()).sum();
        let mut written = 0;
        for step in &self.timesteps {
            for line in &step.raw {
                written += 1;
                if written == total && !self.trailing_newline {
                    write!(f, "{line}")?;
                } else {
                    writeln!(f, "{line}")?;
                }
            }
        }
        Ok(())
    }
}

/// PRO timestamps are `dd.mm.yyyy HH:MM:SS`, unlike SMET's ISO form. They are
/// UTC because every run in this chain sets `TIME_ZONE = 0`.
pub fn parse_pro_timestamp(text: &str) -> Option<DateTime<Utc>> {
    NaiveDateTime::parse_from_str(text, "%d.%m.%Y %H:%M:%S")
        .ok()
        .map(|n| n.and_utc())
}
