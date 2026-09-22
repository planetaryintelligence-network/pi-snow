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
        Some(
            if code == "0513" && values.len() == self.element_count() + 1 {
                &values[..values.len() - 1]
            } else {
                values.as_slice()
            },
        )
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
                let timestamp =
                    parse_pro_timestamp(stamp).ok_or_else(|| IoFormatError::BadValue {
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

        Ok(Self {
            station,
            declared_codes,
            timesteps,
            header_lines,
            trailing_newline,
        })
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

// ---------------------------------------------------------------------------
// Writing from scratch
// ---------------------------------------------------------------------------

use pi_snow_core::constants::NODATA;

/// Station parameters written at the top of a PRO file. The validation harness
/// reads `Altitude` for its elevation filter and the chain's own validator reads
/// `SlopeAngle` and `SlopeAzi`, so all three are load-bearing.
#[derive(Debug, Clone, PartialEq)]
pub struct ProHeader {
    pub station_name: String,
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: f64,
    pub slope_angle_deg: f64,
    pub slope_azimuth_deg: f64,
}

/// Surface hoar standing on the surface, reported in `0514` and as the extra
/// trailing value of `0513`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceHoar {
    pub grain_size_mm: f64,
    pub density: f64,
}

/// The Swiss code for surface hoar, as the reference output writes it in the
/// trailing `0513` value and in `0514`.
pub const SURFACE_HOAR_CODE: u16 = 660;

/// One element of one timestep, in the PRO file's own units. Anything the
/// engine does not compute stays at its default, which is `NODATA` (-999), so it
/// reads back as NaN rather than as a misleading zero.
#[derive(Debug, Clone, PartialEq)]
pub struct ElementRecord {
    pub top_height_cm: f64,
    pub density: f64,
    pub temperature_c: f64,
    pub id: u32,
    pub age_days: f64,
    pub lwc_percent: f64,
    pub dendricity: f64,
    pub sphericity: f64,
    pub coordination_number: f64,
    pub bond_size_mm: f64,
    pub grain_size_mm: f64,
    pub grain_type: u16,
    pub ice_percent: f64,
    pub air_percent: f64,
    pub stress_kpa: f64,
    pub viscosity_gpa_s: f64,
    pub soil_percent: f64,
    pub temperature_gradient: f64,
    pub conductivity: f64,
    pub absorbed_shortwave: f64,
    /// `0523`, in units of 1e-6 s^-1.
    pub deformation_rate: f64,
    pub sdef: f64,
    pub sn38: f64,
    pub sk38: f64,
    pub hand_hardness: f64,
    pub optical_grain_size_mm: f64,
    pub shear_strength_kpa: f64,
    pub grain_size_difference_mm: f64,
    pub hardness_difference: f64,
    pub ssi: f64,
    pub inverse_texture_index: f64,
    pub critical_cut_length_m: f64,
}

impl Default for ElementRecord {
    fn default() -> Self {
        Self {
            top_height_cm: NODATA,
            density: NODATA,
            temperature_c: NODATA,
            id: 0,
            age_days: NODATA,
            lwc_percent: NODATA,
            dendricity: NODATA,
            sphericity: NODATA,
            coordination_number: NODATA,
            bond_size_mm: NODATA,
            grain_size_mm: NODATA,
            grain_type: 0,
            ice_percent: NODATA,
            air_percent: NODATA,
            stress_kpa: NODATA,
            viscosity_gpa_s: NODATA,
            // No soil column in this configuration, so zero is the true value.
            soil_percent: 0.0,
            temperature_gradient: NODATA,
            conductivity: NODATA,
            absorbed_shortwave: NODATA,
            deformation_rate: NODATA,
            sdef: NODATA,
            sn38: NODATA,
            sk38: NODATA,
            hand_hardness: NODATA,
            optical_grain_size_mm: NODATA,
            shear_strength_kpa: NODATA,
            grain_size_difference_mm: NODATA,
            hardness_difference: NODATA,
            ssi: NODATA,
            inverse_texture_index: NODATA,
            critical_cut_length_m: NODATA,
        }
    }
}

/// One timestep to write.
#[derive(Debug, Clone, PartialEq)]
pub struct ProTimestep {
    pub timestamp: DateTime<Utc>,
    /// Bottom element first. Empty for bare ground.
    pub elements: Vec<ElementRecord>,
    pub surface_hoar: Option<SurfaceHoar>,
}

/// The field declarations, in the order and with the descriptions the reference
/// output carries. These are file-format documentation, which the clean-room
/// rule allows us to read and reproduce.
const DECLARATIONS: &[&str] = &[
    "0500,Date",
    "0501,nElems,height [> 0: top, < 0: bottom of elem.] (cm)",
    "0502,nElems,element density (kg m-3)",
    "0503,nElems,element temperature (degC)",
    "0504,nElems,element ID (1)",
    "0505,nElems,element age (days)",
    "0506,nElems,liquid water content by volume (%)",
    "0508,nElems,dendricity (1)",
    "0509,nElems,sphericity (1)",
    "0510,nElems,coordination number (1)",
    "0511,nElems,bond size (mm)",
    "0512,nElems,grain size (mm)",
    "0513,nElems,grain type (Swiss Code F1F2F3)",
    "0514,3,grain type, grain size (mm), and density (kg m-3) of SH at surface",
    "0515,nElems,ice volume fraction (%)",
    "0516,nElems,air volume fraction (%)",
    "0517,nElems,stress in (kPa)",
    "0518,nElems,viscosity (GPa s)",
    "0519,nElems,soil volume fraction (%)",
    "0520,nElems,temperature gradient (K m-1)",
    "0521,nElems,thermal conductivity (W K-1 m-1)",
    "0522,nElems,absorbed shortwave radiation (W m-2)",
    "0523,nElems,viscous deformation rate (1.e-6 s-1)",
    "0530,8,position (cm) and minimum stability indices:",
    "profile type, stability class, z_Sdef, Sdef, z_Sn38, Sn38, z_Sk38, Sk38",
    "0531,nElems,deformation rate stability index Sdef",
    "0532,nElems,natural stability index Sn38",
    "0533,nElems,stability index Sk38",
    "0534,nElems,hand hardness in index steps (1)",
    "0535,nElems,optical equivalent grain size (mm)",
    "0601,nElems,snow shear strength (kPa)",
    "0602,nElems,grain size difference (mm)",
    "0603,nElems,hardness difference (1)",
    "0604,nElems,ssi",
    "0605,nElems,inverse texture index ITI (Mg m-4)",
    "0606,nElems,critical cut length (m)",
];

/// Write the station and header blocks, up to and including `[DATA]`.
pub fn write_header(header: &ProHeader) -> String {
    let mut out = String::new();
    out.push_str("[STATION_PARAMETERS]\n");
    out.push_str(&format!("StationName= {}\n", header.station_name));
    out.push_str(&format!("Latitude= {:.8}\n", header.latitude));
    out.push_str(&format!("Longitude= {:.8}\n", header.longitude));
    out.push_str(&format!("Altitude= {:.0}\n", header.altitude));
    out.push_str(&format!("SlopeAngle= {:.2}\n", header.slope_angle_deg));
    out.push_str(&format!("SlopeAzi= {:.2}\n", header.slope_azimuth_deg));
    out.push_str("\n[HEADER]\n");
    out.push_str(&format!(
        "#pi-snow {} (clean-room engine, see experiments/phase-2-rust-engine-v1)\n",
        env!("CARGO_PKG_VERSION")
    ));
    for line in DECLARATIONS {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("\n[DATA]\n");
    out
}

/// Format one value, writing the nodata sentinel as the literal `-999` so it
/// reads unambiguously whatever the field's usual precision.
fn value(v: f64, precision: usize) -> String {
    if v == NODATA {
        "-999".to_string()
    } else {
        format!("{v:.precision$}")
    }
}

fn scientific(v: f64) -> String {
    if v == NODATA {
        "-999".to_string()
    } else {
        format!("{v:.3e}")
    }
}

fn row<F: Fn(&ElementRecord) -> String>(code: &str, elements: &[ElementRecord], f: F) -> String {
    let values: Vec<String> = elements.iter().map(f).collect();
    format!("{code},{},{}\n", elements.len(), values.join(","))
}

/// Write one timestep's block.
///
/// Bare ground is exactly `0500,<date>` and `0501,1,0`, which is how the
/// reference writes it and how the Python extractor recognises it.
pub fn write_timestep(step: &ProTimestep) -> String {
    let mut out = format!("0500,{}\n", step.timestamp.format("%d.%m.%Y %H:%M:%S"));
    let e = &step.elements;
    if e.is_empty() {
        out.push_str("0501,1,0\n");
        return out;
    }
    out.push_str(&row("0501", e, |r| value(r.top_height_cm, 2)));
    out.push_str(&row("0502", e, |r| value(r.density, 1)));
    out.push_str(&row("0503", e, |r| value(r.temperature_c, 2)));
    out.push_str(&row("0504", e, |r| r.id.to_string()));
    out.push_str(&row("0505", e, |r| value(r.age_days, 2)));
    out.push_str(&row("0506", e, |r| value(r.lwc_percent, 1)));
    out.push_str(&row("0508", e, |r| value(r.dendricity, 2)));
    out.push_str(&row("0509", e, |r| value(r.sphericity, 2)));
    out.push_str(&row("0510", e, |r| value(r.coordination_number, 1)));
    out.push_str(&row("0511", e, |r| value(r.bond_size_mm, 2)));
    out.push_str(&row("0512", e, |r| value(r.grain_size_mm, 2)));
    // nElems + 1 values: the extra trailing one is the surface-hoar grain type,
    // 0 when there is none. 24 values for 23 elements in the reference output.
    let hoar_code = if step.surface_hoar.is_some() {
        SURFACE_HOAR_CODE
    } else {
        0
    };
    let mut types: Vec<String> = e.iter().map(|r| r.grain_type.to_string()).collect();
    types.push(hoar_code.to_string());
    out.push_str(&format!("0513,{},{}\n", e.len() + 1, types.join(",")));
    match step.surface_hoar {
        Some(h) => out.push_str(&format!(
            "0514,3,{SURFACE_HOAR_CODE},{:.1},{:.0}\n",
            h.grain_size_mm, h.density
        )),
        None => out.push_str("0514,3,-999,-999.0,-999.0\n"),
    }
    out.push_str(&row("0515", e, |r| value(r.ice_percent, 0)));
    out.push_str(&row("0516", e, |r| value(r.air_percent, 0)));
    out.push_str(&row("0517", e, |r| scientific(r.stress_kpa)));
    out.push_str(&row("0518", e, |r| scientific(r.viscosity_gpa_s)));
    out.push_str(&row("0519", e, |r| value(r.soil_percent, 0)));
    out.push_str(&row("0520", e, |r| scientific(r.temperature_gradient)));
    out.push_str(&row("0521", e, |r| scientific(r.conductivity)));
    out.push_str(&row("0522", e, |r| value(r.absorbed_shortwave, 1)));
    out.push_str(&row("0523", e, |r| value(r.deformation_rate, 1)));
    // Profile type and stability class are -1 in every one of the 1,571
    // snow-covered timesteps of the reference run: this configuration does not
    // classify profiles. The position and minimum-index slots follow.
    out.push_str("0530,8,-1,-1,-999,-999,-999,-999,-999,-999\n");
    out.push_str(&row("0531", e, |r| value(r.sdef, 2)));
    out.push_str(&row("0532", e, |r| value(r.sn38, 2)));
    out.push_str(&row("0533", e, |r| value(r.sk38, 2)));
    out.push_str(&row("0534", e, |r| value(r.hand_hardness, 1)));
    out.push_str(&row("0535", e, |r| value(r.optical_grain_size_mm, 2)));
    out.push_str(&row("0601", e, |r| value(r.shear_strength_kpa, 2)));
    out.push_str(&row("0602", e, |r| value(r.grain_size_difference_mm, 2)));
    out.push_str(&row("0603", e, |r| value(r.hardness_difference, 2)));
    out.push_str(&row("0604", e, |r| value(r.ssi, 2)));
    out.push_str(&row("0605", e, |r| value(r.inverse_texture_index, 1)));
    out.push_str(&row("0606", e, |r| value(r.critical_cut_length_m, 2)));
    out
}
