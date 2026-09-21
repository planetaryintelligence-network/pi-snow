//! SNO: the initial and final snow profile, written in the SMET container with a
//! fixed layer field list.
//!
//! Round-trip is byte identical for the same reason SMET's is: a run's final
//! state is an input to the next run, and a formatting difference would show up
//! as a spurious divergence.

use std::fmt;

use chrono::{DateTime, Utc};
use pi_snow_core::constants::RHO_ICE;
use pi_snow_core::{Column, Composition, Element, Microstructure};

use crate::error::IoFormatError;
use crate::smet::parse_timestamp;

const FORMAT: &str = "SNO";

/// One layer row of a SNO file, in the file's own units.
#[derive(Debug, Clone, PartialEq)]
pub struct SnoLayer {
    /// Deposition date of the layer.
    pub deposited: DateTime<Utc>,
    /// `Layer_Thick`, m.
    pub thickness: f64,
    /// `T`, K.
    pub temperature: f64,
    /// `Vol_Frac_I`, `Vol_Frac_W`, `Vol_Frac_V`, `Vol_Frac_S` (1).
    pub volume_fraction_ice: f64,
    pub volume_fraction_water: f64,
    pub volume_fraction_void: f64,
    pub volume_fraction_soil: f64,
    /// `Rho_S` kg m^-3, `Conduc_S` W m^-1 K^-1, `HeatCapac_S` J kg^-1 K^-1.
    pub soil_density: f64,
    pub soil_conductivity: f64,
    pub soil_heat_capacity: f64,
    /// `rg`, grain **radius** in mm.
    pub grain_radius_mm: f64,
    /// `rb`, bond **radius** in mm.
    pub bond_radius_mm: f64,
    /// `dd` dendricity and `sp` sphericity (1).
    pub dendricity: f64,
    pub sphericity: f64,
    /// `mk`, the history marker.
    pub marker: u16,
    /// `mass_hoar`, kg m^-2.
    pub mass_hoar: f64,
    /// `ne`, how many elements this row expands into.
    pub elements: usize,
    /// `CDot`, the stress rate, Pa s^-1.
    pub stress_rate: f64,
    /// `metamo`, the accumulated metamorphism indicator.
    pub metamorphism: f64,
    raw: String,
}

/// A parsed SNO file.
#[derive(Debug, Clone, PartialEq)]
pub struct SnoFile {
    pub station_id: String,
    pub slope_angle_deg: f64,
    pub slope_azimuth_deg: f64,
    pub profile_date: Option<DateTime<Utc>>,
    /// `HS_Last`, m.
    pub hs_last: f64,
    pub soil_layer_count: usize,
    pub layers: Vec<SnoLayer>,
    header_lines: Vec<String>,
    trailing_newline: bool,
}

fn header_f64(lines: &[String], key: &'static str) -> Result<f64, IoFormatError> {
    header_str(lines, key)?.parse().map_err(|_| IoFormatError::BadValue {
        format: FORMAT,
        line: 0,
        what: "header number",
        text: key.to_string(),
    })
}

fn header_str<'a>(lines: &'a [String], key: &'static str) -> Result<&'a str, IoFormatError> {
    lines
        .iter()
        .find_map(|l| {
            let (k, v) = l.split_once('=')?;
            (k.trim() == key).then(|| v.trim())
        })
        .ok_or(IoFormatError::MissingHeader { format: FORMAT, key })
}

impl SnoFile {
    pub fn parse(text: &str) -> Result<Self, IoFormatError> {
        let trailing_newline = text.ends_with('\n');
        let mut header_lines = Vec::new();
        let mut lines = text.lines().enumerate();
        let mut saw_data = false;
        for (_, line) in lines.by_ref() {
            header_lines.push(line.to_string());
            if line.trim() == "[DATA]" {
                saw_data = true;
                break;
            }
        }
        if !saw_data {
            return Err(IoFormatError::NoDataSection { format: FORMAT });
        }

        let mut layers = Vec::new();
        for (index, line) in lines {
            if line.trim().is_empty() {
                continue;
            }
            layers.push(parse_layer(line, index + 1)?);
        }

        Ok(Self {
            station_id: header_str(&header_lines, "station_id")?.to_string(),
            slope_angle_deg: header_f64(&header_lines, "SlopeAngle")?,
            slope_azimuth_deg: header_f64(&header_lines, "SlopeAzi")?,
            profile_date: header_str(&header_lines, "ProfileDate").ok().and_then(parse_timestamp),
            hs_last: header_f64(&header_lines, "HS_Last")?,
            soil_layer_count: header_f64(&header_lines, "nSoilLayerData")? as usize,
            layers,
            header_lines,
            trailing_newline,
        })
    }

    /// Expand the layer table into the engine's column state.
    ///
    /// A layer with `ne > 1` becomes `ne` equal elements, which is how the format
    /// expresses a thick homogeneous layer without writing it out element by
    /// element. Nodal temperatures are taken as the layer temperatures at the
    /// interfaces, with the bottom node set from the lowest layer, since the file
    /// records element temperatures and the engine solves on nodes.
    pub fn to_column(&self) -> Column {
        let ground = self.layers.first().map(|l| l.temperature).unwrap_or(pi_snow_core::constants::T_MELT);
        let mut column = Column::bare_ground(ground, self.slope_angle_deg, self.slope_azimuth_deg);
        column.soil_elements = self.soil_layer_count;
        let mut nodes = vec![ground];
        for layer in &self.layers {
            let count = layer.elements.max(1);
            let thickness = layer.thickness / count as f64;
            for _ in 0..count {
                column.elements.push(Element {
                    thickness,
                    composition: Composition {
                        ice: layer.volume_fraction_ice,
                        water: layer.volume_fraction_water,
                        air: layer.volume_fraction_void,
                        soil: layer.volume_fraction_soil,
                    },
                    micro: Microstructure {
                        dendricity: layer.dendricity,
                        sphericity: layer.sphericity,
                        grain_radius: layer.grain_radius_mm * 1e-3,
                        bond_radius: layer.bond_radius_mm * 1e-3,
                        coordination_number: 0.0,
                        marker: layer.marker,
                    },
                    age_days: 0.0,
                    stress: 0.0,
                    stress_rate: layer.stress_rate,
                    metamorphism: layer.metamorphism,
                    soil_density: layer.soil_density,
                    soil_conductivity: layer.soil_conductivity,
                    soil_heat_capacity: layer.soil_heat_capacity,
                });
                nodes.push(layer.temperature);
            }
        }
        if !column.elements.is_empty() {
            column.node_temperature = nodes;
        }
        column.surface_hoar_mass = self.layers.last().map(|l| l.mass_hoar).unwrap_or(0.0);
        debug_assert!(column.is_well_formed());
        let _ = RHO_ICE;
        column
    }
}

fn parse_layer(line: &str, number: usize) -> Result<SnoLayer, IoFormatError> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() != 19 {
        return Err(IoFormatError::WrongArity {
            format: FORMAT,
            line: number,
            expected: 19,
            found: parts.len(),
        });
    }
    let number_at = |index: usize, what: &'static str| -> Result<f64, IoFormatError> {
        parts[index].parse().map_err(|_| IoFormatError::BadValue {
            format: FORMAT,
            line: number,
            what,
            text: parts[index].to_string(),
        })
    };
    Ok(SnoLayer {
        deposited: parse_timestamp(parts[0]).ok_or_else(|| IoFormatError::BadValue {
            format: FORMAT,
            line: number,
            what: "timestamp",
            text: parts[0].to_string(),
        })?,
        thickness: number_at(1, "Layer_Thick")?,
        temperature: number_at(2, "T")?,
        volume_fraction_ice: number_at(3, "Vol_Frac_I")?,
        volume_fraction_water: number_at(4, "Vol_Frac_W")?,
        volume_fraction_void: number_at(5, "Vol_Frac_V")?,
        volume_fraction_soil: number_at(6, "Vol_Frac_S")?,
        soil_density: number_at(7, "Rho_S")?,
        soil_conductivity: number_at(8, "Conduc_S")?,
        soil_heat_capacity: number_at(9, "HeatCapac_S")?,
        grain_radius_mm: number_at(10, "rg")?,
        bond_radius_mm: number_at(11, "rb")?,
        dendricity: number_at(12, "dd")?,
        sphericity: number_at(13, "sp")?,
        marker: number_at(14, "mk")? as u16,
        mass_hoar: number_at(15, "mass_hoar")?,
        elements: number_at(16, "ne")? as usize,
        stress_rate: number_at(17, "CDot")?,
        metamorphism: number_at(18, "metamo")?,
        raw: line.to_string(),
    })
}

impl fmt::Display for SnoFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for line in &self.header_lines {
            writeln!(f, "{line}")?;
        }
        for (index, layer) in self.layers.iter().enumerate() {
            if index + 1 == self.layers.len() && !self.trailing_newline {
                write!(f, "{}", layer.raw)?;
            } else {
                writeln!(f, "{}", layer.raw)?;
            }
        }
        Ok(())
    }
}
