//! New-snow density, deposition, and merging of thin elements.
//!
//! **Density.** The configuration's `LEHNING_NEW` is in Lehning et al. (2002)
//! Part II, which could not be obtained. Three published forms are offered:
//!
//! * `Pahaut`, the default: `109 + 6 (Ta - Tfus) + 26 U^(1/2)`, floor 50 kg
//!   m^-3, Vionnet et al. (2012) GMD 5, read in full. A declared divergence.
//! * `Zwart2007Eq51`: the SNOWPACK density Zwart's (2007) MSc thesis (eq. 5.1,
//!   Table 5.5, read in full) presents as current in 2007. Tested against the
//!   reference run's own freshly deposited elements, it is not what 3.7.0 runs
//!   as `LEHNING_NEW` (bias -74 kg m^-3, and it goes unphysically low), so it is
//!   not the default. The floor applied to it is ours.
//! * `Zwart`: the thesis's own regression III (p. 29, Table 5.4), which the
//!   "uac" arm selects as `HN_DENSITY_MODEL = ZWART`.
//!
//! Pahaut was made the default after that comparison (Pahaut bias +42, RMSE 50;
//! eq. 5.1 bias -74, RMSE 99, against 135 reference fresh elements). No
//! coefficient was changed; a published form was selected.
//!
//! **Deposition.** From the reference's own output (lab notes, black-box
//! test): precipitation waits in a buffer and is deposited as whole elements of
//! `HEIGHT_NEW_ELEM` thickness. That behaviour is copied as an interface
//! observation. Fresh-snow dendricity 1 and sphericity 0.5 are Brun et al.
//! (1992); the grain size is the INI's `NEW_SNOW_GRAIN_SIZE`.
//!
//! **Merging.** `COMBINE_ELEMENTS` with `MINIMUM_L_ELEMENT`: an element thinner
//! than the minimum is merged into the one below (the one above, for the bottom
//! element). The thresholds are the INI's; the rule is ours.

use pi_snow_core::constants::T_MELT;
use pi_snow_core::{Column, Composition, Element, Microstructure};

use crate::water::{node_energies, set_temperatures_from_energies};

/// Floor applied to any density scheme that lacks its own, kg m^-3. Taken from
/// the Pahaut form's published floor; applying it to the others is ours.
pub const MIN_DENSITY: f64 = 50.0;

/// Bond radius given to fresh snow, m. No published source was obtained for
/// SNOWPACK's new-snow bond size; 0.04 mm is the reference output's PRO 0511
/// value for fresh elements (0.08 mm bond size, halved to a radius). Copied
/// from output, flagged in the gap list as unsourced.
pub const NEW_SNOW_BOND_RADIUS: f64 = 0.04e-3;

/// Meteorological conditions at deposition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Conditions {
    /// Air temperature, K.
    pub t_air: f64,
    /// Snow surface temperature, K.
    pub t_surface: f64,
    /// Relative humidity, fraction 0-1.
    pub relative_humidity: f64,
    /// Wind speed, m s^-1.
    pub wind: f64,
}

/// New-snow density parameterisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DensityScheme {
    #[default]
    Pahaut,
    Zwart2007Eq51,
    Zwart,
    Anderson1976,
}

impl DensityScheme {
    /// New-snow density, kg m^-3.
    pub fn density(self, c: &Conditions) -> f64 {
        let ta = c.t_air - T_MELT;
        let rh_percent = 100.0 * c.relative_humidity;
        match self {
            DensityScheme::Pahaut => (109.0 + 6.0 * ta + 26.0 * c.wind.max(0.0).sqrt()).max(50.0),
            DensityScheme::Zwart2007Eq51 => {
                let (tss, vw) = (c.t_surface - T_MELT, c.wind);
                (70.0 + 6.5 * ta + 7.5 * tss + 0.26 * rh_percent + 13.0 * vw
                    - 4.5 * ta * tss
                    - 0.65 * ta * vw
                    - 0.17 * rh_percent * vw
                    + 0.06 * ta * tss * rh_percent)
                    .max(MIN_DENSITY)
            }
            DensityScheme::Zwart => {
                let base = 3.27820894 + 0.03132918 * ta
                    - 0.74783337 * (rh_percent / 100.0).clamp(0.0, 1.0).sqrt().asin()
                    + 0.30943402 * c.wind.max(2.0).log10();
                10f64.powf(if ta < -14.0 { base } else { base - 0.35976206 })
            }
            DensityScheme::Anderson1976 => 50.0 + 1.7 * (ta + 15.0).max(0.0).powf(1.5),
        }
    }
}

/// The deposition buffer: solid precipitation waiting to fill an element.
#[derive(Debug, Clone, PartialEq)]
pub struct Deposition {
    element_height: f64,
    grain_radius: f64,
    buffer: f64,
}

impl Deposition {
    /// `element_height` is `HEIGHT_NEW_ELEM` (m); `grain_radius` is half
    /// `NEW_SNOW_GRAIN_SIZE`, in m.
    pub fn new(element_height: f64, grain_radius: f64) -> Self {
        Self {
            element_height,
            grain_radius,
            buffer: 0.0,
        }
    }

    /// Solid precipitation waiting in the buffer, kg m^-2.
    pub fn buffered(&self) -> f64 {
        self.buffer
    }

    /// Add `solid_mass` kg m^-2 of solid precipitation and deposit every whole
    /// element it completes. Returns the mass deposited, kg m^-2.
    pub fn add(
        &mut self,
        column: &mut Column,
        solid_mass: f64,
        c: &Conditions,
        scheme: DensityScheme,
    ) -> f64 {
        self.buffer += solid_mass.max(0.0);
        let density = scheme.density(c);
        let element_mass = self.element_height * density;
        // A small tolerance so that exactly one element's worth, accumulated in
        // floating point, is not left waiting for want of the last ulp.
        let count = ((self.buffer + 1e-12) / element_mass).floor() as usize;
        if count == 0 {
            return 0.0;
        }
        let node_t = c.t_air.min(T_MELT);
        for _ in 0..count {
            column.elements.push(Element {
                thickness: self.element_height,
                composition: Composition::dry_snow(density),
                micro: Microstructure {
                    dendricity: 1.0,
                    sphericity: 0.5,
                    grain_radius: self.grain_radius,
                    bond_radius: NEW_SNOW_BOND_RADIUS,
                    coordination_number: 1.0,
                    marker: 0,
                },
                age_days: 0.0,
                stress: 0.0,
                stress_rate: 0.0,
                metamorphism: 0.0,
                soil_density: 0.0,
                soil_conductivity: 0.0,
                soil_heat_capacity: 0.0,
                grain_history: crate::grain_classification::NO_HISTORY,
            });
            column.node_temperature.push(node_t);
        }
        let deposited = count as f64 * element_mass;
        self.buffer = (self.buffer - deposited).max(0.0);
        deposited
    }
}

/// Merge every element thinner than `min_length` into a neighbour, conserving
/// mass exactly and nodal energy relative to the melting point.
pub fn merge_thin_elements(column: &mut Column, min_length: f64) {
    if column.elements.len() < 2 {
        return;
    }
    let mut energies = node_energies(column);
    let mut i = 0;
    while i < column.elements.len() && column.elements.len() >= 2 {
        if column.elements[i].thickness >= min_length {
            i += 1;
            continue;
        }
        // Merge element `upper` into `lower`; the node between them goes.
        let (lower, upper) = if i > 0 { (i - 1, i) } else { (0, 1) };
        let shared_node = lower + 1;
        let a = column.elements[lower].clone();
        let b = column.elements.remove(upper);
        let merged = combine(&a, &b);
        column.elements[lower] = merged;
        let node_energy = energies.remove(shared_node);
        column.node_temperature.remove(shared_node);
        // The removed node's energy goes half to each surviving neighbour.
        energies[lower] += 0.5 * node_energy;
        energies[lower + 1] += 0.5 * node_energy;
        i = lower;
    }
    set_temperatures_from_energies(column, &energies);
}

/// Combine two adjacent elements: volumes add, so each fraction is the
/// volume-weighted mean and mass is conserved; microstructure is mass-weighted.
fn combine(a: &Element, b: &Element) -> Element {
    let h = a.thickness + b.thickness;
    let vw = |x: f64, y: f64| (x * a.thickness + y * b.thickness) / h;
    let (ma, mb) = (a.mass(), b.mass());
    let mw = |x: f64, y: f64| {
        if ma + mb > 0.0 {
            (x * ma + y * mb) / (ma + mb)
        } else {
            0.5 * (x + y)
        }
    };
    Element {
        thickness: h,
        composition: Composition {
            ice: vw(a.composition.ice, b.composition.ice),
            water: vw(a.composition.water, b.composition.water),
            air: vw(a.composition.air, b.composition.air),
            soil: vw(a.composition.soil, b.composition.soil),
        },
        micro: Microstructure {
            dendricity: mw(a.micro.dendricity, b.micro.dendricity),
            sphericity: mw(a.micro.sphericity, b.micro.sphericity),
            grain_radius: mw(a.micro.grain_radius, b.micro.grain_radius),
            bond_radius: mw(a.micro.bond_radius, b.micro.bond_radius),
            coordination_number: mw(a.micro.coordination_number, b.micro.coordination_number),
            marker: a.micro.marker.max(b.micro.marker),
        },
        age_days: mw(a.age_days, b.age_days),
        stress: a.stress,
        stress_rate: mw(a.stress_rate, b.stress_rate),
        metamorphism: mw(a.metamorphism, b.metamorphism),
        soil_density: a.soil_density,
        soil_conductivity: a.soil_conductivity,
        soil_heat_capacity: a.soil_heat_capacity,
        grain_history: crate::grain_classification::merge_history(a.grain_history, b.grain_history),
    }
}
