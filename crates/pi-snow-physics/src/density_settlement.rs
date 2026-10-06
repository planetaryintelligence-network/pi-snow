//! Published dry-snow viscosity forms registered by engine-density-v1.

use pi_snow_core::constants::T_MELT;
use pi_snow_core::Element;

use crate::settlement::{Br92, Viscosity};

const DRY_WATER_LIMIT: f64 = 1e-12;
const DENDRITICITY_LIMIT: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    Br92,
    T11,
    K75,
    T11K75,
}

#[derive(Debug, Clone, Copy)]
pub struct DensitySettlementViscosity(pub Scheme);

impl Viscosity for DensitySettlementViscosity {
    fn viscosity(&self, element: &Element, temperature: f64) -> f64 {
        if element.composition.water > DRY_WATER_LIMIT {
            return Br92.viscosity(element, temperature);
        }
        let dendritic = element.micro.dendricity > DENDRITICITY_LIMIT;
        match self.0 {
            Scheme::Br92 => Br92.viscosity(element, temperature),
            Scheme::T11 => t11(element.density(), temperature),
            Scheme::K75 if dendritic => k75(element.density(), temperature),
            Scheme::K75 => Br92.viscosity(element, temperature),
            Scheme::T11K75 if dendritic => k75(element.density(), temperature),
            Scheme::T11K75 => t11(element.density(), temperature),
        }
    }
}

/// Teufelsbauer (2011), Eq. 9 as reproduced by Védrine and Hagenmuller
/// (2026). The published value is MPa s; the engine uses Pa s.
fn t11(density: f64, temperature: f64) -> f64 {
    let temperature_c = temperature - T_MELT;
    let eta_mpa_s = 5e-8
        * density.powf(-0.0371 * temperature_c + 4.4)
        * (1e-4 * (0.018 * density).exp() + 1.0);
    eta_mpa_s * 1e6
}

/// Kojima (1975), Eq. 10 as reproduced by Védrine and Hagenmuller (2026).
/// The published value is MPa s; the engine uses Pa s.
fn k75(density: f64, temperature: f64) -> f64 {
    let temperature_c = temperature - T_MELT;
    7e-9 * density.powf(4.75 - temperature_c / 40.0) * 1e6
}
