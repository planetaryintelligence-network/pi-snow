//! Snow albedo and shortwave absorption with depth.
//!
//! **A declared divergence.** The configuration selects `LEHNING_2` albedo and
//! `MULTI_BAND` absorption, both in Lehning et al. (2002) Part III, which
//! could not be obtained. Implemented instead is Crocus's three-band scheme,
//! Vionnet et al. (2012), GMD 5, 773-791, eq. 13 (optical diameter), eq. 14
//! (exponential extinction) and Table 4, read on the page:
//!
//! * bands 0.3-0.8, 0.8-1.5, 1.5-2.8 um, taking 0.71, 0.21, 0.08 of the
//!   incoming shortwave;
//! * band albedos from the optical diameter of the top 3 cm, the visible band
//!   darkening with the surface age scaled by `min(1, max(P / 870 hPa, 0.5))`;
//! * extinction coefficients `max(40, 0.00192 rho / sqrt(d_opt))` and
//!   `max(100, 0.01098 rho / sqrt(d_opt))` m^-1 for the first two bands, per
//!   element; the third band is absorbed at the surface (beta infinite);
//! * shortwave that passes through a thin pack goes to the ground flux.
//!
//! Brun et al. (1992) note the age coefficient was fitted at Col de Porte with
//! unusually strong dust deposition from nearby forest and should be smaller at
//! higher, cleaner sites; it is used as printed.

use pi_snow_core::{Column, Microstructure};

const BAND_FRACTION: [f64; 3] = [0.71, 0.21, 0.08];
/// Depth over which the surface properties are averaged, m.
const SURFACE_DEPTH: f64 = 0.03;
/// Col de Porte reference pressure, Pa.
const P_CDP: f64 = 87_000.0;

/// Optical diameter, m, eq. 13.
pub fn optical_diameter(m: &Microstructure) -> f64 {
    let (d, s) = (m.dendricity, m.sphericity);
    if d > 0.0 {
        1e-4 * (d + (1.0 - d) * (4.0 - s))
    } else {
        let gs = 2.0 * m.grain_radius;
        gs * s + (1.0 - s) * (4e-4_f64).max(gs / 2.0)
    }
}

/// Where the shortwave went, W m^-2.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Shortwave {
    /// Broadband albedo.
    pub albedo: f64,
    /// Reflected shortwave, `albedo * incoming`.
    pub reflected: f64,
    /// Absorbed at the surface (the third band), W m^-2.
    pub surface: f64,
    /// Absorbed within each element, bottom element first, W m^-2.
    pub per_element: Vec<f64>,
    /// Transmitted through the whole pack to the ground, W m^-2.
    pub to_ground: f64,
}

/// Partition `incoming` shortwave (W m^-2, on the surface) at surface pressure
/// `pressure` (Pa).
pub fn shortwave(column: &Column, incoming: f64, pressure: f64) -> Shortwave {
    let n = column.elements.len();
    // Only an empty column returns early: albedo is a property of the snow,
    // not of the flux, and must be reported at night too.
    if n == 0 {
        return Shortwave {
            per_element: vec![0.0; n],
            ..Shortwave::default()
        };
    }
    let (d_opt, age) = surface_properties(column);
    let root = d_opt.sqrt();
    let aging = (pressure / P_CDP).clamp(0.5, 1.0) * 0.2 * age / 60.0;
    let a1 = (0.92_f64.min(0.96 - 1.58 * root) - aging).max(0.6);
    let a2 = (0.9 - 15.4 * root).max(0.3);
    let dp = d_opt.min(0.0023);
    let a3 = 346.3 * dp - 32.31 * dp.sqrt() + 0.88;
    shortwave_with_band_albedo(column, incoming, [a1, a2, a3])
}

/// Partition shortwave with supplied surface albedos while retaining
/// Vionnet's three-band fractions and within-pack extinction (eq. 14).
pub fn shortwave_with_band_albedo(
    column: &Column,
    incoming: f64,
    band_albedo: [f64; 3],
) -> Shortwave {
    let n = column.elements.len();
    if n == 0 {
        return Shortwave {
            per_element: vec![0.0; n],
            ..Shortwave::default()
        };
    }
    let albedo: f64 = BAND_FRACTION
        .iter()
        .zip(band_albedo)
        .map(|(w, a)| w * a)
        .sum();

    let incoming = incoming.max(0.0);
    let mut per_element = vec![0.0; n];
    // Band 3 is absorbed at the surface.
    let surface = BAND_FRACTION[2] * (1.0 - band_albedo[2]) * incoming;
    let mut to_ground = 0.0;
    for band in 0..2 {
        let mut flux = BAND_FRACTION[band] * (1.0 - band_albedo[band]) * incoming;
        for (index, e) in column.elements.iter().enumerate().rev() {
            let root_e = optical_diameter(&e.micro).sqrt();
            let beta = if band == 0 {
                (0.00192 * e.density() / root_e).max(40.0)
            } else {
                (0.01098 * e.density() / root_e).max(100.0)
            };
            let transmitted = flux * (-beta * e.thickness).exp();
            per_element[index] += flux - transmitted;
            flux = transmitted;
        }
        to_ground += flux;
    }
    Shortwave {
        albedo,
        reflected: albedo * incoming,
        surface,
        per_element,
        to_ground,
    }
}

/// Thickness-weighted optical diameter and age of the top `SURFACE_DEPTH`.
fn surface_properties(column: &Column) -> (f64, f64) {
    let (mut depth, mut d_sum, mut age_sum) = (0.0, 0.0, 0.0);
    for e in column.elements.iter().rev() {
        let take = e.thickness.min(SURFACE_DEPTH - depth);
        if take <= 0.0 {
            break;
        }
        d_sum += optical_diameter(&e.micro) * take;
        age_sum += e.age_days * take;
        depth += take;
    }
    (d_sum / depth, age_sum / depth)
}
