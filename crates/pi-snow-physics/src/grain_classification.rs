//! Grain-type classification of Crocus v3.0.2 (`GRAIN_CLASSIFICATION=LAFAYSSE2026`).
//!
//! Source, read on the page: Lafaysse et al. (2026), *Version 3.0.2 of the
//! Crocus snowpack model*, Geosci. Model Dev. 19, 6273-6334,
//! doi:10.5194/gmd-19-6273-2026:
//!
//! * Eq. 144 (Sect. 3.2, p. 6297): grain type from dendricity, sphericity,
//!   grain size and the historical tracker h;
//! * Table 2 (p. 6277): the meaning of h = 0..5;
//! * Eq. 50 (p. 6284): the update of h at the end of each metamorphism step;
//! * Eqs. D1, D2, D5, D6 (pp. 6306-6307): dendricity and grain size are exact
//!   functions of the optical diameter and sphericity, so Eq. 144 applies to
//!   this engine's (dendricity, sphericity, size) state directly.
//!
//! What the paper leaves open and is assumed here (plan.md K1-K8):
//!
//! * K1: grain size in Eq. 144 is in mm (the paper's other equations use m,
//!   where 0.55 and 1.05 are impossible);
//! * K2: "h in (2; 4)" is h in {2, 4} and "(3; 5)" is {3, 5};
//! * K3: "RG + C" in the h = 1 branch is RG+FC, as the preprint prints it;
//! * K4: in X+Y, X is the primary class (the paper's MEPRA tests, Eqs.
//!   156-158, treat FC+DH as a weak layer and RG+FC as a slab);
//! * K5: marker digit 2 for melt forms refrozen after wetting (h in {4, 5}),
//!   the interface convention of the reference's 772 crust code, else 0;
//! * K6: merging two elements takes the union of their Table 2 flags.
//!
//! * K9: S = 0 and S = 1 are exact tests, as printed, reached through the
//!   clamp at the end of each metamorphism step;
//! * K10: "below the melting point" in Eq. 50 compares the element temperature
//!   (mean of its two nodes after phase change) with this engine's `T_MELT`;
//!   the paper's T0 is the triple point. Unlike Crocus's Eq. 3, pi-snow does
//!   not pin a wet element at the melting point, so a slightly wet element can
//!   pass from h = 2 to 4 while wet; {2, 4} and {3, 5} share every class and
//!   differ only in the K5 marker digit.

use crate::stability::GrainClass;
use pi_snow_core::Microstructure;
use pi_snow_core::constants::T_MELT;

/// Table 2's no-history value, which every new element starts with.
pub const NO_HISTORY: u8 = 0;

/// Liquid water mass fraction above which rounded grains record a liquid
/// history (Eq. 50: l / (rho z) > 0.005).
const WET_HISTORY_FRACTION: f64 = 0.005;

/// Eq. 50: the historical tracker after one metamorphism step.
///
/// `m` is the element's microstructure after the step, `temperature` its
/// temperature in K and `liquid_mass_fraction` its liquid water mass over its
/// total mass. Dendritic snow keeps its history.
pub fn update_history(
    h: u8,
    m: &Microstructure,
    temperature: f64,
    liquid_mass_fraction: f64,
) -> u8 {
    if m.dendricity > 0.0 {
        return h;
    }
    let s = m.sphericity;
    if s == 0.0 && h == 0 {
        1
    } else if s == 1.0 && liquid_mass_fraction > WET_HISTORY_FRACTION {
        match h {
            0 => 2,
            1 => 3,
            _ => h,
        }
    } else if temperature < T_MELT {
        match h {
            2 => 4,
            3 => 5,
            _ => h,
        }
    } else {
        h
    }
}

/// Table 2's three flags: liquid water, faceted crystals, more than one
/// freeze-melt cycle, at any time before now.
fn flags(h: u8) -> (bool, bool, bool) {
    match h {
        0 => (false, false, false),
        1 => (false, true, false),
        2 => (true, false, false),
        3 => (true, true, false),
        4 => (true, false, true),
        _ => (true, true, true),
    }
}

fn from_flags(liquid: bool, faceted: bool, cycles: bool) -> u8 {
    match (liquid, faceted, cycles) {
        (false, false, _) => 0,
        (false, true, _) => 1,
        (true, false, false) => 2,
        (true, true, false) => 3,
        (true, false, true) => 4,
        (true, true, true) => 5,
    }
}

/// K6: the history of two merged elements, the union of their Table 2 flags.
pub fn merge_history(a: u8, b: u8) -> u8 {
    let (la, fa, ca) = flags(a);
    let (lb, fb, cb) = flags(b);
    from_flags(la || lb, fa || fb, ca || cb)
}

/// A primary class, a secondary class, and the Swiss three-digit code.
fn code(primary: GrainClass, secondary: GrainClass, marker: u16) -> u16 {
    let digit = |c: GrainClass| c as u16 + 1;
    100 * digit(primary) + 10 * digit(secondary) + marker
}

/// Eq. 144: the Swiss three-digit grain code of one element.
pub fn lafaysse2026(m: &Microstructure, h: u8) -> u16 {
    use GrainClass::*;
    let (d, s) = (m.dendricity, m.sphericity);
    let gs_mm = 2e3 * m.grain_radius;
    let pure = |c| code(c, c, 0);
    if d > 0.0 {
        return if d < 0.3 {
            if s < 0.55 {
                code(DF, FC, 0)
            } else {
                code(DF, RG, 0)
            }
        } else if d < 0.6 {
            pure(DF)
        } else if d < 0.8 {
            code(PP, DF, 0)
        } else {
            pure(PP)
        };
    }
    let cold_rule = || {
        if s < 0.2 {
            pure(FC)
        } else if s < 0.8 {
            code(RG, FC, 0)
        } else {
            pure(RG)
        }
    };
    let refrozen = matches!(h, 4 | 5);
    let mf = |secondary| code(MF, secondary, if refrozen { 2 } else { 0 });
    match h {
        0 => cold_rule(),
        1 => {
            if gs_mm < 0.55 {
                cold_rule()
            } else if gs_mm < 1.05 {
                if s < 0.55 { code(FC, DH, 0) } else { mf(DH) }
            } else if s < 0.55 {
                pure(DH)
            } else {
                mf(DH)
            }
        }
        2 | 4 => {
            if s < 0.55 {
                mf(FC)
            } else if gs_mm < 0.55 {
                code(RG, MF, 0)
            } else {
                mf(MF)
            }
        }
        _ => {
            if s < 0.55 {
                mf(FC)
            } else {
                mf(DH)
            }
        }
    }
}
