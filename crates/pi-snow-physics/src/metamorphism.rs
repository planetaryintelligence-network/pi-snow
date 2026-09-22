//! Snow metamorphism and grain classification.
//!
//! **A declared divergence.** SNOWPACK's metamorphism (`METAMORPHISM_MODEL =
//! DEFAULT`) is in Lehning et al. (2002) Part II, which could not be obtained.
//! The laws here are Crocus's, read in full:
//!
//! * Dry metamorphism of dendricity and sphericity by temperature-gradient
//!   regime: Brun et al. (1992), J. Glaciol. 38, 13-22, eqs. 3-4, with the
//!   regime split at 5 and 15 K m^-1 and the non-dendritic column of Vionnet et
//!   al. (2012), GMD 5, 773-791, Table 1. Rates are per day, T in K.
//! * Kinetic grain growth for fully faceted grains under G > 15 K m^-1:
//!   Marbouty (1980) as given by Vionnet et al. (2012) Appendix B, eqs. B1-B5,
//!   **implemented as printed**. Eq. B2 as printed jumps from 1.0 to 1.3 at
//!   -6 C and exceeds 1, although the text says f runs from 0 to 1; the page
//!   was checked, the transcription is faithful, and Marbouty's original was not
//!   obtained to settle it. Flagged in the gap list.
//! * Wet metamorphism: Brun et al. (1992) eq. 5, theta the liquid water
//!   content as a **mass** percentage (Vionnet Appendix A), rates per day. The
//!   wet grain-growth law for rounded grains, `dv/dt = v0' + v1' theta^3`,
//!   needs coefficients from Brun (1989) that neither paper prints, so wet
//!   rounded grains do not grow here.
//!
//! Grain size of dendritic snow is held at its deposited value, as the
//! reference's output shows fresh snow at `NEW_SNOW_GRAIN_SIZE`; there is no
//! dry growth of rounded grains, as in Crocus's Table 1. Bond radius has no
//! published law in hand and is held constant.
//!
//! **Classification** into the Swiss three-digit code is our rule, drawn from
//! Brun et al.'s (1992) Fig. 1 diagram placing (dendricity, sphericity, size)
//! against the international classes. It is documented on [`classify`] and is
//! expected to be the largest single source of divergence from the reference.

use pi_snow_core::Microstructure;
use pi_snow_core::constants::T_MELT;

const DAY: f64 = 86_400.0;

/// Local conditions of one element for one step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Local {
    /// Element temperature, K.
    pub temperature: f64,
    /// Absolute temperature gradient across the element, K m^-1.
    pub gradient: f64,
    /// Bulk density, kg m^-3.
    pub density: f64,
    /// Liquid water content as a percentage of the element's mass.
    pub lwc_percent: f64,
}

/// History a grain's shape variables cannot express.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct History {
    /// Liquid water has refrozen in this element.
    pub refrozen: bool,
    /// The element is buried surface hoar.
    pub surface_hoar: bool,
    /// The element holds liquid water now.
    pub wet: bool,
}

/// Advance one element's microstructure by `dt` seconds.
pub fn step(m: &mut Microstructure, l: &Local, history: &mut History, dt: f64) {
    let days = dt / DAY;
    let arrhenius = (-6000.0 / l.temperature).exp();
    let g = l.gradient.abs();
    history.wet = l.lwc_percent > 0.0;

    if l.lwc_percent > 0.0 {
        // Brun et al. (1992) eq. 5.
        let rate = l.lwc_percent.powi(3) / 16.0 * days;
        if m.dendricity > 0.0 {
            m.dendricity -= rate;
            m.sphericity += rate;
        } else if m.sphericity < 1.0 {
            m.sphericity += rate;
        }
    } else if m.dendricity > 0.0 {
        if g <= 5.0 {
            m.dendricity -= 2e8 * arrhenius * days;
            m.sphericity += 1e9 * arrhenius * days;
        } else {
            let rate = 2e8 * arrhenius * g.powf(0.4) * days;
            m.dendricity -= rate;
            m.sphericity -= rate;
        }
    } else if g <= 5.0 {
        m.sphericity += 1e9 * arrhenius * days;
    } else if m.sphericity > 0.0 || g <= 15.0 {
        m.sphericity -= 2e8 * arrhenius * g.powf(0.4) * days;
    } else {
        // Fully faceted under a strong gradient: kinetic growth of the
        // diameter, Vionnet Appendix B. The state carries a radius.
        let growth =
            marbouty_f(l.temperature) * marbouty_h(l.density) * marbouty_g(g) * 1.0417e-9 * dt;
        m.grain_radius += 0.5 * growth;
    }
    m.dendricity = m.dendricity.clamp(0.0, 1.0);
    m.sphericity = m.sphericity.clamp(0.0, 1.0);
}

/// Eq. B2 as printed, `T - T_fus` in degrees C.
fn marbouty_f(t: f64) -> f64 {
    let c = t - T_MELT;
    if c < -40.0 {
        0.0
    } else if c < -22.0 {
        0.011 * (c + 40.0)
    } else if c < -6.0 {
        0.2 + 0.05 * (c + 22.0)
    } else {
        1.0 - 0.05 * c
    }
}

/// Eq. B3, density in kg m^-3.
fn marbouty_h(rho: f64) -> f64 {
    if rho < 150.0 {
        1.0
    } else if rho < 400.0 {
        1.0 - 0.004 * (rho - 150.0)
    } else {
        0.0
    }
}

/// Eq. B4, G in K m^-1.
fn marbouty_g(g: f64) -> f64 {
    if g < 15.0 {
        0.0
    } else if g < 25.0 {
        0.01 * (g - 15.0)
    } else if g < 40.0 {
        0.1 + 0.037 * (g - 25.0)
    } else if g < 50.0 {
        0.65 + 0.02 * (g - 40.0)
    } else if g < 70.0 {
        0.85 + 0.0075 * (g - 50.0)
    } else {
        1.0
    }
}

/// Swiss three-digit grain code: primary class, secondary class, marker.
///
/// Our rule, from Brun et al. (1992) Fig. 1, which places fresh snow at the
/// apex (d = 1, s = 0.5), decomposition along falling dendricity, and the two
/// non-dendritic branches at s = 0 (facets, growing to depth hoar with size)
/// and s = 1 (rounds). Class digits are the interface mapping established in
/// the stability module (1 PP ... 9 FCxr). Thresholds below are ours:
///
/// * refrozen history: 772 (melt forms, crust marker 2), as the reference
///   writes for melt-freeze crusts; the frozen extractor's crust test reads
///   exactly this code;
/// * currently wet: 770;
/// * buried surface hoar: 660;
/// * d >= 0.75: 110; 0.5 <= d < 0.75: 120; 0 < d < 0.5: 220, secondary 3 or 4
///   by whether s is above or below 0.5;
/// * d = 0, s >= 0.5: rounds, 330, or 334 for s < 0.75;
/// * d = 0, s < 0.5: facets 440 (443 for s >= 0.3), or depth hoar 550 once the
///   grain is at least 1.5 mm and s < 0.2.
pub fn classify(m: &Microstructure, h: &History, _density: f64) -> u16 {
    let (d, s) = (m.dendricity, m.sphericity);
    let size_mm = 2e3 * m.grain_radius;
    if h.refrozen {
        return 772;
    }
    if h.wet {
        return 770;
    }
    if h.surface_hoar {
        return 660;
    }
    if d >= 0.75 {
        110
    } else if d >= 0.5 {
        120
    } else if d > 0.0 {
        if s >= 0.5 { 223 } else { 224 }
    } else if s >= 0.5 {
        if s >= 0.75 { 330 } else { 334 }
    } else if size_mm >= 1.5 && s < 0.2 {
        550
    } else if s >= 0.3 {
        443
    } else {
        440
    }
}
