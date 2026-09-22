//! Overburden stress and viscous settlement.
//!
//! The settlement law is Newtonian: strain rate equals stress over viscosity,
//! `dD/D = -(sigma / eta) dt` (Vionnet et al. 2012, *The detailed snowpack
//! scheme Crocus and its implementation in SURFEX v7.2*, GMD 5, 773-791, Eq. 5).
//! Vionnet states the same law underlies both Crocus and SNOWPACK below about
//! 0.4 MPa; what differs between models is the viscosity.
//!
//! **The viscosity here is not SNOWPACK's.** The configuration we compare
//! against selects `VISCOSITY_MODEL = DEFAULT`, whose published form lives in
//! Lehning et al. (2002) Part II, which nobody on this project has been able to
//! read. Implemented instead is Crocus's BR92 law, verified from two
//! independent papers. This is a declared divergence, and it is the reason the
//! stage 2 golden comparison is expected to disagree rather than hoped to agree.
//!
//! A closer candidate exists and is deliberately **not** implemented: the
//! Kojima form Védrine and Hagenmuller (2026, The Cryosphere 20, 1257-1277,
//! Eq. 10) report SNOWPACK using for dendritic new snow,
//! `eta = 7e-9 * rho^(4.75 - T/40)` with T in degrees Celsius. Their text leaves
//! the unit of `eta` ambiguous between Pa s and MPa s, and a factor of a million
//! is not something to settle by picking the reading that best matches the
//! reference's output. See the experiment's gap list.

use pi_snow_core::constants::{GRAVITY, RHO_ICE, RHO_WATER, T_MELT};
use pi_snow_core::{Column, Element};

/// A snow viscosity parameterisation, Pa s.
pub trait Viscosity {
    /// Viscosity of `element` at temperature `temperature` (K).
    fn viscosity(&self, element: &Element, temperature: f64) -> f64;
}

/// The BR92 compaction viscosity of Crocus.
///
/// `eta = f1 f2 eta0 (rho / c_eta) exp(a_eta (T_fus - T) + b_eta rho)`
///
/// Vionnet et al. (2012) Eq. 7 with coefficients from its Appendix A, and
/// reproduced with identical coefficients as "BR92" in Védrine and Hagenmuller
/// (2026) Eq. 8. Both trace it to Brun et al. (1992), which was not itself read.
/// Védrine and Hagenmuller note it settles too fast above a solid fraction of
/// about 0.2 relative to their reinterpreted laboratory data, which bounds how
/// far to trust it in old, dense layers.
#[derive(Debug, Clone, Copy, Default)]
pub struct Br92;

impl Br92 {
    /// Reference viscosity, Pa s. Printed as 7.62237e6 kg s^-1 m^-1.
    const ETA0: f64 = 7.622_37e6;
    /// Temperature coefficient, K^-1.
    const A_ETA: f64 = 0.1;
    /// Density coefficient, m^3 kg^-1.
    const B_ETA: f64 = 0.023;
    /// Density normalisation, kg m^-3.
    const C_ETA: f64 = 250.0;

    /// `f1 = 1 / (1 + 60 theta_w)`, Vionnet et al. (2012) Eq. 8, where
    /// `W_liq / (rho_w D)` in the paper is the volumetric liquid water content.
    pub fn liquid_water_factor(volumetric_water: f64) -> f64 {
        1.0 / (1.0 + 60.0 * volumetric_water)
    }

    /// `f2 = min(4, exp(min(g1, gs - g2) / g3))`, Vionnet et al. (2012) Eq. 9,
    /// with g1 = 0.4 mm, g2 = 0.2 mm, g3 = 0.1 mm.
    ///
    /// `grain_size` is in metres. Crocus's `gs` is an optical diameter; the
    /// engine carries a grain radius, and passes twice it. That conversion is a
    /// known approximation: optical and geometric grain size are not the same
    /// quantity, and the round-1 critique flags radius-versus-diameter as a
    /// factor-of-two trap across the whole equation set.
    pub fn grain_factor(grain_size: f64) -> f64 {
        const G1: f64 = 0.4e-3;
        const G2: f64 = 0.2e-3;
        const G3: f64 = 0.1e-3;
        (G1.min(grain_size - G2) / G3).exp().min(4.0)
    }
}

impl Viscosity for Br92 {
    fn viscosity(&self, element: &Element, temperature: f64) -> f64 {
        let rho = element.density();
        let f1 = Self::liquid_water_factor(element.composition.water);
        let f2 = Self::grain_factor(2.0 * element.micro.grain_radius);
        f1 * f2 * Self::ETA0 * (rho / Self::C_ETA)
            * (Self::A_ETA * (T_MELT - temperature) + Self::B_ETA * rho).exp()
    }
}

/// Recompute every element's overburden stress, Pa, negative in compression.
///
/// Each element is evaluated at its own mid-height: the weight of everything
/// above it plus half its own. Vionnet et al. (2012) Eq. 6 states the half-own
/// weight only for the top layer; applying it to every layer is our choice.
///
/// Thickness is vertical (`PERP_TO_SLOPE = false`), so the slope-normal
/// component carries `cos^2(psi)`: one cosine converts vertical mass per
/// horizontal area to mass per slope area, the other takes the normal component
/// of its weight. This agrees with Vionnet's `g cos(psi) rho D` once his
/// slope-normal `D` is written as `h cos(psi)`.
pub fn update_stress(column: &mut Column) {
    let factor = GRAVITY * column.slope_angle_deg.to_radians().cos().powi(2);
    let mut above = 0.0;
    for element in column.elements.iter_mut().rev() {
        let own = element.mass();
        element.stress = -factor * (above + 0.5 * own);
        above += own;
    }
}

/// Settle one element under `stress` (Pa, negative in compression) with
/// viscosity `viscosity` (Pa s) over `dt` seconds.
///
/// Integrated exactly for a constant viscosity over the step, as
/// `D <- D exp(sigma dt / eta)`, so a large step cannot overshoot to negative
/// thickness the way forward Euler can. Mass is conserved by scaling every
/// volume fraction by the inverse thickness ratio. The element is not allowed
/// to compact past solid ice plus its liquid water.
pub fn settle(element: &mut Element, stress: f64, viscosity: f64, dt: f64) {
    let ratio = (stress * dt / viscosity).exp();
    let c = &element.composition;
    // Thinnest the element can become: zero air.
    let solid = c.ice + c.water + c.soil;
    let ratio = ratio.max(solid);
    element.thickness *= ratio;
    let c = &mut element.composition;
    c.ice /= ratio;
    c.water /= ratio;
    c.soil /= ratio;
    c.air = (1.0 - c.ice - c.water - c.soil).max(0.0);
    debug_assert!(c.is_consistent(1e-9), "settlement broke the fractions: {c:?}");
    debug_assert!(c.ice * RHO_ICE <= RHO_ICE + 1e-9 && c.water * RHO_WATER >= 0.0);
}

/// One settlement step over the whole column.
///
/// Stress is recomputed first from the current state, so the step uses the
/// load at its start. The viscous deformation rate PRO 0523 reports is the
/// strain rate `sigma / eta` from this same evaluation.
pub fn step<V: Viscosity>(column: &mut Column, dt: f64, viscosity: V) {
    update_stress(column);
    let temperatures = column.element_temperatures();
    for (element, temperature) in column.elements.iter_mut().zip(temperatures) {
        let eta = viscosity.viscosity(element, temperature);
        let stress = element.stress;
        settle(element, stress, eta, dt);
    }
}
