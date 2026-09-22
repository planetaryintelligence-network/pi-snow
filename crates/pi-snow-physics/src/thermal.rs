//! Heat conduction on the layered column.
//!
//! One-dimensional transient conduction, discretised with linear two-node
//! finite elements and integrated with the backward-Euler (theta = 1) scheme.
//! Backward Euler is chosen over Crank-Nicolson deliberately: the element stack
//! spans two and a half orders of magnitude in thickness (2.5 mm to tens of
//! centimetres), which makes the system stiff, and Crank-Nicolson rings on stiff
//! systems rather than damping. The cost is first-order accuracy in time, which
//! at a 10-minute step is small beside the error already introduced by splitting
//! phase change out of the thermal solve.
//!
//! **Phase change is not here.** Following the decoupled pattern that Wever et
//! al. (2014, The Cryosphere 8, 257-274, section 3.3.1) describe and that this
//! configuration selects with `COUPLEDPHASECHANGES = false`, the conduction
//! solve holds the volumetric heat capacity fixed across the step, and melting
//! and refreezing are applied afterwards from the resulting temperature profile.
//! That split is what makes this module testable against closed-form solutions.
//!
//! Sign convention: `z` increases upward from the ground, node 0 is the ground
//! and the last node is the snow surface. A positive flux is energy entering the
//! snow.

use pi_snow_core::constants::{C_AIR, C_ICE, C_WATER, K_ICE, RHO_AIR, RHO_ICE, RHO_WATER};
use pi_snow_core::{Column, Element};

/// Which published effective-thermal-conductivity parameterisation to use.
///
/// Both implemented forms are Yen's, and both were read from a source.
/// Sturm et al. (1997) and Calonne et al. (2011) are the other two standard
/// choices; their coefficients have only been seen second-hand, so they are
/// absent rather than implemented, since a wrong coefficient here propagates
/// into every temperature in the profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[derive(Default)]
pub enum Conductivity {
    /// `lambda = 2.22362 * rho^1.885`, rho in Mg m^-3.
    ///
    /// Yen (1981), *Review of thermal properties of snow, ice and sea ice*,
    /// CRREL Report 81-10, eq. 34, p. 16, read in the original: his regression
    /// on the combined data of seven earlier investigators, correlation 0.8614.
    /// Yen notes it runs low as density approaches that of ice. The default,
    /// because it is the primary source.
    #[default]
    Yen1981,
    /// `k = k_ice * (rho / rho_water)^1.88`.
    ///
    /// Vionnet et al. (2012), *The detailed snowpack scheme Crocus and its
    /// implementation in SURFEX v7.2*, Geoscientific Model Development 5,
    /// 773-791, Eq. 18, where it is attributed to Yen (1981). Read from
    /// Vionnet, not from Yen's CRREL report, so it is a secondary citation.
    ///
    /// Note it does **not** tend to `k_ice` as density tends to ice density; at
    /// 917 kg m^-3 it gives 0.8497 of it. That is a property of the fit, not
    /// a transcription error, and there is a test asserting it stays that way.
    ///
    /// With this crate's `K_ICE` of 2.3 it runs 3 to 4 % above Yen's original
    /// coefficients; kept as a labelled variant for comparison with Crocus.
    YenCrocus,
}

/// Exponent in the Yen form as printed in Vionnet et al. (2012) Eq. 18.
const YEN_EXPONENT: f64 = 1.88;
/// Coefficient and exponent of Yen (1981) eq. 34 as printed, rho in Mg m^-3.
const YEN_1981_COEFFICIENT: f64 = 2.22362;
const YEN_1981_EXPONENT: f64 = 1.885;

impl Conductivity {
    /// Effective thermal conductivity of snow of bulk density `density`
    /// (kg m^-3), in W m^-1 K^-1.
    pub fn of_density(self, density: f64) -> f64 {
        match self {
            Conductivity::Yen1981 => {
                YEN_1981_COEFFICIENT * (density / 1000.0).powf(YEN_1981_EXPONENT)
            }
            Conductivity::YenCrocus => K_ICE * (density / RHO_WATER).powf(YEN_EXPONENT),
        }
    }
}

/// Volumetric heat capacity of an element, J m^-3 K^-1.
///
/// The mixture rule of Bouvet et al. (2024), *Multiscale modeling of heat and
/// mass transfer in dry snow*, The Cryosphere 18, 4285-4313, Eq. 25, extended
/// with the liquid-water term for wet snow. Soil is included for completeness
/// though this configuration has no soil column.
pub fn volumetric_heat_capacity(element: &Element) -> f64 {
    let c = &element.composition;
    c.ice * RHO_ICE * C_ICE
        + c.water * RHO_WATER * C_WATER
        + c.air * RHO_AIR * C_AIR
        + c.soil * element.soil_density * element.soil_heat_capacity
}

/// Column-integrated sensible internal energy relative to 0 K, J m^-2.
///
/// Relative to an arbitrary datum, which is all a conservation test needs. It
/// carries no latent term, because at this stage no phase change happens, and
/// adding one would hide a leak rather than expose it.
pub fn internal_energy(column: &Column) -> f64 {
    element_temperatures(column)
        .iter()
        .zip(&column.elements)
        .map(|(&t, e)| volumetric_heat_capacity(e) * e.thickness * t)
        .sum()
}

fn element_temperatures(column: &Column) -> Vec<f64> {
    column
        .node_temperature
        .windows(2)
        .map(|w| 0.5 * (w[0] + w[1]))
        .collect()
}

/// Advance one step with both ends held at a prescribed temperature.
///
/// `bottom` and `top` are in kelvin, `dt` in seconds. Used by the analytic
/// tests and by nothing else: the production upper boundary is the energy
/// balance, because this configuration sets `CHANGE_BC = false`.
pub fn step_dirichlet(column: &mut Column, bottom: f64, top: f64, dt: f64, k: Conductivity) {
    solve(
        column,
        dt,
        k,
        Boundary::Dirichlet(bottom),
        Boundary::Dirichlet(top),
    );
}

/// Advance one step with zero flux at both ends.
///
/// The conservation oracle. A Dirichlet boundary legitimately adds and removes
/// energy, so it cannot detect a leak; this can.
pub fn step_adiabatic(column: &mut Column, dt: f64, k: Conductivity) {
    solve(column, dt, k, Boundary::Flux(0.0), Boundary::Flux(0.0));
}

/// Advance one step with a prescribed flux at the top and at the bottom.
///
/// This is the production entry point. The surface energy balance supplies the
/// top flux and `GEO_HEAT` the bottom one. Both are positive into the snow.
pub fn step_neumann(
    column: &mut Column,
    ground_flux: f64,
    surface_flux: f64,
    dt: f64,
    k: Conductivity,
) {
    solve(
        column,
        dt,
        k,
        Boundary::Flux(ground_flux),
        Boundary::Flux(surface_flux),
    );
}

/// Advance one step with the geothermal flux at the bottom and, at the top, a
/// flux that depends linearly on the new surface temperature,
/// `F = a + b T_surface` (W m^-2, into the snow, `b <= 0`). Used by the
/// surface energy balance, which linearises itself about the current state.
pub fn step_surface_linear(
    column: &mut Column,
    ground_flux: f64,
    a: f64,
    b: f64,
    dt: f64,
    k: Conductivity,
) {
    step_surface_linear_with_sources(column, ground_flux, a, b, &[], dt, k);
}

/// As [`step_surface_linear`], with a heat source in each element (W m^-2,
/// bottom element first), such as absorbed shortwave. Each element's source is
/// split equally between its two nodes, the same lumping as the capacitance.
/// An empty slice means no sources.
pub fn step_surface_linear_with_sources(
    column: &mut Column,
    ground_flux: f64,
    a: f64,
    b: f64,
    sources: &[f64],
    dt: f64,
    k: Conductivity,
) {
    solve_with_sources(
        column,
        dt,
        k,
        Boundary::Flux(ground_flux),
        Boundary::Linear { a, b },
        sources,
    );
}

/// The lower boundary of a step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ground {
    /// A prescribed flux into the snow, W m^-2 (`GEO_HEAT`).
    Flux(f64),
    /// A prescribed ground temperature, K. The reference behaves as if its
    /// lower boundary were held at the forcing's `TSG` (lab notes): its bottom
    /// element never rises above 0 C and averages -0.04 to -0.09 C under a
    /// surface at -14 to -34 C, which a 0.06 W m^-2 flux could not produce.
    Temperature(f64),
}

/// Advance one step with a chosen lower boundary, the linear surface flux
/// `a + b T` on top, and per-element heat sources (W m^-2).
pub fn step_general(
    column: &mut Column,
    ground: Ground,
    a: f64,
    b: f64,
    sources: &[f64],
    dt: f64,
    k: Conductivity,
) {
    let bottom = match ground {
        Ground::Flux(f) => Boundary::Flux(f),
        Ground::Temperature(t) => Boundary::Dirichlet(t),
    };
    solve_with_sources(column, dt, k, bottom, Boundary::Linear { a, b }, sources);
}

#[derive(Debug, Clone, Copy)]
enum Boundary {
    Dirichlet(f64),
    /// W m^-2, positive into the snow.
    Flux(f64),
    /// `a + b T` W m^-2 at the boundary node, evaluated implicitly.
    Linear {
        a: f64,
        b: f64,
    },
}

/// Assemble and solve one backward-Euler step.
///
/// The element capacitance matrix is **lumped** (the row sums of the consistent
/// matrix placed on the diagonal) rather than consistent. Lumping costs some
/// accuracy on a smooth solution but removes the spurious under- and overshoot
/// that a consistent mass matrix produces at a sharp thermal front, and a sharp
/// front at the snow surface on a clear night is the normal case here, not the
/// exception. It also keeps the system tridiagonal, which the Thomas algorithm
/// below solves in O(n).
fn solve(column: &mut Column, dt: f64, k: Conductivity, bottom: Boundary, top: Boundary) {
    solve_with_sources(column, dt, k, bottom, top, &[]);
}

fn solve_with_sources(
    column: &mut Column,
    dt: f64,
    k: Conductivity,
    bottom: Boundary,
    top: Boundary,
    sources: &[f64],
) {
    let n = column.elements.len();
    if n == 0 {
        return;
    }
    let nodes = n + 1;

    // Tridiagonal system A T^{new} = b.
    let mut sub = vec![0.0; nodes];
    let mut diag = vec![0.0; nodes];
    let mut sup = vec![0.0; nodes];
    let mut rhs = vec![0.0; nodes];

    for (index, element) in column.elements.iter().enumerate() {
        let conductance = k.of_density(element.density()) / element.thickness;
        // Lumped capacitance: half of each element's heat capacity to each of
        // its two nodes.
        let half_capacity = 0.5 * volumetric_heat_capacity(element) * element.thickness / dt;

        diag[index] += conductance + half_capacity;
        sup[index] -= conductance;
        rhs[index] += half_capacity * column.node_temperature[index];

        diag[index + 1] += conductance + half_capacity;
        sub[index + 1] -= conductance;
        rhs[index + 1] += half_capacity * column.node_temperature[index + 1];
    }

    for (index, source) in sources.iter().enumerate() {
        rhs[index] += 0.5 * source;
        rhs[index + 1] += 0.5 * source;
    }

    apply(&mut sub, &mut diag, &mut sup, &mut rhs, 0, bottom);
    apply(&mut sub, &mut diag, &mut sup, &mut rhs, nodes - 1, top);

    thomas(&sub, &diag, &sup, &rhs, &mut column.node_temperature);
}

fn apply(
    sub: &mut [f64],
    diag: &mut [f64],
    sup: &mut [f64],
    rhs: &mut [f64],
    node: usize,
    boundary: Boundary,
) {
    match boundary {
        Boundary::Dirichlet(value) => {
            // Replace the row rather than penalising it, so the imposed value is
            // exact rather than exact-to-a-big-number.
            sub[node] = 0.0;
            sup[node] = 0.0;
            diag[node] = 1.0;
            rhs[node] = value;
        }
        // A Neumann condition is a load on the boundary node; the row already
        // carries the correct capacitance and conductance from assembly.
        Boundary::Flux(flux) => rhs[node] += flux,
        // Implicit in T: the b T part moves to the left-hand side.
        Boundary::Linear { a, b } => {
            rhs[node] += a;
            diag[node] -= b;
        }
    }
}

/// Thomas algorithm for a tridiagonal system. The matrix here is symmetric,
/// weakly diagonally dominant and has positive diagonal, so it needs no
/// pivoting; a zero pivot would mean a zero-thickness element, which layer
/// management is responsible for never producing.
fn thomas(sub: &[f64], diag: &[f64], sup: &[f64], rhs: &[f64], out: &mut [f64]) {
    let n = diag.len();
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];

    c[0] = sup[0] / diag[0];
    d[0] = rhs[0] / diag[0];
    for i in 1..n {
        let denominator = diag[i] - sub[i] * c[i - 1];
        c[i] = sup[i] / denominator;
        d[i] = (rhs[i] - sub[i] * d[i - 1]) / denominator;
    }
    out[n - 1] = d[n - 1];
    for i in (0..n - 1).rev() {
        out[i] = d[i] - c[i] * out[i + 1];
    }
}
