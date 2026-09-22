//! Stage 2 gate: overburden stress and viscous settlement.
//!
//! Analytic and property tests only. The viscosity implemented is Crocus's
//! BR92 law (Vionnet et al. 2012, Eq. 7 and Appendix A), not SNOWPACK's
//! `VISCOSITY_MODEL = DEFAULT`, whose published form could not be read. That
//! is a declared divergence from the reference, so no golden test here can be
//! expected to agree closely, and none is attempted at this stage.

use approx::assert_relative_eq;
use pi_snow_core::constants::*;
use pi_snow_core::{Column, Composition, Element, Microstructure};
use pi_snow_physics::settlement::{self, Br92, Viscosity};

fn element(thickness: f64, density: f64, grain_radius: f64) -> Element {
    Element {
        thickness,
        composition: Composition::dry_snow(density),
        micro: Microstructure {
            dendricity: 0.0,
            sphericity: 1.0,
            grain_radius,
            bond_radius: 0.4 * grain_radius,
            coordination_number: 3.0,
            marker: 0,
        },
        age_days: 0.0,
        stress: 0.0,
        stress_rate: 0.0,
        metamorphism: 0.0,
        soil_density: 0.0,
        soil_conductivity: 0.0,
        soil_heat_capacity: 0.0,
    }
}

fn column(n: usize, thickness: f64, density: f64, slope_deg: f64) -> Column {
    let mut c = Column::bare_ground(263.15, slope_deg, 0.0);
    for _ in 0..n {
        c.elements.push(element(thickness, density, 0.25e-3));
    }
    c.node_temperature = vec![263.15; n + 1];
    c
}

#[test]
fn overburden_on_flat_ground_is_the_weight_above_plus_half_its_own() {
    // Vionnet et al. (2012) Eq. 6 sums the layers strictly above and gives the
    // top layer half its own weight. Evaluating every element at its own
    // mid-depth generalises that; the generalisation is an implementation
    // choice, recorded as such, not a published formula.
    let (n, h, rho) = (10, 0.05, 200.0);
    let mut c = column(n, h, rho, 0.0);
    settlement::update_stress(&mut c);
    for (index, e) in c.elements.iter().enumerate() {
        let layers_above = (n - 1 - index) as f64;
        let expected = -GRAVITY * rho * h * (layers_above + 0.5);
        assert_relative_eq!(e.stress, expected, max_relative = 1e-12);
    }
}

#[test]
fn overburden_on_a_slope_scales_by_cos_squared_with_vertical_thickness() {
    // The configuration sets PERP_TO_SLOPE = false, so thickness is vertical.
    // Mass per unit slope area is rho h cos(psi), and its slope-normal
    // component is that times g cos(psi): cos^2 overall. Vionnet writes
    // g cos(psi) rho D with slope-normal D = h cos(psi), which is the same
    // thing. Getting this wrong is a cos(38) = 0.79 error in every stress.
    let flat = {
        let mut c = column(8, 0.04, 250.0, 0.0);
        settlement::update_stress(&mut c);
        c
    };
    for slope in [10.0_f64, 30.0, 34.0, 38.0] {
        let mut c = column(8, 0.04, 250.0, slope);
        settlement::update_stress(&mut c);
        let factor = slope.to_radians().cos().powi(2);
        for (s, f) in c.elements.iter().zip(&flat.elements) {
            assert_relative_eq!(s.stress, f.stress * factor, max_relative = 1e-12);
        }
    }
}

#[test]
fn constant_viscosity_gives_exact_exponential_compaction() {
    // dD/D = -(sigma / eta) dt integrates to D0 exp(-sigma t / eta) for constant
    // eta. The step multiplies by exp(-sigma dt / eta), so this holds to
    // rounding over any number of steps, not just in the limit dt -> 0.
    let mut e = element(0.10, 150.0, 0.25e-3);
    let (sigma, eta, dt, steps) = (-500.0, 5.0e9, 600.0, 1_000);
    let (h0, m0) = (e.thickness, e.mass());
    for _ in 0..steps {
        settlement::settle(&mut e, sigma, eta, dt);
    }
    let t = dt * steps as f64;
    assert_relative_eq!(e.thickness, h0 * (sigma * t / eta).exp(), max_relative = 1e-10);
    assert_relative_eq!(e.mass(), m0, max_relative = 1e-12);
}

#[test]
fn settlement_conserves_mass_and_keeps_fractions_consistent() {
    let mut c = column(20, 0.03, 120.0, 34.0);
    let before: Vec<f64> = c.elements.iter().map(Element::mass).collect();
    for _ in 0..(6 * 24 * 10) {
        settlement::step(&mut c, 600.0, Br92);
    }
    for (e, m) in c.elements.iter().zip(before) {
        assert_relative_eq!(e.mass(), m, max_relative = 1e-10);
        assert!(e.composition.is_consistent(1e-12), "{:?}", e.composition);
        assert!(e.density() <= RHO_ICE, "density {} above ice", e.density());
    }
    // Deeper elements carry more load and must have compacted more.
    assert!(c.elements[0].density() > c.elements[19].density());
}

#[test]
fn br92_liquid_water_factor_matches_its_published_form() {
    // f1 = 1 / (1 + 60 W_liq / (rho_w D)), Vionnet et al. (2012) Eq. 8, where
    // W_liq / (rho_w D) is the volumetric liquid water content.
    assert_relative_eq!(Br92::liquid_water_factor(0.0), 1.0);
    assert_relative_eq!(Br92::liquid_water_factor(0.05), 1.0 / (1.0 + 60.0 * 0.05));
}

#[test]
fn br92_grain_factor_is_capped_at_four() {
    // f2 = min(4, exp(min(g1, gs - g2) / g3)), g1 = 0.4 mm, g2 = 0.2 mm,
    // g3 = 0.1 mm (Vionnet et al. 2012, Eq. 9). The inner min caps the
    // exponent at 4, and exp(4) is well above the outer cap of 4.
    assert_relative_eq!(Br92::grain_factor(0.2e-3), 1.0, max_relative = 1e-12);
    assert_relative_eq!(Br92::grain_factor(0.3e-3), 1.0_f64.exp(), max_relative = 1e-12);
    assert_relative_eq!(Br92::grain_factor(5.0e-3), 4.0);
    assert!(Br92::grain_factor(0.1e-3) < 1.0, "small grains settle faster");
}

#[test]
fn br92_viscosity_rises_with_density_and_with_cold() {
    let e = |rho: f64| element(0.05, rho, 0.2e-3);
    let warm = T_MELT - 1.0;
    let cold = T_MELT - 15.0;
    let mut previous = 0.0;
    for rho in [60.0, 100.0, 200.0, 300.0, 450.0] {
        let eta = Br92.viscosity(&e(rho), warm);
        assert!(eta > previous, "viscosity must increase with density");
        previous = eta;
        assert!(Br92.viscosity(&e(rho), cold) > eta, "colder snow is stiffer");
    }
}
