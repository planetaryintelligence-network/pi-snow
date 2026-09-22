//! Stage 5 gate: metamorphism and grain classification.
//!
//! Rates are Crocus's published laws (Brun et al. 1992; Vionnet et al. 2012
//! Table 1, Table 2, Appendix B), a declared divergence from SNOWPACK's
//! unobtainable ones. The Swiss-code classification is our rule, drawn from
//! Brun's (1992) Fig. 1 diagram; the tests pin it so any change is visible.

use approx::assert_relative_eq;
use pi_snow_core::Microstructure;
use pi_snow_physics::metamorphism::{self, History, Local};

fn micro(d: f64, s: f64, grain_mm: f64) -> Microstructure {
    Microstructure {
        dendricity: d,
        sphericity: s,
        grain_radius: 0.5e-3 * grain_mm,
        bond_radius: 0.04e-3,
        coordination_number: 2.0,
        marker: 0,
    }
}

fn local(t: f64, gradient: f64) -> Local {
    Local {
        temperature: t,
        gradient,
        density: 150.0,
        lwc_percent: 0.0,
    }
}

const DAY: f64 = 86_400.0;

#[test]
fn low_gradient_dendritic_rates_are_brun_eq_3() {
    // |grad T| <= 5 K/m: dd/dt = -2e8 exp(-6000/T), ds/dt = +1e9 exp(-6000/T),
    // per day (Brun et al. 1992 eq. 3; Vionnet et al. 2012 Table 1).
    let t: f64 = 263.15;
    let k = (-6000.0 / t).exp();
    let mut m = micro(1.0, 0.5, 0.3);
    metamorphism::step(&mut m, &local(t, 2.0), &mut History::default(), DAY);
    assert_relative_eq!(m.dendricity, 1.0 - 2e8 * k, max_relative = 1e-12);
    assert_relative_eq!(m.sphericity, 0.5 + 1e9 * k, max_relative = 1e-12);
}

#[test]
fn moderate_gradient_rates_are_brun_eq_4() {
    // |grad T| > 5 K/m: dd/dt = ds/dt = -2e8 exp(-6000/T) G^0.4.
    let (t, g) = (263.15, 10.0_f64);
    let rate = 2e8 * (-6000.0_f64 / t).exp() * g.powf(0.4);
    let mut m = micro(0.8, 0.5, 0.3);
    metamorphism::step(&mut m, &local(t, g), &mut History::default(), DAY);
    assert_relative_eq!(m.dendricity, 0.8 - rate, max_relative = 1e-12);
    assert_relative_eq!(m.sphericity, 0.5 - rate, max_relative = 1e-12);
}

#[test]
fn shape_variables_stay_in_the_unit_interval() {
    let mut m = micro(0.01, 0.99, 0.3);
    for _ in 0..200 {
        metamorphism::step(&mut m, &local(272.0, 1.0), &mut History::default(), DAY);
    }
    assert_eq!(m.dendricity, 0.0);
    assert_eq!(m.sphericity, 1.0);
}

#[test]
fn kinetic_growth_is_marbouty_via_vionnet_appendix_b() {
    // Only for non-dendritic, fully faceted grains (s = 0) under G > 15 K/m:
    // d gs/dt = f(T) h(rho) g(G) Phi, Phi = 1.0417e-9 m/s. At T - Tfus = -4 C,
    // rho = 200, G = 30: f = 1 - 0.05*(-4) ... that branch is "otherwise",
    // f = 1 - 0.05 (T - Tfus) = 1.2; h = 1 - 0.004*50 = 0.8;
    // g = 0.1 + 0.037*5 = 0.285.
    let mut m = micro(0.0, 0.0, 1.0);
    let l = Local {
        temperature: 273.15 - 4.0,
        gradient: 30.0,
        density: 200.0,
        lwc_percent: 0.0,
    };
    metamorphism::step(&mut m, &l, &mut History::default(), 3600.0);
    let growth = 1.2 * 0.8 * 0.285 * 1.0417e-9 * 3600.0;
    assert_relative_eq!(2.0 * m.grain_radius, 1.0e-3 + growth, max_relative = 1e-9);
}

#[test]
fn wet_dendritic_snow_loses_dendricity_as_theta_cubed() {
    // Brun et al. (1992) eq. 5: dd/dt = -theta^3 / 16 per day, theta in %.
    let mut m = micro(0.9, 0.5, 0.3);
    let l = Local {
        temperature: 273.15,
        gradient: 0.0,
        density: 250.0,
        lwc_percent: 2.0,
    };
    metamorphism::step(&mut m, &l, &mut History::default(), DAY / 24.0);
    assert_relative_eq!(m.dendricity, 0.9 - 8.0 / 16.0 / 24.0, max_relative = 1e-9);
}

#[test]
fn classification_follows_the_documented_rule() {
    use metamorphism::classify;
    let dry = History::default();
    assert_eq!(
        classify(&micro(1.0, 0.5, 0.3), &dry, 100.0),
        110,
        "fresh snow, as the reference writes"
    );
    assert_eq!(
        classify(&micro(0.3, 0.6, 0.3), &dry, 150.0) / 100,
        2,
        "decomposing"
    );
    assert_eq!(
        classify(&micro(0.0, 0.9, 0.35), &dry, 250.0) / 100,
        3,
        "rounds"
    );
    assert_eq!(
        classify(&micro(0.0, 0.1, 0.8), &dry, 220.0) / 100,
        4,
        "facets"
    );
    assert_eq!(
        classify(&micro(0.0, 0.0, 2.5), &dry, 250.0) / 100,
        5,
        "depth hoar"
    );
    let refrozen = History {
        refrozen: true,
        ..History::default()
    };
    assert_eq!(
        classify(&micro(0.0, 1.0, 0.8), &refrozen, 550.0),
        772,
        "melt-freeze crust"
    );
    let hoar = History {
        surface_hoar: true,
        ..History::default()
    };
    assert_eq!(
        classify(&micro(0.0, 0.0, 2.0), &hoar, 120.0),
        660,
        "surface hoar"
    );
}
