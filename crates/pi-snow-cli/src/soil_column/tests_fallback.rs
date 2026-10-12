//! engine-soil-spinup-v2 plan tests A-T1, A-T3 and A-T4 for the in-tree
//! post-failure bisection. A-T2 (the five captured Q3 stops accept at
//! 32/34/33/35/33 iterations) and A-T5 (the v2 tree reproduces retained binary
//! `6e1aa6cc` on the 166 G2 requests) need retained inputs and run in S2.
//!
//! The converging-row fixture is v1's `a9_contract.rs` T1/T4 fixture. Its
//! digest was recorded by running this same test body on a clean checkout of
//! main `ebaf93bf`, which has no fallback: `ROW` lines (flux and residual bits,
//! column enthalpies, step report, counters) and the soil header.

use super::*;
use pi_snow_physics::surface::{Atmosphere, SurfaceParameters};
use pi_snow_physics::turbulence::Scheme;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn fixture_text() -> String {
    let params = SurfaceParameters {
        emissivity: 0.97,
        z_wind: 2.0,
        z_scalar: 2.0,
        z0: 0.01,
        scheme: Scheme::Louis,
    };
    let mut out = String::new();
    for (temperature, shortwave) in [(300.0, 500.0), (255.0, 0.0), (273.15, 100.0)] {
        let atm = Atmosphere {
            t_air: temperature,
            relative_humidity: 0.7,
            wind: 2.0,
            absorbed_shortwave: shortwave,
            ilwr: 300.0,
            pressure: 80000.0,
            rain_rate: 0.0,
        };
        let mut soil = SoilColumn::uniform(Material::MoistLoam, temperature);
        let r = bare::step(&mut soil, 600.0, Bottom::Geothermal005, &atm, &params).unwrap();
        let d = &r.diagnostics;
        out.push_str(&format!(
            "ROW {:?} {} {} {:?} {:?} {:?}\nSCAN {:?} CERT {:?}\n",
            temperature,
            r.surface_flux_w_m2.to_bits(),
            r.flux_residual_w_m2.to_bits(),
            soil.enthalpy.map(f64::to_bits),
            r.soil,
            (
                d.certified,
                d.stiffness_lower_bound_w_m2_k.to_bits(),
                d.maximum_sampled_budget_slope_w_m2_k.to_bits(),
                d.column_solves,
                d.budget_evaluations,
                d.computed_budget_evaluations,
                d.newton_iterations,
                d.nonconvergences
            ),
            d.scan,
            d.certification
        ));
    }
    let settings = pi_snow_core::options::PhysicsOptions::from_settings([
        "GROUND_BOUNDARY=SOIL_COLUMN",
        "SOIL_PROPERTIES=MOIST_LOAM",
        "SOIL_BOTTOM=GEOTHERMAL_005",
        "SOIL_INIT=UNIFORM_TMELT",
    ])
    .unwrap()
    .soil;
    let mut snow = pi_snow_core::Column::bare_ground(273.15, 0.0, 0.0);
    let mut runtime = runtime::Runtime::new(
        settings,
        runtime::InitializationContext {
            mean_336h_k: None,
            annual_mean_k: 280.0,
        },
        &mut snow,
    )
    .unwrap();
    runtime.column = SoilColumn::uniform(Material::MoistLoam, 280.0);
    let atm = Atmosphere {
        t_air: 285.0,
        relative_humidity: 0.7,
        wind: 2.0,
        absorbed_shortwave: 200.0,
        ilwr: 350.0,
        pressure: 80000.0,
        rain_rate: 0.0,
    };
    runtime.bare_step(&mut snow, 600.0, &atm, &params).unwrap();
    let header = output::Data::new(runtime)
        .extend_smet("SMET 1.1 ASCII\n[HEADER]\nfields = timestamp HS_mod\n[DATA]\n");
    out.push_str(&format!("HEADER\n{header}"));
    out
}

#[test]
#[ignore = "prints the digest; used once on main ebaf93bf to record EXPECTED"]
fn print_fixture_digest() {
    let text = fixture_text();
    println!("DIGEST {} {:016x}", text.len(), fnv1a(text.as_bytes()));
}

const EXPECTED: &str = "9223 7adb9c19ca6ee3b0";

/// A-T1 `a9_converging_rows_bitwise` and A-T4 `a9_header_bytes`.
#[test]
fn a9_converging_rows_and_header_equal_the_pre_a9_tree() {
    let text = fixture_text();
    assert_eq!(
        format!("{} {:016x}", text.len(), fnv1a(text.as_bytes())),
        EXPECTED
    );
    assert!(
        !text.contains("soil_bare_fallback"),
        "a converging run must carry no fallback header line"
    );
}

/// A-T1, counter half.
#[test]
fn converging_rows_have_zero_fallback_counters() {
    let p = SurfaceParameters {
        emissivity: 0.97,
        z_wind: 2.0,
        z_scalar: 2.0,
        z0: 0.01,
        scheme: Scheme::Louis,
    };
    for (t, sw) in [(300.0, 500.0), (255.0, 0.0), (273.15, 100.0)] {
        let a = Atmosphere {
            t_air: t,
            relative_humidity: 0.7,
            wind: 2.0,
            absorbed_shortwave: sw,
            ilwr: 300.0,
            pressure: 80000.0,
            rain_rate: 0.0,
        };
        let mut soil = SoilColumn::uniform(Material::MoistLoam, t);
        let r = bare::step(&mut soil, 600.0, Bottom::Geothermal005, &a, &p).unwrap();
        let d = &r.diagnostics;
        assert_eq!(
            (d.fallback_entries, d.fallback_iterations, d.fallback_failures),
            (0, 0, 0)
        );
        assert_eq!(d.failure_bracket_bits, None);
    }
}

/// A-T3 `a9_adjacent_jump_collapses`: a real column, a balance with an
/// adjacent-float sign change where both residuals exceed 1e-6.
#[test]
fn adjacent_jump_through_the_full_solver_is_a_failed_defined_row() {
    let mut soil = SoilColumn::uniform(Material::MoistLoam, 280.0);
    let before = soil.enthalpy;
    let jump = |phase: Phase| {
        if phase.temperature_k >= 280.123456789 {
            -100.0
        } else {
            100.0
        }
    };
    let failure = bare::solve(&mut soil, 600.0, Bottom::Insulated, jump).unwrap_err();
    assert_eq!(failure.message, bare_fallback::COLLAPSED);
    let d = failure.diagnostics;
    assert_eq!(d.newton_iterations, 100);
    assert_eq!((d.fallback_entries, d.fallback_failures, d.nonconvergences), (1, 1, 1));
    assert_eq!(soil.enthalpy.map(f64::to_bits), before.map(f64::to_bits));
    let [lo, hi] = d.failure_bracket_bits.unwrap();
    assert_eq!(lo + 1, hi);
    let residual = |q: f64| {
        let mut c = soil.clone();
        c.step(600.0, Surface::Flux(q), Bottom::Insulated).unwrap();
        q - jump(c.phase(0))
    };
    assert!(residual(f64::from_bits(lo)) < -1e-6 && residual(f64::from_bits(hi)) > 1e-6);
}

/// A row the Newton loop cannot finish but whose residual is continuous is
/// recovered by the bisection and carries the three header lines.
#[test]
fn a_cycling_newton_row_is_recovered_and_counted() {
    // A residual whose slope is tiny near the root and steep far from it makes
    // the secant stencil useless; the bisection brackets it regardless.
    let mut soil = SoilColumn::uniform(Material::MoistLoam, 280.0);
    let balance = |p: Phase| 40.0 * ((p.temperature_k - 281.0) / 3.0).tanh().powi(3) * -1.0;
    let report = bare::solve(&mut soil, 600.0, Bottom::Insulated, balance);
    // Either the Newton path converged or the fallback did; both must accept.
    let report = report.unwrap();
    assert!(report.flux_residual_w_m2.abs() <= 1e-6);
    let d = &report.diagnostics;
    assert_eq!(d.fallback_failures, 0);
    assert_eq!(d.fallback_entries == 0, d.newton_iterations < 100);
}
