//! engine-surface-v1's two surface-layer solves (`SURFACE_LAYER=...`), against
//! the Phase 2 two-height solve on the cases `turbulence_decoupling.rs` pins.

use pi_snow_physics::turbulence::{self, Scheme, SurfaceLayer};

fn layer(wind: f64, t_air: f64, t_surface: f64) -> SurfaceLayer {
    SurfaceLayer {
        wind,
        z_wind: 9.5,
        z_scalar: 1.5,
        z0: 0.002,
        t_air,
        t_surface,
        q_air: 2.0e-3,
        q_surface: 1.5e-3,
        pressure: 73_000.0,
    }
}

#[test]
fn near_neutral_all_three_share_the_neutral_coefficient() {
    // The log-law reduction leaves k^2 / (ln(z_w/z0) ln(z_s/z0)) unchanged, so
    // with a vanishing gradient the fluxes agree to the stability correction.
    let l = layer(5.0, 268.001, 268.0);
    let phase2 = turbulence::fluxes(&l, Scheme::MoHoltslag).sensible;
    let single = turbulence::fluxes(&l, Scheme::MoHoltslagScalarHeight).sensible;
    assert!((single / phase2 - 1.0).abs() < 2e-3, "{single} {phase2}");
}

#[test]
fn where_phase_2_decouples_both_keep_a_flux_into_the_snow() {
    // U 2 m/s, 5 K inversion: no root in the two-height system (Phase 2 gives 0).
    let l = layer(2.0, 268.0, 263.0);
    assert!(turbulence::fluxes(&l, Scheme::MoHoltslag).sensible.abs() < 1e-6);
    let single = turbulence::fluxes(&l, Scheme::MoHoltslagScalarHeight);
    let limited = turbulence::fluxes(&l, Scheme::MoHoltslagZetaLimited);
    assert!(
        single.sensible > 1.0 && single.obukhov_length.is_finite(),
        "{single:?}"
    );
    assert!(limited.sensible > 1.0, "{limited:?}");
}

#[test]
fn the_clm5_limit_holds_z_over_l_at_the_wind_height() {
    for (u, ta, ts) in [
        (2.0, 268.0, 263.0),
        (1.0, 270.0, 255.0),
        (8.0, 268.0, 267.5),
    ] {
        let f = turbulence::fluxes(&layer(u, ta, ts), Scheme::MoHoltslagZetaLimited);
        let zeta = 9.5 / f.obukhov_length;
        assert!(
            (0.01 - 1e-9..=2.0 + 1e-9).contains(&zeta),
            "zeta {zeta} at {u} {ta} {ts}"
        );
    }
    let unstable = turbulence::fluxes(&layer(3.0, 263.0, 268.0), Scheme::MoHoltslagZetaLimited);
    let zeta = 9.5 / unstable.obukhov_length;
    assert!((-100.0..=-0.01 + 1e-9).contains(&zeta), "{zeta}");
}

#[test]
fn inside_the_clm5_bounds_the_limit_changes_nothing() {
    // U 8 m/s, 0.5 K: Phase 2 converges with z_w / L inside 0.01..2. (At 3
    // m/s and 3 K it already converges at z_w / L of about 5.4.)
    let l = layer(8.0, 268.0, 267.5);
    let phase2 = turbulence::fluxes(&l, Scheme::MoHoltslag);
    let zeta = 9.5 / phase2.obukhov_length;
    assert!((0.01..2.0).contains(&zeta), "{zeta}");
    let limited = turbulence::fluxes(&l, Scheme::MoHoltslagZetaLimited);
    assert!((limited.sensible - phase2.sensible).abs() < 1e-6 * phase2.sensible.abs());
}

#[test]
fn scalar_height_is_the_phase_2_solve_when_both_heights_coincide() {
    let mut l = layer(3.0, 268.0, 265.0);
    l.z_wind = l.z_scalar;
    let phase2 = turbulence::fluxes(&l, Scheme::MoHoltslag);
    let single = turbulence::fluxes(&l, Scheme::MoHoltslagScalarHeight);
    assert_eq!(phase2, single);
}

#[test]
fn in_stable_air_the_scalar_height_solve_never_lowers_the_flux() {
    // Dropping the momentum correction above z_s can only raise u* (review S2).
    for (u, ta, ts) in [
        (8.0, 268.0, 267.5),
        (3.0, 268.0, 265.0),
        (2.0, 268.0, 263.0),
        (5.0, 270.0, 262.0),
    ] {
        let l = layer(u, ta, ts);
        let phase2 = turbulence::fluxes(&l, Scheme::MoHoltslag);
        let scalar = turbulence::fluxes(&l, Scheme::MoHoltslagScalarHeight);
        assert!(scalar.u_star >= phase2.u_star - 1e-12, "{u} {ta} {ts}");
        assert!(scalar.sensible >= phase2.sensible - 1e-9, "{u} {ta} {ts}");
    }
}
