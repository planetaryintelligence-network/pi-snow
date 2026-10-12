//! engine-soil-spinup-v2 plan test Z-T7 `zc_soil_path_wiring`, and the budget
//! level of the retained failure: the bare-ground balance the A9 fallback
//! could not root is discontinuous at T_s = T_a under the transplant and
//! continuous under `Scheme::MoHoltslagZetaContinuous`.

use super::bare_cache::PredicateCache;
use super::*;
use pi_snow_physics::atmosphere::{air_density, specific_humidity, vapour_pressure_water};
use pi_snow_physics::surface::{Atmosphere, SurfaceParameters};
use pi_snow_physics::turbulence::{self, Scheme, SurfaceLayer};

fn rig_like(t_air: f64, wind: f64, rh: f64) -> Atmosphere {
    Atmosphere {
        t_air,
        relative_humidity: rh,
        wind,
        absorbed_shortwave: 600.0,
        ilwr: 300.0,
        pressure: 80_000.0,
        rain_rate: 0.0,
    }
}

fn params(scheme: Scheme) -> SurfaceParameters {
    SurfaceParameters {
        emissivity: 0.97,
        z_wind: 10.0,
        z_scalar: 2.0,
        z0: 0.002,
        scheme,
    }
}

fn warm(t: f64) -> Phase {
    Phase {
        temperature_k: t,
        liquid_fraction: 1.0,
        dtemperature_denthalpy: 0.0,
    }
}

fn next_up(t: f64) -> f64 {
    f64::from_bits(t.to_bits() + 1)
}

#[test]
fn bare_budget_uses_the_continuous_scheme_it_is_given() {
    for (t_air, wind, rh, dt) in [
        (288.15, 5.0, 0.22, 3.0),
        (284.0, 2.5, 0.6, -4.0),
        (293.0, 8.0, 0.15, 10.0),
    ] {
        let atm = rig_like(t_air, wind, rh);
        let ts = t_air + dt;
        let b = bare::budget(&atm, &params(Scheme::MoHoltslagZetaContinuous), warm(ts));
        let layer = SurfaceLayer {
            wind,
            z_wind: 10.0,
            z_scalar: 2.0,
            z0: 0.002,
            t_air,
            t_surface: ts,
            q_air: specific_humidity(rh * vapour_pressure_water(t_air), 80_000.0),
            q_surface: specific_humidity(vapour_pressure_water(ts), 80_000.0),
            pressure: 80_000.0,
        };
        let direct = turbulence::fluxes(&layer, Scheme::MoHoltslagZetaContinuous);
        assert_eq!(b.sensible.to_bits(), direct.sensible.to_bits());
        let probe = SurfaceLayer { q_air: 1.0, q_surface: 0.0, ..layer };
        let rho = air_density(80_000.0, 0.5 * (t_air + ts));
        let exchange =
            turbulence::fluxes(&probe, Scheme::MoHoltslagZetaContinuous).vapour_mass / rho;
        let expected = direct.vapour_mass / (1.0 + 50.0 * exchange);
        assert_eq!(b.vapour_mass.to_bits(), expected.to_bits());
        // And it is not the transplant's value unless the limit never binds.
        let limited = bare::budget(&atm, &params(Scheme::MoHoltslagZetaLimited), warm(ts));
        println!("dt {dt}: continuous {} limited {}", b.total, limited.total);
    }
}

/// The retained failure at the level of the balance the solver roots.
#[test]
fn bare_balance_jumps_under_the_transplant_and_not_under_the_continuous_scheme() {
    let mut transplant_min = f64::INFINITY;
    let mut continuous_max = 0.0_f64;
    for (t_air, wind, rh) in [
        (288.15, 4.8, 0.22),
        (289.15, 5.9, 0.22),
        (290.0, 3.0, 0.3),
        (285.0, 5.0, 0.15),
    ] {
        let atm = rig_like(t_air, wind, rh);
        // One ulp below and above T_a, both on the same (liquid) phase branch.
        let below = f64::from_bits(t_air.to_bits() - 1);
        let above = next_up(t_air);
        let step = |scheme: Scheme| {
            let p = params(scheme);
            (bare::budget(&atm, &p, warm(above)).total - bare::budget(&atm, &p, warm(below)).total)
                .abs()
        };
        transplant_min = transplant_min.min(step(Scheme::MoHoltslagZetaLimited));
        continuous_max = continuous_max.max(step(Scheme::MoHoltslagZetaContinuous));
    }
    println!("balance step at T_s = T_a: transplant min {transplant_min:e}, continuous max {continuous_max:e} W/m2");
    // The A9 acceptance tolerance is 1e-6 W/m2 and a collapse needs a step
    // above 2e-6. The transplant steps by watts; the continuous scheme by
    // far less than the tolerance.
    assert!(transplant_min > 0.5, "{transplant_min:e}");
    assert!(continuous_max <= 1e-9, "{continuous_max:e}");
}

#[test]
fn predicate_cache_keeps_the_two_schemes_apart() {
    let shared = PredicateCache::shared();
    let atm = rig_like(288.15, 5.0, 0.22);
    let mut guard = shared.lock().unwrap();
    let a = guard.row(7, &atm, &params(Scheme::MoHoltslagZetaLimited));
    a.maximum_slope = Some(1.25);
    let b = guard.row(7, &atm, &params(Scheme::MoHoltslagZetaContinuous));
    assert_eq!(b.maximum_slope, None, "continuous scheme served the transplant's row");
    b.maximum_slope = Some(2.5);
    let a_again = guard.row(7, &atm, &params(Scheme::MoHoltslagZetaLimited));
    assert_eq!(a_again.maximum_slope, Some(1.25));
}

#[test]
fn snow_surface_budget_uses_the_continuous_scheme() {
    let atm = rig_like(268.0, 4.0, 0.5);
    let ts = 266.5;
    let p = params(Scheme::MoHoltslagZetaContinuous);
    let b = pi_snow_physics::surface::budget(&atm, &p, ts);
    let layer = SurfaceLayer {
        wind: 4.0,
        z_wind: 10.0,
        z_scalar: 2.0,
        z0: 0.002,
        t_air: 268.0,
        t_surface: ts,
        q_air: specific_humidity(0.5 * vapour_pressure_water(268.0), 80_000.0),
        q_surface: specific_humidity(
            pi_snow_physics::atmosphere::vapour_pressure_ice(ts),
            80_000.0,
        ),
        pressure: 80_000.0,
    };
    let direct = turbulence::fluxes(&layer, Scheme::MoHoltslagZetaContinuous);
    assert_eq!(b.sensible.to_bits(), direct.sensible.to_bits());
    assert_eq!(b.vapour_mass.to_bits(), direct.vapour_mass.to_bits());
}
