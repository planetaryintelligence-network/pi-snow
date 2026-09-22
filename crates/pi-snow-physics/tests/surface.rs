//! Stage 1b gate: the surface energy balance coupled to conduction.

use approx::assert_relative_eq;
use pi_snow_core::constants::*;
use pi_snow_core::{Column, Composition, Element, Microstructure};
use pi_snow_physics::surface::{self, Atmosphere, SurfaceParameters};
use pi_snow_physics::thermal::{self, Conductivity};
use pi_snow_physics::turbulence::Scheme;

fn column(n: usize, dz: f64, rho: f64, t0: f64) -> Column {
    let mut c = Column::bare_ground(t0, 0.0, 0.0);
    for _ in 0..n {
        c.elements.push(Element {
            thickness: dz,
            composition: Composition::dry_snow(rho),
            micro: Microstructure {
                dendricity: 0.0,
                sphericity: 1.0,
                grain_radius: 0.5e-3,
                bond_radius: 0.2e-3,
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
        });
    }
    c.node_temperature = vec![t0; n + 1];
    c
}

fn params() -> SurfaceParameters {
    SurfaceParameters {
        emissivity: surface::SNOW_EMISSIVITY,
        z_wind: 10.0,
        z_scalar: 2.0,
        z0: 0.002,
        scheme: Scheme::MoHoltslag,
    }
}

fn calm(ilwr: f64) -> Atmosphere {
    Atmosphere {
        t_air: 260.0,
        relative_humidity: 0.8,
        wind: 0.0,
        absorbed_shortwave: 0.0,
        ilwr,
        pressure: 70_000.0,
        rain_rate: 0.0,
    }
}

#[test]
fn emissivity_is_the_middle_of_dozier_and_warren() {
    // Dozier and Warren (1982): all-wave emissivity 0.985 to 0.990, independent
    // of grain size. The midpoint is our choice within their range.
    const { assert!(surface::SNOW_EMISSIVITY >= 0.985 && surface::SNOW_EMISSIVITY <= 0.990) };
}

#[test]
fn a_calm_dark_surface_relaxes_to_radiative_equilibrium() {
    // No wind, no sun, no ground flux: the only surface flux is longwave, so
    // at steady state eps ILWR = eps sigma Ts^4 and Ts = (ILWR / sigma)^(1/4),
    // whatever the emissivity. The column is adiabatic at the bottom.
    let ilwr = 200.0;
    let expected = (ilwr / STEFAN_BOLTZMANN).powf(0.25);
    let mut c = column(20, 0.02, 250.0, 265.0);
    for _ in 0..40_000 {
        surface::step(
            &mut c,
            &calm(ilwr),
            &params(),
            0.0,
            600.0,
            Conductivity::Yen1981,
        );
    }
    let ts = *c.node_temperature.last().unwrap();
    assert_relative_eq!(ts, expected, epsilon = 0.01);
    // And the whole column follows it, since the bottom is adiabatic.
    assert_relative_eq!(c.node_temperature[0], expected, epsilon = 0.05);
}

#[test]
fn the_step_conserves_energy_against_the_boundary_fluxes() {
    // Over a cold, windy night with no phase change, the change in column
    // internal energy equals the time-integrated surface flux plus the ground
    // flux. The surface flux is evaluated at the end-of-step surface
    // temperature, which is what the implicit boundary imposes.
    let mut c = column(15, 0.03, 200.0, 263.15);
    let atm = Atmosphere {
        wind: 4.0,
        t_air: 255.0,
        ilwr: 180.0,
        ..calm(180.0)
    };
    let (ground, dt) = (0.06, 600.0);
    let before = thermal::internal_energy(&c);
    let mut injected = 0.0;
    for _ in 0..144 {
        let flux = surface::step(&mut c, &atm, &params(), ground, dt, Conductivity::Yen1981);
        injected += (flux.total + ground) * dt;
    }
    let after = thermal::internal_energy(&c);
    assert_relative_eq!(after - before, injected, max_relative = 1e-6);
    assert!(after < before, "a cold windy night must cool the snow");
}

#[test]
fn warm_humid_windy_air_pulls_the_surface_above_radiative_equilibrium() {
    let ilwr = 200.0;
    let radiative = (ilwr / STEFAN_BOLTZMANN).powf(0.25);
    let mut c = column(20, 0.02, 250.0, 265.0);
    let atm = Atmosphere {
        wind: 5.0,
        t_air: 268.0,
        relative_humidity: 0.9,
        ..calm(ilwr)
    };
    for _ in 0..2_000 {
        surface::step(&mut c, &atm, &params(), 0.0, 600.0, Conductivity::Yen1981);
    }
    let ts = *c.node_temperature.last().unwrap();
    assert!(
        ts > radiative && ts < atm.t_air,
        "ts {ts} radiative {radiative}"
    );
}

#[test]
fn reported_longwave_terms_are_consistent() {
    // OLWR includes the reflected part of the incoming: (1 - eps) ILWR +
    // eps sigma Ts^4. LWR_net = ILWR - OLWR.
    let b = surface::budget(&calm(220.0), &params(), 260.0);
    let eps = surface::SNOW_EMISSIVITY;
    let olwr = eps * STEFAN_BOLTZMANN * 260.0_f64.powi(4) + (1.0 - eps) * 220.0;
    assert_relative_eq!(b.olwr, olwr, max_relative = 1e-12);
    assert_relative_eq!(b.lwr_net, 220.0 - olwr, max_relative = 1e-12);
}

#[test]
fn absorbed_shortwave_inside_the_pack_is_conserved() {
    // Sources inside elements add exactly their energy: the internal-energy
    // change equals boundary fluxes plus the summed sources.
    let mut c = column(10, 0.03, 200.0, 263.15);
    let atm = Atmosphere {
        wind: 2.0,
        t_air: 262.0,
        ilwr: 230.0,
        ..calm(230.0)
    };
    let sources: Vec<f64> = (0..10).map(|i| if i >= 7 { 20.0 } else { 0.0 }).collect();
    let (ground, dt) = (0.06, 600.0);
    let before = thermal::internal_energy(&c);
    let mut injected = 0.0;
    for _ in 0..48 {
        let b = surface::step_with_sources(
            &mut c,
            &atm,
            &params(),
            ground,
            &sources,
            dt,
            Conductivity::Yen1981,
        );
        injected += (b.total + ground + sources.iter().sum::<f64>()) * dt;
    }
    assert_relative_eq!(
        thermal::internal_energy(&c) - before,
        injected,
        max_relative = 1e-6
    );
}

#[test]
fn a_fixed_ground_temperature_holds_the_bottom_node_and_its_flux_closes_the_budget() {
    // With the ground held at 273.15 K under a cold night, the bottom node
    // stays there, heat flows up out of the ground, and the ground flux, taken
    // as the energy residual, is positive (into the snow) and closes exactly.
    use pi_snow_physics::thermal::Ground;
    let mut c = column(15, 0.02, 200.0, 268.0);
    let atm = Atmosphere {
        wind: 3.0,
        t_air: 255.0,
        ilwr: 170.0,
        ..calm(170.0)
    };
    let dt = 600.0;
    let mut ground_total = 0.0;
    for _ in 0..144 {
        let before = thermal::internal_energy(&c);
        let b = surface::step_general(
            &mut c,
            &atm,
            &params(),
            Ground::Temperature(273.15),
            &[],
            dt,
            Conductivity::Yen1981,
        );
        let qg0 = (thermal::internal_energy(&c) - before) / dt - b.total;
        ground_total += qg0;
        assert_relative_eq!(c.node_temperature[0], 273.15, epsilon = 1e-9);
    }
    assert!(
        ground_total > 0.0,
        "a warm ground must heat the cold snow from below"
    );
}
