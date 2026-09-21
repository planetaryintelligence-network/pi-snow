//! Stage 1a gate: heat conduction with a prescribed surface temperature.
//!
//! Every test here is **analytic**: the answer is known independently of any
//! snow model, so a failure proves we are wrong rather than merely different.
//! No golden-file comparison appears at this stage, deliberately. Golden tests
//! can only show agreement or divergence, and starting with them invites tuning
//! toward the reference instead of toward the physics.

use approx::assert_relative_eq;
use pi_snow_core::constants::*;
use pi_snow_core::{Column, Composition, Element, Microstructure};
use pi_snow_physics::thermal::{self, Conductivity};

/// A uniform column of `n` elements of dry snow, isothermal at `t0`.
fn uniform_column(n: usize, thickness: f64, density: f64, t0: f64) -> Column {
    let mut column = Column::bare_ground(t0, 0.0, 0.0);
    for _ in 0..n {
        column.elements.push(Element {
            thickness,
            composition: Composition::dry_snow(density),
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
    column.node_temperature = vec![t0; n + 1];
    column
}

#[test]
fn yen_conductivity_does_not_reach_ice_at_ice_density() {
    // k = k_ice * (rho / rho_water)^1.88, Vionnet et al. (2012) Eq. 18, attributed
    // there to Yen (1981). At rho = rho_ice this gives 0.85 k_ice, not
    // k_ice. That is a real property of the parameterisation, not a transcription
    // error, and the obvious "fix" of forcing it to k_ice would be wrong.
    let k = Conductivity::YenCrocus.of_density(RHO_ICE);
    assert_relative_eq!(k, K_ICE * (RHO_ICE / RHO_WATER).powf(1.88), max_relative = 1e-12);
    assert!(k < K_ICE, "Yen form must stay below k_ice at ice density");
    // (917/1000)^1.88 = 0.84968. The derivation round's own sanity note quoted
    // "about 0.858 k_ice, about 1.97 W/m/K" for this; that is a small slip in
    // the note, not in the formula, and the number here is the arithmetic.
    assert_relative_eq!(k / K_ICE, 0.849_677_955_948_175_5, max_relative = 1e-12);
    assert_relative_eq!(k, 1.954, max_relative = 1e-3);
}

#[test]
fn yen_conductivity_is_monotonic_and_bounded() {
    let mut previous = 0.0;
    for density in [50.0, 100.0, 200.0, 300.0, 500.0, 700.0, RHO_ICE] {
        let k = Conductivity::YenCrocus.of_density(density);
        assert!(k > previous, "conductivity must increase with density");
        assert!(k < K_ICE, "snow cannot conduct better than solid ice");
        previous = k;
    }
}

#[test]
fn two_slab_steady_state_is_the_exact_linear_profile() {
    // With both ends held fixed, no source and uniform conductivity, the steady
    // profile is exactly linear. Any deviation is an assembly or boundary bug,
    // not a physics disagreement, because this has a closed form.
    let (n, dz) = (20, 0.02);
    let (t_top, t_bottom) = (263.15, 273.15);
    let mut column = uniform_column(n, dz, 250.0, 268.15);

    for _ in 0..200_000 {
        thermal::step_dirichlet(&mut column, t_bottom, t_top, 10.0, Conductivity::YenCrocus);
    }

    let height = n as f64 * dz;
    for (index, &temperature) in column.node_temperature.iter().enumerate() {
        let z = index as f64 * dz;
        let expected = t_bottom + (t_top - t_bottom) * z / height;
        assert_relative_eq!(temperature, expected, epsilon = 1e-6);
    }
}

#[test]
fn semi_infinite_step_cooling_follows_the_erf_solution() {
    // A deep column initially isothermal at T0, surface dropped to Ts at t=0:
    //   T(z,t) = Ts + (T0 - Ts) * erf( z / (2 sqrt(alpha t)) )
    // measuring z downward from the surface. Valid while the disturbance has not
    // reached the far boundary, which is why the column is made deliberately
    // deeper than the diffusion length over the test duration.
    let (n, dz, density) = (400, 0.01, 250.0);
    let (t_initial, t_surface) = (273.15, 253.15);
    let mut column = uniform_column(n, dz, density, t_initial);

    let k = Conductivity::YenCrocus.of_density(density);
    let heat_capacity = thermal::volumetric_heat_capacity(&column.elements[0]);
    let alpha = k / heat_capacity;

    let (dt, steps) = (60.0, 360); // six hours
    for _ in 0..steps {
        thermal::step_dirichlet(&mut column, t_initial, t_surface, dt, Conductivity::YenCrocus);
    }
    let elapsed = dt * steps as f64;
    let diffusion_length = (alpha * elapsed).sqrt();
    assert!(
        6.0 * diffusion_length < n as f64 * dz,
        "column must stay semi-infinite over the test: 6*sqrt(alpha t) = {:.2} m",
        6.0 * diffusion_length
    );

    let surface_node = column.node_temperature.len() - 1;
    for depth_nodes in [5usize, 10, 20, 40, 80] {
        let z = depth_nodes as f64 * dz;
        let expected = t_surface + (t_initial - t_surface) * erf(z / (2.0 * diffusion_length));
        let got = column.node_temperature[surface_node - depth_nodes];
        assert_relative_eq!(got, expected, epsilon = 0.05);
    }
}

#[test]
fn a_closed_box_conserves_energy() {
    // Zero flux at both ends and no source: the column-integrated internal
    // energy must not drift. This isolates leaks in the assembly and in the
    // boundary bookkeeping, which a Dirichlet test cannot see because Dirichlet
    // boundaries legitimately add and remove energy.
    let (n, dz) = (30, 0.02);
    let mut column = uniform_column(n, dz, 250.0, 268.15);
    // A non-uniform start, so there is something to conduct.
    for (index, temperature) in column.node_temperature.iter_mut().enumerate() {
        *temperature = 263.15 + 10.0 * (index as f64 / n as f64);
    }

    let before = thermal::internal_energy(&column);
    // The fundamental relaxation time of a 0.6 m column of 250 kg/m3 snow is
    // L^2 / (pi^2 alpha) = 1.08e5 s, about 30 hours. Run to 278 hours, roughly
    // nine e-foldings, so the initial 10 K spread decays to about 1 mK and the
    // relaxation assertion below is not measuring the transient it was meant to
    // outlast. An earlier version of this test ran for 28 hours and failed at a
    // 3.2 K spread, which was the physically correct answer, not a solver bug.
    for _ in 0..100_000 {
        thermal::step_adiabatic(&mut column, 10.0, Conductivity::YenCrocus);
    }
    let after = thermal::internal_energy(&column);
    assert_relative_eq!(after, before, max_relative = 1e-9);

    // It must actually have relaxed, or the conservation check above would pass
    // trivially on a solver that does nothing at all.
    let spread = column.node_temperature.iter().cloned().fold(f64::MIN, f64::max)
        - column.node_temperature.iter().cloned().fold(f64::MAX, f64::min);
    assert!(spread < 0.01, "column should have relaxed; spread is {spread} K");

    // And it must have relaxed to the right temperature: with a uniform
    // composition, the equilibrium is the energy-weighted mean of the start.
    let height: f64 = column.elements.iter().map(|e| e.thickness).sum();
    let capacity = thermal::volumetric_heat_capacity(&column.elements[0]);
    assert_relative_eq!(
        column.node_temperature[0],
        before / (capacity * height),
        epsilon = 0.01
    );
}

#[test]
fn volumetric_heat_capacity_matches_the_published_mixture_rule() {
    // (rho C)_eff = (1 - phi) rho_i C_i + phi rho_a C_a, Bouvet et al. (2024)
    // Eq. 25, with C_i = 2000 and C_a = 1005 J kg^-1 K^-1 at -10 C.
    let element = &uniform_column(1, 0.1, 250.0, 268.15).elements[0];
    let ice_fraction = 250.0 / RHO_ICE;
    let expected = ice_fraction * RHO_ICE * C_ICE + (1.0 - ice_fraction) * RHO_AIR * C_AIR;
    assert_relative_eq!(thermal::volumetric_heat_capacity(element), expected, max_relative = 1e-12);
}

/// Abramowitz and Stegun 7.1.26, max absolute error 1.5e-7. Present only so the
/// analytic oracle above does not need a dependency.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}
