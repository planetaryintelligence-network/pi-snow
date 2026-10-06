use super::*;

#[test]
fn bare_soil_surface_solve() {
    use pi_snow_physics::surface::{Atmosphere, SurfaceParameters};
    use pi_snow_physics::turbulence::Scheme;
    let params = SurfaceParameters {
        emissivity: 0.97,
        z_wind: 2.0,
        z_scalar: 2.0,
        z0: 0.01,
        scheme: Scheme::Louis,
    };
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
        let report = bare::step(&mut soil, 600.0, Bottom::Geothermal005, &atm, &params).unwrap();
        assert!(
            (report.surface_flux_w_m2
                - bare::budget(&atm, &params, soil.phase(0).temperature_k).total)
                .abs()
                <= 1e-6
        );
        assert!(report.soil.ledger_residual_j_m2.abs() <= 1e-3);
        assert!(
            (bare::budget(&atm, &params, 300.0).total - bare::budget(&atm, &params, 301.0).total)
                .abs()
                > 1.0
        );
    }
    let mut soil = SoilColumn::uniform(Material::MoistLoam, 280.0);
    let bump = |t: f64| -15.0 * (t - 280.0) + 1000.0 * (-(t - 290.0).powi(2)).exp();
    let report = bare::solve(&mut soil, 600.0, Bottom::Insulated, bump).unwrap();
    assert_eq!(report.monotonicity_violations, 1);
    assert!((report.surface_flux_w_m2 - bump(soil.phase(0).temperature_k)).abs() <= 1e-6);
}

#[test]
fn soil_column_periodic_damping() {
    let material = Material::MoistLoam;
    let conductivity = material.conductivity(FREEZING_K, 1.0);
    let diffusivity = conductivity / material.capacity(1.0);
    for (days, depths, tolerance, lag_tolerance) in [
        (1.0, [0.05, 0.20], 0.10, 7200.0),
        (30.0, [0.20, 0.50], 0.05, 86400.0),
    ] {
        let period = days * 86400.0;
        let omega = 2.0 * std::f64::consts::PI / period;
        let damping = (diffusivity * period / std::f64::consts::PI).sqrt();
        let mut soil = SoilColumn::uniform(material, 283.15);
        // Initialise the exact half-space profile. Prescribe its bottom flux
        // so the test measures discretisation, without a reflected 1 m wave.
        for i in 0..10 {
            let temperature =
                283.15 + (-DEPTHS_M[i] / damping).exp() * (-DEPTHS_M[i] / damping).sin();
            soil.enthalpy[i] =
                material.enthalpy_at_temperature(temperature) * SoilColumn::node_volume(i);
        }
        let mut sine = [0.0; 2];
        let mut cosine = [0.0; 2];
        let steps_per_period = (period / 600.0) as usize;
        for step in 1..=steps_per_period * 4 {
            let phase = omega * step as f64 * 600.0;
            let bottom_phase = phase - 1.0 / damping;
            let bottom_flux = conductivity / damping
                * (-1.0 / damping).exp()
                * (bottom_phase.sin() + bottom_phase.cos());
            soil.step(
                600.0,
                Surface::Temperature(283.15 + phase.sin()),
                Bottom::Flux(bottom_flux),
            )
            .unwrap();
            if step > steps_per_period * 3 {
                for (i, depth) in depths.into_iter().enumerate() {
                    let anomaly = soil.temperature_at_depth(depth) - 283.15;
                    sine[i] += anomaly * phase.sin() * 2.0 / steps_per_period as f64;
                    cosine[i] += anomaly * phase.cos() * 2.0 / steps_per_period as f64;
                }
            }
        }
        for (i, depth) in depths.into_iter().enumerate() {
            let measured = sine[i].hypot(cosine[i]);
            let expected = (-depth / damping).exp();
            assert!(
                (measured / expected - 1.0).abs() <= tolerance,
                "period {days} d, depth {depth}: amplitude {measured}, expected {expected}"
            );
            let lag = (-cosine[i]).atan2(sine[i]) / omega;
            let expected_lag = depth / damping / omega;
            assert!(
                (lag - expected_lag).abs() <= lag_tolerance,
                "period {days} d, depth {depth}: lag {lag}, expected {expected_lag}"
            );
        }
    }
}

fn erf(x: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26, absolute error below 1.5e-7.
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t;
    x.signum() * (1.0 - p * (-x * x).exp())
}

#[test]
fn soil_stefan_neumann() {
    let material = Material::MoistLoam;
    let k_frozen = material.conductivity(FREEZING_K - 5.0, 0.0);
    let k_liquid = material.conductivity(FREEZING_K, 1.0);
    let a_frozen = k_frozen / material.capacity(0.0);
    let a_liquid = k_liquid / material.capacity(1.0);
    let latent = material.water_kg_m3() * LATENT_J_KG;
    // Two-phase Neumann solution, with the +2 C unfrozen sensible term.
    let balance = |lambda: f64| {
        let nu = lambda * (a_frozen / a_liquid).sqrt();
        k_frozen * 5.0 * (-lambda * lambda).exp()
            / (std::f64::consts::PI.sqrt() * a_frozen.sqrt() * erf(lambda))
            - k_liquid * 2.0 * (-nu * nu).exp()
                / (std::f64::consts::PI.sqrt() * a_liquid.sqrt() * (1.0 - erf(nu)))
            - latent * lambda * a_frozen.sqrt()
    };
    let (mut lo, mut hi) = (0.0001, 1.0);
    assert!(balance(lo) > 0.0 && balance(hi) < 0.0);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if balance(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let lambda = 0.5 * (lo + hi);
    let nu = lambda * (a_frozen / a_liquid).sqrt();
    let reference = 2.0 * lambda * (a_frozen * 10.0 * 86400.0).sqrt();
    let mut soil = SoilColumn::uniform(material, FREEZING_K + 2.0);
    let mut insulated = soil.clone();
    for step in 1..=1440 {
        let time = step as f64 * 600.0;
        let exact_bottom = -k_liquid * 2.0 * (-1.0 / (4.0 * a_liquid * time)).exp()
            / ((1.0 - erf(nu)) * (std::f64::consts::PI * a_liquid * time).sqrt());
        soil.step(
            600.0,
            Surface::Temperature(FREEZING_K - 5.0),
            Bottom::Flux(exact_bottom),
        )
        .unwrap();
        insulated
            .step(
                600.0,
                Surface::Temperature(FREEZING_K - 5.0),
                Bottom::Insulated,
            )
            .unwrap();
        if step % 144 == 0 {
            let (volume, interp) = freezing_fronts(&soil);
            let analytic = 2.0 * lambda * (a_frozen * time).sqrt();
            eprintln!(
                "U5 day {}: ice-equivalent {volume:.12} m, 50%-liquid {interp:.12} m, Neumann {analytic:.12} m, volume error {:.6}%",
                step / 144,
                100.0 * (volume / analytic - 1.0)
            );
        }
    }
    let (frozen_volume, measured) = freezing_fronts(&soil);
    eprintln!(
        "U5 insulated diagnostic: {:?}; expected amended volume about 0.45416 m (-1.37%)",
        freezing_fronts(&insulated)
    );
    eprintln!(
        "U5 final half-space: volume {frozen_volume:.12} m, 50%-liquid {measured:.12} m, Neumann {reference:.12} m"
    );
    assert!(
        (frozen_volume / reference - 1.0).abs() <= 0.05,
        "10 d ice-equivalent freezing front {frozen_volume} m, Neumann {reference} m"
    );
}

fn freezing_fronts(soil: &SoilColumn) -> (f64, f64) {
    let crossing = (0..9)
        .find(|&i| soil.phase(i).liquid_fraction <= 0.5 && soil.phase(i + 1).liquid_fraction > 0.5)
        .unwrap();
    let f0 = soil.phase(crossing).liquid_fraction;
    let f1 = soil.phase(crossing + 1).liquid_fraction;
    let measured =
        DEPTHS_M[crossing] + (DEPTHS_M[crossing + 1] - DEPTHS_M[crossing]) * (0.5 - f0) / (f1 - f0);
    let frozen_volume: f64 = (0..10)
        .map(|i| SoilColumn::node_volume(i) * (1.0 - soil.phase(i).liquid_fraction))
        .sum();
    (frozen_volume, measured)
}

#[test]
fn soil_initial_profile() {
    for material in [Material::MoistLoam, Material::DryLoam] {
        let warm = SoilColumn::air_profile(material, 275.0, 285.0, 3.0);
        let cold = SoilColumn::air_profile(material, 275.0, 285.0, -3.0);
        let damping =
            (material.conductivity(FREEZING_K, 1.0) / material.capacity(1.0) * 30.0 * 86400.0
                / std::f64::consts::PI)
                .sqrt();
        for i in 0..10 {
            let expected = 275.0 + 10.0 * (-DEPTHS_M[i] / damping).exp() + 3.0;
            assert!((warm.phase(i).temperature_k - expected).abs() < 1e-12);
            assert!(
                (warm.phase(i).temperature_k - cold.phase(i).temperature_k - 6.0).abs() < 1e-12
            );
        }
        let uniform = SoilColumn::uniform(material, FREEZING_K);
        for i in 0..10 {
            assert_eq!(uniform.phase(i).liquid_fraction, 1.0);
        }
    }
}

#[test]
fn soil_enthalpy_ledger() {
    let mut soil = SoilColumn::uniform(Material::MoistLoam, 271.15);
    let mut seed = 17_u64;
    let water = soil.layer_water_kg_m2();
    for i in 0..10_000 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let flux = (seed >> 32) as f64 / u32::MAX as f64 * 600.0 - 300.0;
        let report = soil
            .step(600.0, Surface::Flux(flux), Bottom::Geothermal005)
            .unwrap_or_else(|e| panic!("step {i}: {e}"));
        assert!(report.ledger_residual_j_m2.abs() <= 1e-3);
        assert_eq!(soil.layer_water_kg_m2(), water);
        for n in 0..10 {
            let p = soil.phase(n);
            assert!((0.0..=1.0).contains(&p.liquid_fraction));
            if p.temperature_k < FREEZING_K {
                assert_eq!(p.liquid_fraction, 0.0);
            }
            if p.temperature_k > FREEZING_K {
                assert_eq!(p.liquid_fraction, 1.0);
            }
            if p.liquid_fraction > 0.0 && p.liquid_fraction < 1.0 {
                assert_eq!(p.temperature_k, FREEZING_K);
            }
        }
    }
}

#[test]
fn soil_memory_eigen_decay() {
    let mut soil = SoilColumn::uniform(Material::MoistLoam, 284.15);
    for day in 1..=14 {
        for _ in 0..144 {
            soil.step(
                600.0,
                Surface::Robin {
                    transfer_w_m2_k: 15.0,
                    air_k: 283.15,
                },
                Bottom::Deep {
                    temperature_k: 283.15,
                },
            )
            .unwrap();
        }
        if day == 7 || day == 14 {
            let expected = if day == 7 {
                [0.107, 0.200, 0.352]
            } else {
                [0.038, 0.071, 0.125]
            };
            for (depth, reference) in [0.0508, 0.2032, 0.508].into_iter().zip(expected) {
                let anomaly = soil.temperature_at_depth(depth) - 283.15;
                assert!(
                    (anomaly - reference).abs() <= 0.10 * reference,
                    "day {day}, depth {depth}: {anomaly}, expected {reference}"
                );
            }
        }
    }
}

#[test]
fn soil_phase_complementarity() {
    for material in [Material::MoistLoam, Material::DryLoam] {
        let latent = material.water_kg_m3() * LATENT_J_KG;
        for h in [
            -latent,
            0.0,
            latent * 0.3,
            latent * 0.8,
            latent,
            2.0 * latent,
        ] {
            let phase = material.phase(h);
            if h <= 0.0 {
                assert_eq!(phase.liquid_fraction, 0.0);
            } else if h >= latent {
                assert_eq!(phase.liquid_fraction, 1.0);
            } else {
                assert_eq!(phase.temperature_k, FREEZING_K);
            }
        }
    }
}

#[test]
fn soil_water_fixed() {
    for material in [Material::MoistLoam, Material::DryLoam] {
        let mut soil = SoilColumn::uniform(material, 278.15);
        let before = soil.layer_water_kg_m2();
        for _ in 0..1440 {
            soil.step(600.0, Surface::Flux(-150.0), Bottom::Insulated)
                .unwrap();
            for (a, b) in soil.layer_water_kg_m2().iter().zip(before) {
                assert!((a - b).abs() <= 1e-12 * b);
            }
        }
    }
}
