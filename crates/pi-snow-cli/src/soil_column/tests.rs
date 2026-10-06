use super::*;

#[test]
fn soil_column_periodic_damping() {
    let material = Material::MoistLoam;
    let conductivity = material.conductivity(1.0);
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
    let k_frozen = material.conductivity(0.0);
    let k_liquid = material.conductivity(1.0);
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
    let reference = (lo + hi) * (a_frozen * 10.0 * 86400.0).sqrt();
    let mut soil = SoilColumn::uniform(material, FREEZING_K + 2.0);
    for _ in 0..1440 {
        soil.step(
            600.0,
            Surface::Temperature(FREEZING_K - 5.0),
            Bottom::Insulated,
        )
        .unwrap();
    }
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
    eprintln!(
        "Neumann diagnostics: liquid fractions {:?}, frozen-volume front {frozen_volume} m",
        std::array::from_fn::<_, 10, _>(|i| soil.phase(i).liquid_fraction)
    );
    assert!(
        (measured / reference - 1.0).abs() <= 0.05,
        "10 d freezing front {measured} m, Neumann {reference} m"
    );
}

#[test]
fn soil_initial_profile() {
    for material in [Material::MoistLoam, Material::DryLoam] {
        let warm = SoilColumn::air_profile(material, 275.0, 285.0, 3.0);
        let cold = SoilColumn::air_profile(material, 275.0, 285.0, -3.0);
        let damping = (material.conductivity(1.0) / material.capacity(1.0) * 30.0 * 86400.0
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
