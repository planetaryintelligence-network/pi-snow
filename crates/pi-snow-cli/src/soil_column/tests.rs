use super::*;

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
