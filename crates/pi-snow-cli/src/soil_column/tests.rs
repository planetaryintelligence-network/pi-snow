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
            (report.surface_flux_w_m2 - bare::budget(&atm, &params, soil.phase(0)).total).abs()
                <= 1e-6
        );
        assert!(report.soil.ledger_residual_j_m2.abs() <= 1e-3);
        assert!(
            (bare::budget(&atm, &params, bare::pure_phase(300.0)).total
                - bare::budget(&atm, &params, bare::pure_phase(301.0)).total)
                .abs()
                > 1.0
        );
    }
    let mut soil = SoilColumn::uniform(Material::MoistLoam, 280.0);
    let bump = |p: Phase| {
        let t = p.temperature_k;
        -15.0 * (t - 280.0) + 1000.0 * (-(t - 290.0).powi(2)).exp()
    };
    let report = bare::solve(&mut soil, 600.0, Bottom::Insulated, bump).unwrap();
    assert!(!report.diagnostics.certified);
    assert!((report.surface_flux_w_m2 - bump(soil.phase(0))).abs() <= 1e-6);
}

fn bare_corner_fixture() -> (SoilColumn, f64) {
    let soil = SoilColumn::uniform(Material::MoistLoam, FREEZING_K - 0.5);
    let mut trial = soil.clone();
    let full_liquid = trial
        .step(600.0, Surface::Temperature(FREEZING_K), Bottom::Insulated)
        .unwrap()
        .top_flux_w_m2;
    let qa = full_liquid
        - soil.material.water_kg_m3() * LATENT_J_KG * SoilColumn::node_volume(0) / 600.0;
    (soil, qa)
}

fn scan_bare_fixture(soil: &SoilColumn, balance: &impl Fn(Phase) -> f64) -> bare::RootScan {
    let imposed = |t| bare_scan::dirichlet(soil, 600.0, Bottom::Insulated, t).unwrap();
    let lower = solver::surface_stiffness_lower_bound(soil, 600.0, Bottom::Insulated);
    let width = soil.material.water_kg_m3() * LATENT_J_KG * SoilColumn::node_volume(0) / 600.0;
    let certificate = bare_scan::certify(balance, 0.0, lower, width).unwrap();
    bare_scan::scan(
        soil,
        600.0,
        Bottom::Insulated,
        balance,
        lower,
        [imposed(210.0), imposed(340.0)],
        &certificate,
    )
    .unwrap()
}

fn assert_root_cells(scan: &bare::RootScan, expected: &[(bare_scan::Segment, f64, f64)]) {
    assert_eq!(scan.roots.len(), expected.len());
    for (root, &(segment, lo, hi)) in scan.roots.iter().zip(expected) {
        assert_eq!(root.cells.len(), 1);
        assert_eq!(root.cells[0].segment, segment);
        assert_eq!(root.cells[0].lo, lo);
        assert_eq!(root.cells[0].hi, hi);
    }
}

#[test]
fn bare_soil_weighted_phase_corners() {
    use bare_scan::Segment::{Frozen, Plateau};
    for (jump, exact_root) in [(4.0, 41.84656390635812), (-4.0, 45.62953441483138)] {
        let (mut soil, qa) = bare_corner_fixture();
        let balance = |p: Phase| {
            qa - jump / 2.0 - 10.0 * (p.temperature_k - FREEZING_K) + jump * p.liquid_fraction
        };
        let mut scan = scan_bare_fixture(&soil, &balance);
        let report = bare::solve(&mut soil, 600.0, Bottom::Insulated, balance).unwrap();
        assert!(report.diagnostics.certified);
        assert_eq!(
            report.diagnostics.certification.as_ref().unwrap().failed,
            [false; 3]
        );
        assert!(
            report
                .diagnostics
                .certification
                .as_ref()
                .unwrap()
                .jumps
                .iter()
                .all(|j| j.abs() <= bare_scan::TAU)
        );
        assert!(report.flux_residual_w_m2.abs() <= 1e-6);
        assert_eq!(
            report.surface_flux_w_m2, exact_root,
            "A4 must not change root selection"
        );
        let expected = if jump > 0.0 {
            (Frozen, 273.1, FREEZING_K)
        } else {
            (Plateau, 0.0, 0.01)
        };
        assert_root_cells(&scan, &[expected]);
        assert_eq!(scan.jump_crossings, [false; 2]);
        scan.assess(Some(soil.phase(0)));
        assert_eq!(scan.agreement, Some(bare::Agreement::Agree));
        eprintln!(
            "U8 weighted corner jump {jump}: qa {qa}, root {}, roots {}, resolve solves {}",
            report.surface_flux_w_m2,
            scan.roots.len(),
            scan.resolve_solves
        );
    }
}

#[test]
fn bare_soil_discontinuous_evaporation_scan() {
    use bare_scan::Segment::{Frozen, Plateau};
    let (mut soil, qa) = bare_corner_fixture();
    let balance = |p: Phase| {
        qa - 2.0 - 10.0 * (p.temperature_k - FREEZING_K)
            + if p.temperature_k >= FREEZING_K {
                4.0
            } else {
                0.0
            }
    };
    let report = bare::solve(&mut soil, 600.0, Bottom::Insulated, balance).unwrap();
    let d = &report.diagnostics;
    assert_eq!(
        d.certification.as_ref().unwrap().failed,
        [false, false, true]
    );
    assert!((d.certification.as_ref().unwrap().jumps[0] - 4.0).abs() <= 1e-6);
    assert!(d.multiple_roots());
    let scan = d.scan.as_ref().unwrap();
    assert_root_cells(scan, &[(Frozen, 273.1, FREEZING_K), (Plateau, 0.0, 0.01)]);
    assert_eq!(scan.jump_crossings, [true, false]);
    assert_eq!(scan.agreement, Some(bare::Agreement::Agree));
    eprintln!(
        "U8 switched evaporation: qa {qa}, root {}, roots {}, resolve solves {}",
        report.surface_flux_w_m2,
        scan.roots.len(),
        scan.resolve_solves
    );
}

#[test]
fn bare_soil_discontinuous_condensation_refusal() {
    let (mut soil, qa) = bare_corner_fixture();
    let before = soil.clone();
    let balance = |p: Phase| {
        qa + 2.0
            - 10.0 * (p.temperature_k - FREEZING_K)
            - if p.temperature_k >= FREEZING_K {
                4.0
            } else {
                0.0
            }
    };
    let failure = bare::solve(&mut soil, 600.0, Bottom::Insulated, balance).unwrap_err();
    assert_eq!(failure.diagnostics.nonconvergences, 1);
    assert_eq!(soil.enthalpy, before.enthalpy);
    assert_eq!(
        failure.diagnostics.certification.as_ref().unwrap().failed,
        [false, false, true]
    );
    assert!((failure.diagnostics.certification.as_ref().unwrap().jumps[0] + 4.0).abs() <= 1e-6);
    let scan = failure.diagnostics.scan.as_ref().unwrap();
    assert!(scan.roots.is_empty());
    assert_eq!(scan.jump_crossings, [true, false]);
    assert_eq!(scan.agreement, Some(bare::Agreement::Agree));
    eprintln!(
        "U8 switched condensation: {}, roots 0, resolve solves {}",
        failure.message, scan.resolve_solves
    );
}

#[test]
fn bare_soil_schur_slope_predicate() {
    use bare_scan::Segment::Liquid;
    let mut soil = SoilColumn::uniform(Material::MoistLoam, 280.0);
    let balance = |p: Phase| {
        let t = p.temperature_k;
        if t < 280.0 {
            -15.0 * (t - 280.0)
        } else if t <= 282.0 {
            120.0 * (t - 280.0)
        } else {
            240.0 - 15.0 * (t - 282.0)
        }
    };
    let report = bare::solve(&mut soil, 600.0, Bottom::Insulated, balance).unwrap();
    assert!(!report.diagnostics.certified);
    assert!(report.diagnostics.maximum_sampled_budget_slope_w_m2_k > 119.9);
    assert!(report.diagnostics.stiffness_lower_bound_w_m2_k < 120.0);
    assert_eq!(
        report.diagnostics.certification.as_ref().unwrap().failed,
        [true, false, false]
    );
    let scan = report.diagnostics.scan.as_ref().unwrap();
    assert_root_cells(scan, &[(Liquid, 280.0, 280.0), (Liquid, 282.7, 282.8)]);
    assert_eq!(scan.nodes.each_ref().map(|n| n.len()), [633, 101, 670]);
    assert_eq!(scan.jump_crossings, [false; 2]);
    assert!(report.diagnostics.multiple_roots());
    assert!(report.surface_flux_w_m2.abs() <= 1e-6);
    assert_eq!(scan.agreement, Some(bare::Agreement::Agree));
    eprintln!(
        "U8 +120 slope: roots {}, resolve solves {}",
        scan.roots.len(),
        scan.resolve_solves
    );
}

fn bare_bump(p: Phase) -> f64 {
    let t = p.temperature_k;
    -15.0 * (t - 280.0) + 1000.0 * (-(t - 290.0).powi(2)).exp()
}

#[test]
fn bare_soil_bump_analytic_root_count() {
    use bare_scan::Segment::Liquid;
    let mut soil = SoilColumn::uniform(Material::MoistLoam, 280.0);
    let mut linear = soil.clone();
    linear
        .step(600.0, Surface::Flux(100.0), Bottom::Insulated)
        .unwrap();
    let stiffness = 100.0 / (linear.phase(0).temperature_k - 280.0);
    let analytic_residual = |t| stiffness * (t - 280.0) - bare_bump(bare::pure_phase(t));
    assert!(analytic_residual(279.0) < 0.0 && analytic_residual(281.0) > 0.0);
    assert!(
        analytic_residual(289.0) > 0.0
            && analytic_residual(290.0) < 0.0
            && analytic_residual(291.0) > 0.0
    );
    let report = bare::solve(&mut soil, 600.0, Bottom::Insulated, bare_bump).unwrap();
    let scan = report.diagnostics.scan.as_ref().unwrap();
    assert_eq!(
        report.diagnostics.certification.as_ref().unwrap().failed,
        [true, false, false]
    );
    assert_root_cells(
        scan,
        &[
            (Liquid, 280.0, 280.0),
            (Liquid, 289.8, 289.9),
            (Liquid, 290.0, 290.1),
        ],
    );
    assert_eq!(scan.jump_crossings, [false; 2]);
    assert_eq!(scan.agreement, Some(bare::Agreement::Agree));
    for (lo, hi) in [(289.8, 289.9), (290.0, 290.1)] {
        assert!(analytic_residual(lo) * analytic_residual(hi) < 0.0);
    }
    assert!(analytic_residual(280.0).abs() <= 1e-6);
    eprintln!(
        "U8 bump analytic roots 3, path scan roots {}, reduced stiffness {stiffness}, resolve solves {}",
        scan.roots.len(),
        scan.resolve_solves
    );
}

#[test]
fn bare_soil_envelope_soundness() {
    let soil = SoilColumn::uniform(Material::MoistLoam, 280.0);
    let scan = scan_bare_fixture(&soil, &bare_bump);
    let mut checked = 0;
    for node in scan.nodes[0].iter().chain(scan.nodes[2].iter()) {
        if node.coordinate == FREEZING_K {
            continue;
        }
        let q = bare_scan::dirichlet(&soil, 600.0, Bottom::Insulated, node.coordinate).unwrap();
        assert_eq!(
            node.sign,
            Some(bare_scan::sign(q - node.budget)),
            "envelope sign at {} K",
            node.coordinate
        );
        checked += 1;
    }
    assert_eq!(checked, 1301);
    eprintln!(
        "U8 envelope: all {checked} node signs exact, resolve solves {}",
        scan.resolve_solves
    );
}

#[test]
fn bare_soil_path_coordinate_identity() {
    let (soil, qa) = bare_corner_fixture();
    let width = soil.material.water_kg_m3() * LATENT_J_KG * SoilColumn::node_volume(0) / 600.0;
    let qb = qa + width;
    for f in [0.25, 0.5, 0.75] {
        let mut column = soil.clone();
        column
            .step(
                600.0,
                Surface::Flux(qb - (1.0 - f) * width),
                Bottom::Insulated,
            )
            .unwrap();
        assert_eq!(column.phase(0).temperature_k, FREEZING_K);
        assert!((column.phase(0).liquid_fraction - f).abs() <= 1e-9);
    }
    let q = bare_scan::dirichlet(&soil, 600.0, Bottom::Insulated, 272.0).unwrap();
    let mut column = soil.clone();
    column
        .step(600.0, Surface::Flux(q), Bottom::Insulated)
        .unwrap();
    assert!((column.phase(0).temperature_k - 272.0).abs() <= 1e-8);
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

fn coupled_test_snow(thickness: f64, temperature: f64) -> pi_snow_core::Column {
    let profile = pi_snow_io::sno::SnoFile::parse(include_str!(
        "../../../pi-snow-io/tests/fixtures/layered_CLN2.sno"
    ))
    .unwrap();
    let mut column = profile.to_column();
    column.elements.truncate(1);
    column.elements[0].thickness = thickness;
    column.elements[0].composition = pi_snow_core::Composition::dry_snow(250.0);
    column.node_temperature = vec![temperature; 2];
    column
}

#[test]
fn interface_series_conductance() {
    use pi_snow_physics::thermal::Conductivity;
    let mut soil = SoilColumn::uniform(Material::MoistLoam, 280.0);
    let mut snow = coupled_test_snow(0.03, 280.0);
    let conductivity = Conductivity::Yen1981;
    let k_snow = conductivity.of_density(snow.elements[0].density());
    let k_soil = soil.material.conductivity(280.0, 1.0);
    let q = 0.5;
    for (i, depth) in DEPTHS_M.into_iter().enumerate() {
        soil.enthalpy[i] = soil
            .material
            .enthalpy_at_temperature(280.0 - q * depth / k_soil)
            * SoilColumn::node_volume(i);
    }
    snow.node_temperature[1] = 280.0 + q * snow.elements[0].thickness / k_snow;
    let surface = snow.node_temperature[1];
    let bottom = Bottom::Deep {
        temperature_k: 280.0 - q * 8.0 / k_soil,
    };
    let report = coupled::linear_step(
        &mut snow,
        &mut soil,
        600.0,
        bottom,
        q + 15.0 * surface,
        -15.0,
        &[0.0],
        0.0,
        conductivity,
    )
    .unwrap();
    let snow_center = 0.5 * (snow.node_temperature[0] + snow.node_temperature[1]);
    let soil_center = 0.5 * (soil.phase(0).temperature_k + soil.phase(1).temperature_k);
    let series_flux = (snow_center - soil_center)
        / (snow.elements[0].thickness / (2.0 * k_snow)
            + (DEPTHS_M[1] - DEPTHS_M[0]) / (2.0 * k_soil));
    assert!((series_flux - q).abs() <= 1e-10);
    assert!((report.soil.top_flux_w_m2 - series_flux).abs() <= 1e-10);
    assert!(report.combined_ledger_residual_j_m2.abs() <= 1e-3);
}

#[test]
fn coupled_thermal_ledger_and_shared_node() {
    use pi_snow_physics::thermal::Conductivity;
    for thickness in [4e-9, 1e-6, 0.03] {
        for temperature in [268.15, FREEZING_K, 278.15] {
            let mut soil = SoilColumn::uniform(Material::MoistLoam, temperature);
            let mut snow = coupled_test_snow(thickness, temperature);
            let report = coupled::linear_step(
                &mut snow,
                &mut soil,
                600.0,
                Bottom::Geothermal005,
                15.0 * FREEZING_K + 80.0,
                -15.0,
                &[12.0],
                3.0,
                Conductivity::Yen1981,
            )
            .unwrap();
            assert_eq!(snow.node_temperature[0], soil.phase(0).temperature_k);
            assert!(
                report.combined_ledger_residual_j_m2.abs() <= 1e-3,
                "h={thickness}, T={temperature}, residual={}",
                report.combined_ledger_residual_j_m2
            );
            assert!(report.soil.ledger_residual_j_m2.abs() <= 1e-3);
        }
    }
}

#[test]
fn soil_phase_shared_storage_and_transport() {
    use pi_snow_core::options::WaterRetention;
    use pi_snow_physics::water;
    for thickness in [4e-9, 1e-6, 0.03] {
        for temperature in [268.15, FREEZING_K, 278.15] {
            let mut soil = SoilColumn::uniform(Material::MoistLoam, temperature);
            let mut snow = coupled_test_snow(thickness, temperature);
            snow.elements[0].composition.water = 0.1;
            snow.elements[0].composition.air -= 0.1;
            let mut native = snow.clone();
            let reference = water::phase_change(&mut native);
            let report = phase::phase_change(&mut snow, &mut soil).unwrap();
            assert_eq!(report.snow, reference, "snow phase amounts changed");
            assert_eq!(
                snow.elements, native.elements,
                "snow composition operator changed"
            );
            assert_eq!(&snow.node_temperature[1..], &native.node_temperature[1..]);
            assert_eq!(snow.node_temperature[0], soil.phase(0).temperature_k);
            assert!(report.combined_ledger_residual_j_m2.abs() <= 1e-6);
            let before = water::energy_relative_to_melt(&snow) + soil.energy_j_m2();
            let (runoff, _) =
                phase::cascade(&mut snow, &mut soil, WaterRetention::Wever2014Printed).unwrap();
            let after = water::energy_relative_to_melt(&snow) + soil.energy_j_m2();
            assert!((after - before + pi_snow_core::constants::LH_FUSION * runoff).abs() <= 1e-6);
            assert_eq!(snow.node_temperature[0], soil.phase(0).temperature_k);
        }
    }
}

#[test]
fn thin_element_on_soil() {
    use pi_snow_core::options::WaterRetention;
    use pi_snow_physics::surface::{Atmosphere, SurfaceParameters};
    use pi_snow_physics::thermal::Conductivity;
    use pi_snow_physics::turbulence::Scheme;
    let mut soil = SoilColumn::uniform(Material::MoistLoam, 279.15);
    let mut snow = coupled_test_snow(4e-9, 279.15);
    snow.elements[0].composition = pi_snow_core::Composition::dry_snow(600.0);
    assert!((snow.swe() - 2.4e-6).abs() <= 1e-20);
    let water = soil.layer_water_kg_m2();
    let params = SurfaceParameters {
        emissivity: 0.97,
        z_wind: 2.0,
        z_scalar: 2.0,
        z0: 0.01,
        scheme: Scheme::Louis,
    };
    let atm = Atmosphere {
        t_air: 285.15,
        relative_humidity: 0.7,
        wind: 2.0,
        absorbed_shortwave: 200.0,
        ilwr: 350.0,
        pressure: 80000.0,
        rain_rate: 0.0,
    };
    let mut cleared = None;
    for step in 1..=144 {
        if snow.elements.is_empty() {
            bare::step(&mut soil, 600.0, Bottom::Geothermal005, &atm, &params).unwrap();
            snow.node_temperature[0] = soil.phase(0).temperature_k;
        } else {
            let (_, report) = coupled::step(
                &mut snow,
                &mut soil,
                600.0,
                Bottom::Geothermal005,
                &atm,
                &params,
                &[0.0],
                0.0,
                Conductivity::Yen1981,
            )
            .unwrap();
            assert!(report.combined_ledger_residual_j_m2.abs() <= 1e-3);
            let phase = phase::phase_change(&mut snow, &mut soil).unwrap();
            assert!(phase.combined_ledger_residual_j_m2.abs() <= 1e-6);
            phase::cascade(&mut snow, &mut soil, WaterRetention::Wever2014Printed).unwrap();
            if snow.elements.is_empty() {
                cleared = Some(step);
            }
        }
        for t in soil
            .temperatures()
            .into_iter()
            .chain(snow.node_temperature.iter().copied())
        {
            assert!(t.is_finite() && (210.0..=340.0).contains(&t));
        }
        assert_eq!(soil.layer_water_kg_m2(), water);
    }
    assert!(
        snow.elements.is_empty(),
        "4 nm sole element did not clear within 24 h"
    );
    eprintln!(
        "U9 4 nm / 2.4 mg m-2: cleared at 600 s step {:?}, final soil top {} K",
        cleared,
        soil.phase(0).temperature_k
    );
}

#[test]
fn soil_key_refusals() {
    use crate::engine::{self, RunError, Site};
    use pi_snow_core::{Column, PhysicsOptions};
    use pi_snow_io::{
        ini::{Ini, RunConfig},
        smet::Smet,
    };
    let cfg = RunConfig::from_ini(
        &Ini::parse(include_str!(
            "../../../pi-snow-io/tests/fixtures/run_ULAMB1.ini"
        ))
        .unwrap(),
    )
    .unwrap();
    let forcing = Smet::parse(
        "SMET 1.1 ASCII\n[HEADER]\nfields = timestamp TA\n[DATA]\n2025-10-01T00:00 280\n",
    )
    .unwrap();
    let site = Site {
        station_id: "U12".into(),
        station_name: "U12".into(),
        latitude: 40.0,
        longitude: -111.0,
        altitude: 2500.0,
        slope_angle: 0.0,
        slope_azimuth: 0.0,
        soil_albedo: 0.09,
    };
    for settings in [
        vec!["SOIL_PROPERTIES=MOIST_LOAM"],
        vec!["SOIL_BOTTOM=GEOTHERMAL_005"],
        vec!["SOIL_INIT=UNIFORM_TMELT"],
        vec!["SOIL_TBOT_K=280"],
        vec!["SOIL_INIT_OFFSET_K=0"],
        vec![
            "GROUND_BOUNDARY=SOIL_COLUMN",
            "TOTAL_MELT=CROCUS_EQ116",
            "SOIL_PROPERTIES=MOIST_LOAM",
            "SOIL_BOTTOM=GEOTHERMAL_005",
            "SOIL_INIT=UNIFORM_TMELT",
        ],
    ] {
        let options = PhysicsOptions::from_settings(settings.iter().copied()).unwrap();
        let error = engine::run_with_options(
            &cfg,
            &options,
            &site,
            &forcing,
            Column::bare_ground(273.15, 0.0, 0.0),
            forcing.rows[0].timestamp,
        )
        .unwrap_err();
        assert!(
            matches!(error, RunError::IncompatiblePhysics(_)),
            "{settings:?}: {error}"
        );
        assert!(error.to_string().contains("SOIL_"));
    }
    for offset in ["0", "+3", "-3"] {
        let mut options = PhysicsOptions::from_settings([
            "GROUND_BOUNDARY=SOIL_COLUMN",
            "SOIL_PROPERTIES=DRY_LOAM",
            "SOIL_BOTTOM=DEEP_TBOT_8M",
            "SOIL_TBOT_K=281.25",
            "SOIL_INIT=AIR_PROFILE",
        ])
        .unwrap();
        options
            .set(&format!("SOIL_INIT_OFFSET_K={offset}"))
            .unwrap();
        options.soil.validate(true, false).unwrap();
        assert_eq!(options.soil.offset_k(), offset.parse::<f64>().unwrap());
        assert_eq!(options.soil.tbot_k(), Some(281.25));
    }
    assert!(PhysicsOptions::from_settings(["SOIL_TBOT_K=NaN"]).is_err());
    assert!(PhysicsOptions::from_settings(["SOIL_INIT_OFFSET_K=1"]).is_err());
    assert!(PhysicsOptions::from_settings(["SOIL_MAYBE=1"]).is_err());
    let options = PhysicsOptions::from_settings([
        "GROUND_BOUNDARY=SOIL_COLUMN",
        "SOIL_PROPERTIES=MOIST_LOAM",
        "SOIL_BOTTOM=DEEP_TBOT_8M",
        "SOIL_INIT=UNIFORM_TMELT",
    ])
    .unwrap();
    assert!(options.soil.validate(true, false).is_err());
    let options = PhysicsOptions::from_settings([
        "GROUND_BOUNDARY=SOIL_COLUMN",
        "SOIL_PROPERTIES=MOIST_LOAM",
        "SOIL_BOTTOM=GEOTHERMAL_005",
        "SOIL_INIT=UNIFORM_TMELT",
        "SOIL_INIT_OFFSET_K=0",
    ])
    .unwrap();
    assert!(options.soil.validate(true, false).is_err());
    // GPU U12 is checked separately against the existing native argument parser.
}

#[test]
fn soil_runtime_initialization_context() {
    use pi_snow_core::{Column, PhysicsOptions};
    use runtime::{InitializationContext, Runtime};
    let options = PhysicsOptions::from_settings([
        "GROUND_BOUNDARY=SOIL_COLUMN",
        "SOIL_PROPERTIES=MOIST_LOAM",
        "SOIL_BOTTOM=DEEP_TBOT_8M",
        "SOIL_TBOT_K=280",
        "SOIL_INIT=AIR_PROFILE",
        "SOIL_INIT_OFFSET_K=+3",
    ])
    .unwrap();
    let mut snow = Column::bare_ground(273.15, 0.0, 0.0);
    let context = InitializationContext {
        mean_336h_k: None,
        annual_mean_k: 280.0,
    };
    assert!(
        Runtime::new(options.soil, context, &mut snow)
            .unwrap_err()
            .contains("336")
    );
    let context = InitializationContext {
        mean_336h_k: Some(285.0),
        ..context
    };
    let runtime = Runtime::new(options.soil, context, &mut snow).unwrap();
    assert_eq!(
        runtime.column.temperatures(),
        SoilColumn::air_profile(Material::MoistLoam, 280.0, 285.0, 3.0).temperatures()
    );
    assert_eq!(
        snow.node_temperature[0],
        runtime.column.phase(0).temperature_k
    );
    assert_eq!(runtime.initial_graft_j_m2, 0.0);
    assert_eq!(runtime.ledger_residual_j_m2(), 0.0);
}

#[test]
fn soil_output_rows_keep_native_columns_and_weighted_fluxes() {
    use chrono::{TimeZone, Utc};
    use pi_snow_io::smet::{self, OutputRow, Smet};
    let timestamp = Utc.with_ymd_and_hms(2025, 10, 1, 1, 0, 0).unwrap();
    let native = smet::write_output_smet(
        "O",
        "O",
        40.0,
        -111.0,
        2500.0,
        0.0,
        0.0,
        &[OutputRow::nodata(timestamp)],
    );
    let soil = SoilColumn::uniform(Material::MoistLoam, 280.0);
    let mut rows = output::Rows::default();
    rows.accumulate(300.0, 3000.0, -15.0);
    rows.accumulate(600.0, 12000.0, -30.0);
    rows.push(timestamp, &soil);
    let text = rows.extend_smet(&native);
    let result = Smet::parse(&text).unwrap();
    let reference = Smet::parse(&native).unwrap();
    assert_eq!(&result.fields[..reference.fields.len()], &reference.fields);
    assert_eq!(
        &result.rows[0].values[..reference.rows[0].values.len()],
        &reference.rows[0].values
    );
    assert_eq!(
        result.rows[0].values[result.field_index("TSOIL_005").unwrap()],
        Some(280.0)
    );
    assert!(
        (result.rows[0].values[result.field_index("G_SNOW_SOIL").unwrap()].unwrap()
            - 15000.0 / 900.0)
            .abs()
            < 1e-9
    );
    assert_eq!(
        result.rows[0].values[result.field_index("G_BOT").unwrap()],
        Some(-0.05)
    );
    for key in ["units_offset", "units_multiplier"] {
        assert_eq!(
            result.header[key].split_whitespace().count(),
            reference.header[key].split_whitespace().count() + 7
        );
    }
    assert_eq!(
        text.split("[DATA]\n")
            .nth(1)
            .unwrap()
            .strip_suffix('\n')
            .unwrap()
            .strip_prefix(native.split("[DATA]\n").nth(1).unwrap().trim_end())
            .unwrap()
            .split_whitespace()
            .count(),
        7
    );
}

#[test]
fn soil_shared_node_reconciles_native_merge_energy() {
    use pi_snow_physics::{new_snow, thermal, water};
    let mut snow = coupled_test_snow(1e-6, 280.0);
    snow.elements.push(snow.elements[0].clone());
    snow.elements[1].thickness = 0.03;
    snow.node_temperature.push(278.0);
    let mut soil = SoilColumn::uniform(Material::MoistLoam, 280.0);
    let before = water::energy_relative_to_melt(&snow) + soil.energy_j_m2();
    new_snow::merge_thin_elements(&mut snow, 0.001);
    assert_eq!(snow.elements.len(), 1);
    let after_native = water::energy_relative_to_melt(&snow) + soil.energy_j_m2();
    assert!((after_native - before).abs() < 1e-6);
    let exchange = phase::reconcile(&mut snow, &mut soil).unwrap();
    let after_shared = water::energy_relative_to_melt(&snow) + soil.energy_j_m2();
    assert!((after_shared - after_native).abs() < 1e-6);
    assert!(exchange.is_finite());
    assert_eq!(snow.node_temperature[0], soil.phase(0).temperature_k);
    assert!(thermal::internal_energy(&snow).is_finite());
}

#[test]
fn soil_runtime_bare_transfer_and_cost_counters() {
    use pi_snow_core::{Column, PhysicsOptions};
    use pi_snow_physics::{
        surface::{Atmosphere, SurfaceParameters},
        turbulence::Scheme,
    };
    let settings = PhysicsOptions::from_settings([
        "GROUND_BOUNDARY=SOIL_COLUMN",
        "SOIL_PROPERTIES=MOIST_LOAM",
        "SOIL_BOTTOM=GEOTHERMAL_005",
        "SOIL_INIT=UNIFORM_TMELT",
    ])
    .unwrap()
    .soil;
    let mut snow = Column::bare_ground(273.15, 0.0, 0.0);
    let context = runtime::InitializationContext {
        mean_336h_k: None,
        annual_mean_k: 280.0,
    };
    let mut state = runtime::Runtime::new(settings, context, &mut snow).unwrap();
    let atmosphere = Atmosphere {
        t_air: 285.0,
        relative_humidity: 0.7,
        wind: 2.0,
        absorbed_shortwave: 200.0,
        ilwr: 350.0,
        pressure: 80000.0,
        rain_rate: 0.0,
    };
    let parameters = SurfaceParameters {
        emissivity: 0.97,
        z_wind: 2.0,
        z_scalar: 2.0,
        z0: 0.01,
        scheme: Scheme::Louis,
    };
    let budget = state
        .bare_step(&mut snow, 600.0, &atmosphere, &parameters)
        .unwrap();
    assert_eq!(state.counters.bare_rows, 1);
    assert_eq!(state.counters.bare_seconds, 600.0);
    assert_eq!(
        state.counters.bare_turbulence_calls,
        2 * state.counters.budget_evaluations
    );
    // Certification, bracket trials, entry balance, Newton acceptance and final budget.
    assert!(state.counters.budget_evaluations >= 2705 + 2 + 1 + 1 + 1);
    assert!(state.ledger_residual_j_m2().abs() <= 1e-3);
    assert!((state.bottom_energy_j_m2 + 30.0).abs() <= 1e-12);
    assert!((state.top_energy_j_m2 - budget.total * 600.0).abs() <= 1e-3);
    assert_eq!(
        snow.node_temperature[0],
        state.column.phase(0).temperature_k
    );
    let before = (state.column.temperatures(), state.column.energy_j_m2());
    assert!(
        state
            .bare_step(&mut snow, 0.0, &atmosphere, &parameters)
            .is_err()
    );
    assert_eq!(
        (state.column.temperatures(), state.column.energy_j_m2()),
        before
    );
    assert_eq!(state.counters.predicate_not_computed_rows, 1);
    assert_eq!(state.counters.nonconvergences, 1);
    assert_eq!(
        state.counters.certified_rows + state.counters.uncertified_rows,
        1
    );
}

fn engine_soil_fixture() -> (
    pi_snow_io::ini::RunConfig,
    crate::engine::Site,
    pi_snow_io::smet::Smet,
) {
    let cfg = pi_snow_io::ini::RunConfig::from_ini(
        &pi_snow_io::ini::Ini::parse(include_str!(
            "../../../pi-snow-io/tests/fixtures/run_ULAMB1.ini"
        ))
        .unwrap(),
    )
    .unwrap();
    let site = crate::engine::Site {
        station_id: "SOIL_UNIT".into(),
        station_name: "soil unit test".into(),
        latitude: 40.0,
        longitude: -111.0,
        altitude: 2500.0,
        slope_angle: 0.0,
        slope_azimuth: 0.0,
        soil_albedo: 0.09,
    };
    let forcing = pi_snow_io::smet::Smet::parse(
        "SMET 1.1 ASCII\n[HEADER]\nsoil_mean_336h_k = 280\nfields = timestamp TA RH VW DW ISWR ILWR PSUM TSG\n[DATA]\n2025-10-01T00:00 280 0.6 2 0 0 310 0 273.15\n2025-10-01T00:10 280 0.6 2 0 0 310 0 273.15\n2025-10-01T01:00 280 0.6 2 0 0 310 0 273.15\n"
    ).unwrap();
    (cfg, site, forcing)
}

fn engine_soil_options() -> pi_snow_core::PhysicsOptions {
    pi_snow_core::PhysicsOptions::from_settings([
        "GROUND_BOUNDARY=SOIL_COLUMN",
        "SOIL_PROPERTIES=MOIST_LOAM",
        "SOIL_BOTTOM=DEEP_TBOT_8M",
        "SOIL_TBOT_K=280",
        "SOIL_INIT=AIR_PROFILE",
    ])
    .unwrap()
}

#[test]
fn soil_engine_dispatch_bare_snow_and_replayed_state() {
    let (cfg, site, forcing) = engine_soil_fixture();
    let options = engine_soil_options();
    let end = forcing.rows.last().unwrap().timestamp;
    for snow in [
        pi_snow_core::Column::bare_ground(FREEZING_K, 0.0, 0.0),
        coupled_test_snow(0.1, FREEZING_K - 1.0),
    ] {
        let bare = snow.elements.is_empty();
        let out =
            crate::engine::run_with_options(&cfg, &options, &site, &forcing, snow.clone(), end)
                .unwrap();
        let data = out.soil.unwrap();
        assert_eq!(data.output.rows.len(), 2);
        assert!(data.maximum_row_ledger_residual_j_m2 <= 1e-3);
        assert!(data.runtime.ledger_residual_j_m2().abs() <= 1e-3);
        assert_eq!(data.maximum_relative_soil_water_change, 0.0);
        assert_eq!(data.phase_complementarity_failures, 0);
        assert_eq!(data.runtime.counters.bare_consistency_failures, 0);
        assert_eq!(data.runtime.counters.bare_rows > 0, bare);
        assert_eq!(data.runtime.counters.snow_rows > 0, !bare);
        assert!(
            out.smet
                .contains("TSOIL_005 TSOIL_010 TSOIL_020 TSOIL_050 SOIL_H G_SNOW_SOIL G_BOT")
        );
        assert!(out.smet.contains("soil_max_row_ledger_residual_j_m2 ="));
        let mut without_mean = forcing.clone();
        without_mean.header.remove("soil_mean_336h_k");
        let replay = crate::engine::run_with_soil_state(
            &cfg,
            &options,
            &site,
            &without_mean,
            snow,
            end,
            None,
            SoilColumn::air_profile(Material::MoistLoam, 280.0, 280.0, 0.0),
        )
        .unwrap();
        assert_eq!(out.smet, replay.smet);
        assert_eq!(out.pro, replay.pro);
    }
}

#[test]
fn soil_engine_requires_bound_initialization_history() {
    let (cfg, site, mut forcing) = engine_soil_fixture();
    let options = engine_soil_options();
    forcing.header.remove("soil_mean_336h_k");
    let error = crate::engine::run_with_options(
        &cfg,
        &options,
        &site,
        &forcing,
        pi_snow_core::Column::bare_ground(FREEZING_K, 0.0, 0.0),
        forcing.rows.last().unwrap().timestamp,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        crate::engine::RunError::IncompatiblePhysics(_)
    ));
    assert!(error.to_string().contains("soil_mean_336h_k"));
}

#[test]
fn soil_engine_solver_stop_retains_completed_intervals() {
    let (cfg, site, mut forcing) = engine_soil_fixture();
    let i = forcing.field_index("ILWR").unwrap();
    forcing.rows[2].values[i] = Some(10000.0);
    let error = crate::engine::run_with_options(
        &cfg,
        &engine_soil_options(),
        &site,
        &forcing,
        pi_snow_core::Column::bare_ground(FREEZING_K, 0.0, 0.0),
        forcing.rows.last().unwrap().timestamp,
    )
    .unwrap_err();
    let crate::engine::RunError::SoilSolver {
        step,
        time,
        partial,
        ..
    } = error
    else {
        panic!("unexpected soil stop: {error}");
    };
    assert_eq!(step, 2);
    assert_eq!(time, forcing.rows[2].timestamp);
    let data = partial.soil.unwrap();
    assert_eq!(data.output.rows.len(), 1);
    assert_eq!(data.output.rows[0].timestamp, forcing.rows[1].timestamp);
    assert!(partial.smet.contains("2025-10-01T00:10"));
    assert!(!partial.smet.contains("2025-10-01T01:00"));
    assert_eq!(data.runtime.counters.nonconvergences, 1);
}
