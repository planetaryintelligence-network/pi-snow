//! Albedo and shortwave absorption with depth: Crocus's three-band scheme,
//! Vionnet et al. (2012) eqs. 13-14 and Table 4, checked on the page.

use approx::assert_relative_eq;
use pi_snow_core::constants::T_MELT;
use pi_snow_core::{Column, Composition, Element, Microstructure};
use pi_snow_physics::albedo;

fn element(h: f64, rho: f64, d: f64, s: f64, grain_mm: f64, age: f64) -> Element {
    Element {
        thickness: h,
        composition: Composition::dry_snow(rho),
        micro: Microstructure {
            dendricity: d,
            sphericity: s,
            grain_radius: 0.5e-3 * grain_mm,
            bond_radius: 0.04e-3,
            coordination_number: 2.0,
            marker: 0,
        },
        age_days: age,
        stress: 0.0,
        stress_rate: 0.0,
        metamorphism: 0.0,
        soil_density: 0.0,
        soil_conductivity: 0.0,
        soil_heat_capacity: 0.0,
    }
}

fn column(elements: Vec<Element>) -> Column {
    let n = elements.len();
    let mut c = Column::bare_ground(T_MELT - 5.0, 0.0, 0.0);
    c.elements = elements;
    c.node_temperature = vec![T_MELT - 5.0; n + 1];
    c
}

#[test]
fn optical_diameter_is_eq_13() {
    // Dendritic: 1e-4 [d + (1 - d)(4 - s)] m. Non-dendritic:
    // gs s + (1 - s) max(4e-4, gs / 2), gs the grain diameter in m.
    let m = |d, s, g| element(0.01, 100.0, d, s, g, 0.0).micro;
    assert_relative_eq!(albedo::optical_diameter(&m(1.0, 0.5, 0.3)), 1e-4);
    assert_relative_eq!(
        albedo::optical_diameter(&m(0.5, 0.5, 0.3)),
        1e-4 * (0.5 + 0.5 * 3.5)
    );
    assert_relative_eq!(albedo::optical_diameter(&m(0.0, 1.0, 0.6)), 0.6e-3);
    assert_relative_eq!(albedo::optical_diameter(&m(0.0, 0.0, 0.6)), 4e-4);
}

#[test]
fn fresh_fine_snow_band_albedos_are_table_4() {
    let c = column(vec![element(0.10, 100.0, 1.0, 0.5, 0.3, 0.0)]);
    let r = albedo::shortwave(&c, 500.0, 70_000.0);
    let root = (1e-4_f64).sqrt();
    let a1 = (0.96_f64 - 1.58 * root).min(0.92);
    let a2 = (0.9 - 15.4 * root).max(0.3);
    let a3 = 346.3 * 1e-4 - 32.31 * root + 0.88;
    assert_relative_eq!(
        r.albedo,
        0.71 * a1 + 0.21 * a2 + 0.08 * a3,
        max_relative = 1e-12
    );
    assert!(r.albedo > 0.84 && r.albedo < 0.87, "{}", r.albedo);
}

#[test]
fn ageing_darkens_the_visible_band_scaled_by_pressure() {
    // Delta_age = min(1, max(P / 870 hPa, 0.5)) 0.2 A / 60. At 700 hPa and 30
    // days: 0.8046 * 0.2 * 0.5 = 0.0805 off band 1.
    let fresh = albedo::shortwave(
        &column(vec![element(0.1, 100.0, 1.0, 0.5, 0.3, 0.0)]),
        500.0,
        70_000.0,
    );
    let aged = albedo::shortwave(
        &column(vec![element(0.1, 100.0, 1.0, 0.5, 0.3, 30.0)]),
        500.0,
        70_000.0,
    );
    let delta = (700.0_f64 / 870.0).clamp(0.5, 1.0) * 0.2 * 30.0 / 60.0;
    assert_relative_eq!(
        fresh.albedo - aged.albedo,
        0.71 * delta,
        max_relative = 1e-9
    );
}

#[test]
fn absorbed_energy_is_conserved_across_surface_snow_and_ground() {
    let c = column(vec![
        element(0.02, 300.0, 0.0, 0.8, 0.6, 20.0),
        element(0.01, 150.0, 0.2, 0.6, 0.4, 5.0),
        element(0.01, 100.0, 1.0, 0.5, 0.3, 0.0),
    ]);
    let sw = 600.0;
    let r = albedo::shortwave(&c, sw, 70_000.0);
    let total = r.surface + r.per_element.iter().sum::<f64>() + r.to_ground;
    assert_relative_eq!(total, (1.0 - r.albedo) * sw, max_relative = 1e-12);
    assert_eq!(r.per_element.len(), 3);
    assert!(
        r.to_ground > 0.0,
        "4 cm of snow lets some visible light through"
    );
}

#[test]
fn a_deep_pack_absorbs_everything_before_the_ground() {
    let layers = (0..40)
        .map(|_| element(0.05, 250.0, 0.0, 0.9, 0.5, 10.0))
        .collect();
    let r = albedo::shortwave(&column(layers), 600.0, 70_000.0);
    assert!(r.to_ground < 1e-6, "{}", r.to_ground);
}
