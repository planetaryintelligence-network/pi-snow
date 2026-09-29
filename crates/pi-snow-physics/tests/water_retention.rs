//! Holding capacity from Coleou and Lesaffre (1998, Ann. Glaciol. 26, 64-68),
//! checked against the paper's own numbers, and the Phase 2 form kept as the
//! default.

use approx::assert_relative_eq;
use pi_snow_core::constants::{RHO_ICE, RHO_WATER};
use pi_snow_core::options::WaterRetention;
use pi_snow_core::{Column, Composition, Element, Microstructure};
use pi_snow_physics::water::{self, CL98_MEASURED, CL98_NARROW};

const CL: WaterRetention = WaterRetention::ColeouLesaffre1998;

/// theta_i at P/(1-P) = x, with P = 1 - theta_i.
fn ice_at(x: f64) -> f64 {
    1.0 / (1.0 + x)
}

#[test]
fn the_regression_reproduces_the_papers_checks() {
    // "about 7 % of the mass" at P/(1-P) ~ 0.9 and "about 12 %" at ~ 1.8.
    assert_relative_eq!(water::cl98_mass_fraction(0.9), 0.06839, epsilon = 1e-9);
    assert_relative_eq!(water::cl98_mass_fraction(1.8), 0.11978, epsilon = 1e-9);
    // "for P = 0.5, the regression gives W_pi = 7.3 %", which holds with W as a
    // fraction of wet mass (7.34 %) and not of dry mass (6.79 %).
    let share = water::cl98_pore_share(1.0);
    assert_relative_eq!(
        share,
        0.0741 / (1.0 - 0.0741) * RHO_ICE / RHO_WATER,
        epsilon = 1e-12
    );
    assert!((share - 0.073).abs() < 0.0005, "{share}");
    // "when W_mi varies from 5 to 15 % of the mass, W_pi varies from 6.5 to 8.5 %".
    for w in [0.05, 0.15] {
        let s = water::cl98_pore_share((w - 0.017) / 0.0571);
        assert!((0.065..=0.085).contains(&s), "{w} -> {s}");
    }
}

#[test]
fn inside_the_measured_range_the_regression_applies() {
    // theta_i = 0.5: W = 0.0741, theta_w = W/(1-W) (917/1000) 0.5.
    let expect = 0.0741 / (1.0 - 0.0741) * 0.917 * 0.5;
    assert_relative_eq!(water::capacity(CL, 0.5), expect, max_relative = 1e-12);
    assert_relative_eq!(expect, 0.0366939, epsilon = 1e-7);
}

#[test]
fn outside_it_a_constant_share_of_pore_volume_applies() {
    // Below the lightest sample (theta_i < 0.3051) and above the densest
    // (theta_i > 0.6083), the regression's share at the nearer end.
    assert_relative_eq!(water::cl98_pore_share(2.278), 0.06941, epsilon = 5e-6);
    assert_relative_eq!(water::cl98_pore_share(0.644), 0.08092, epsilon = 5e-6);
    for ice in [0.01, 0.1, 0.2, 0.3] {
        assert_relative_eq!(
            water::capacity(CL, ice),
            water::cl98_pore_share(2.278) * (1.0 - ice),
            max_relative = 1e-12
        );
    }
    for ice in [0.62, 0.7, 0.8, 0.9] {
        assert_relative_eq!(
            water::capacity(CL, ice),
            water::cl98_pore_share(0.644) * (1.0 - ice),
            max_relative = 1e-12
        );
    }
    assert_eq!(water::capacity(CL, 1.0), 0.0);
    // No ice at all: the limit of the constant share.
    assert_relative_eq!(water::capacity(CL, 0.0), water::cl98_pore_share(2.278));
}

#[test]
fn the_form_is_continuous_at_both_ends_of_either_range() {
    for (retention, range) in [
        (CL, CL98_MEASURED),
        (WaterRetention::ColeouLesaffre1998Narrow, CL98_NARROW),
    ] {
        for x in [range.low, range.high] {
            let ice = ice_at(x);
            let a = water::capacity(retention, ice * (1.0 - 1e-12));
            let b = water::capacity(retention, ice * (1.0 + 1e-12));
            assert_relative_eq!(a, b, max_relative = 1e-9);
        }
    }
}

#[test]
fn capacity_stays_inside_the_pores_and_falls_as_the_snow_densifies() {
    let mut previous = f64::INFINITY;
    for i in 0..=1000 {
        let ice = i as f64 / 1000.0;
        let c = water::capacity(CL, ice);
        assert!(c >= 0.0 && c <= 0.09 * (1.0 - ice) + 1e-15, "{ice} -> {c}");
        assert!(c <= previous + 1e-15, "not monotone at {ice}");
        previous = c;
    }
}

#[test]
fn the_narrow_range_matches_the_primary_inside_it() {
    for x in [0.9, 1.0, 1.4, 1.8] {
        let ice = ice_at(x);
        assert_relative_eq!(
            water::capacity(WaterRetention::ColeouLesaffre1998Narrow, ice),
            water::capacity(CL, ice),
            max_relative = 1e-12
        );
    }
    // Between the two lower ends they differ, by at most 0.0061 of pore volume.
    let ice = ice_at(0.644);
    let gap =
        water::capacity(CL, ice) - water::capacity(WaterRetention::ColeouLesaffre1998Narrow, ice);
    assert!(gap > 0.0 && gap / (1.0 - ice) < 0.0062, "{gap}");
}

fn wet_element(ice: f64, water_fraction: f64) -> Element {
    Element {
        thickness: 0.05,
        composition: Composition {
            ice,
            water: water_fraction,
            air: 1.0 - ice - water_fraction,
            soil: 0.0,
        },
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
        grain_history: 0,
    }
}

fn wet_column() -> Column {
    let mut c = Column::bare_ground(273.15, 0.0, 0.0);
    c.elements = vec![
        wet_element(0.8, 0.05),
        wet_element(0.45, 0.12),
        wet_element(0.25, 0.2),
    ];
    c.node_temperature = vec![273.15; 4];
    c
}

#[test]
fn the_default_cascade_is_the_phase_2_cascade_bit_for_bit() {
    let (mut a, mut b) = (wet_column(), wet_column());
    let ra = water::cascade(&mut a);
    let rb = water::cascade_with(&mut b, WaterRetention::default());
    assert_eq!(ra.to_bits(), rb.to_bits());
    for (x, y) in a.elements.iter().zip(&b.elements) {
        assert_eq!(x.composition.water.to_bits(), y.composition.water.to_bits());
    }
    for ice in [0.05, 0.23, 0.5, 0.812, 0.9] {
        assert_eq!(
            water::capacity(WaterRetention::Wever2014Printed, ice).to_bits(),
            water::holding_capacity(ice).to_bits()
        );
    }
}

#[test]
fn the_cascade_keeps_exactly_the_capacity_and_conserves_water() {
    let mut c = wet_column();
    let before = c.liquid_water();
    let runoff = water::cascade_with(&mut c, CL);
    assert_relative_eq!(c.liquid_water() + runoff, before, max_relative = 1e-12);
    for e in &c.elements {
        assert_relative_eq!(
            e.composition.water,
            water::capacity(CL, e.composition.ice),
            max_relative = 1e-12
        );
    }
    // Dense snow (theta_i 0.8, 734 kg m^-3) now holds water; the printed form
    // held almost none (0.0012 by volume).
    assert!(water::capacity(CL, 0.8) > 0.015);
    assert!(water::holding_capacity(0.8) < 0.002);
}
