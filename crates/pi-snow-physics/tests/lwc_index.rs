//! The liquid water content index (SMET `S5` under `S5=LWC_INDEX_MITTERER2013`).
//!
//! Mitterer et al. (2013) eq. 1: the average volumetric liquid water content of
//! the entire snowpack over 0.03. Hendrick et al. (2023) eq. 1 make the average
//! layer-height weighted. The paper gives no worked numbers, so these tests
//! check eq. 1 itself on hand-built columns.

use approx::assert_relative_eq;
use pi_snow_physics::stability::{LWC_INDEX_THRESHOLD, lwc_index};

#[test]
fn threshold_is_the_papers_three_percent() {
    assert_eq!(LWC_INDEX_THRESHOLD, 0.03);
}

#[test]
fn a_dry_pack_is_zero() {
    assert_eq!(lwc_index([(0.4, 0.0), (0.2, 0.0)]), Some(0.0));
}

#[test]
fn a_pack_uniformly_at_three_percent_is_one() {
    assert_relative_eq!(
        lwc_index([(0.1, 0.03), (0.5, 0.03), (0.02, 0.03)]).unwrap(),
        1.0,
        epsilon = 1e-12
    );
}

#[test]
fn the_average_is_weighted_by_layer_thickness() {
    // 0.9 m dry under 0.1 m at 6 %: mean theta_w = 0.006, index 0.2. An
    // unweighted mean of the two layers would give 1.0.
    assert_relative_eq!(
        lwc_index([(0.9, 0.0), (0.1, 0.06)]).unwrap(),
        0.2,
        epsilon = 1e-12
    );
}

#[test]
fn equals_total_water_over_snow_depth() {
    // The form the reference output satisfies: MS_Water / (rho_w HS) / 0.03.
    let layers = [(0.12, 0.011), (0.35, 0.047), (0.08, 0.002), (0.61, 0.0)];
    let depth: f64 = layers.iter().map(|l| l.0).sum();
    let water_kg_m2: f64 = layers.iter().map(|l| l.0 * l.1 * 1000.0).sum();
    assert_relative_eq!(
        lwc_index(layers).unwrap(),
        water_kg_m2 / (1000.0 * depth) / 0.03,
        epsilon = 1e-12
    );
}

#[test]
fn no_snow_has_no_index() {
    assert_eq!(lwc_index(std::iter::empty()), None);
    assert_eq!(lwc_index([(0.0, 0.0)]), None);
}
