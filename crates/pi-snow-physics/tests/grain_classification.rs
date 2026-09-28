//! `GRAIN_CLASSIFICATION=LAFAYSSE2026`: Crocus v3.0.2's grain-type diagnosis.
//!
//! Every cell and boundary of Lafaysse et al. (2026) Eq. 144 (p. 6297), every
//! transition of Eq. 50 (p. 6284), and the identity of the optical-diameter
//! round trip (Eqs. D1, D2, D5, D6) that lets Eq. 144 read this engine's state
//! directly. Codes are primary digit, secondary digit, marker (1 PP ... 7 MF).

use pi_snow_core::Microstructure;
use pi_snow_physics::grain_classification::{lafaysse2026, merge_history, update_history};

fn micro(d: f64, s: f64, grain_mm: f64) -> Microstructure {
    Microstructure {
        dendricity: d,
        sphericity: s,
        grain_radius: 0.5e-3 * grain_mm,
        bond_radius: 0.04e-3,
        coordination_number: 2.0,
        marker: 0,
    }
}

fn class(d: f64, s: f64, grain_mm: f64, h: u8) -> u16 {
    lafaysse2026(&micro(d, s, grain_mm), h)
}

#[test]
fn dendritic_branch_follows_dendricity_bins() {
    // delta in [0; 0.3[: DF+FC below S 0.55, DF+RG from 0.55.
    assert_eq!(class(0.01, 0.0, 0.3, 0), 240);
    assert_eq!(class(0.29, 0.5499, 0.3, 0), 240);
    assert_eq!(class(0.29, 0.55, 0.3, 0), 230);
    // [0.3; 0.6[: DF; [0.6; 0.8[: PP+DF; [0.8; 1.0]: PP.
    assert_eq!(class(0.3, 0.1, 0.3, 0), 220);
    assert_eq!(class(0.5999, 0.9, 0.3, 0), 220);
    assert_eq!(class(0.6, 0.5, 0.3, 0), 120);
    assert_eq!(class(0.7999, 0.5, 0.3, 0), 120);
    assert_eq!(class(0.8, 0.5, 0.3, 0), 110);
    assert_eq!(class(1.0, 0.5, 0.1, 0), 110);
}

#[test]
fn dendritic_snow_ignores_history_and_size() {
    for h in 0..=5 {
        assert_eq!(class(0.5, 0.2, 3.0, h), 220);
    }
}

#[test]
fn non_dendritic_without_history_splits_on_sphericity() {
    assert_eq!(class(0.0, 0.0, 0.4, 0), 440);
    assert_eq!(class(0.0, 0.1999, 2.0, 0), 440);
    assert_eq!(class(0.0, 0.2, 0.4, 0), 340);
    assert_eq!(class(0.0, 0.7999, 0.4, 0), 340);
    assert_eq!(class(0.0, 0.8, 0.4, 0), 330);
    assert_eq!(class(0.0, 1.0, 5.0, 0), 330);
}

#[test]
fn faceted_history_grows_to_depth_hoar_with_size() {
    // g_s < 0.55 mm: as h = 0 (with K3, "RG + C" read as RG+FC).
    assert_eq!(class(0.0, 0.1, 0.5499, 1), 440);
    assert_eq!(class(0.0, 0.5, 0.5, 1), 340);
    assert_eq!(class(0.0, 0.9, 0.5, 1), 330);
    // [0.55, 1.05[: FC+DH below S 0.55, MF+DH from 0.55.
    assert_eq!(class(0.0, 0.0, 0.55, 1), 450);
    assert_eq!(class(0.0, 0.5499, 1.0499, 1), 450);
    assert_eq!(class(0.0, 0.55, 0.8, 1), 750);
    // >= 1.05: DH below S 0.55, MF+DH from 0.55.
    assert_eq!(class(0.0, 0.0, 1.05, 1), 550);
    assert_eq!(class(0.0, 0.54, 4.0, 1), 550);
    assert_eq!(class(0.0, 0.55, 4.0, 1), 750);
}

#[test]
fn liquid_history_without_facets() {
    // h = 2 (wetted, not yet refrozen): marker 0.
    assert_eq!(class(0.0, 0.5499, 0.3, 2), 740);
    assert_eq!(class(0.0, 0.55, 0.5499, 2), 370);
    assert_eq!(class(0.0, 0.55, 0.55, 2), 770);
    assert_eq!(class(0.0, 0.1, 2.0, 2), 740);
    // h = 4 (refrozen): the same classes, melt forms carry marker 2 (K5).
    assert_eq!(class(0.0, 0.3, 0.3, 4), 742);
    assert_eq!(class(0.0, 1.0, 0.3, 4), 370);
    assert_eq!(class(0.0, 1.0, 1.2, 4), 772);
}

#[test]
fn liquid_history_with_facets() {
    assert_eq!(class(0.0, 0.5499, 0.3, 3), 740);
    assert_eq!(class(0.0, 0.55, 3.0, 3), 750);
    assert_eq!(class(0.0, 0.0, 0.3, 5), 742);
    assert_eq!(class(0.0, 1.0, 0.3, 5), 752);
}

#[test]
fn eq_50_transitions() {
    let cold = 263.15;
    let melt = 273.15;
    let nd = |s| micro(0.0, s, 0.4);
    // Faceted: S = 0 records facets only from no history.
    assert_eq!(update_history(0, &nd(0.0), cold, 0.0), 1);
    assert_eq!(update_history(2, &nd(0.0), cold, 0.0), 4);
    assert_eq!(update_history(4, &nd(0.0), cold, 0.0), 4);
    // Wet rounded grains above 0.5 % liquid by mass.
    assert_eq!(update_history(0, &nd(1.0), melt, 0.0051), 2);
    assert_eq!(update_history(1, &nd(1.0), melt, 0.0051), 3);
    assert_eq!(update_history(4, &nd(1.0), melt, 0.02), 4);
    assert_eq!(update_history(0, &nd(1.0), melt, 0.005), 0);
    assert_eq!(update_history(0, &nd(0.99), melt, 0.05), 0);
    // Refreezing below the melting point.
    assert_eq!(update_history(2, &nd(1.0), melt - 0.01, 0.0), 4);
    assert_eq!(update_history(3, &nd(0.5), cold, 0.0), 5);
    assert_eq!(update_history(2, &nd(0.9), melt, 0.001), 2);
    assert_eq!(update_history(1, &nd(0.5), cold, 0.0), 1);
    // Dendritic snow keeps its history, even at S = 0.
    assert_eq!(update_history(0, &micro(0.2, 0.0, 0.4), cold, 0.0), 0);
    assert_eq!(update_history(2, &micro(0.2, 1.0, 0.4), cold, 0.0), 2);
}

#[test]
fn merged_history_is_the_union_of_table_2_flags() {
    assert_eq!(merge_history(0, 0), 0);
    assert_eq!(merge_history(0, 1), 1);
    assert_eq!(merge_history(1, 2), 3);
    assert_eq!(merge_history(2, 4), 4);
    assert_eq!(merge_history(1, 4), 5);
    assert_eq!(merge_history(3, 4), 5);
    for a in 0..=5 {
        for b in 0..=5 {
            assert_eq!(merge_history(a, b), merge_history(b, a));
        }
        assert_eq!(merge_history(a, a), a);
    }
}

/// Eq. D1 (dendritic) and D5 (non-dendritic, B21): optical diameter, m.
fn optical(delta: f64, s: f64, gs: f64) -> f64 {
    if delta > 0.0 {
        1e-4 * (delta + (1.0 - delta) * (4.0 - s))
    } else {
        gs * s + (1.0 - s) * (4e-4 + gs) / 2.0
    }
}

#[test]
fn optical_diameter_round_trip_is_the_identity() {
    for &s in &[0.0, 0.2, 0.5, 0.55, 0.8, 1.0] {
        for &delta in &[0.01, 0.3, 0.6, 0.8, 1.0] {
            let d = optical(delta, s, 0.0);
            assert!(d < 1e-4 * (4.0 - s), "D1 dendritic test");
            let back = (1e4 * d - 4.0 + s) / (s - 3.0); // D2
            assert!((back - delta).abs() < 1e-12);
        }
        for &gs in &[3e-4, 5.5e-4, 1.05e-3, 4e-3] {
            let d = optical(0.0, s, gs);
            let back = 2.0 * (d - 2e-4 * (1.0 - s)) / (1.0 + s); // D6
            assert!((back - gs).abs() < 1e-15);
        }
    }
}
