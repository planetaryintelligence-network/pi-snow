//! Stage 6 gate: shear strength, hand hardness and the stability indices.
//!
//! Every function here takes grain type as an input rather than computing it,
//! so this stage is testable before metamorphism (stage 5) exists. The grain
//! type the engine eventually feeds in is where the divergence will come from;
//! these tests fix what happens once it arrives.

use approx::assert_relative_eq;
use pi_snow_physics::stability::{self, GrainClass, Layer};

const RHO_ICE: f64 = 917.0;

fn layer(depth_top: f64, thickness: f64, density: f64, class: GrainClass) -> Layer {
    Layer {
        depth_top,
        thickness,
        density,
        grain_type: class.code(),
        grain_size_mm: 1.0,
        hand_hardness: 2.0,
        temperature_c: -5.0,
        age_days: 5.0,
    }
}

#[test]
fn swiss_primary_digit_mapping_is_the_one_the_interface_implies() {
    // Established from interface evidence only: the reference writes 660 for
    // surface hoar, 772 for a melt-freeze crust, and the frozen extractor's
    // hardness rows match Monti et al. (2014) Table 2 in exactly this order.
    use GrainClass::*;
    let expected = [
        (1, PP),
        (2, DF),
        (3, RG),
        (4, FC),
        (5, DH),
        (6, SH),
        (7, MF),
        (8, IF),
        (9, FCxr),
    ];
    for (digit, class) in expected {
        assert_eq!(GrainClass::from_primary_digit(digit), Some(class));
    }
    assert_eq!(GrainClass::from_code(772), Some(MF));
    assert_eq!(GrainClass::from_code(660), Some(SH));
    assert_eq!(GrainClass::from_primary_digit(0), None);
}

#[test]
fn group_one_strength_is_the_jamieson_johnston_2001_power_law() {
    // Table 8, Group I (PP, DF, RG), Model A: 14.5 (rho / rho_ice)^1.73 kPa.
    for class in [GrainClass::PP, GrainClass::DF, GrainClass::RG] {
        let s = stability::shear_strength_kpa(&layer(0.5, 0.05, 200.0, class), 0.0, 1.0);
        assert_relative_eq!(s, 14.5 * (200.0 / RHO_ICE).powf(1.73), max_relative = 1e-12);
    }
}

#[test]
fn group_two_strength_is_the_jamieson_johnston_2001_log_form() {
    // Table 8, Group II (FC, DH), Model B, the paper's better fit for that
    // group (R2 0.54 against 0.37): 18.5 (rho / rho_ice)^2.11 kPa.
    for class in [GrainClass::FC, GrainClass::DH] {
        let s = stability::shear_strength_kpa(&layer(0.5, 0.05, 250.0, class), 0.0, 1.0);
        assert_relative_eq!(s, 18.5 * (250.0 / RHO_ICE).powf(2.11), max_relative = 1e-12);
    }
}

#[test]
fn buried_surface_hoar_strength_is_the_lehning_2004_regression() {
    // Lehning et al. (2004) Table 1, their re-analysis without E_min:
    // 0.429 + 0.0138 t + 1.12 sigma_zz - 0.00596 H + 0.000785 Hs
    //   - 0.271 Thick + 0.0202 Twl   (kPa; t days, sigma_zz kPa, H/Hs/Thick cm,
    //   Twl degrees C)
    let mut l = layer(0.40, 0.01, 150.0, GrainClass::SH);
    l.age_days = 12.0;
    l.temperature_c = -6.0;
    let (sigma_zz, hs_m) = (0.9, 1.3);
    let expected = 0.429 + 0.0138 * 12.0 + 1.12 * sigma_zz - 0.00596 * 40.0 + 0.000785 * 130.0
        - 0.271 * 1.0
        + 0.0202 * -6.0;
    let s = stability::shear_strength_kpa(&l, sigma_zz, hs_m);
    assert_relative_eq!(s, expected, max_relative = 1e-12);
}

#[test]
fn skier_stress_at_38_degrees_by_each_published_form() {
    // Schweizer et al. (2006) eq. 1: 155 / (h - P) Pa. Jamieson and Johnston
    // (1998): 0.14 / h kPa for a 500 N/m line load. Either removes the need
    // for Fohn's alpha_max table, which was not obtained.
    use stability::SkierForm::*;
    assert_relative_eq!(
        stability::skier_shear_stress_38_kpa(Schweizer2006, 1.0, 0.0),
        0.155
    );
    assert_relative_eq!(
        stability::skier_shear_stress_38_kpa(JamiesonJohnston1998, 1.0, 0.0),
        0.14
    );
    assert_relative_eq!(
        stability::skier_shear_stress_38_kpa(Schweizer2006, 0.8, 0.3),
        0.155 / 0.5
    );
}

#[test]
fn penetration_by_each_published_form() {
    // Schweizer (2006) p. 54: P = 34.6 / rho_30, rho_30 the mean density of the
    // upper 30 cm. 34.6 = 0.8 * 43.3 to three figures: the frozen extractor's
    // 0.8 is this. J&J (1998): P = 42.4 / rho_30 with rho_30 estimated from
    // the mean slab density.
    use stability::SkierForm::*;
    assert_eq!(stability::SkierForm::default(), Schweizer2006);
    assert_relative_eq!(
        stability::ski_penetration(Schweizer2006, 160.0, 999.0, 9.9),
        34.6 / 160.0
    );
    let rho_30 = 200.0 + 150.0 * (0.3 - 0.5);
    assert_relative_eq!(
        stability::ski_penetration(JamiesonJohnston1998, 999.0, 200.0, 1.0),
        42.4 / rho_30,
        max_relative = 1e-12
    );
}

#[test]
fn upper_30_cm_density_weights_by_thickness_and_stops_at_30_cm() {
    let profile = vec![
        layer(0.35, 0.50, 400.0, GrainClass::RG), // below 30 cm: ignored
        layer(0.10, 0.25, 200.0, GrainClass::RG), // contributes 0.20 m
        layer(0.0, 0.10, 100.0, GrainClass::PP),  // contributes 0.10 m
    ];
    let expected = (100.0 * 0.10 + 200.0 * 0.20) / 0.30;
    assert_relative_eq!(
        stability::upper_mean_density(&profile, 0.30),
        expected,
        max_relative = 1e-12
    );
}

#[test]
fn sk38_above_the_ski_penetration_depth_is_the_ceiling_as_the_reference_writes() {
    // Jamieson and Johnston (1998) define S_K as 0 where the ski penetrates past
    // the layer. The reference writes 6.00 there instead (stage 6 comparison,
    // lab notes), and the RF1.1 `Ss` feature is a profile minimum, so writing 0
    // would pull that feature to zero for a bookkeeping reason. Copied.
    let profile = vec![
        layer(0.10, 0.02, 250.0, GrainClass::FC),
        layer(0.0, 0.10, 90.0, GrainClass::PP),
    ];
    let idx = stability::indices(&profile, 1.0);
    // A 10 cm slab of 90 kg/m3 snow: penetration 34.6 / 90 = 0.38 m.
    assert_eq!(idx[0].sk38, Some(stability::INDEX_CAP));
}

#[test]
fn natural_index_follows_the_38_degree_stress_geometry() {
    // Sn38 = (Sigma + sigma_zz phi) / sigma_xz with sigma_zz = W cos^2 38 and
    // sigma_xz = W sin 38 cos 38, W the slab weight per horizontal area.
    // Faceted weak layer, so phi' = 0 for a persistent class (J&J 1998).
    let slab = layer(0.0, 0.8, 200.0, GrainClass::RG);
    let weak = layer(0.8, 0.02, 250.0, GrainClass::FC);
    let profile = vec![weak.clone(), slab];
    let idx = stability::indices(&profile, 0.82);
    let w = 200.0 * 0.8 * 9.80665 / 1000.0; // kPa
    let psi = 38.0_f64.to_radians();
    let strength = 18.5 * (250.0 / RHO_ICE).powf(2.11);
    let expected = strength / (w * psi.sin() * psi.cos());
    assert_relative_eq!(idx[0].sn38.unwrap(), expected.min(6.0), max_relative = 1e-9);
}

#[test]
fn indices_are_capped_at_six_and_shallow_layers_report_the_cap() {
    // The reference writes 6.00 as its ceiling and reports 6.00 for layers too
    // near the surface to evaluate (MIN_DEPTH_SUBSURF = 0.07 m in the INI).
    // Both are interface conventions read from its output, not physics.
    let profile = vec![
        layer(0.05, 0.03, 300.0, GrainClass::RG),
        layer(0.0, 0.05, 100.0, GrainClass::PP),
    ];
    let idx = stability::indices(&profile, 0.08);
    for i in &idx {
        assert!(i.sn38.unwrap() <= 6.0 && i.sk38.unwrap() <= 6.0);
    }
    assert_eq!(idx[1].sn38, Some(6.0), "top layer above MIN_DEPTH_SUBSURF");
}

#[test]
fn ssi_adds_one_per_missing_structural_contrast() {
    // SSI = SK38 + dR* + dE*, dR* = 1 if hardness difference < 1.5, dE* = 1 if
    // grain size difference < 0.5 mm (Bellaire et al. 2006, restating
    // Schweizer et al. 2006). Contrast is taken across the interface with the
    // layer above, which is our reading of "across a layer interface".
    assert_eq!(stability::structural_penalty(1.0, 2.0, 0.6, 0.7), 2.0);
    assert_eq!(stability::structural_penalty(1.0, 3.0, 0.6, 0.7), 1.0);
    assert_eq!(stability::structural_penalty(1.0, 3.0, 0.6, 1.5), 0.0);
}

#[test]
fn critical_cut_length_is_the_richter_2019_form() {
    // r_c = sqrt(E' D_sl F_wl 2 tau_p / sigma_n), E' = E / (1 - nu^2),
    // E = 5.07e9 (rho_sl / rho_ice)^5.13 Pa, nu = 0.2,
    // F_wl = 4.6e-9 ((rho_wl / rho_ice)(gs_wl / 0.00125))^-2 m/Pa,
    // sigma_n = rho_sl g D_sl.  Richter et al. (2019) eqs. 1-3.
    let (rho_sl, d_sl, rho_wl, gs_wl, tau_p_kpa) = (220.0, 0.6, 200.0, 0.0015, 0.8);
    let e = 5.07e9 * (rho_sl / RHO_ICE).powf(5.13);
    let e_prime = e / (1.0 - 0.2_f64.powi(2));
    let f_wl = 4.6e-9 * ((rho_wl / RHO_ICE) * (gs_wl / 0.00125)).powf(-2.0);
    let sigma_n = rho_sl * 9.80665 * d_sl;
    let expected = (e_prime * d_sl * f_wl * 2.0 * tau_p_kpa * 1e3 / sigma_n).sqrt();
    let got = stability::critical_cut_length_m(rho_sl, d_sl, rho_wl, gs_wl * 1e3, tau_p_kpa);
    assert_relative_eq!(got, expected, max_relative = 1e-12);
    // r_c grows as rho_slab^2.06 here (E ~ rho^5.13 over sigma_n ~ rho): a
    // stiffer slab needs a longer cut before a crack propagates.
    assert!(stability::critical_cut_length_m(300.0, d_sl, rho_wl, 1.5, tau_p_kpa) > got);
}

#[test]
fn monti_hardness_matches_the_published_table_rows() {
    // Monti, Schweizer and Fierz (2014) Table 2. Upper bounds of R1..R4 for RG:
    // 189, 277, 368, 443; R5 from 444.
    use GrainClass::RG;
    for (rho, r) in [
        (150.0, 1.0),
        (189.0, 1.0),
        (190.0, 2.0),
        (300.0, 3.0),
        (400.0, 4.0),
        (500.0, 5.0),
    ] {
        assert_eq!(stability::monti_hardness(RG, rho), Some(r), "RG at {rho}");
    }
    // Surface hoar and ice are not in the published table.
    assert_eq!(stability::monti_hardness(GrainClass::SH, 150.0), None);
    assert_eq!(stability::monti_hardness(GrainClass::IF, 800.0), None);
}
