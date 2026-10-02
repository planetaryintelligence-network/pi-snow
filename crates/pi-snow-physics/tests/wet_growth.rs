//! Brun (1989) wet grain growth (`WET_GRAIN_GROWTH=BRUN1989`), against the
//! numbers printed on p. 26 of the paper.

use pi_snow_core::Microstructure;
use pi_snow_physics::wet_growth;

fn rounded(radius_m: f64) -> Microstructure {
    Microstructure {
        dendricity: 0.0,
        sphericity: 1.0,
        grain_radius: radius_m,
        bond_radius: 0.05e-3,
        coordination_number: 3.0,
        marker: 0,
    }
}

#[test]
fn the_rate_is_the_printed_law_and_levels_off_at_ten_per_cent() {
    assert!((wet_growth::rate_mm3_per_s(0.0) - 1.28e-8).abs() < 1e-22);
    assert!((wet_growth::rate_mm3_per_s(5.0) - (1.28e-8 + 4.22e-10 * 125.0)).abs() < 1e-20);
    let at_ten = wet_growth::rate_mm3_per_s(10.0);
    assert!((at_ten - 4.348e-7).abs() < 1e-12, "{at_ten}");
    assert_eq!(wet_growth::rate_mm3_per_s(15.0), at_ten);
}

#[test]
fn a_wet_rounded_grain_gains_exactly_the_law_s_volume() {
    let mut m = rounded(0.15e-3);
    let dt = 86_400.0;
    wet_growth::step(&mut m, 5.0, dt);
    let v = |r_m: f64| 4.0 / 3.0 * std::f64::consts::PI * (1e3 * r_m).powi(3);
    let gained = v(m.grain_radius) - v(0.15e-3);
    assert!(
        (gained - wet_growth::rate_mm3_per_s(5.0) * dt).abs() < 1e-12,
        "{gained}"
    );
    assert!(m.grain_radius > 0.15e-3);
}

#[test]
fn dry_dendritic_or_unrounded_grains_do_not_grow() {
    for (lwc, d, s) in [(0.0, 0.0, 1.0), (5.0, 0.2, 1.0), (5.0, 0.0, 0.9)] {
        let mut m = rounded(0.15e-3);
        m.dendricity = d;
        m.sphericity = s;
        wet_growth::step(&mut m, lwc, 600.0);
        assert_eq!(m.grain_radius, 0.15e-3, "lwc {lwc} d {d} s {s}");
    }
}

#[test]
fn ten_days_of_ten_minute_steps_give_the_closed_form_radius() {
    // Review N8: integrating step by step at constant L must land on
    // v0 + v' t with t in seconds; a days-for-seconds slip (Vionnet's Table 2
    // caption) would miss by a factor of 86,400.
    let (dt, steps, lwc) = (600.0, 1440, 5.0);
    let mut m = rounded(0.15e-3);
    for _ in 0..steps {
        wet_growth::step(&mut m, lwc, dt);
    }
    let v0 = 4.0 / 3.0 * std::f64::consts::PI * 0.15_f64.powi(3);
    let v = v0 + wet_growth::rate_mm3_per_s(lwc) * dt * steps as f64;
    let r_mm = (3.0 * v / (4.0 * std::f64::consts::PI)).cbrt();
    assert!(
        (1e3 * m.grain_radius - r_mm).abs() < 1e-9,
        "{} {}",
        1e3 * m.grain_radius,
        r_mm
    );
    // About 0.257 mm after ten wet days at 5 %: d grows from 0.30 to about 0.51 mm.
    assert!((r_mm - 0.2565).abs() < 5e-4, "{r_mm}");
}
