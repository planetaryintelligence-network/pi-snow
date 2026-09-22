//! Stage 3 gate: new-snow density, deposition and layer management.

use approx::assert_relative_eq;
use pi_snow_core::constants::*;
use pi_snow_core::{Column, Element};
use pi_snow_physics::new_snow::{self, Conditions, DensityScheme, Deposition};

fn conditions(t_air_c: f64, rh: f64, wind: f64) -> Conditions {
    Conditions {
        t_air: T_MELT + t_air_c,
        t_surface: T_MELT + t_air_c - 1.0,
        relative_humidity: rh,
        wind,
    }
}

#[test]
fn pahaut_density_is_vionnets_form_with_its_floor() {
    // Vionnet et al. (2012), GMD 5: rho = 109 + 6 (Ta - Tfus) + 26 U^(1/2),
    // floor 50 kg/m3.
    let c = conditions(-5.0, 0.8, 4.0);
    assert_relative_eq!(
        DensityScheme::Pahaut.density(&c),
        109.0 - 30.0 + 52.0,
        max_relative = 1e-12
    );
    assert_eq!(
        DensityScheme::Pahaut.density(&conditions(-20.0, 0.8, 0.0)),
        50.0
    );
}

#[test]
fn zwart_density_is_the_thesis_regression_iii() {
    // Zwart (2007) MSc thesis p. 29 and Table 5.4: log10 rho = b0 + b1 Ta +
    // b2 asin(sqrt(RH%/100)) + b3 log10(VW), with b0(2) added above -14 C and
    // wind floored at 2 m/s.
    let (b01, b1, b2, b3, b02) = (3.27820894, 0.03132918, -0.74783337, 0.30943402, -0.35976206);
    let expect = |ta: f64, rh: f64, vw: f64| {
        let base = b01 + b1 * ta + b2 * (rh).sqrt().asin() + b3 * vw.max(2.0).log10();
        10f64.powf(if ta < -14.0 { base } else { base + b02 })
    };
    for (ta, rh, vw) in [(-5.0, 0.8, 4.0), (-18.0, 0.9, 1.0), (-10.0, 0.75, 6.0)] {
        assert_relative_eq!(
            DensityScheme::Zwart.density(&conditions(ta, rh, vw)),
            expect(ta, rh, vw),
            max_relative = 1e-12
        );
    }
}

#[test]
fn eq_5_1_takes_relative_humidity_in_percent() {
    // Zwart (2007) eq. 5.1 and Table 5.5, the SNOWPACK density of 2007: RH is in
    // percent in the regression, a fraction in our forcing.
    let c = conditions(-4.0, 0.85, 3.0);
    let (ta, tss, rh, vw): (f64, f64, f64, f64) = (-4.0, -5.0, 85.0, 3.0);
    let expected = 70.0 + 6.5 * ta + 7.5 * tss + 0.26 * rh + 13.0 * vw
        - 4.5 * ta * tss
        - 0.65 * ta * vw
        - 0.17 * rh * vw
        + 0.06 * ta * tss * rh;
    assert_relative_eq!(
        DensityScheme::Zwart2007Eq51.density(&c),
        expected.max(new_snow::MIN_DENSITY),
        max_relative = 1e-12
    );
}

#[test]
fn precipitation_waits_until_it_fills_a_whole_element() {
    // The reference deposits snow in whole HEIGHT_NEW_ELEM elements from a
    // buffer (black-box finding, lab notes). Below one element's mass nothing
    // is deposited; at or above it, whole elements are, and the remainder waits.
    let mut column = Column::bare_ground(T_MELT - 5.0, 0.0, 0.0);
    let mut d = Deposition::new(0.005, 0.15e-3);
    let c = conditions(-5.0, 0.8, 3.0);
    let rho = DensityScheme::Pahaut.density(&c);
    let element_mass = 0.005 * rho;

    let deposited = d.add(&mut column, 0.4 * element_mass, &c, DensityScheme::Pahaut);
    assert_eq!(deposited, 0.0);
    assert!(column.elements.is_empty());

    let deposited = d.add(&mut column, 2.3 * element_mass, &c, DensityScheme::Pahaut);
    assert_eq!(column.elements.len(), 2);
    assert_relative_eq!(deposited, 2.0 * element_mass, max_relative = 1e-12);
    assert_relative_eq!(d.buffered(), 0.7 * element_mass, max_relative = 1e-9);
    assert!(column.is_well_formed());
}

#[test]
fn deposited_elements_carry_fresh_snow_microstructure() {
    // Brun et al. (1992): fresh snow has dendricity 1 and sphericity 0.5. The
    // new-snow grain size is NEW_SNOW_GRAIN_SIZE (0.3 mm, a diameter, so a
    // 0.15 mm radius).
    let mut column = Column::bare_ground(T_MELT - 3.0, 0.0, 0.0);
    let mut d = Deposition::new(0.005, 0.15e-3);
    let c = conditions(-3.0, 0.8, 2.0);
    d.add(&mut column, 1.0, &c, DensityScheme::Pahaut);
    let e: &Element = column.elements.last().unwrap();
    assert_eq!((e.micro.dendricity, e.micro.sphericity), (1.0, 0.5));
    assert_relative_eq!(e.micro.grain_radius, 0.15e-3);
    assert_relative_eq!(e.thickness, 0.005);
    assert_relative_eq!(
        e.density(),
        DensityScheme::Pahaut.density(&c),
        max_relative = 1e-12
    );
    assert!(*column.node_temperature.last().unwrap() <= T_MELT);
}

#[test]
fn mass_is_conserved_between_buffer_and_snowpack() {
    let mut column = Column::bare_ground(T_MELT - 5.0, 0.0, 0.0);
    let mut d = Deposition::new(0.005, 0.15e-3);
    let mut added = 0.0;
    for i in 0..500 {
        let c = conditions(
            -2.0 - (i % 7) as f64,
            0.7 + 0.004 * (i % 50) as f64,
            (i % 9) as f64,
        );
        let p = 0.013 * ((i % 13) as f64);
        added += p;
        d.add(&mut column, p, &c, DensityScheme::Pahaut);
    }
    assert_relative_eq!(column.swe() + d.buffered(), added, max_relative = 1e-12);
}

#[test]
fn thin_elements_merge_and_conserve_mass() {
    // COMBINE_ELEMENTS with MINIMUM_L_ELEMENT = 2.5 mm: an element thinner than
    // that is merged into its neighbour below (or above, for the bottom one).
    // The merge rule is ours; the thresholds are the INI's.
    let mut column = Column::bare_ground(T_MELT - 5.0, 0.0, 0.0);
    let mut d = Deposition::new(0.005, 0.15e-3);
    d.add(
        &mut column,
        3.0,
        &conditions(-5.0, 0.8, 3.0),
        DensityScheme::Pahaut,
    );
    let n = column.elements.len();
    column.elements[n - 2].thickness *= 0.3; // make one element too thin
    let ice = column.elements[n - 2].composition.ice / 0.3;
    column.elements[n - 2].composition.ice = ice;
    column.elements[n - 2].composition.air = 1.0 - ice;
    let mass_before = column.swe();
    new_snow::merge_thin_elements(&mut column, 0.0025);
    assert_eq!(column.elements.len(), n - 1);
    assert!(column.is_well_formed());
    assert_relative_eq!(column.swe(), mass_before, max_relative = 1e-12);
    assert!(
        column
            .elements
            .iter()
            .all(|e| e.thickness >= 0.0025 && e.composition.is_consistent(1e-9))
    );
}
