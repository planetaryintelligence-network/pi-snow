//! Atmospheric thermodynamics, gated on the check tables the sources print.

use approx::assert_relative_eq;
use pi_snow_physics::atmosphere as atm;

#[test]
fn vapour_pressures_reproduce_murphy_koop_table_c1() {
    // Murphy and Koop (2005) Table C1, "values recommended for checking computer
    // codes", p. 1561. Their footnote: double precision reproduces them; the
    // last printed digit bounds the tolerance.
    for (t, ice, liquid) in [
        (240.0, 27.272, 37.667),
        (210.0, 0.70202, 1.2335),
        (273.15, 611.154, 611.213),
        (273.16, 611.657, 611.657),
    ] {
        assert_relative_eq!(atm::vapour_pressure_ice(t), ice, max_relative = 2e-4);
        assert_relative_eq!(atm::vapour_pressure_water(t), liquid, max_relative = 2e-4);
    }
    assert_relative_eq!(
        atm::vapour_pressure_water(300.0),
        3536.8,
        max_relative = 2e-4
    );
}

#[test]
fn supercooled_water_holds_more_vapour_than_ice() {
    // Why FORCE_RH_WATER matters: below freezing the same relative humidity
    // means more vapour over water than over ice.
    for t in [253.15, 263.15, 272.15] {
        assert!(atm::vapour_pressure_water(t) > atm::vapour_pressure_ice(t));
    }
}

#[test]
fn pressure_reproduces_the_us_standard_atmosphere_table() {
    // U.S. Standard Atmosphere 1976, Table I, geopotential altitude, pp. 52 and
    // 54. Our altitudes are geometric; at 3 km the geometric-geopotential
    // difference is about 1.4 m, 0.02 % in pressure, inside this tolerance.
    assert_relative_eq!(
        atm::pressure_at_altitude(2500.0),
        74_682.0,
        max_relative = 5e-4
    );
    assert_relative_eq!(
        atm::pressure_at_altitude(3000.0),
        70_108.0,
        max_relative = 5e-4
    );
    assert_relative_eq!(
        atm::pressure_at_altitude(0.0),
        101_325.0,
        max_relative = 1e-12
    );
}

#[test]
fn air_density_and_heat_capacity_follow_from_the_printed_constants() {
    // rho = P M0 / (R* T); cp = gamma / (gamma - 1) R*/M0, gamma = 1.400.
    let r = 8.314_32e3 / 28.9644;
    assert_relative_eq!(
        atm::air_density(101_325.0, 288.15),
        101_325.0 / (r * 288.15),
        max_relative = 1e-12
    );
    assert_relative_eq!(atm::CP_AIR, 3.5 * r, max_relative = 1e-12);
    // US Standard Atmosphere sea-level density is 1.2250 kg/m3.
    assert_relative_eq!(
        atm::air_density(101_325.0, 288.15),
        1.2250,
        max_relative = 2e-4
    );
}

#[test]
fn specific_humidity_is_consistent_with_vapour_pressure() {
    // q = eps e / (P - (1 - eps) e); at e = 0 it is zero, and it rises with e.
    assert_eq!(atm::specific_humidity(0.0, 70_000.0), 0.0);
    let q = atm::specific_humidity(600.0, 70_000.0);
    assert!(q > 0.005 && q < 0.006, "q {q}");
}
