//! Atmospheric thermodynamics for the surface energy balance.
//!
//! * Vapour pressure over ice and over liquid water: Murphy and Koop (2005),
//!   *Review of the vapour pressures of ice and supercooled water for
//!   atmospheric applications*, QJRMS 131, 1539-1565, eqs. 7 and 10, read in
//!   full. Eq. 10 is valid down to 123 K, which is what the configuration's
//!   `FORCE_RH_WATER = true` needs: relative humidity is taken over
//!   supercooled water at every temperature.
//! * Pressure with altitude, air density and heat capacity: U.S. Standard
//!   Atmosphere 1976 (NOAA-S/T 76-1562), eqs. 1 and 33a with the constants of
//!   its Tables 2A-2C and 4, read in full. The document does not print c_p;
//!   it is derived here from its printed gamma = 1.400 and R*/M0.

/// Universal gas constant, J kmol^-1 K^-1 (US Standard Atmosphere, Table 2A).
const R_STAR: f64 = 8.314_32e3;
/// Mean molecular weight of dry air, kg kmol^-1 (US Standard Atmosphere, 1.2.4).
const M0: f64 = 28.9644;
/// Molar mass of water, kg kmol^-1. A physical constant, used only for the
/// vapour/air molecular weight ratio.
const M_WATER: f64 = 18.015;
/// Specific gas constant of dry air, J kg^-1 K^-1.
pub const R_DRY: f64 = R_STAR / M0;
/// Specific heat of dry air at constant pressure, J kg^-1 K^-1, from
/// gamma / (gamma - 1) * R_dry with the Standard Atmosphere's gamma = 1.400.
pub const CP_AIR: f64 = 1.4 / 0.4 * R_DRY;
/// Ratio of the molecular weights of water vapour and dry air.
pub const EPSILON: f64 = M_WATER / M0;

/// Saturation vapour pressure over ice, Pa, T in K. Murphy and Koop eq. 7,
/// valid above 110 K.
pub fn vapour_pressure_ice(t: f64) -> f64 {
    (9.550426 - 5723.265 / t + 3.53068 * t.ln() - 0.00728332 * t).exp()
}

/// Saturation vapour pressure over liquid (including supercooled) water, Pa,
/// T in K. Murphy and Koop eq. 10, valid 123 K to 332 K.
pub fn vapour_pressure_water(t: f64) -> f64 {
    (54.842763 - 6763.22 / t - 4.210 * t.ln()
        + 0.000367 * t
        + (0.0415 * (t - 218.8)).tanh() * (53.878 - 1331.22 / t - 9.44523 * t.ln() + 0.014025 * t))
        .exp()
}

/// Pressure, Pa, at geometric altitude `z` metres, from the Standard
/// Atmosphere's tropospheric layer (eq. 33a, b = 0): sea level 101325 Pa and
/// 288.15 K, gradient -6.5 K km^-1. Geometric altitude is used in place of
/// geopotential; the difference is under 2 m below 3.5 km.
pub fn pressure_at_altitude(z: f64) -> f64 {
    const P0: f64 = 101_325.0;
    const T0: f64 = 288.15;
    const LAPSE: f64 = -0.0065;
    const G0: f64 = 9.80665;
    P0 * (T0 / (T0 + LAPSE * z)).powf(G0 * M0 / (R_STAR * LAPSE))
}

/// Dry-air density, kg m^-3, from the ideal gas law (eq. 1).
pub fn air_density(pressure: f64, temperature: f64) -> f64 {
    pressure / (R_DRY * temperature)
}

/// Specific humidity, kg kg^-1, from vapour pressure `e` and pressure `p`.
pub fn specific_humidity(e: f64, p: f64) -> f64 {
    EPSILON * e / (p - (1.0 - EPSILON) * e)
}
