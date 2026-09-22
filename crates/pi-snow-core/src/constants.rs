//! Physical constants.
//!
//! Every value here is a measured property of a material or a defined SI
//! constant, not a model parameter. Model parameters live beside the
//! parameterisation that uses them, so that each one sits next to its citation.

/// Density of pure ice at 0 degrees C, kg m^-3.
pub const RHO_ICE: f64 = 917.0;
/// Density of liquid water at 0 degrees C, kg m^-3.
pub const RHO_WATER: f64 = 1000.0;
/// Density of dry air at 0 degrees C and 1013.25 hPa, kg m^-3.
pub const RHO_AIR: f64 = 1.1;

/// Specific heat capacity of ice at -10 degrees C, J kg^-1 K^-1.
///
/// Bouvet et al. (2024), *Multiscale modeling of heat and mass transfer in dry
/// snow*, The Cryosphere 18, 4285-4313, Table 1. Ice's heat capacity varies
/// with temperature by roughly 0.5 % per kelvin, so the -10 C value is a choice,
/// not a constant of nature; it is the one the source used and the one the
/// conductivity parameterisation was fitted alongside.
pub const C_ICE: f64 = 2000.0;
/// Specific heat capacity of liquid water, J kg^-1 K^-1.
pub const C_WATER: f64 = 4190.0;
/// Specific heat capacity of air at constant pressure, J kg^-1 K^-1.
///
/// Bouvet et al. (2024) Table 1, at -10 degrees C.
pub const C_AIR: f64 = 1005.0;

/// Thermal conductivity of pure ice at -10 degrees C, W m^-1 K^-1.
///
/// Bouvet et al. (2024) Table 1. Paired with `C_ICE` and `C_AIR` above: the
/// three are used together in the same mixture rule, so taking them from one
/// source at one temperature keeps them mutually consistent.
pub const K_ICE: f64 = 2.3;
/// Thermal conductivity of liquid water, W m^-1 K^-1.
pub const K_WATER: f64 = 0.5984;
/// Thermal conductivity of still air, W m^-1 K^-1.
pub const K_AIR: f64 = 0.026;

/// Latent heat of fusion at 0 degrees C and 1 atm, J kg^-1.
///
/// Yen (1981), CRREL Report 81-10, p. 14, citing Rossini et al. (1952). The
/// scan's text layer reads "33.5 kJ/kg", a dropped digit; 333.5 is confirmed
/// independently by Murphy and Koop's (2005) triple-point latent heats.
pub const LH_FUSION: f64 = 333_500.0;
/// Latent heat of sublimation, J kg^-1.
///
/// Yen (1981): practically constant at 2838 kJ/kg between 213 and 272 K. Agrees
/// within 0.15 % with Murphy and Koop (2005), 51059 J/mol at the triple point.
pub const LH_SUBLIMATION: f64 = 2_838_000.0;
/// Latent heat of vaporisation, J kg^-1.
pub const LH_VAPORISATION: f64 = LH_SUBLIMATION - LH_FUSION;

/// Melting point of water at standard pressure, K.
pub const T_MELT: f64 = 273.15;
/// Stefan-Boltzmann constant, W m^-2 K^-4.
pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;
/// Von Karman constant (dimensionless).
pub const VON_KARMAN: f64 = 0.4;
/// Standard gravity, m s^-2.
pub const GRAVITY: f64 = 9.80665;
/// Specific gas constant for dry air, J kg^-1 K^-1.
pub const R_DRY_AIR: f64 = 287.05;
/// Specific gas constant for water vapour, J kg^-1 K^-1.
pub const R_VAPOUR: f64 = 461.5;

/// The nodata sentinel SMET and PRO files use.
pub const NODATA: f64 = -999.0;
