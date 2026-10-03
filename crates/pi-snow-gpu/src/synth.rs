//! Synthetic forcing for developing the kernel without touching registered inputs:
//! cold storms, clear nights, spring warming into rain and melt, on a few slopes.

use chrono::{Duration, TimeZone, Utc};
use pi_snow_core::Column;
use pi_snow_core::constants::T_MELT;
use pi_snow_io::ini::{Ini, RunConfig};

use crate::stepper::{ForcingRow, Site};

/// The physics keys of the v3b station INI; paths are unused here.
pub const INI: &str = "[Input]
METEOPATH = ./input
STATION1 = SYN
PSUM_PH::PRECSPLITTING::snow = 273.35
PSUM_PH::PRECSPLITTING::rain = 275.35
[Output]
METEOPATH = ./output
EXPERIMENT = synthetic
TS_DAYS_BETWEEN = 0.041666666666666664
PROF_DAYS_BETWEEN = 0.041666666666666664
[Snowpack]
HEIGHT_OF_WIND_VALUE = 10.0
HEIGHT_OF_METEO_VALUES = 2.0
ATMOSPHERIC_STABILITY = MO_HOLTSLAG
ROUGHNESS_LENGTH = 0.002
CALCULATION_STEP_LENGTH = 10.0
";

pub fn config() -> RunConfig {
    RunConfig::from_ini(&Ini::parse(INI).unwrap()).unwrap()
}

/// The same with non-default per-site keys, as the referee's INIs carry: a different
/// rain threshold, measurement heights and roughness (plan.md section 2).
pub fn config_variant() -> RunConfig {
    let text = INI
        .replace("HEIGHT_OF_WIND_VALUE = 10.0", "HEIGHT_OF_WIND_VALUE = 6.5")
        .replace("HEIGHT_OF_METEO_VALUES = 2.0", "HEIGHT_OF_METEO_VALUES = 3.0")
        .replace("ROUGHNESS_LENGTH = 0.002", "ROUGHNESS_LENGTH = 0.003")
        + "[SnowpackAdvanced]\nTHRESH_RAIN = 0.4\n";
    let cfg = RunConfig::from_ini(&Ini::parse(&text).unwrap()).unwrap();
    assert_eq!(cfg.thresh_rain_c, 0.4);
    assert_eq!(cfg.height_of_wind_value, 6.5);
    assert_eq!(cfg.height_of_meteo_values, 3.0);
    assert_eq!(cfg.roughness_length, 0.003);
    cfg
}

#[cfg(test)]
mod tests {
    #[test]
    fn variant_ini_uses_all_non_default_referee_values() {
        let cfg = super::config_variant();
        assert_eq!(cfg.thresh_rain_c, 0.4);
        assert_eq!(cfg.height_of_wind_value, 6.5);
        assert_eq!(cfg.height_of_meteo_values, 3.0);
        assert_eq!(cfg.roughness_length, 0.003);
    }
}

pub struct Synthetic {
    pub sites: Vec<Site>,
    pub rows: Vec<Vec<ForcingRow>>,
}

/// `columns` synthetic columns of `days` days at 10-minute steps.
pub fn generate(columns: usize, days: usize) -> Synthetic {
    let start = Utc.with_ymd_and_hms(2025, 11, 1, 0, 0, 0).unwrap();
    let mut sites = Vec::new();
    let mut rows = Vec::new();
    for c in 0..columns {
        let slope = [0.0, 30.0, 38.0, 45.0, 20.0, 35.0, 10.0, 42.0][c % 8];
        let aspect = [0.0, 0.0, 90.0, 180.0, 270.0, 45.0, 315.0, 135.0][c % 8];
        let wet = 0.5 + 0.25 * (c % 7) as f64;
        sites.push(Site {
            station_id: format!("SYN{c}"),
            station_name: format!("synthetic {c}"),
            latitude: 40.55 + 0.01 * c as f64,
            longitude: -111.65 + 0.005 * c as f64,
            altitude: 2400.0 + 60.0 * c as f64,
            slope_angle: slope,
            slope_azimuth: aspect,
            soil_albedo: 0.09,
        });
        let mut col = Vec::new();
        for s in 0..=days * 144 {
            let t = start + Duration::minutes(10 * s as i64);
            let d = s as f64 / 144.0;
            let h = (s % 144) as f64 / 6.0;
            let storm = if (d as usize + c) % 6 < 2 && h < 18.0 { 1.0 } else { 0.0 };
            let ta = 266.0 + 5.0 * (2.0 * std::f64::consts::PI * (h - 9.0) / 24.0).sin()
                + 14.0 * d / days as f64
                - 3.0 * storm
                - 0.004 * (sites[c].altitude - 2400.0);
            let sun = (800.0 * (2.0 * std::f64::consts::PI * (h - 19.0) / 24.0).cos() - 250.0).max(0.0);
            let iswr = sun * (1.0 - 0.6 * storm) * (0.7 + 0.5 * d / days as f64);
            let psum = if storm > 0.0 { 0.18 * wet } else { 0.0 };
            col.push(ForcingRow {
                t,
                ta: Some(ta),
                rh: Some(0.7 + 0.25 * storm),
                vw: Some(1.5 + 4.0 * storm + 1.0 * (h / 3.0).sin().abs()),
                dw: Some(270.0),
                iswr: Some(iswr),
                ilwr: Some(245.0 + 55.0 * storm + 3.0 * (ta - 266.0)),
                psum: Some(if s == 0 { 0.0 } else { psum }),
                tsg: Some(T_MELT),
            });
        }
        rows.push(col);
    }
    Synthetic { sites, rows }
}

pub fn bare(site: &Site) -> Column {
    Column::bare_ground(T_MELT, site.slope_angle, site.slope_azimuth)
}
