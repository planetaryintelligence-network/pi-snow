//! The run configuration: the engine reads the same INI the reference is
//! invoked with, and refuses anything it does not implement.
//!
//! Refusing matters more than parsing. An engine that silently ignored
//! `WATERTRANSPORTMODEL_SNOW = RICHARDSEQUATION` would run the bucket and report
//! numbers as if it had solved Richards' equation, and the comparison would be
//! against physics nobody asked for.

use pi_snow_io::ini::{Ini, NewSnowDensity, RunConfig, Stability};

fn fixture() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/run_ULAMB1.ini"
    ))
    .unwrap()
}

/// The "uac" physics arm, built exactly as `terrain_snowpack.apply_uac_physics`
/// builds it from the same template.
fn uac_arm(text: &str) -> String {
    text.replace(
        "CALCULATION_STEP_LENGTH = 10.0",
        "CALCULATION_STEP_LENGTH = 15.0",
    )
    .replace(
        "PSUM::accumulate::period = 600",
        "PSUM::accumulate::period = 900",
    )
    .replace(
        "ATMOSPHERIC_STABILITY = MO_HOLTSLAG",
        "ATMOSPHERIC_STABILITY = RICHARDSON",
    )
    .replace("SOIL_FLUX = true", "SOIL_FLUX = false")
    .replace(
        "[SnowpackAdvanced]",
        "[SnowpackAdvanced]\nHN_DENSITY_MODEL = ZWART\nALBEDO_PARAMETERIZATION = LEHNING_2",
    )
}

#[test]
fn sections_and_keys_are_case_insensitive() {
    // The input file writes [Snowpack] and TA::resample; the expanded output the
    // reference writes back says [SNOWPACK] and TA::RESAMPLE. Same keys.
    let ini = Ini::parse(&fixture()).unwrap();
    assert_eq!(ini.get("snowpack", "calculation_step_length"), Some("10.0"));
    assert_eq!(ini.get("SNOWPACK", "CALCULATION_STEP_LENGTH"), Some("10.0"));
    assert_eq!(
        ini.get("Interpolations1D", "PSUM::ACCUMULATE::PERIOD"),
        Some("600")
    );
}

#[test]
fn the_ours_arm_input_ini_maps_to_the_configuration_the_plan_registers() {
    let c = RunConfig::from_ini(&Ini::parse(&fixture()).unwrap()).unwrap();
    assert_eq!(c.station_id, "ULAMB1");
    assert_eq!(c.experiment, "hrrr-terrain-v1");
    assert_eq!(c.step_seconds, 600.0);
    assert_eq!(c.stability, Stability::MoHoltslag);
    assert_eq!(c.new_snow_density, NewSnowDensity::LehningNew);
    assert_eq!(c.roughness_length, 0.002);
    assert_eq!(
        (c.height_of_meteo_values, c.height_of_wind_value),
        (2.0, 10.0)
    );
    assert_eq!(c.geo_heat, 0.06);
    assert_eq!(c.output_interval_seconds, 3600.0);
    assert_eq!(
        (c.precip_snow_below_k, c.precip_rain_above_k),
        (273.35, 275.35)
    );
    // Defaults the input file leaves to the engine, taken from the expanded INI
    // the reference writes back beside its output.
    assert_eq!(c.height_new_element, 0.005);
    assert_eq!(c.minimum_element_length, 0.0025);
    assert_eq!(c.new_snow_grain_size_mm, 0.3);
    assert_eq!(
        (c.hoar_thresh_rh, c.hoar_thresh_ta_c, c.hoar_thresh_vw),
        (0.97, 1.2, 3.5)
    );
}

#[test]
fn the_uac_arm_maps_to_richardson_zwart_and_a_fifteen_minute_step() {
    let c = RunConfig::from_ini(&Ini::parse(&uac_arm(&fixture())).unwrap()).unwrap();
    assert_eq!(c.step_seconds, 900.0);
    assert_eq!(c.stability, Stability::Richardson);
    assert_eq!(c.new_snow_density, NewSnowDensity::Zwart);
}

#[test]
fn unimplemented_physics_is_refused_with_its_name() {
    for (from, to, key) in [
        (
            "SNOW_EROSION = false",
            "SNOW_EROSION = true",
            "SNOW_EROSION",
        ),
        ("CANOPY = false", "CANOPY = true", "CANOPY"),
        ("SNP_SOIL = false", "SNP_SOIL = true", "SNP_SOIL"),
        ("CHANGE_BC = false", "CHANGE_BC = true", "CHANGE_BC"),
        ("NUMBER_SLOPES = 1", "NUMBER_SLOPES = 5", "NUMBER_SLOPES"),
        (
            "ATMOSPHERIC_STABILITY = MO_HOLTSLAG",
            "ATMOSPHERIC_STABILITY = MO_MICHLMAYR",
            "ATMOSPHERIC_STABILITY",
        ),
        (
            "[SnowpackAdvanced]",
            "[SnowpackAdvanced]\nWATERTRANSPORTMODEL_SNOW = RICHARDSEQUATION",
            "WATERTRANSPORTMODEL_SNOW",
        ),
        (
            "[SnowpackAdvanced]",
            "[SnowpackAdvanced]\nENABLE_VAPOUR_TRANSPORT = true",
            "ENABLE_VAPOUR_TRANSPORT",
        ),
    ] {
        let text = fixture().replace(from, to);
        assert_ne!(text, fixture(), "fixture must contain {from:?}");
        let err = RunConfig::from_ini(&Ini::parse(&text).unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains(key), "error for {key} was: {err}");
    }
}
