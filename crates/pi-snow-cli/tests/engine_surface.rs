//! engine-surface-v1's keys at the engine seam: the default is the Phase 2
//! engine, wet snow grows under `WET_GRAIN_GROWTH=BRUN1989`, and a
//! `SURFACE_LAYER` value that cannot apply is refused rather than ignored.

use chrono::{Duration, TimeZone, Utc};
use pi_snow_cli::engine::{RunError, Site, run, run_with_options};
use pi_snow_core::{Column, Composition, Element, Microstructure, PhysicsOptions};
use pi_snow_io::ini::{Ini, RunConfig};
use pi_snow_io::smet::Smet;

fn config() -> RunConfig {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/snowpack.ini"
    ))
    .unwrap();
    RunConfig::from_ini(&Ini::parse(&text).unwrap()).unwrap()
}

fn site() -> Site {
    Site {
        station_id: "TEST".into(),
        station_name: "test".into(),
        latitude: 40.6,
        longitude: -111.6,
        altitude: 2600.0,
        slope_angle: 0.0,
        slope_azimuth: 0.0,
        soil_albedo: 0.09,
    }
}

/// Ten spring days: warm afternoons, freezing nights, no precipitation.
fn forcing() -> Smet {
    let mut text = String::from(
        "SMET 1.1 ASCII\n[HEADER]\nstation_id = TEST\nlatitude = 40.6\nlongitude = -111.6\naltitude = 2600\n\
         nodata = -999\ntz = 0\nfields = timestamp TA RH TSG VW DW ISWR ILWR PSUM\n[DATA]\n",
    );
    let start = Utc.with_ymd_and_hms(2026, 4, 20, 0, 0, 0).unwrap();
    for i in 0..(10 * 144) {
        let t = start + Duration::minutes(10 * i);
        let hour = (i % 144) as f64 / 6.0;
        let day = (std::f64::consts::PI * (hour - 13.0) / 12.0).cos();
        let ta = 273.15 + 2.0 + 6.0 * day;
        let iswr = (800.0 * (std::f64::consts::PI * (hour - 19.0) / 12.0).cos()).max(0.0);
        text.push_str(&format!(
            "{} {ta} 0.6 273.15 3.0 180 {iswr} 260 0\n",
            t.format("%Y-%m-%dT%H:%M")
        ));
    }
    Smet::parse(&text).unwrap()
}

fn start_column() -> Column {
    let mut c = Column::bare_ground(273.15, 0.0, 0.0);
    let n = 20;
    c.elements = (0..n)
        .map(|_| Element {
            thickness: 0.05,
            composition: Composition {
                ice: 0.4,
                water: 0.0,
                air: 0.6,
                soil: 0.0,
            },
            micro: Microstructure {
                dendricity: 0.0,
                sphericity: 1.0,
                grain_radius: 0.15e-3,
                bond_radius: 0.05e-3,
                coordination_number: 3.0,
                marker: 0,
            },
            age_days: 10.0,
            stress: 0.0,
            stress_rate: 0.0,
            metamorphism: 0.0,
            soil_density: 0.0,
            soil_conductivity: 0.0,
            soil_heat_capacity: 0.0,
            grain_history: 0,
        })
        .collect();
    c.node_temperature = vec![272.9; n + 1];
    c
}

fn end() -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 4, 30, 0, 0, 0).unwrap()
}

/// Largest grain diameter (mm, PRO 0512) in the last profile.
fn last_max_grain_mm(pro: &str) -> f64 {
    let line = pro.lines().rfind(|l| l.starts_with("0512,")).unwrap();
    line.split(',')
        .skip(2)
        .map(|v| v.parse::<f64>().unwrap())
        .fold(0.0, f64::max)
}

#[test]
fn the_default_options_are_the_phase_2_engine() {
    let a = run(&config(), &site(), &forcing(), start_column(), end()).unwrap();
    let explicit =
        PhysicsOptions::from_settings(["SURFACE_LAYER=PHASE2", "WET_GRAIN_GROWTH=NONE"]).unwrap();
    let b = run_with_options(
        &config(),
        &explicit,
        &site(),
        &forcing(),
        start_column(),
        end(),
    )
    .unwrap();
    assert_eq!(a.smet, b.smet);
    assert_eq!(a.pro, b.pro);
}

#[test]
fn wet_rounded_grains_grow_only_under_brun1989() {
    let none = run(&config(), &site(), &forcing(), start_column(), end()).unwrap();
    assert!(
        (last_max_grain_mm(&none.pro) - 0.30).abs() < 0.006,
        "{}",
        last_max_grain_mm(&none.pro)
    );
    let options = PhysicsOptions::from_settings(["WET_GRAIN_GROWTH=BRUN1989"]).unwrap();
    let grown = run_with_options(
        &config(),
        &options,
        &site(),
        &forcing(),
        start_column(),
        end(),
    )
    .unwrap();
    assert!(
        last_max_grain_mm(&grown.pro) > 0.32,
        "{}",
        last_max_grain_mm(&grown.pro)
    );
}

#[test]
fn both_surface_layer_values_run_and_change_the_surface() {
    let phase2 = run(&config(), &site(), &forcing(), start_column(), end()).unwrap();
    for value in ["SCALAR_HEIGHT_NEUTRAL_WIND", "ZETA_LIMIT_CLM5"] {
        let options =
            PhysicsOptions::from_settings([format!("SURFACE_LAYER={value}").as_str()]).unwrap();
        let out = run_with_options(
            &config(),
            &options,
            &site(),
            &forcing(),
            start_column(),
            end(),
        )
        .unwrap();
        assert_ne!(out.smet, phase2.smet, "{value}");
    }
}

#[test]
fn a_surface_layer_value_with_louis_is_refused() {
    let options =
        PhysicsOptions::from_settings(["TURBULENCE=LOUIS1979", "SURFACE_LAYER=ZETA_LIMIT_CLM5"])
            .unwrap();
    let result = run_with_options(
        &config(),
        &options,
        &site(),
        &forcing(),
        start_column(),
        end(),
    );
    assert!(matches!(result, Err(RunError::IncompatiblePhysics(_))));
}
