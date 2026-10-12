//! engine-soil-spinup-v2 plan Z-T8 (engine half) and the match-site listing of
//! plan A1.1: every place in `rust/` that matches `SurfaceLayer::ZetaLimitClm5`
//! either selects the continuous scheme when `SURFACE_LAYER_NEUTRAL=CONTINUOUS`
//! is set or refuses. The sites at the S1 commit, from
//! `rg "SurfaceLayer::ZetaLimitClm5" -g '*.rs' rust/`:
//!
//! | site | with the key | tested by |
//! |---|---|---|
//! | `pi-snow-cli/src/engine.rs` scheme mapping | selects `MoHoltslagZetaContinuous` | `engine_*` below |
//! | `pi-snow-cli/src/engine.rs` AOKI and CHAN checks | unchanged, satisfied by `ZetaLimitClm5` | `engine_runs_continuous_with_aoki` |
//! | `pi-snow-cli/src/engine_cold_crash_candidate.rs` mapping | selects it | `candidate_*` below |
//! | `pi-snow-cli/src/engine_cold_crash_candidate.rs` AOKI check | unchanged | (same mapping test) |
//! | `pi-snow-core/src/options.rs` parser and its unit test | parses the key separately | `tests/surface_layer_neutral.rs` |
//!
//! The soil kernel has no match on `ZetaLimitClm5`: it takes the scheme the
//! engine selected (`SurfaceParameters::scheme`).

use chrono::{Duration, TimeZone, Utc};
use pi_snow_cli::engine::{RunError, Site, run_with_options};
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



fn run_with(settings: &[&str]) -> Result<pi_snow_cli::engine::RunOutput, RunError> {
    let options = PhysicsOptions::from_settings(settings.iter().copied()).unwrap();
    run_with_options(&config(), &options, &site(), &forcing(), start_column(), end())
}

#[test]
fn engine_refuses_the_key_without_zeta_limit_clm5() {
    for other in [
        &["SURFACE_LAYER_NEUTRAL=CONTINUOUS"][..],
        &["SURFACE_LAYER=PHASE2", "SURFACE_LAYER_NEUTRAL=CONTINUOUS"],
        &["SURFACE_LAYER_NEUTRAL=CONTINUOUS", "SURFACE_LAYER=SCALAR_HEIGHT_NEUTRAL_WIND"],
    ] {
        match run_with(other) {
            Err(RunError::IncompatiblePhysics(m)) => {
                assert!(m.contains("SURFACE_LAYER_NEUTRAL requires SURFACE_LAYER=ZETA_LIMIT_CLM5"), "{m}")
            }
            Err(e) => panic!("{other:?}: wrong error {e:?}"),
            Ok(_) => panic!("{other:?}: ran"),
        }
    }
}

#[test]
fn engine_refuses_the_key_with_louis_like_clm5() {
    let r = run_with(&[
        "TURBULENCE=LOUIS1979",
        "SURFACE_LAYER=ZETA_LIMIT_CLM5",
        "SURFACE_LAYER_NEUTRAL=CONTINUOUS",
    ]);
    assert!(matches!(r, Err(RunError::IncompatiblePhysics(_))));
}

#[test]
fn engine_runs_continuous_in_either_order_and_matches_itself() {
    let a = run_with(&["SURFACE_LAYER=ZETA_LIMIT_CLM5", "SURFACE_LAYER_NEUTRAL=CONTINUOUS"]).unwrap();
    let b = run_with(&["SURFACE_LAYER_NEUTRAL=CONTINUOUS", "SURFACE_LAYER=ZETA_LIMIT_CLM5"]).unwrap();
    assert_eq!(a.smet, b.smet);
    assert_eq!(a.pro, b.pro);
    let native = run_with(&["SURFACE_LAYER=ZETA_LIMIT_CLM5"]).unwrap();
    println!("continuous smet differs from native CLM5 on this fixture: {}", a.smet != native.smet);
}

#[test]
fn engine_runs_continuous_with_aoki() {
    let r = run_with(&[
        "SURFACE_LAYER=ZETA_LIMIT_CLM5",
        "SURFACE_LAYER_NEUTRAL=CONTINUOUS",
        "SNOW_ALBEDO=AOKI2003_TEMP_RATE",
    ]);
    assert!(r.is_ok(), "{:?}", r.err());
}

#[path = "../src/engine_cold_crash_candidate.rs"]
#[allow(dead_code)]
mod candidate;

#[test]
fn candidate_engine_maps_and_refuses_like_the_main_engine() {
    let native = PhysicsOptions::from_settings(["SURFACE_LAYER=ZETA_LIMIT_CLM5"]).unwrap();
    let cont = PhysicsOptions::from_settings([
        "SURFACE_LAYER=ZETA_LIMIT_CLM5",
        "SURFACE_LAYER_NEUTRAL=CONTINUOUS",
    ])
    .unwrap();
    let alone = PhysicsOptions::from_settings(["SURFACE_LAYER_NEUTRAL=CONTINUOUS"]).unwrap();
    let csite = candidate::Site {
        station_id: "TEST".into(),
        station_name: "test".into(),
        latitude: 40.6,
        longitude: -111.6,
        altitude: 2600.0,
        slope_angle: 0.0,
        slope_azimuth: 0.0,
        soil_albedo: 0.09,
    };
    let go = |o: &PhysicsOptions| {
        candidate::run_with_options(&config(), o, &csite, &forcing(), start_column(), end())
    };
    assert!(go(&native).is_ok());
    assert!(go(&cont).is_ok());
    assert!(matches!(go(&alone), Err(candidate::RunError::IncompatiblePhysics(_))));
    // The candidate engine and the main engine give the same continuous run.
    let main = run_with(&["SURFACE_LAYER=ZETA_LIMIT_CLM5", "SURFACE_LAYER_NEUTRAL=CONTINUOUS"]).unwrap();
    let cand = go(&cont).unwrap();
    assert_eq!(main.smet, cand.smet);
}
