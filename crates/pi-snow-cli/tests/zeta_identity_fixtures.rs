//! engine-soil-spinup-v2 plan Z-T1 `zc_key_off_identity`.
//!
//! With `SURFACE_LAYER_NEUTRAL` absent the engine must reproduce main
//! `ebaf93bf` byte for byte. The expected digests below were produced by this
//! file's own `print_digests` test run on a clean checkout of `ebaf93bf`
//! (which has no neutral key), on the fixture `tests/fixtures/snowpack.ini`
//! (`rust/crates/pi-snow-cli/tests/fixtures/snowpack.ini`, SHA-256
//! 892c3bd0b3b0bb10b69f789753f22b9b704769769f469b3c12c6a5560b6da4a6)
//! with the ten-day forcing built in `engine_surface.rs`, whose air
//! temperature crosses the snow surface temperature daily.
//!
//! Digest: FNV-1a 64 over the bytes, plus the length. Cargo.lock may not
//! change, so no SHA-2 crate is available here; the SHA-256 identity gate runs
//! in the Python step that compares retained records.

use chrono::{Duration, TimeZone, Utc};
use pi_snow_cli::engine::{Site, run_with_options};
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

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

const CASES: [(&str, &[&str]); 5] = [
    ("default", &[]),
    ("clm5", &["SURFACE_LAYER=ZETA_LIMIT_CLM5"]),
    (
        "scalar_height",
        &["SURFACE_LAYER=SCALAR_HEIGHT_NEUTRAL_WIND"],
    ),
    (
        "clm5_aoki",
        &[
            "SURFACE_LAYER=ZETA_LIMIT_CLM5",
            "SNOW_ALBEDO=AOKI2003_TEMP_RATE",
        ],
    ),
    (
        "clm5_remove_element",
        &[
            "SURFACE_LAYER=ZETA_LIMIT_CLM5",
            "SUBLIMATION_MELT_OUT=REMOVE_ELEMENT",
        ],
    ),
];

fn digest(settings: &[&str]) -> String {
    let options = PhysicsOptions::from_settings(settings.iter().copied()).unwrap();
    let out = run_with_options(
        &config(),
        &options,
        &site(),
        &forcing(),
        start_column(),
        end(),
    )
    .unwrap();
    format!(
        "smet {} {:016x} pro {} {:016x}",
        out.smet.len(),
        fnv1a(out.smet.as_bytes()),
        out.pro.len(),
        fnv1a(out.pro.as_bytes())
    )
}

#[test]
#[ignore = "prints the digests; used once on main ebaf93bf to record EXPECTED"]
fn print_digests() {
    for (name, settings) in CASES {
        println!("DIGEST {name} {}", digest(settings));
    }
}

const EXPECTED: [(&str, &str); 5] = [
    (
        "default",
        "smet 152870 2c1cb01eb4f5cd21 pro 877928 8d7d9a8f1ef619d1",
    ),
    (
        "clm5",
        "smet 152870 811fc654f25cd594 pro 870720 f59ca93f93028167",
    ),
    (
        "scalar_height",
        "smet 152870 48443d19664232f6 pro 869230 5d405c0a47f2e814",
    ),
    (
        "clm5_aoki",
        "smet 152870 748ac2b72afb5c38 pro 873501 cb499f7eb6a43196",
    ),
    (
        "clm5_remove_element",
        "smet 152870 811fc654f25cd594 pro 870720 f59ca93f93028167",
    ),
];

#[test]
fn zc_key_off_identity() {
    for ((name, settings), (ename, expected)) in CASES.iter().zip(EXPECTED.iter()) {
        assert_eq!(name, ename);
        assert_eq!(
            &digest(settings),
            expected,
            "{name} differs from main ebaf93bf"
        );
    }
}
