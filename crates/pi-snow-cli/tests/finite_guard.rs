//! engine-finite-guard-v1: a non-finite value in the state or the surface
//! budget while the column holds snow stops the run with a non-zero exit and a
//! message naming the time step and the field. Before the guard, a NaN passed the T_CRAZY check
//! (it compares with `<` and `>`) and the run wrote NaN and exited 0.
//!
//! Synthetic forcing: the fixtures' ULAMB1 INI and SNO (bare ground), six days
//! of 10-minute forcing with snow on day 0, and one value replaced by `NaN` or
//! `inf` (SMET values are read with `str::parse::<f64>`, which accepts both).

use std::path::{Path, PathBuf};
use std::process::Command;

use chrono::{TimeZone, Utc};
use pi_snow_cli::engine::{RunError, Site, run};
use pi_snow_cli::finite;
use pi_snow_core::{Column, Composition, Element, Microstructure};
use pi_snow_io::ini::{Ini, RunConfig};
use pi_snow_io::smet::Smet;
use pi_snow_physics::surface::Budget;

const END: &str = "2026-01-07T00:00";

fn scratch(name: &str) -> PathBuf {
    // Cargo's per-target scratch dir (rust/target/tmp), never /tmp; a fresh name per run.
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "pi-snow-finite-{}-{name}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Six days of forcing; `bad` replaces column `field` of data row `row` with `value`.
fn forcing(bad: Option<(usize, &str, &str)>) -> String {
    let fields = ["TA", "RH", "TSG", "VW", "DW", "ISWR", "ILWR", "PSUM"];
    let mut text = String::from(
        "SMET 1.1 ASCII\n[HEADER]\nstation_id = ULAMB1\nstation_name = Synthetic\n\
         latitude = 39.59478\nlongitude = -111.22507\naltitude = 2754\nnodata = -999\ntz = 0\n\
         fields = timestamp TA RH TSG VW DW ISWR ILWR PSUM\n[DATA]\n",
    );
    for step in 0..(6 * 144) {
        let minutes = step * 10;
        let (hour, day) = ((minutes / 60) % 24, minutes / 1440);
        let phase = (minutes % 1440) as f64 / 1440.0 * std::f64::consts::TAU;
        let ta = 267.0 + 4.0 * (phase - 2.0).sin();
        let iswr = (500.0 * (phase - 1.6).sin()).max(0.0);
        let psum = if day == 0 && (8..14).contains(&hour) {
            0.5
        } else {
            0.0
        };
        let mut values: Vec<String> = [ta, 0.8, 273.15, 3.0, 200.0, iswr, 240.0, psum]
            .iter()
            .map(|v| format!("{v:.4}"))
            .collect();
        if let Some((row, field, value)) = bad
            && row == step
        {
            values[fields.iter().position(|f| *f == field).unwrap()] = value.to_string();
        }
        text += &format!(
            "2026-01-{:02}T{:02}:{:02} {}\n",
            1 + day,
            hour,
            minutes % 60,
            values.join(" ")
        );
    }
    text
}

fn stage(name: &str, bad: Option<(usize, &str, &str)>) -> PathBuf {
    let unit = scratch(name);
    std::fs::create_dir_all(unit.join("input")).unwrap();
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    std::fs::copy(fixtures.join("snowpack.ini"), unit.join("snowpack.ini")).unwrap();
    std::fs::copy(fixtures.join("ULAMB1.sno"), unit.join("input/ULAMB1.sno")).unwrap();
    std::fs::write(unit.join("input/ULAMB1.smet"), forcing(bad)).unwrap();
    unit
}

fn pi_snow(unit: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_pi-snow"))
        .args(["-c", "snowpack.ini", "-e", END])
        .current_dir(unit)
        .output()
        .unwrap()
}

#[test]
fn finite_forcing_runs_to_the_end_with_no_non_finite_output() {
    let unit = stage("clean", None);
    let out = pi_snow(&unit);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let smet = std::fs::read_to_string(unit.join("output/ULAMB1_hrrr-terrain-v1.smet")).unwrap();
    assert!(!smet.to_lowercase().contains("nan") && !smet.contains("inf"));
}

#[test]
fn the_nan_run_the_old_binary_finished_now_fails() {
    // Pinned by running the pre-guard binary 547c49ed on this forcing (lab notes):
    // exit 0 with NaN in 77 SMET rows. Here: the guarded binary on it must not.
    let unit = stage("nan-snow-2", Some((400, "TA", "NaN")));
    assert!(!pi_snow(&unit).status.success());
}

#[test]
fn a_nan_under_snow_exits_non_zero_naming_the_step_and_the_field() {
    // Data row 400 is 2026-01-03T18:40, the 400th step (the first row starts the clock).
    let unit = stage("nan-snow", Some((400, "TA", "NaN")));
    let out = pi_snow(&unit);
    assert!(!out.status.success(), "a NaN state must not exit 0");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("non-finite state at time step 400"), "{err}");
    assert!(err.contains("2026-01-03 18:40:00"), "{err}");
    assert!(err.contains("NaN"), "{err}");
    let smet = std::fs::read_to_string(unit.join("output/ULAMB1_hrrr-terrain-v1.smet")).unwrap();
    let last = smet.lines().last().unwrap();
    assert!(last.starts_with("2026-01-03T18:00"), "{last}");
    assert!(unit.join("output/ULAMB1_hrrr-terrain-v1.pro").is_file());
}

#[test]
fn a_nan_on_bare_ground_is_not_stopped_by_design() {
    // Row 3, 00:30 on day 0, before the first snowfall: the column is empty, which
    // the scoped guard does not check (option D), and bare ground resets the node.
    let unit = stage("nan-bare", Some((3, "ILWR", "NaN")));
    let out = pi_snow(&unit);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn an_infinite_input_never_exits_zero() {
    // An infinity may surface as a NaN (inf - inf) or trip T_CRAZY; either way
    // the run must fail.
    let unit = stage("inf", Some((400, "ILWR", "inf")));
    let out = pi_snow(&unit);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("non-finite state") || err.contains("T_CRAZY"),
        "{err}"
    );
}

#[test]
fn the_library_returns_the_step_and_field() {
    let cfg = RunConfig::from_ini(
        &Ini::parse(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/snowpack.ini"
            ))
            .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let site = Site {
        station_id: "ULAMB1".into(),
        station_name: "Synthetic".into(),
        latitude: 39.59478,
        longitude: -111.22507,
        altitude: 2754.0,
        slope_angle: 0.0,
        slope_azimuth: 0.0,
        soil_albedo: 0.09,
    };
    let forcing = Smet::parse(&forcing(Some((400, "RH", "NaN")))).unwrap();
    let end = Utc.with_ymd_and_hms(2026, 1, 7, 0, 0, 0).unwrap();
    match run(
        &cfg,
        &site,
        &forcing,
        Column::bare_ground(273.15, 0.0, 0.0),
        end,
    ) {
        Err(RunError::NonFinite { step, field, .. }) => {
            assert_eq!(step, 400);
            assert!(field.contains("NaN"), "{field}");
        }
        Err(e) => panic!("wrong error: {e}"),
        Ok(_) => panic!("a NaN run returned Ok"),
    }
}

fn element() -> Element {
    Element {
        thickness: 0.05,
        composition: Composition {
            ice: 0.3,
            water: 0.0,
            air: 0.7,
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
        age_days: 1.0,
        stress: 0.0,
        stress_rate: 0.0,
        metamorphism: 0.0,
        soil_density: 0.0,
        soil_conductivity: 0.0,
        soil_heat_capacity: 0.0,
        grain_history: 0,
    }
}

#[test]
fn the_checkers_name_fields_a_temperature_check_would_miss() {
    let mut c = Column::bare_ground(270.0, 0.0, 0.0);
    c.elements = vec![element(), element()];
    c.node_temperature = vec![270.0; 3];
    assert_eq!(finite::column(&c), None);
    c.elements[1].micro.grain_radius = f64::NAN;
    assert_eq!(
        finite::column(&c).as_deref(),
        Some(
            "element 1 (from the bottom) grain_radius = NaN (column: 2 elements, 3 nodes, total thickness 0.1 m)"
        )
    );
    let b = Budget::default();
    assert_eq!(finite::budget(&b, 0.0), None);
    let mut bad = b;
    bad.latent = f64::INFINITY;
    assert_eq!(finite::budget(&bad, 0.0).as_deref(), Some("Ql = inf"));
    assert_eq!(
        finite::budget(&b, f64::NAN).as_deref(),
        Some("deposition buffer = NaN")
    );
}
