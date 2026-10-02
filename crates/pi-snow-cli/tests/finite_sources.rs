//! engine-finite-guard-v1's two source fixes, on synthetic forcing:
//! `SUBLIMATION_MELT_OUT=REMOVE_ELEMENT` (a thin dry pack sublimating away) and
//! `SURFACE_LINEARISATION=RADIATIVE_FLOOR` (the slope never shallower than the
//! emitted longwave's).

use chrono::{Duration, TimeZone, Utc};
use pi_snow_cli::engine::{Site, run, run_with_options};
use pi_snow_core::{Column, Composition, Element, Microstructure, PhysicsOptions};
use pi_snow_io::ini::{Ini, RunConfig};
use pi_snow_io::smet::Smet;
use pi_snow_physics::surface::{Atmosphere, SNOW_EMISSIVITY, SurfaceParameters};
use pi_snow_physics::surface_numerics::linearise;
use pi_snow_physics::turbulence::Scheme;

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

/// Two dry, windy, cloudless nights and a day: nothing melts, a lot sublimates.
fn dry_wind() -> Smet {
    let mut text = String::from(
        "SMET 1.1 ASCII\n[HEADER]\nstation_id = TEST\nlatitude = 40.6\nlongitude = -111.6\naltitude = 2600\n\
         nodata = -999\ntz = 0\nfields = timestamp TA RH TSG VW DW ISWR ILWR PSUM\n[DATA]\n",
    );
    let start = Utc.with_ymd_and_hms(2026, 1, 10, 0, 0, 0).unwrap();
    for i in 0..(3 * 144) {
        let t = start + Duration::minutes(10 * i);
        text.push_str(&format!(
            "{} 270.15 0.15 273.15 9.0 270 0 200 0\n",
            t.format("%Y-%m-%dT%H:%M")
        ));
    }
    Smet::parse(&text).unwrap()
}

/// One thin dry element of rounded snow: about 0.9 kg/m2 of ice.
fn thin_pack() -> Column {
    let mut c = Column::bare_ground(272.0, 0.0, 0.0);
    c.elements = vec![Element {
        thickness: 0.004,
        composition: Composition {
            ice: 0.25,
            water: 0.0,
            air: 0.75,
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
        age_days: 5.0,
        stress: 0.0,
        stress_rate: 0.0,
        metamorphism: 0.0,
        soil_density: 0.0,
        soil_conductivity: 0.0,
        soil_heat_capacity: 0.0,
        grain_history: 0,
    }];
    c.node_temperature = vec![271.0, 270.0];
    c
}

fn end() -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 1, 13, 0, 0, 0).unwrap()
}

fn nan_rows(smet: &str) -> usize {
    smet.lines()
        .skip_while(|l| *l != "[DATA]")
        .filter(|l| {
            l.split_whitespace()
                .any(|v| v == "NaN" || v == "inf" || v == "-inf")
        })
        .count()
}

#[test]
fn phase_2_leaves_a_nan_hour_when_sublimation_takes_the_last_ice() {
    // Pins the defect the fix is for: the pack sublimates away and the empty
    // column's node is NaN for a step (stored Phase 2 runs show the same rows).
    let out = run(&config(), &site(), &dry_wind(), thin_pack(), end()).unwrap();
    assert!(
        nan_rows(&out.smet) > 0,
        "the synthetic pack must reproduce the Phase 2 NaN hour"
    );
}

#[test]
fn remove_element_sublimates_the_pack_away_with_no_nan() {
    let options = PhysicsOptions::from_settings(["SUBLIMATION_MELT_OUT=REMOVE_ELEMENT"]).unwrap();
    let out = run_with_options(
        &config(),
        &options,
        &site(),
        &dry_wind(),
        thin_pack(),
        end(),
    )
    .unwrap();
    assert_eq!(
        nan_rows(&out.smet),
        0,
        "{}",
        out.smet.lines().find(|l| l.contains("NaN")).unwrap_or("")
    );
    let last = out.smet.lines().last().unwrap();
    assert!(last.split_whitespace().count() > 10, "{last}");
}

#[test]
fn the_floored_slope_is_never_shallower_than_the_longwave_slope() {
    // Across calm, windy, stable, unstable and melting cases, including the
    // two-height decoupling regime where Phase 2's slope can be zero.
    for (wind, t_air, t0) in [
        (0.5, 268.0, 250.0),
        (2.0, 268.0, 263.0),
        (8.0, 260.0, 270.0),
        (3.0, 275.0, 273.1),
        (0.0, 250.0, 240.0),
    ] {
        let atm = Atmosphere {
            t_air,
            relative_humidity: 0.7,
            wind,
            absorbed_shortwave: 50.0,
            ilwr: 200.0,
            pressure: 72_000.0,
            rain_rate: 0.0,
        };
        let p = SurfaceParameters {
            emissivity: SNOW_EMISSIVITY,
            z_wind: 9.5,
            z_scalar: 1.5,
            z0: 0.002,
            scheme: Scheme::MoHoltslag,
        };
        let (a, b) = linearise(&atm, &p, t0);
        let t = t0.min(273.15 - 0.01);
        let floor = -4.0 * SNOW_EMISSIVITY * 5.670_374_419e-8 * t.powi(3);
        assert!(
            b <= floor + 1e-12,
            "b {b} above floor {floor} at {wind} {t_air} {t0}"
        );
        assert!(a.is_finite() && b.is_finite());
    }
}
