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

/// Reconstruct a deterministic 420-layer synthetic pack from the preceding
/// 18:00 pi-snow diagnostic profile. The 0.5 cm top layer is thin against the
/// 171.8 cm column and retains the branch-sensitive thermal profile.
fn deep_thin_topped_pack() -> Column {
    let fields: std::collections::HashMap<&str, Vec<f64>> =
        include_str!("fixtures/tprut1-deep-pack-20260219-1800.txt")
            .lines()
            .map(|line| {
                let mut parts = line.split(',');
                let code = parts.next().unwrap();
                let count: usize = parts.next().unwrap().parse().unwrap();
                let values: Vec<f64> = parts.map(|v| v.parse().unwrap()).collect();
                assert_eq!(values.len(), count, "PRO field {code}");
                (
                    match code {
                        "0501" => "height",
                        "0502" => "density",
                        "0503" => "temperature",
                        "0505" => "age",
                        "0506" => "water",
                        "0508" => "dendricity",
                        "0509" => "sphericity",
                        "0510" => "coordination",
                        "0511" => "bond",
                        "0512" => "grain",
                        "0515" => "ice",
                        "0516" => "air",
                        "0517" => "stress",
                        "0519" => "soil",
                        "0520" => "gradient",
                        _ => panic!("unexpected PRO field {code}"),
                    },
                    values,
                )
            })
            .collect();
    let get = |name: &str| fields.get(name).unwrap();
    let n = get("height").len();
    assert_eq!(n, 420);
    let mut c = Column::bare_ground(273.15, 25.8, 8.9);
    c.elements = (0..n)
        .map(|i| {
            let thickness =
                (get("height")[i] - if i == 0 { 0.0 } else { get("height")[i - 1] }) / 100.0;
            let density = get("density")[i];
            let ice = (density - 1.2) / (917.0 - 1.2);
            Element {
                thickness,
                composition: Composition {
                    ice,
                    water: 0.0,
                    air: 1.0 - ice,
                    soil: 0.0,
                },
                micro: Microstructure {
                    dendricity: get("dendricity")[i],
                    sphericity: get("sphericity")[i],
                    grain_radius: get("grain")[i] / 2_000.0,
                    bond_radius: get("bond")[i] / 2_000.0,
                    coordination_number: get("coordination")[i],
                    marker: 0,
                },
                age_days: get("age")[i],
                stress: get("stress")[i] * 1_000.0,
                stress_rate: 0.0,
                metamorphism: 0.0,
                soil_density: 0.0,
                soil_conductivity: 0.0,
                soil_heat_capacity: 0.0,
                grain_history: 0,
            }
        })
        .collect();
    let temperatures: Vec<f64> = get("temperature").iter().map(|t| t + 273.15).collect();
    let gradients = get("gradient");
    c.node_temperature = Vec::with_capacity(n + 1);
    c.node_temperature
        .push(temperatures[0] - gradients[0] * c.elements[0].thickness / 2.0);
    for i in 0..n {
        c.node_temperature
            .push(c.node_temperature[i] + gradients[i] * c.elements[i].thickness);
    }
    c
}

fn tprut1_excursion_forcing() -> Smet {
    Smet::parse(
        "SMET 1.1 ASCII\n[HEADER]\nstation_id = TPRUT1\nstation_name = synthetic deep pack\n\
         latitude = 40.848590\nlongitude = -111.038360\naltitude = 2929\n\
         nodata = -999\ntz = 0\nfields = timestamp TA RH TSG VW DW ISWR ILWR PSUM\n[DATA]\n\
         2026-02-19T18:00 257.4625527572 0.7513340530 273.15 4.4158356455 231.4510014351 648.5140918378 142.1859166022 0.0001125867\n\
         2026-02-19T18:10 257.7617352895 0.7562556383 273.15 4.8757708001 231.1249464175 644.1238240301 147.5718874789 0\n\
         2026-02-19T18:20 258.0609178218 0.7611772236 273.15 5.3358366303 230.8551100226 639.7335562224 152.9578583556 0\n",
    )
    .unwrap()
}

fn tprut1_site() -> Site {
    Site {
        station_id: "TPRUT1".into(),
        station_name: "synthetic deep pack".into(),
        latitude: 40.848590,
        longitude: -111.038360,
        altitude: 2929.0,
        slope_angle: 25.80,
        slope_azimuth: 8.90,
        soil_albedo: 0.09,
    }
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

#[test]
fn radiative_floor_finishes_the_synthetic_deep_thin_topped_surface_step() {
    let cfg = config();
    let forcing = tprut1_excursion_forcing();
    let site = tprut1_site();
    let end = Utc.with_ymd_and_hms(2026, 2, 19, 18, 20, 0).unwrap();

    let phase2 = run(&cfg, &site, &forcing, deep_thin_topped_pack(), end);
    match phase2 {
        Err(pi_snow_cli::engine::RunError::NonFinite { field, .. }) => {
            assert!(field.contains("NaN") || field.contains("inf"), "{field}");
        }
        Err(error) => panic!("Phase 2 stopped for another reason: {error}"),
        Ok(_) => panic!("Phase 2 did not reproduce the surface-solve excursion"),
    }

    let options = PhysicsOptions::from_settings(["SURFACE_LINEARISATION=RADIATIVE_FLOOR"]).unwrap();
    let guarded = run_with_options(
        &cfg,
        &options,
        &site,
        &forcing,
        deep_thin_topped_pack(),
        end,
    )
    .expect("the radiative floor must finish the synthetic step");
    assert!(!guarded.smet.to_lowercase().contains("nan"));
    assert!(!guarded.smet.to_lowercase().contains("inf"));
}
