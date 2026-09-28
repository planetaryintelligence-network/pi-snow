//! `MASS_REPORT=LEDGER` and `WATER_RETENTION` through the whole driver, on a
//! synthetic three-week season: a storm, a cold clear spell, a humid spell
//! with deposition, then warm days with rain on snow and melt.

use chrono::{DateTime, Duration, TimeZone, Utc};
use pi_snow_cli::engine::{self, RunOutput, Site};
use pi_snow_core::PhysicsOptions;
use pi_snow_io::ini::{Ini, RunConfig};
use pi_snow_io::smet::Smet;
use pi_snow_io::sno::SnoFile;

const INI: &str = include_str!("../../pi-snow-io/tests/fixtures/run_ULAMB1.ini");
const SNO: &str = include_str!("../../pi-snow-io/tests/fixtures/init_ULAMB.sno");

fn start() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 1, 30, 23, 0, 0).unwrap()
}

/// Ten-minute forcing, 21 days. Fields as the pipeline writes them (K, 1, m/s,
/// W m^-2, kg m^-2 per step).
fn forcing() -> Smet {
    let mut text = String::from(
        "SMET 1.1 ASCII\n[HEADER]\nstation_id = ULAMB1\nstation_name = synthetic\n\
         latitude = 39.59478\nlongitude = -111.22507\naltitude = 2754\nnodata = -999\n\
         tz = 0\nfields = timestamp TA RH TSG VW DW ISWR ILWR PSUM\n[DATA]\n",
    );
    for step in 0..(21 * 144) {
        let t = start() + Duration::minutes(10 * step);
        let day = step as f64 / 144.0;
        let hour = (day.fract() * 24.0 + 17.0) % 24.0; // local solar-ish hour
        let sun = (std::f64::consts::PI * (hour - 6.0) / 12.0).sin().max(0.0);
        let (ta, rh, vw, iswr, ilwr, psum) = if day < 3.0 {
            (266.0, 0.95, 2.0, 80.0 * sun, 260.0, 0.2)
        } else if day < 8.0 {
            (260.0 + 4.0 * sun, 0.6, 3.0, 550.0 * sun, 190.0, 0.0)
        } else if day < 11.0 {
            (271.5, 1.0, 1.5, 150.0 * sun, 300.0, 0.0)
        } else if day < 14.0 {
            (276.0 + 6.0 * sun, 0.8, 2.5, 750.0 * sun, 290.0, 0.0)
        } else if day < 15.0 {
            (277.0, 0.98, 3.0, 100.0 * sun, 310.0, 0.3)
        } else {
            (279.0 + 7.0 * sun, 0.6, 2.5, 800.0 * sun, 280.0, 0.0)
        };
        text.push_str(&format!(
            "{} {ta} {rh} 273.15 {vw} 180 {iswr} {ilwr} {psum}\n",
            t.format("%Y-%m-%dT%H:%M")
        ));
    }
    Smet::parse(&text).unwrap()
}

fn run(settings: &[&str]) -> RunOutput {
    let cfg = RunConfig::from_ini(&Ini::parse(INI).unwrap()).unwrap();
    let site = Site {
        station_id: "ULAMB1".into(),
        station_name: "synthetic".into(),
        latitude: 39.59478,
        longitude: -111.22507,
        altitude: 2754.0,
        slope_angle: 30.0,
        slope_azimuth: 180.0,
        soil_albedo: 0.09,
    };
    let options = PhysicsOptions::from_settings(settings.iter().copied()).unwrap();
    let column = SnoFile::parse(SNO).unwrap().to_column();
    engine::run_with_options(
        &cfg,
        &options,
        &site,
        &forcing(),
        column,
        start() + Duration::days(21),
    )
    .unwrap()
}

struct Row {
    swe: f64,
    net: f64,
    deposited: f64,
    runoff: f64,
    sublimation: f64,
    deposition: f64,
}

fn ledger_rows(text: &str) -> Vec<Row> {
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let col = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let (swe, dep, rp, ro, dr, sub, depo) = (
        col("swe"),
        col("deposited"),
        col("rain_to_pack"),
        col("runoff"),
        col("drained"),
        col("sublimation_applied"),
        col("deposition_applied"),
    );
    lines
        .map(|l| {
            let v: Vec<f64> = l.split(',').skip(1).map(|x| x.parse().unwrap()).collect();
            let g = |i: usize| v[i - 1];
            Row {
                swe: g(swe),
                net: g(dep) + g(rp) - g(ro) - g(dr) - g(sub) + g(depo),
                deposited: g(dep),
                runoff: g(ro) + g(dr),
                sublimation: g(sub),
                deposition: g(depo),
            }
        })
        .collect()
}

#[test]
fn the_ledger_changes_no_byte_of_smet_or_pro() {
    let plain = run(&[]);
    let explicit = run(&["MASS_REPORT=PHASE2", "WATER_RETENTION=WEVER2014_PRINTED"]);
    let ledger = run(&["MASS_REPORT=LEDGER"]);
    assert!(plain.mass.is_none() && explicit.mass.is_none());
    assert_eq!(plain.smet, explicit.smet);
    assert_eq!(plain.pro, explicit.pro);
    assert_eq!(plain.smet, ledger.smet);
    assert_eq!(plain.pro, ledger.pro);
    let rows = ledger.mass.expect("ledger written");
    // One ledger row per SMET data row.
    let smet_rows = plain
        .smet
        .split("[DATA]")
        .nth(1)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty());
    assert_eq!(rows.lines().count() - 1, smet_rows.count());
}

#[test]
fn the_ledger_closes_the_budget_every_hour() {
    let rows = ledger_rows(&run(&["MASS_REPORT=LEDGER"]).mass.unwrap());
    // The season exercises every term, or closing would prove little.
    let total = |f: fn(&Row) -> f64| rows.iter().map(f).sum::<f64>();
    assert!(total(|r| r.deposited) > 50.0, "storm deposited");
    assert!(total(|r| r.sublimation) > 0.1, "sublimation");
    assert!(total(|r| r.deposition) > 0.01, "deposition");
    assert!(total(|r| r.runoff) > 10.0, "runoff");
    let mut worst: f64 = 0.0;
    for pair in rows.windows(2) {
        worst = worst.max((pair[1].swe - pair[0].swe - pair[1].net).abs());
    }
    assert!(worst < 1e-6, "largest hourly residual {worst} kg m^-2");
}

#[test]
fn coleou_lesaffre_retention_moves_water_and_conserves_mass() {
    let old = run(&["MASS_REPORT=LEDGER"]);
    let new = run(&["MASS_REPORT=LEDGER", "WATER_RETENTION=COLEOU_LESAFFRE_1998"]);
    assert_ne!(
        old.smet, new.smet,
        "the retention law must reach the output"
    );
    let rows = ledger_rows(&new.mass.unwrap());
    for pair in rows.windows(2) {
        assert!((pair[1].swe - pair[0].swe - pair[1].net).abs() < 1e-6);
    }
}
