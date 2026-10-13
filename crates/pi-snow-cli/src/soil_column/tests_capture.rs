//! engine-soil-spinup-v2 plan test Z-T6 `retained_collapse_cases`.
//!
//! The captures are written by the S2 capture build (`s2_rig.apply_capture_patch`)
//! when a bare solve in a replay of the retained state-2 arms ends in a
//! collapsed bracket: the entry column, atmosphere, parameters, the collapsed
//! bracket bits and the stop message, as text. This test reads every
//! `capture-*.txt` under `$SOIL_V2_CAPTURES` and asserts, for each:
//!
//!  (a) under the captured scheme (the CLM5 transplant) the post-failure
//!      bisection collapses again, with the recorded bracket bits;
//!  (b) the same inputs under `MoHoltslagZetaContinuous` accept with
//!      `|r| <= 1e-6 W/m2`, by Newton or by the fallback within its cap.
//!
//! Without the variable the test does nothing, so the ordinary suite is
//! unaffected. Capture fidelity (the capture build reproduces the retained
//! stops and bracket bits) is checked in Python before this test is run.

use super::*;
use pi_snow_physics::surface::{Atmosphere, SurfaceParameters};
use pi_snow_physics::turbulence::Scheme;
use std::collections::HashMap;

fn f(hex: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(hex, 16).unwrap())
}

struct Capture {
    dt: f64,
    material: Material,
    bottom: Bottom,
    enthalpy: [f64; 10],
    atm: Atmosphere,
    params: SurfaceParameters,
    scheme: Scheme,
    bracket: [u64; 2],
}

fn parse(text: &str) -> Capture {
    let mut map: HashMap<&str, Vec<&str>> = HashMap::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        if let Some(key) = it.next() {
            map.insert(key, it.collect());
        }
    }
    let material = match map["material"][0] {
        "MoistLoam" => Material::MoistLoam,
        "DryLoam" => Material::DryLoam,
        other => panic!("material {other}"),
    };
    let bottom = match map["bottom"][0] {
        "Deep" => Bottom::Deep {
            temperature_k: f(map["bottom"][1]),
        },
        "Geothermal005" => Bottom::Geothermal005,
        "Insulated" => Bottom::Insulated,
        "Flux" => Bottom::Flux(f(map["bottom"][1])),
        other => panic!("bottom {other}"),
    };
    let a: Vec<f64> = map["atm"].iter().map(|h| f(h)).collect();
    let p: Vec<f64> = map["params"][..4].iter().map(|h| f(h)).collect();
    let scheme = match map["params"][4] {
        "MoHoltslagZetaLimited" => Scheme::MoHoltslagZetaLimited,
        "MoHoltslag" => Scheme::MoHoltslag,
        "MoHoltslagZetaContinuous" => Scheme::MoHoltslagZetaContinuous,
        other => panic!("scheme {other}"),
    };
    let mut enthalpy = [0.0; 10];
    for (i, h) in map["enthalpy"].iter().enumerate() {
        enthalpy[i] = f(h);
    }
    Capture {
        dt: f(map["dt"][0]),
        material,
        bottom,
        enthalpy,
        atm: Atmosphere {
            t_air: a[0],
            relative_humidity: a[1],
            wind: a[2],
            absorbed_shortwave: a[3],
            ilwr: a[4],
            pressure: a[5],
            rain_rate: a[6],
        },
        params: SurfaceParameters {
            emissivity: p[0],
            z_wind: p[1],
            z_scalar: p[2],
            z0: p[3],
            scheme,
        },
        scheme,
        bracket: [
            u64::from_str_radix(map["bracket"][0], 16).unwrap(),
            u64::from_str_radix(map["bracket"][1], 16).unwrap(),
        ],
    }
}

fn column(c: &Capture) -> SoilColumn {
    let mut s = SoilColumn::uniform(c.material, 280.0);
    s.enthalpy = c.enthalpy;
    s
}

struct Outcome {
    name: String,
    bracket: [u64; 2],
    collapse_reproduced: bool,
    bracket_equal: bool,
    zn_accepted: bool,
    residual: f64,
    fallback_entries: usize,
}

fn outcome(path: &std::path::Path) -> Outcome {
    let c = parse(&std::fs::read_to_string(path).unwrap());
    // (a) the original failure is preserved: same scheme, collapses again.
    let mut soil = column(&c);
    let (collapse_reproduced, bracket_equal) =
        match bare::step(&mut soil, c.dt, c.bottom, &c.atm, &c.params) {
            Err(f) => (
                f.message == bare_fallback::COLLAPSED,
                f.diagnostics.failure_bracket_bits == Some(c.bracket),
            ),
            Ok(_) => (false, false),
        };
    // (b) under ZN the same inputs have a root the unchanged tolerance accepts.
    assert_ne!(c.scheme, Scheme::MoHoltslagZetaContinuous);
    let params = SurfaceParameters {
        scheme: Scheme::MoHoltslagZetaContinuous,
        ..c.params
    };
    let mut soil = column(&c);
    let (zn_accepted, residual, fallback_entries) =
        match bare::step(&mut soil, c.dt, c.bottom, &c.atm, &params) {
            Ok(r) => (
                r.flux_residual_w_m2.abs() <= 1e-6,
                r.flux_residual_w_m2,
                r.diagnostics.fallback_entries,
            ),
            Err(_) => (false, f64::NAN, 0),
        };
    Outcome {
        name: path.file_name().unwrap().to_string_lossy().into_owned(),
        bracket: c.bracket,
        collapse_reproduced,
        bracket_equal,
        zn_accepted,
        residual,
        fallback_entries,
    }
}

/// Reads `$SOIL_V2_CAPTURES` and the six retained bracket pairs from
/// `$SOIL_V2_RETAINED_BRACKETS` (decimal `lo hi` per line). Exactly six
/// captures, whose bracket pairs equal the retained six. All six (a) and (b)
/// outcomes are collected and printed before anything is asserted, so a first
/// failure does not hide the other five.
#[test]
fn retained_collapse_cases() {
    let Ok(dir) = std::env::var("SOIL_V2_CAPTURES") else {
        eprintln!("SOIL_V2_CAPTURES unset: Z-T6 runs in S2 on the capture build's output");
        return;
    };
    let retained_file = std::env::var("SOIL_V2_RETAINED_BRACKETS")
        .expect("SOIL_V2_RETAINED_BRACKETS must name the file of retained bracket pairs");
    let mut retained: Vec<[u64; 2]> = std::fs::read_to_string(&retained_file)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split_whitespace().map(|x| x.parse::<u64>().unwrap());
            [it.next().unwrap(), it.next().unwrap()]
        })
        .collect();
    retained.sort();
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("capture-"))
        })
        .collect();
    files.sort();
    let outcomes: Vec<Outcome> = files.iter().map(|p| outcome(p)).collect();
    for o in &outcomes {
        println!(
            "Z-T6-ROW file={} bracket={}:{} collapse_reproduced={} bracket_equal={} zn_accepted={} residual={:e} fallback_entries={}",
            o.name,
            o.bracket[0],
            o.bracket[1],
            o.collapse_reproduced,
            o.bracket_equal,
            o.zn_accepted,
            o.residual,
            o.fallback_entries
        );
    }
    let mut got: Vec<[u64; 2]> = outcomes.iter().map(|o| o.bracket).collect();
    got.sort();
    println!(
        "Z-T6-SUMMARY captures={} retained={}",
        outcomes.len(),
        retained.len()
    );
    assert_eq!(outcomes.len(), 6, "exactly six captures are registered");
    assert_eq!(
        got, retained,
        "capture bracket pairs differ from the six retained pairs"
    );
    for o in &outcomes {
        assert!(
            o.collapse_reproduced,
            "{}: transplant collapse not reproduced",
            o.name
        );
        assert!(o.bracket_equal, "{}: collapsed bracket bits differ", o.name);
        assert!(o.zn_accepted, "{}: ZN did not accept", o.name);
    }
}

#[test]
fn capture_text_parses_to_the_bits_it_was_written_with() {
    let bits = |x: f64| format!("{:016x}", x.to_bits());
    let h: Vec<String> = (0..10).map(|i| bits(1.0e5 * (i as f64 + 1.0))).collect();
    let text = format!(
        "stamp 1474310400\ndt {}\nmaterial MoistLoam\nbottom Deep {}\nenthalpy {}\n\
         atm {} {} {} {} {} {} {}\nparams {} {} {} {} MoHoltslagZetaLimited\n\
         bracket {} {}\nmessage x\n",
        bits(600.0),
        bits(274.25),
        h.join(" "),
        bits(288.15),
        bits(0.22),
        bits(5.0),
        bits(400.0),
        bits(300.0),
        bits(80_000.0),
        bits(0.0),
        bits(0.97),
        bits(10.0),
        bits(2.0),
        bits(0.002),
        format!("{:016x}", 200.5_f64.to_bits()),
        format!("{:016x}", 200.5_f64.to_bits() + 1),
    );
    let c = parse(&text);
    assert_eq!(c.dt, 600.0);
    assert_eq!(c.atm.relative_humidity.to_bits(), 0.22_f64.to_bits());
    assert_eq!(c.params.z0.to_bits(), 0.002_f64.to_bits());
    assert_eq!(c.scheme, Scheme::MoHoltslagZetaLimited);
    assert_eq!(c.enthalpy[9].to_bits(), 1.0e6_f64.to_bits());
    assert!(matches!(c.bottom, Bottom::Deep { temperature_k } if temperature_k == 274.25));
    assert_eq!(c.bracket[1], c.bracket[0] + 1);
}
