//! Single-pixel report path with a hash-bound physics passthrough.
//! It reads one F32 forcing block in memory and never calls `unit::write_output`.
use std::fs::File;
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Duration, TimeZone, Utc};
use pi_snow_cli::batch::UnitReport;
use pi_snow_cli::diagnostic::{self, DiagnosticMode, Minimum, Refusal, RunDiagnostics};
use pi_snow_cli::engine::{self, RunError, Site};
use pi_snow_cli::unit::parse_end;
use pi_snow_core::PhysicsOptions;
use pi_snow_io::ini::{Ini, RunConfig};
use pi_snow_io::smet::Smet;
use pi_snow_io::sno::SnoFile;

const HOURS: usize = 4344;
const FIELDS: usize = 7;
const BLOCK_BYTES: usize = HOURS * FIELDS * 4;

#[derive(Clone)]
struct Pixel {
    index: usize,
    key: String,
    row: u32,
    column: u32,
    latitude: String,
    longitude: String,
    altitude: String,
    slope: String,
    azimuth: String,
}

fn value(args: &[String], name: &str) -> Result<String> {
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].clone())
        .with_context(|| format!("missing {name}"))
}

fn pixel(args: &[String]) -> Result<Pixel> {
    Ok(Pixel {
        index: value(args, "--index")?.parse()?,
        key: value(args, "--key")?,
        row: value(args, "--row")?.parse()?,
        column: value(args, "--column")?.parse()?,
        latitude: value(args, "--latitude")?,
        longitude: value(args, "--longitude")?,
        altitude: value(args, "--altitude")?,
        slope: value(args, "--slope")?,
        azimuth: value(args, "--azimuth")?,
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let physics = pi_snow_cli::unit::parse_physics(&args)?;
    let p = pixel(&args)?;
    let forcing = PathBuf::from(value(&args, "--forcing")?);
    let template = PathBuf::from(value(&args, "--ini-template")?);
    let end = parse_end(&value(&args, "--end")?)?;
    if args.iter().any(|arg| arg == "--dump-smet") {
        print!("{}", forcing_text(&p, &forcing)?);
        return Ok(());
    }
    report(&p, &forcing, &template, end, &physics)
}

fn read_block(path: &Path, index: usize) -> Result<Vec<f32>> {
    let mut file = File::open(path).with_context(|| path.display().to_string())?;
    let mut bytes = vec![0u8; BLOCK_BYTES];
    file.seek(std::io::SeekFrom::Start((index * BLOCK_BYTES) as u64))?;
    file.read_exact(&mut bytes)?;
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}


fn wind_to_vector(speed: f64, direction_from: f64) -> (f64, f64) {
    let r = direction_from.to_radians();
    (-speed * r.sin(), -speed * r.cos())
}

fn py_mod_360(x: f64) -> f64 {
    let r = x % 360.0;
    if r == 0.0 { 0.0 } else if r < 0.0 { r + 360.0 } else { r }
}

fn vector_to_wind(u: f64, v: f64) -> (f64, f64) {
    if u == 0.0 && v == 0.0 { return (0.0, 0.0); }
    (u.hypot(v), py_mod_360(270.0 - v.atan2(u).to_degrees()))
}

struct Hourly {
    ta: Vec<f64>, rh: Vec<f64>, iswr: Vec<f64>, ilwr: Vec<f64>,
    u: Vec<f64>, v: Vec<f64>, psum: Vec<f64>,
}

fn hourly(raw: &[f32]) -> Result<Hourly> {
    let at = |h: usize, f: usize| raw[h * FIELDS + f] as f64;
    let known: Vec<bool> = (0..HOURS).map(|h| [0, 1, 4, 5, 2, 3].iter().all(|&f| at(h, f).is_finite())).collect();
    if !known[0] || !known[HOURS - 1] { bail!("forcing is missing at the first or last hour"); }
    let state = |h: usize| { let (u, v) = wind_to_vector(at(h, 2), at(h, 3)); [at(h, 0), at(h, 1), at(h, 4), at(h, 5), u, v] };
    let mut cols: [Vec<f64>; 6] = Default::default();
    let mut psum = Vec::with_capacity(HOURS);
    let mut previous = 0usize;
    for h in 0..HOURS {
        let s = if known[h] { previous = h; state(h) } else {
            let following = (h + 1..HOURS).find(|&k| known[k]).context("forcing gap reaches season end")?;
            let weight = (h - previous) as f64 / (following - previous) as f64;
            let (a, b) = (state(previous), state(following));
            std::array::from_fn(|i| a[i] + (b[i] - a[i]) * weight)
        };
        for (c, x) in cols.iter_mut().zip(s) { c.push(x); }
        let p = at(h, 6);
        if !p.is_finite() { psum.push(0.0); }
        else if p < 0.0 { bail!("negative PSUM at hour {h}"); }
        else { psum.push(p); }
    }
    let [ta, rh, iswr, ilwr, u, v] = cols;
    Ok(Hourly { ta, rh, iswr, ilwr, u, v, psum })
}

struct ForcingRow { time: DateTime<Utc>, fields: [f64; 8] }

fn expand(h: &Hourly) -> Vec<ForcingRow> {
    let start = Utc.with_ymd_and_hms(2025, 11, 1, 0, 0, 0).unwrap();
    let mut rows = Vec::with_capacity((HOURS - 1) * 6 + 1);
    let mut add = |time, ta, rh, u, v, iswr, ilwr, psum| {
        let (vw, dw) = vector_to_wind(u, v);
        rows.push(ForcingRow { time, fields: [ta, rh, 273.15, vw, dw, iswr, ilwr, psum] });
    };
    add(start, h.ta[0], h.rh[0], h.u[0], h.v[0], h.iswr[0], h.ilwr[0], 0.0);
    for i in 1..HOURS {
        let end = start + Duration::hours(i as i64);
        for step in 1..=6 {
            let w = step as f64 / 6.0;
            let lerp = |a: &[f64]| a[i - 1] + (a[i] - a[i - 1]) * w;
            add(end - Duration::minutes(10 * (6 - step) as i64), lerp(&h.ta), lerp(&h.rh),
                lerp(&h.u), lerp(&h.v), lerp(&h.iswr), lerp(&h.ilwr), h.psum[i] / 6.0);
        }
    }
    rows
}

fn smet_text(p: &Pixel, rows: &[ForcingRow]) -> String {
    let mut s = format!("SMET 1.1 ASCII\n[HEADER]\nstation_id = {}\nstation_name = grid pixel {},{}\nlatitude = {}\nlongitude = {}\naltitude = {}\nnodata = -999\ntz = 0\nsource = engine-gpu-v1 forcing B, station-lane 10-minute expansion\nfields = timestamp TA RH TSG VW DW ISWR ILWR PSUM\n[DATA]\n", p.key, p.row, p.column, p.latitude, p.longitude, p.altitude);
    for r in rows {
        s.push_str(&format!("{} {} {} {} {} {} {} {} {}\n", r.time.format("%Y-%m-%dT%H:%M"),
            r.fields[0], r.fields[1], r.fields[2], r.fields[3], r.fields[4], r.fields[5], r.fields[6], r.fields[7]));
    }
    s
}

fn sno_text(p: &Pixel) -> String {
    format!("SMET 1.1 ASCII\n[HEADER]\nstation_id = {}\nstation_name = grid pixel {},{}\nlongitude = {}\nlatitude = {}\naltitude = {}\nnodata = -999\ntz = 0\nsource = pi-pod grid empty autumn initialization\nProfileDate = 2025-11-01T00:00\nHS_Last = 0\nSlopeAngle = {}\nSlopeAzi = {}\nnSoilLayerData = 0\nnSnowLayerData = 0\nSoilAlbedo = 0.09\nBareSoil_z0 = 0.200\nCanopyHeight = 0.00\nCanopyLeafAreaIndex = 0.00\nCanopyDirectThroughfall = 1.00\nWindScalingFactor = 1.00\nErosionLevel = 0\nTimeCountDeltaHS = 0.000000\nfields = timestamp Layer_Thick T Vol_Frac_I Vol_Frac_W Vol_Frac_V Vol_Frac_S Rho_S Conduc_S HeatCapac_S rg rb dd sp mk mass_hoar ne CDot metamo\n[DATA]\n",
        p.key, p.row, p.column, p.longitude, p.latitude, p.altitude, p.slope, p.azimuth)
}

fn ini_text(template: &str, key: &str) -> Result<String> {
    let mut out = String::new(); let mut seen = (0, 0, 0);
    for line in template.lines() {
        let k = line.split('=').next().unwrap_or("").trim().to_ascii_uppercase();
        let new = match k.as_str() {
            "STATION1" => { seen.0 += 1; format!("STATION1 = {key}") },
            "SNOWFILE1" => { seen.1 += 1; format!("SNOWFILE1 = {key}") },
            "EXPERIMENT" => { seen.2 += 1; "EXPERIMENT = grid".to_string() },
            _ => line.to_string(),
        };
        out.push_str(&new); out.push('\n');
    }
    if seen != (1, 1, 1) { bail!("INI template must carry STATION1, SNOWFILE1 and EXPERIMENT once each"); }
    Ok(out)
}

fn extract(error: &RunError) -> Option<(usize, Option<RunDiagnostics>)> {
    match error {
        RunError::Crazy { partial, .. } => Some(&**partial).map(|p| (p.steps, p.diagnostics.clone())),
        RunError::NonFinite { partial, .. } => Some((partial.steps, partial.diagnostics.clone())),
        _ => None,
    }
}

fn report(p: &Pixel, forcing_path: &Path, ini_template: &Path, end: DateTime<Utc>,
          physics: &PhysicsOptions) -> Result<()> {
    let smet = Smet::parse(&forcing_text(p, forcing_path)?)?;
    let sno = SnoFile::parse(&sno_text(p))?;
    let template = std::fs::read_to_string(ini_template)?;
    let cfg = RunConfig::from_ini(&Ini::parse(&ini_text(&template, &p.key)?)?)?;
    let site = Site { station_id: p.key.clone(), station_name: format!("grid pixel {},{}", p.row, p.column),
        latitude: p.latitude.parse()?, longitude: p.longitude.parse()?, altitude: p.altitude.parse()?,
        slope_angle: sno.slope_angle_deg, slope_azimuth: sno.slope_azimuth_deg, soil_albedo: 0.09 };
    let outcome = engine::run_with_diagnostics(&cfg, physics, &site, &smet,
        sno.to_column(), end, Some(DiagnosticMode::Minimum));
    let (steps, diagnostics, error) = match outcome {
        Ok(output) => (output.steps, output.diagnostics, None),
        Err(e) => { let (steps, d) = extract(&e).context("unclassified engine failure")?;
            (steps, d, Some(e.to_string())) }
    };
    let diagnostics = diagnostics.context("minimum diagnostic absent")?;
    let unit = UnitReport { unit_dir: PathBuf::from(&p.key), station_id: Some(p.key.clone()), steps,
        seconds: 0.0, error, minimum: diagnostics.minimum.map(|m| Minimum { kelvin: m.kelvin,
            time: m.time, step: m.step, node: m.node }), refusal: diagnostics.refusal.map(|r| Refusal {
            kind: r.kind, kelvin: r.kelvin, time: r.time, step: r.step, node: r.node, detail: r.detail }) };
    print!("{}", diagnostic::minimum_report_json(&[unit], 1, 0.0));
    Ok(())
}

fn forcing_text(p: &Pixel, forcing_path: &Path) -> Result<String> {
    let raw = read_block(forcing_path, p.index)?;
    Ok(smet_text(p, &expand(&hourly(&raw)?)))
}
