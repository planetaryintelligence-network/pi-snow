//! `pi-snow -c <ini> -e <YYYY-MM-DDTHH:MM>`, the same invocation the pipeline
//! uses for the reference engine. Paths in the INI are relative to the INI's
//! directory, which is also the working directory the pipeline sets.
//!
//! `--physics KEY=VALUE` (repeatable) selects a labelled physics variant; with
//! none given the engine is exactly the Phase 2 engine. See
//! `pi_snow_core::options`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, NaiveDateTime, Utc};
use pi_snow_cli::engine::{self, Site};
use pi_snow_core::PhysicsOptions;
use pi_snow_io::ini::{Ini, RunConfig};
use pi_snow_io::smet::Smet;
use pi_snow_io::sno::SnoFile;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let ini_path = PathBuf::from(flag("-c").context("usage: pi-snow -c <ini> -e <end>")?);
    let end_text = flag("-e").context("usage: pi-snow -c <ini> -e <end>")?;
    let end: DateTime<Utc> = NaiveDateTime::parse_from_str(&end_text, "%Y-%m-%dT%H:%M")
        .with_context(|| format!("bad end time {end_text:?}"))?
        .and_utc();

    // A mistyped flag must fail: silently dropping it would run Phase 2
    // physics under an arm's label.
    let mut settings = Vec::new();
    let mut rest = args.iter().skip(1);
    while let Some(a) = rest.next() {
        if a == "--physics" {
            settings.push(rest.next().context("--physics needs KEY=VALUE")?.as_str());
        } else if a.starts_with("--physics") {
            bail!("write --physics KEY=VALUE as two arguments, got {a:?}");
        }
    }
    let physics = PhysicsOptions::from_settings(settings)?;

    let root = ini_path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let cfg = RunConfig::from_ini(&Ini::parse(&std::fs::read_to_string(&ini_path)?)?)?;
    let meteo = root.join(&cfg.meteo_path);
    let forcing_path = meteo.join(format!("{}.smet", cfg.station_id));
    let forcing = Smet::parse(
        &std::fs::read_to_string(&forcing_path)
            .with_context(|| forcing_path.display().to_string())?,
    )?;
    let sno_path = meteo.join(format!("{}.sno", cfg.station_id));
    let sno = SnoFile::parse(
        &std::fs::read_to_string(&sno_path).with_context(|| sno_path.display().to_string())?,
    )?;

    let header = |k: &str| forcing.header.get(k).cloned();
    let number = |k: &str| -> Result<f64> {
        match header(k) {
            Some(v) => Ok(v.parse()?),
            None => bail!("forcing header lacks {k}"),
        }
    };
    let site = Site {
        station_id: cfg.station_id.clone(),
        station_name: header("station_name").unwrap_or_else(|| cfg.station_id.clone()),
        latitude: number("latitude")?,
        longitude: number("longitude")?,
        altitude: number("altitude")?,
        slope_angle: sno.slope_angle_deg,
        slope_azimuth: sno.slope_azimuth_deg,
        soil_albedo: 0.09,
    };

    let started = std::time::Instant::now();
    let out = engine::run_with_options(&cfg, &physics, &site, &forcing, sno.to_column(), end)?;
    let output = root.join(&cfg.output_path);
    std::fs::create_dir_all(&output)?;
    let stem = output.join(format!("{}_{}", cfg.station_id, cfg.experiment));
    std::fs::write(stem.with_extension("smet"), out.smet)?;
    std::fs::write(stem.with_extension("pro"), out.pro)?;
    if let Some(mass) = out.mass {
        std::fs::write(stem.with_extension("mass.csv"), mass)?;
    }
    eprintln!(
        "pi-snow {}: {} steps for {} in {:.3} s",
        env!("CARGO_PKG_VERSION"),
        out.steps,
        cfg.station_id,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
