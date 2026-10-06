//! A unit directory as `pi-snow -c` reads it: `snowpack.ini`, `input/<id>.smet` and
//! `input/<id>.sno`. Mirrors `pi_snow_cli::unit::run_unit`'s loading, which is not
//! reusable from outside because it also runs the engine and writes files.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, NaiveDateTime, Utc};
use pi_snow_core::Column;
use pi_snow_io::ini::{Ini, RunConfig};
use pi_snow_io::smet::Smet;
use pi_snow_io::sno::SnoFile;

use crate::stepper::Site;

pub struct Unit {
    pub root: PathBuf,
    pub cfg: RunConfig,
    pub forcing: Smet,
    pub site: Site,
    pub start: Column,
}

pub fn parse_end(text: &str) -> Result<DateTime<Utc>> {
    Ok(NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M")
        .with_context(|| format!("bad end time {text:?}"))?
        .and_utc())
}

/// Load a unit given the path of its INI.
pub fn load(ini_path: &Path) -> Result<Unit> {
    let root = ini_path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let cfg = RunConfig::from_ini(&Ini::parse(
        &std::fs::read_to_string(ini_path).with_context(|| ini_path.display().to_string())?,
    )?)?;
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
    let start = sno.to_column();
    Ok(Unit {
        root,
        cfg,
        forcing,
        site,
        start,
    })
}

/// `<output>/<station>_<experiment>` without extension, as `pi-snow` names its files.
pub fn output_stem(unit: &Unit) -> PathBuf {
    unit.root
        .join(&unit.cfg.output_path)
        .join(format!("{}_{}", unit.cfg.station_id, unit.cfg.experiment))
}
