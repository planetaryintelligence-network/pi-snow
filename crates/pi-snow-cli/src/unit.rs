//! One column, start to finish: read the INI, forcing and SNO, run the engine,
//! write PRO and SMET. `pi-snow -c ... -e ...` and `pi-snow batch` both call
//! `run_unit`, so a column run on a thread of a batch writes what the same
//! column run as its own process writes.
//!
//! Nothing here depends on the working directory: every path is joined to the
//! INI's own directory, so many units can run in one process.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, NaiveDateTime, Utc};
use pi_snow_core::PhysicsOptions;
use pi_snow_io::ini::{Ini, RunConfig};
use pi_snow_io::smet::Smet;
use pi_snow_io::sno::SnoFile;

use crate::engine::{self, RunError, RunOutput, Site};

/// What one finished column reports.
pub struct UnitRun {
    pub station_id: String,
    pub steps: usize,
    pub seconds: f64,
}

pub fn parse_end(text: &str) -> Result<DateTime<Utc>> {
    Ok(NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M")
        .with_context(|| format!("bad end time {text:?}"))?
        .and_utc())
}

/// Every `--physics KEY=VALUE` in `args`. A mistyped flag must fail: silently
/// dropping it would run Phase 2 physics under an arm's label.
pub fn parse_physics(args: &[String]) -> Result<PhysicsOptions> {
    let mut settings = Vec::new();
    let mut rest = args.iter();
    while let Some(a) = rest.next() {
        if a == "--physics" {
            settings.push(rest.next().context("--physics needs KEY=VALUE")?.as_str());
        } else if a.starts_with("--physics") {
            bail!("write --physics KEY=VALUE as two arguments, got {a:?}");
        }
    }
    Ok(PhysicsOptions::from_settings(settings)?)
}

pub fn run_unit(ini_path: &Path, end: DateTime<Utc>, physics: &PhysicsOptions) -> Result<UnitRun> {
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

    let started = std::time::Instant::now();
    let out = match engine::run_with_options(&cfg, physics, &site, &forcing, sno.to_column(), end) {
        Ok(out) => out,
        Err(error) => {
            if let RunError::Crazy { partial, .. } | RunError::NonFinite { partial, .. } = &error {
                write_output(&root, &cfg, partial)?;
            }
            return Err(error.into());
        }
    };
    write_output(&root, &cfg, &out)?;
    Ok(UnitRun {
        station_id: cfg.station_id,
        steps: out.steps,
        seconds: started.elapsed().as_secs_f64(),
    })
}

fn write_output(root: &Path, cfg: &RunConfig, out: &RunOutput) -> Result<()> {
    let output: PathBuf = root.join(&cfg.output_path);
    std::fs::create_dir_all(&output)?;
    let stem = output.join(format!("{}_{}", cfg.station_id, cfg.experiment));
    if cfg.ts_write {
        std::fs::write(stem.with_extension("smet"), &out.smet)?;
    }
    if cfg.prof_write {
        std::fs::write(stem.with_extension("pro"), &out.pro)?;
    }
    // The ledger (MASS_REPORT=LEDGER) is written here, not in main, so batch
    // mode carries it too.
    if let Some(mass) = &out.mass {
        std::fs::write(stem.with_extension("mass.csv"), mass)?;
    }
    if let Some(events) = &out.density_events {
        std::fs::write(stem.with_extension("density-events.csv"), events)?;
    }
    if let Some(diagnostic) = &out.meltout_diagnostic {
        std::fs::write(stem.with_extension("meltout-diagnostic.csv"), diagnostic)?;
    }
    if let Some(diagnostic) = &out.dirichlet_storage_diagnostic {
        std::fs::write(
            stem.with_extension("dirichlet-storage-diagnostic.csv"),
            diagnostic,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_physics;

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_well_formed_physics_flag_parses() {
        let o = parse_physics(&args(&[
            "-c",
            "x.ini",
            "--physics",
            "S5=LWC_INDEX_MITTERER2013",
        ]))
        .unwrap();
        assert_ne!(o, pi_snow_core::PhysicsOptions::default());
    }

    #[test]
    fn the_equals_spelling_is_refused() {
        let e = parse_physics(&args(&["-c", "x.ini", "--physics=S5=NONE"])).unwrap_err();
        assert!(e.to_string().contains("two arguments"), "{e}");
    }

    #[test]
    fn a_trailing_bare_flag_is_refused() {
        let e = parse_physics(&args(&["-c", "x.ini", "--physics"])).unwrap_err();
        assert!(e.to_string().contains("needs KEY=VALUE"), "{e}");
    }

    #[test]
    fn an_unknown_value_is_refused() {
        assert!(parse_physics(&args(&["--physics", "S5=BOGUS"])).is_err());
    }
}
