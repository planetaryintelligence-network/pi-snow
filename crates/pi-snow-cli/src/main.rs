//! `pi-snow -c <ini> -e <YYYY-MM-DDTHH:MM>`, the same invocation the pipeline
//! uses for the reference engine. Paths in the INI are relative to the INI's
//! directory, which is also the working directory the pipeline sets.
//!
//! `--physics KEY=VALUE` (repeatable) selects a labelled physics variant; with
//! none given the engine is exactly the Phase 2 engine. See
//! `pi_snow_core::options`.
//! `SNOW_ALBEDO=CHAN2022_MERRA2` also requires `--merra2-deposition CSV` with
//! hourly BC and dust deposition rates from the registered source.
//!
//! `pi-snow batch ...` runs many columns in one process; see
//! `pi_snow_cli::batch`.

use std::path::PathBuf;

use anyhow::{Context, Result};
use pi_snow_cli::unit::{parse_end, parse_physics, run_unit_with_deposition};
use pi_snow_physics::chan_albedo::DepositionSeries;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("batch") {
        return pi_snow_cli::batch::main(&args[2..]);
    }
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let ini_path = PathBuf::from(flag("-c").context("usage: pi-snow -c <ini> -e <end>")?);
    let end_text = flag("-e").context("usage: pi-snow -c <ini> -e <end>")?;
    let end = parse_end(&end_text)?;
    let physics = parse_physics(&args[1..])?;
    let merra2 = flag("--merra2-deposition")
        .map(|path| {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading MERRA-2 deposition CSV {path}"))?;
            DepositionSeries::from_csv(&text).map_err(anyhow::Error::msg)
        })
        .transpose()?;

    let run = run_unit_with_deposition(&ini_path, end, &physics, merra2.as_ref())?;
    eprintln!(
        "pi-snow {}: {} steps for {} in {:.3} s",
        env!("CARGO_PKG_VERSION"),
        run.steps,
        run.station_id,
        run.seconds
    );
    Ok(())
}
