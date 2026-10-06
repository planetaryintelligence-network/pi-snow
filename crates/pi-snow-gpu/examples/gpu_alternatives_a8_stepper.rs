use std::fs::{File, read_to_string};
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use pi_snow_core::Column;
use pi_snow_core::PhysicsOptions;
use pi_snow_gpu::{forcing, stepper::{ForcingIndex, Outcome, SeasonWriter, Stepper}};
use pi_snow_io::{ini::{Ini, RunConfig}, smet::Smet};

fn state(elements: &[pi_snow_core::Element]) -> String {
    elements.iter().map(|e| format!("{:.17},{:.17},{:.17},{:.17},{:.17}",
        e.thickness, e.micro.dendricity, e.micro.grain_radius, e.stress, e.ice_mass()))
        .collect::<Vec<_>>().join(";")
}

fn run_pixel(pixel: &forcing::Pixel, forcing_dir: &Path, template: &str,
    expected_smet: &Path, out: &Path) -> Result<()> {
    let mut raw = File::open(forcing_dir.join("forcing.f32"))?;
    let hourly = forcing::hourly_forcing(&forcing::read_hourly(&mut raw, pixel.index)?)?;
    let rows = forcing::expand(&hourly, 1, forcing::HOURS - 1);
    let meteo_text = forcing::smet_text(pixel, &rows);
    let sno_text = forcing::sno_text(pixel);
    std::fs::create_dir_all(out)?;
    std::fs::write(out.join(format!("{}.smet", pixel.key)), &meteo_text)?;
    std::fs::write(out.join(format!("{}.sno", pixel.key)), &sno_text)?;
    let meteo = Smet::parse(&meteo_text)?;
    let ini_text = forcing::ini_text(template, &pixel.key)?;
    let cfg = RunConfig::from_ini(&Ini::parse(&ini_text)?)?;
    let site = pixel.site()?;
    let start = Column::bare_ground(forcing::GROUND_TEMPERATURE_K, site.slope_angle, site.slope_azimuth);
    let index = ForcingIndex::of(&meteo)?;
    let mut stepper = Stepper::new(&cfg, &PhysicsOptions::default(), &site, start)?;
    let mut writer = SeasonWriter::new(&site);
    let trace_path = out.join(format!("{}-per-step.tsv", pixel.key));
    let mut trace = BufWriter::new(File::create(trace_path)?);
    writeln!(trace, "timestamp\tbefore\tafter")?;
    for row in &meteo.rows {
        let before = state(&stepper.column.elements);
        let forcing_row = index.row(row);
        match stepper.step(&forcing_row)? {
            Outcome::Advanced { output } => {
                let after = state(&stepper.column.elements);
                writeln!(trace, "{}\t{}\t{}", row.timestamp.to_rfc3339(), before, after)?;
                if output { writer.push(&mut stepper, row.timestamp); }
            }
            Outcome::NoStep => {}
            Outcome::Skipped => bail!("unexpected missing forcing at {}", row.timestamp),
        }
    }
    trace.flush()?;
    let output = writer.finish(&site, stepper.steps);
    let generated_smet = out.join(format!("{}-hourly.smet", pixel.key));
    std::fs::write(&generated_smet, &output.smet)?;
    let expected = read_to_string(expected_smet).with_context(|| expected_smet.display().to_string())?;
    if output.smet != expected {
        let parsed_generated = Smet::parse(&output.smet)?;
        let parsed_expected = Smet::parse(&expected)?;
        let generated_times: Vec<_> = parsed_generated.rows.iter().map(|r| r.timestamp).collect();
        let expected_times: Vec<_> = parsed_expected.rows.iter().map(|r| r.timestamp).collect();
        bail!("{} hourly verification failed: byte_equal=false generated_rows={} expected_rows={} timestamps_equal={}",
            pixel.key, generated_times.len(), expected_times.len(), generated_times == expected_times);
    }
    println!("{} forcing_rows={} advanced={} hourly_smet=exact", pixel.key, meteo.rows.len(), stepper.steps);
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 { bail!("usage: a8-stepper FORCING_DIR INI_TEMPLATE WATCH_SMET_DIR OUT_DIR PIXELS_TSV"); }
    let forcing_dir = Path::new(&args[1]);
    let template = read_to_string(&args[2])?;
    let watch_smet = Path::new(&args[3]);
    let out = Path::new(&args[4]);
    let pixels = forcing::read_pixels(Path::new(&args[5]))?;
    for key in ["px_320_261", "px_354_260"] {
        let pixel = pixels.iter().find(|p| p.key == key).with_context(|| format!("missing registered pixel {key}"))?;
        let expected = watch_smet.join("cpu/smet").join(format!("{key}_grid.smet"));
        run_pixel(pixel, forcing_dir, &template, &expected, out)?;
    }
    Ok(())
}
