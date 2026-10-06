use std::fs::{File, read_to_string};
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use pi_snow_core::{
    Column, Composition, Element, Microstructure, PhysicsOptions, constants::T_MELT,
};
use pi_snow_gpu::{
    forcing,
    stepper::{ForcingIndex, Outcome, SeasonWriter, Stepper},
};
use pi_snow_io::{
    ini::{Ini, RunConfig},
    smet::Smet,
};
use pi_snow_physics::{grain_classification, metamorphism, thermal};

fn combine(a: &Element, b: &Element) -> Element {
    let h = a.thickness + b.thickness;
    let vw = |x: f64, y: f64| (x * a.thickness + y * b.thickness) / h;
    let (ma, mb) = (a.mass(), b.mass());
    let mw = |x: f64, y: f64| {
        if ma + mb > 0.0 {
            (x * ma + y * mb) / (ma + mb)
        } else {
            0.5 * (x + y)
        }
    };
    Element {
        thickness: h,
        composition: Composition {
            ice: vw(a.composition.ice, b.composition.ice),
            water: vw(a.composition.water, b.composition.water),
            air: vw(a.composition.air, b.composition.air),
            soil: vw(a.composition.soil, b.composition.soil),
        },
        micro: Microstructure {
            dendricity: mw(a.micro.dendricity, b.micro.dendricity),
            sphericity: mw(a.micro.sphericity, b.micro.sphericity),
            grain_radius: mw(a.micro.grain_radius, b.micro.grain_radius),
            bond_radius: mw(a.micro.bond_radius, b.micro.bond_radius),
            coordination_number: mw(a.micro.coordination_number, b.micro.coordination_number),
            marker: a.micro.marker.max(b.micro.marker),
        },
        age_days: mw(a.age_days, b.age_days),
        stress: a.stress,
        stress_rate: mw(a.stress_rate, b.stress_rate),
        metamorphism: mw(a.metamorphism, b.metamorphism),
        soil_density: a.soil_density,
        soil_conductivity: a.soil_conductivity,
        soil_heat_capacity: a.soil_heat_capacity,
        grain_history: grain_classification::merge_history(a.grain_history, b.grain_history),
    }
}

fn node_energies(c: &Column) -> Vec<f64> {
    let mut cap = vec![0.0; c.elements.len() + 1];
    for (i, e) in c.elements.iter().enumerate() {
        let half = 0.5 * thermal::volumetric_heat_capacity(e) * e.thickness;
        cap[i] += half;
        cap[i + 1] += half;
    }
    cap.iter()
        .zip(&c.node_temperature)
        .map(|(k, t)| k * (t - T_MELT))
        .collect()
}

fn set_from_energy(c: &mut Column, energy: &[f64]) {
    let mut cap = vec![0.0; c.elements.len() + 1];
    for (i, e) in c.elements.iter().enumerate() {
        let half = 0.5 * thermal::volumetric_heat_capacity(e) * e.thickness;
        cap[i] += half;
        cap[i + 1] += half;
    }
    for ((t, u), k) in c.node_temperature.iter_mut().zip(energy).zip(cap) {
        if k > 0.0 {
            *t = T_MELT + u / k;
        }
    }
}

fn class(e: &Element) -> u16 {
    metamorphism::classify(&e.micro, &pi_snow_gpu::stepper::history(e), e.density())
}

#[derive(Default)]
struct CapStats {
    merges: usize,
    class_change_merges: usize,
    max_deep: usize,
    mass_error: f64,
    energy_error: f64,
    deep_height_hours: f64,
    class_flip_height_hours: f64,
}

fn class_layers(c: &Column, count: usize) -> Vec<(f64, f64, u16)> {
    let mut bottom = 0.0;
    c.elements
        .iter()
        .take(count)
        .map(|e| {
            let top = bottom + e.thickness;
            let layer = (bottom, top, class(e));
            bottom = top;
            layer
        })
        .collect()
}

fn class_flip_height(before: &[(f64, f64, u16)], after: &[(f64, f64, u16)]) -> f64 {
    let (mut i, mut j, mut changed) = (0, 0, 0.0);
    while i < before.len() && j < after.len() {
        let (a0, a1, ac) = before[i];
        let (b0, b1, bc) = after[j];
        let overlap = (a1.min(b1) - a0.max(b0)).max(0.0);
        if ac != bc {
            changed += overlap;
        }
        if a1 <= b1 {
            i += 1;
        }
        if b1 <= a1 {
            j += 1;
        }
    }
    changed
}

fn cap(
    c: &mut Column,
    stats: &mut CapStats,
    log: &mut impl Write,
    key: &str,
    timestamp: &str,
) -> Result<()> {
    let hs = c.snow_height();
    let deep_limit = hs - 1.0;
    let mut deep = 0;
    let mut top = 0.0;
    for e in &c.elements {
        top += e.thickness;
        if top <= deep_limit {
            deep += 1;
        } else {
            break;
        }
    }
    stats.max_deep = stats.max_deep.max(deep);
    let before_classes = class_layers(c, deep);
    let deep_height = before_classes.last().map_or(0.0, |layer| layer.1);
    while deep > 50 {
        let mut all = Vec::with_capacity(deep - 1);
        let mut same = Vec::new();
        for i in 0..deep - 1 {
            let a = class(&c.elements[i]);
            let b = class(&c.elements[i + 1]);
            let choice = (
                c.elements[i].thickness + c.elements[i + 1].thickness,
                i,
                a,
                b,
            );
            all.push(choice);
            if a == b {
                same.push(choice);
            }
        }
        let choices = if same.is_empty() { &all } else { &same };
        let (_, i, class_a, class_b) = *choices
            .iter()
            .min_by(|a, b| a.0.total_cmp(&b.0).then_with(|| b.1.cmp(&a.1)))
            .unwrap();

        let mass_before = c.elements[i].mass() + c.elements[i + 1].mass();
        let energy_before = node_energies(c);
        let energy_sum_before: f64 = energy_before.iter().sum();
        let internal_node = i + 1;
        let node_u = energy_before[internal_node];
        let a = c.elements[i].clone();
        let b = c.elements.remove(i + 1);
        c.elements[i] = combine(&a, &b);
        c.node_temperature.remove(internal_node);
        let mut remaining = energy_before;
        remaining.remove(internal_node);
        remaining[i] += 0.5 * node_u;
        remaining[i + 1] += 0.5 * node_u;
        set_from_energy(c, &remaining);

        let merged_class = class(&c.elements[i]);
        let mass_error = (mass_before - c.elements[i].mass()).abs();
        let energy_error = (energy_sum_before - node_energies(c).iter().sum::<f64>()).abs();
        stats.mass_error = stats.mass_error.max(mass_error);
        stats.energy_error = stats.energy_error.max(energy_error);
        if mass_error > 1e-10 * mass_before.max(1.0)
            || energy_error > 1e-9 * energy_sum_before.abs().max(1.0)
        {
            bail!(
                "A6 conservation failure for {key} at {timestamp}: mass={mass_error} energy={energy_error}"
            );
        }
        let changed = class_a != class_b || merged_class != class_a;
        stats.class_change_merges += usize::from(changed);
        stats.merges += 1;
        writeln!(
            log,
            "{key}\t{timestamp}\t{i}\t{class_a}\t{class_b}\t{merged_class}\t{mass_error:.12e}\t{energy_error:.12e}"
        )?;
        deep -= 1;
    }
    if deep_height > 0.0 {
        let after_classes = class_layers(c, deep);
        stats.deep_height_hours += deep_height / 6.0;
        stats.class_flip_height_hours += class_flip_height(&before_classes, &after_classes) / 6.0;
    }
    Ok(())
}

fn run_pixel(
    pixel: &forcing::Pixel,
    forcing_dir: &Path,
    template: &str,
    out: &Path,
) -> Result<CapStats> {
    let mut raw = File::open(forcing_dir.join("forcing.f32"))?;
    let hourly = forcing::hourly_forcing(&forcing::read_hourly(&mut raw, pixel.index)?)?;
    let rows = forcing::expand(&hourly, 1, forcing::HOURS - 1);
    let meteo_text = forcing::smet_text(pixel, &rows);
    let meteo = Smet::parse(&meteo_text)?;
    let ini_text = forcing::ini_text(template, &pixel.key)?;
    let cfg = RunConfig::from_ini(&Ini::parse(&ini_text)?)?;
    let site = pixel.site()?;
    let start = Column::bare_ground(
        forcing::GROUND_TEMPERATURE_K,
        site.slope_angle,
        site.slope_azimuth,
    );
    let index = ForcingIndex::of(&meteo)?;
    let mut stepper = Stepper::new(&cfg, &PhysicsOptions::default(), &site, start)?;
    let mut writer = SeasonWriter::new(&site);
    let mut stats = CapStats::default();
    let mut cap_log = BufWriter::new(File::create(out.join(format!("{}-merges.tsv", pixel.key)))?);
    for row in &meteo.rows {
        match stepper.step(&index.row(row))? {
            Outcome::Advanced { output } => {
                cap(
                    &mut stepper.column,
                    &mut stats,
                    &mut cap_log,
                    &pixel.key,
                    &row.timestamp.to_rfc3339(),
                )?;
                if output {
                    writer.push(&mut stepper, row.timestamp);
                }
            }
            Outcome::NoStep => {}
            Outcome::Skipped => bail!("unexpected missing forcing at {}", row.timestamp),
        }
    }
    cap_log.flush()?;
    let output = writer.finish(&site, stepper.steps);
    std::fs::write(out.join(format!("{}_grid.smet", pixel.key)), output.smet)?;
    std::fs::write(out.join(format!("{}_grid.pro", pixel.key)), output.pro)?;
    Ok(stats)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        bail!("usage: a6-cap FORCING_DIR INI_TEMPLATE PIXELS_TSV KEYS_FILE OUT_DIR");
    }
    let forcing_dir = Path::new(&args[1]);
    let template = read_to_string(&args[2])?;
    let pixels = forcing::read_pixels(Path::new(&args[3]))?;
    let key_list = read_to_string(&args[4])?;
    let keys: Vec<_> = key_list
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();
    let out = Path::new(&args[5]);
    std::fs::create_dir_all(out)?;
    let mut summary = BufWriter::new(File::create(out.join("cap-summary.tsv"))?);
    writeln!(
        summary,
        "pixel\tsteps\tmerges\tclass_change_merges\tmax_deep_elements\tdeep_height_hours_m\tclass_flip_height_hours_m\tmax_mass_error_kg_m2\tmax_energy_error_J_m2"
    )?;
    for key in &keys {
        let pixel = pixels
            .iter()
            .find(|p| p.key == *key)
            .with_context(|| format!("missing pixel {key}"))?;
        let stats = run_pixel(pixel, forcing_dir, &template, out)?;
        writeln!(
            summary,
            "{key}\t{}\t{}\t{}\t{}\t{:.12e}\t{:.12e}\t{:.12e}\t{:.12e}",
            (forcing::HOURS - 1) * 6,
            stats.merges,
            stats.class_change_merges,
            stats.max_deep,
            stats.deep_height_hours,
            stats.class_flip_height_hours,
            stats.mass_error,
            stats.energy_error
        )?;
        println!(
            "{key}: merges={} max_deep={} class_flip_height_hours_m={:.5}",
            stats.merges, stats.max_deep, stats.class_flip_height_hours
        );
    }
    summary.flush()?;
    Ok(())
}
