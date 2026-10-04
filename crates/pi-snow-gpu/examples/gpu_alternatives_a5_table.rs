use std::fs::{File, read_to_string};
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use pi_snow_core::constants::{GRAVITY, T_MELT, VON_KARMAN};
use pi_snow_core::{Column, PhysicsOptions};
use pi_snow_gpu::{forcing, stepper::{self, ForcingRow, Stepper}, unit};
use pi_snow_io::ini::{Ini, RunConfig};
use pi_snow_physics::new_snow::Conditions;
use pi_snow_physics::turbulence::{psi_h, psi_m};

const PLAN: &str = "cb93626f18faabb87bf5a967ba5a48fbae092f42968809ce8f55d878ac00a358";
const RI_N: usize = 1024;
const HEIGHT_N: usize = 32;
const RI_SCALE: f64 = 0.01;

#[derive(Default)]
struct Bounds {
    ri_min: f64,
    ri_max: f64,
    zw_min: f64,
    zw_max: f64,
    zs_min: f64,
    zs_max: f64,
    samples: u64,
    skipped_wind: u64,
    invalid_coordinates: u64,
    first_invalid: Option<String>,
}

impl Bounds {
    fn new() -> Self {
        Self { ri_min: f64::INFINITY, ri_max: f64::NEG_INFINITY,
            zw_min: f64::INFINITY, zw_max: f64::NEG_INFINITY,
            zs_min: f64::INFINITY, zs_max: f64::NEG_INFINITY, ..Self::default() }
    }

    fn add(&mut self, stepper: &Stepper<'_>, row: &ForcingRow, source: &str) -> Result<()> {
        if stepper.previous.is_none() { return Ok(()); }
        let (Some(ta), Some(rh), Some(wind), Some(_), Some(_)) =
            (row.ta, row.rh, row.vw, row.iswr, row.ilwr) else { return Ok(()) };
        if wind <= 0.0 { self.skipped_wind += 1; return Ok(()); }
        let psum = row.psum.unwrap_or(0.0).max(0.0);

        // Mirror Stepper::step's deposition before its surface-height calculation.
        let mut column = stepper.column.clone();
        let mut deposition = stepper.deposition.clone();
        let t_surface_before = *column.node_temperature.last().context("surface node")?;
        let conditions = Conditions { t_air: ta, t_surface: t_surface_before,
            relative_humidity: rh, wind };
        let (solid, _) = if ta < T_MELT + stepper.cfg.thresh_rain_c { (psum, 0.0) } else { (0.0, psum) };
        deposition.add(&mut column, solid, &conditions, stepper.constants.density_scheme);

        let hs = column.snow_height();
        let z_wind = (stepper.cfg.height_of_wind_value - hs).max(stepper::MIN_MEASUREMENT_HEIGHT);
        let z_scalar = (stepper.cfg.height_of_meteo_values - hs).max(stepper::MIN_MEASUREMENT_HEIGHT);
        let t_surface = if column.elements.is_empty() {
            row.tsg.unwrap_or(T_MELT)
        } else {
            *column.node_temperature.last().context("post-deposition surface node")?
        };
        let (z0, mean_t) = (stepper.cfg.roughness_length, 0.5 * (ta + t_surface));
        let ri = GRAVITY * z_wind * (ta - t_surface) / (mean_t * wind * wind);
        let (zw, zs) = (z_wind / z0, z_scalar / z0);
        if !ri.is_finite() || !zw.is_finite() || !zs.is_finite() || zw <= 0.0 || zs <= 0.0 {
            self.invalid_coordinates += 1;
            if self.first_invalid.is_none() {
                self.first_invalid = Some(format!("{source} {} TA={ta} T_surface={t_surface} HS={hs} z0={z0} Ri={ri} z_wind/z0={zw} z_scalar/z0={zs}", row.t));
            }
            return Ok(());
        }
        self.ri_min = self.ri_min.min(ri);
        self.ri_max = self.ri_max.max(ri);
        self.zw_min = self.zw_min.min(zw);
        self.zw_max = self.zw_max.max(zw);
        self.zs_min = self.zs_min.min(zs);
        self.zs_max = self.zs_max.max(zs);
        self.samples += 1;
        Ok(())
    }
}

fn scan_unit(ini: &Path, bounds: &mut Bounds) -> Result<()> {
    let u = unit::load(ini)?;
    let index = stepper::ForcingIndex::of(&u.forcing)?;
    let mut st = Stepper::new(&u.cfg, &PhysicsOptions::default(), &u.site, u.start.clone())?;
    for row in &u.forcing.rows {
        let r = index.row(row);
        bounds.add(&st, &r, ini.parent().and_then(Path::file_name).and_then(|s| s.to_str()).unwrap_or("unknown"))?;
        st.step(&r)?;
    }
    Ok(())
}

fn scan_w_pixel(forcing_dir: &Path, template: &str, pixel: &forcing::Pixel, bounds: &mut Bounds) -> Result<()> {
    let mut raw = File::open(forcing_dir.join("forcing.f32"))?;
    let hourly = forcing::hourly_forcing(&forcing::read_hourly(&mut raw, pixel.index)?)?;
    let rows = forcing::expand(&hourly, 1, forcing::HOURS - 1);
    let cfg = RunConfig::from_ini(&Ini::parse(&forcing::ini_text(template, &pixel.key)?)?)?;
    let site = pixel.site()?;
    let start = Column::bare_ground(T_MELT, site.slope_angle, site.slope_azimuth);
    let mut st = Stepper::new(&cfg, &PhysicsOptions::default(), &site, start)?;
    for r in &rows {
        bounds.add(&st, r, &pixel.key)?;
        st.step(r)?;
    }
    Ok(())
}

fn axis_ri(lo: f64, hi: f64) -> Vec<f64> {
    let a = (lo / RI_SCALE).asinh();
    let b = (hi / RI_SCALE).asinh();
    let mut out: Vec<f64> = (0..RI_N).map(|i| RI_SCALE * (a + (b - a) * i as f64 / (RI_N - 1) as f64).sinh()).collect();
    out.extend([0.05, 0.06]);
    out.sort_by(f64::total_cmp);
    out.dedup_by(|x, y| x.to_bits() == y.to_bits());
    out
}

fn axis_log(lo: f64, hi: f64) -> Vec<f64> {
    let (a, b) = (lo.ln(), hi.ln());
    (0..HEIGHT_N).map(|i| (a + (b - a) * i as f64 / (HEIGHT_N - 1) as f64).exp()).collect()
}

fn corrections(ri: f64, zw: f64, zs: f64, z0: f64) -> (f64, f64) {
    // Only dimensionless height ratios enter psi and dm/dh, so this fixed z0
    // is a unit scale, not a station-specific physical roughness.
    let (wind, mean_t) = (1.0, 263.15);
    let dt = ri * mean_t * wind * wind / (GRAVITY * zw);
    let mut inverse_l = 0.0;
    for _ in 0..200 {
        let dm = (zw / z0).ln() - psi_m(zw * inverse_l) + psi_m(z0 * inverse_l);
        let dh = (zs / z0).ln() - psi_h(zs * inverse_l) + psi_h(z0 * inverse_l);
        let u_star = VON_KARMAN * wind / dm;
        let theta_star = VON_KARMAN * dt / dh;
        let target = VON_KARMAN * GRAVITY * theta_star / (u_star * u_star * mean_t);
        let next = 0.5 * (inverse_l + target);
        if (next - inverse_l).abs() <= 1e-10 * next.abs().max(1e-6) { inverse_l = next; break; }
        inverse_l = next;
    }
    (
        (zw / z0).ln() - psi_m(zw * inverse_l) + psi_m(z0 * inverse_l),
        (zs / z0).ln() - psi_h(zs * inverse_l) + psi_h(z0 * inverse_l),
    )
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 7 { bail!("usage: a5-table UNIT_INI_LIST FORCING_DIR INI_TEMPLATE W_PIXEL_INDICES OUT_DIR ROUGHNESS_LENGTH") }
    let units = read_to_string(&a[1])?;
    let mut unit_count = 0;
    let mut bounds = Bounds::new();
    for ini in units.lines().filter(|s| !s.trim().is_empty()) { scan_unit(Path::new(ini), &mut bounds)?; unit_count += 1; }
    if unit_count != 65 { bail!("expected 65 v3b units, found {unit_count}"); }
    let forcing_dir = Path::new(&a[2]);
    let template = read_to_string(&a[3])?;
    let pixel_rows = forcing::read_pixels(&forcing_dir.join("pixels.tsv"))?;
    let indices: Vec<usize> = read_to_string(&a[4])?.lines().filter(|s| !s.trim().is_empty()).map(str::parse).collect::<Result<_, _>>()?;
    for index in &indices {
        let pixel = pixel_rows.get(*index).with_context(|| format!("W pixel index {index} is absent"))?;
        scan_w_pixel(forcing_dir, &template, pixel, &mut bounds)?;
    }
    if bounds.samples == 0 || bounds.ri_min >= bounds.ri_max { bail!("empty or degenerate A5 sample range"); }
    let ri_pad = 0.1 * (bounds.ri_max - bounds.ri_min);
    let (ri_lo, ri_hi) = (bounds.ri_min - ri_pad, bounds.ri_max + ri_pad);
    let (zw_lo, zw_hi) = (bounds.zw_min / 1.1, bounds.zw_max * 1.1);
    let (zs_lo, zs_hi) = (bounds.zs_min / 1.1, bounds.zs_max * 1.1);
    let ri_axis = axis_ri(ri_lo, ri_hi);
    let zw_axis = axis_log(zw_lo, zw_hi);
    let zs_axis = axis_log(zs_lo, zs_hi);
    let output = Path::new(&a[5]);
    std::fs::create_dir_all(output)?;
    let z0: f64 = a[6].parse()?;
    let file = File::create(output.join("mo-table-f64.bin"))?;
    let mut writer = BufWriter::new(file);
    for ri in &ri_axis { for zw in &zw_axis { for zs in &zs_axis {
        let (dm, dh) = corrections(*ri, *zw * z0, *zs * z0, z0);
        if !dm.is_finite() || !dh.is_finite() { bail!("non-finite table node ri={ri} zw/z0={zw} zs/z0={zs}"); }
        writer.write_all(&dm.to_le_bytes())?;
        writer.write_all(&dh.to_le_bytes())?;
    } } }
    writer.flush()?;

    let fmt = |v: f64| format!("{v:.16e}");
    let ri_text = ri_axis.iter().map(|v| fmt(*v)).collect::<Vec<_>>().join(",");
    let zw_text = zw_axis.iter().map(|v| fmt(*v)).collect::<Vec<_>>().join(",");
    let zs_text = zs_axis.iter().map(|v| fmt(*v)).collect::<Vec<_>>().join(",");
    let manifest = format!(
        "{{\n  \"plan_sha256\":\"{PLAN}\",\n  \"source_units\":{unit_count},\n  \"w_pixels\":{},\n  \"samples\":{},\n  \"skipped_wind_nonpositive\":{},\n  \"invalid_coordinate_fallbacks\":{},\n  \"first_invalid_coordinate\":{},\n  \"observed_ri\":[{},{}],\n  \"ri_bounds_padded\":[{},{}],\n  \"observed_z_wind_over_z0\":[{},{}],\n  \"z_wind_over_z0_bounds_padded\":[{},{}],\n  \"observed_z_scalar_over_z0\":[{},{}],\n  \"z_scalar_over_z0_bounds_padded\":[{},{}],\n  \"axis_sizes\":[{}, {}, {}],\n  \"ri_nodes\":[{}],\n  \"z_wind_over_z0_nodes\":[{}],\n  \"z_scalar_over_z0_nodes\":[{}],\n  \"payload\":\"f64 little-endian pairs dm,dh; RI-major, wind-ratio, scalar-ratio\"\n}}\n",
        indices.len(), bounds.samples, bounds.skipped_wind,
        bounds.invalid_coordinates, bounds.first_invalid.as_ref().map_or("null".into(), |s| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))),
        fmt(bounds.ri_min), fmt(bounds.ri_max), fmt(ri_lo), fmt(ri_hi),
        fmt(bounds.zw_min), fmt(bounds.zw_max), fmt(zw_lo), fmt(zw_hi),
        fmt(bounds.zs_min), fmt(bounds.zs_max), fmt(zs_lo), fmt(zs_hi),
        ri_axis.len(), zw_axis.len(), zs_axis.len(), ri_text, zw_text, zs_text);
    std::fs::write(output.join("table.json"), manifest)?;
    println!("units={unit_count} W={} samples={} invalid_coordinate_fallbacks={} Ri=[{},{}] z_wind/z0=[{},{}] z_scalar/z0=[{},{}] table_bytes={}",
        indices.len(), bounds.samples, bounds.invalid_coordinates, fmt(ri_lo), fmt(ri_hi), fmt(zw_lo), fmt(zw_hi), fmt(zs_lo), fmt(zs_hi),
        std::fs::metadata(output.join("mo-table-f64.bin"))?.len());
    let _ = PLAN;
    Ok(())
}
