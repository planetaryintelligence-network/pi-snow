//! Registered 3.2 hourly int16 forcing texture arm.
use std::{collections::HashSet, fs::File, path::PathBuf};

use anyhow::{Context, Result, bail};
use pi_snow_core::{Column, PhysicsOptions, constants::T_MELT};
use pi_snow_gpu_alternatives::{
    forcing::{self, Hourly, HOURS},
    forcing_texture::{self, PackedForcing, TextureGpu},
    runner::{self, ColumnInput, OutputTargets},
    stepper::ForcingRow,
};
use pi_snow_io::ini::{Ini, RunConfig};

const G1C_KEYS: [&str; 3] = ["px_166_353", "px_324_306", "px_462_47"];
const NMAX: usize = 2048;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (command, values) = parse(&args)?;
    match command.as_str() {
        "preflight" => preflight(&values),
        "nvrtc" => { println!("{}", forcing_texture::compile_nvrtc()?); Ok(()) }
        "g1c" => g1c(&values),
        _ => bail!("usage: pi-snow-gpu-32 preflight|nvrtc|g1c --forcing-dir DIR --pixels TSV --ini FILE --packing-manifest JSON --out DIR"),
    }
}

fn parse(args: &[String]) -> Result<(String, std::collections::BTreeMap<String, String>)> {
    let command = args.first().context("command is required")?.clone();
    let mut values = std::collections::BTreeMap::new();
    let mut it = args[1..].iter();
    while let Some(key) = it.next() {
        if !key.starts_with("--") { bail!("unexpected argument {key}"); }
        let value = it.next().with_context(|| format!("{key} needs a value"))?.clone();
        if values.insert(key.clone(), value).is_some() { bail!("duplicate option {key}"); }
    }
    for key in ["--forcing-dir", "--pixels", "--ini", "--packing-manifest", "--out"] {
        if !values.contains_key(key) { bail!("{key} is required"); }
    }
    Ok((command, values))
}

fn arg<'a>(v: &'a std::collections::BTreeMap<String, String>, k: &str) -> &'a str { &v[k] }

fn load_inputs(v: &std::collections::BTreeMap<String, String>) -> Result<(Vec<forcing::Pixel>, Vec<Hourly>, PackedForcing)> {
    let forcing_dir = PathBuf::from(arg(v, "--forcing-dir"));
    let pixels = forcing::read_pixels(&PathBuf::from(arg(v, "--pixels")))?;
    let indices = G1C_KEYS.iter().map(|key| {
        pixels.iter().find(|p| p.key == *key).map(|p| p.index)
            .with_context(|| format!("registered G1c pixel {key} absent from pixel table"))
    }).collect::<Result<Vec<_>>>()?;
    let mut file = File::open(forcing_dir.join("forcing.f32"))?;
    let hourly = indices.iter().map(|&index| {
        forcing::hourly_forcing(&forcing::read_hourly(&mut file, index)?)
    }).collect::<Result<Vec<_>>>()?;
    if hourly.iter().any(|h| h.ta.len() != HOURS) { bail!("forcing season hour count changed"); }
    let selected = indices.iter().map(|&index| pixels.iter().find(|p| p.index == index).unwrap().clone()).collect();
    let packed = PackedForcing::new(&hourly)?;
    Ok((selected, hourly, packed))
}

fn preflight(v: &std::collections::BTreeMap<String, String>) -> Result<()> {
    let (_, hourly, packed) = load_inputs(v)?;
    let result = packed_result(&hourly, &packed, None, None, None, None)
        .replacen("\"state\":\"scored\"", "\"state\":\"packing-preflight\"", 1);
    std::fs::write(PathBuf::from(arg(v, "--out")), result.as_bytes())?;
    println!("{}", result.trim());
    Ok(())
}

fn g1c(v: &std::collections::BTreeMap<String, String>) -> Result<()> {
    let (pixels, hourly, packed) = load_inputs(v)?;
    let gpu = TextureGpu::new(&packed)?;
    let rows: Vec<Vec<ForcingRow>> = hourly.iter().map(|h| forcing::expand(h, 1, HOURS - 1)).collect();
    let n_rows = rows[0].len();
    if rows.iter().any(|r| r.len() != n_rows) { bail!("G1c pixels have unequal expanded row counts"); }
    let tex = gpu.probe_rows(n_rows, rows.len())?;
    let forcing_result = packed_result(&hourly, &packed, Some(&rows), Some(&tex), None, Some(gpu.device_limits));
    let gate_pass = forcing_result.contains("\"g1c_forcing_pass\":true");
    let summaries = if gate_pass {
        let template = std::fs::read_to_string(arg(v, "--ini"))?;
        let cfg = RunConfig::from_ini(&Ini::parse(&forcing::ini_text(&template, "px")?)?)?;
        let options = PhysicsOptions::default();
        let sites = pixels.iter().map(|pixel| pixel.site()).collect::<Result<Vec<_>>>()?;
        let mut inputs = Vec::with_capacity(pixels.len());
        for (site, h) in sites.iter().zip(&hourly) {
            let start = Column::bare_ground(T_MELT, site.slope_angle, site.slope_azimuth);
            inputs.push(ColumnInput {
                cfg: &cfg, site: &site, start,
                rows: forcing::expand(h, 1, HOURS - 1),
                profile_times: Some(HashSet::new()),
            });
        }
        let out = PathBuf::from(arg(v, "--out")).parent().context("output path has no parent")?.join("model-output");
        std::fs::create_dir(&out).context("model-output must be new")?;
        let targets: Vec<OutputTargets> = pixels.iter().map(|p| OutputTargets {
            smet: out.join(format!("{}.smet", p.key)),
            pro: out.join(format!("{}.pro", p.key)),
        }).collect();
        Some(runner::run_hourly_streaming(&gpu.gpu, &inputs, NMAX, &options, &targets)?)
    } else { None };
    let text = packed_result(&hourly, &packed, Some(&rows), Some(&tex), summaries.as_deref(), Some(gpu.device_limits));
    std::fs::write(PathBuf::from(arg(v, "--out")), text.as_bytes())?;
    println!("{}", text.trim());
    Ok(())
}

fn packed_result(hourly: &[Hourly], packed: &PackedForcing,
                 rows: Option<&[Vec<ForcingRow>]>, tex: Option<&[f32]>,
                 summaries: Option<&[runner::ColumnSummary]>,
                 limits: Option<forcing_texture::TextureDeviceLimits>) -> String {
    let mut fields = String::new();
    let mut all_pass = false;
    if let (Some(rows), Some(tex)) = (rows, tex) {
        let columns = rows.len();
        let n_rows = rows[0].len();
        all_pass = true;
        for f in 0..6 {
            let mut maximum = 0.0f64;
            let mut max_hourly_term = 0.0f64;
            let mut violations = 0usize;
            for r in 0..n_rows {
                let h = (r / 6).min(HOURS - 1);
                for c in 0..columns {
                    let exact = row_field(&rows[c][r], f);
                    let value = f64::from(tex[(r * 6 + f) * columns + c]);
                    let left = value_at(&hourly[c], h, f);
                    let right = value_at(&hourly[c], (h + 1).min(HOURS - 1), f);
                    let hourly_term = (right - left).abs() / 256.0;
                    max_hourly_term = max_hourly_term.max(hourly_term);
                    let error = (value - exact).abs();
                    maximum = maximum.max(error);
                    if error > hourly_term + forcing_texture::SCALE[f] { violations += 1; }
                }
            }
            let pass = violations == 0;
            all_pass &= pass;
            if f > 0 { fields.push(','); }
            fields.push_str(&format!("\"{}\":{{\"samples\":{},\"maximum_abs_error\":{},\"maximum_hourly_delta_over_256\":{},\"packing_step\":{},\"violations\":{},\"pass\":{}}}",
                forcing_texture::FIELDS[f], n_rows * columns, maximum, max_hourly_term,
                forcing_texture::SCALE[f], violations, pass));
        }
    }
    let mut psum_error = 0.0f64;
    if let Some(rows) = rows {
        for (col, rs) in rows.iter().enumerate() {
            let expanded: f64 = rs.iter().map(|r| r.psum.unwrap_or(0.0)).sum();
            let source: f64 = hourly[col].psum.iter().sum();
            psum_error = psum_error.max((expanded - source).abs());
        }
    }
    let packing = forcing_texture::FIELDS.iter().enumerate().map(|(i,n)| format!("\"{n}\":{}", packed.max_pack_error[i])).collect::<Vec<_>>().join(",");
    let physics = summaries.map(|s| s.iter().zip(G1C_KEYS).map(|(x,key)| format!(
        "{{\"key\":\"{key}\",\"steps\":{},\"max_elements\":{},\"failed\":{},\"daily_rows\":{}}}",
        x.steps, x.max_elements, x.failure.is_some(), x.daily.len())).collect::<Vec<_>>().join(","));
    let physics_json = physics.unwrap_or_default();
    let dispatched = summaries.is_some();
    let device_limits = limits.map(|x| format!("{{\"max_2d_width\":{},\"max_2d_height\":{},\"max_2d_layered_width\":{},\"max_2d_layered_height\":{},\"max_2d_layers\":{}}}", x.width_2d, x.height_2d, x.width_2d_layered, x.height_2d_layered, x.layers_2d)).unwrap_or_else(|| "null".into());
    format!("{{\"arm\":\"3.2 forcing-int16-texture\",\"state\":\"scored\",\"columns\":{},\"hours\":{},\"texture_extent\":{{\"width_elements\":{},\"height_rows\":{},\"row_bytes\":{},\"source_pitch_bytes\":{}}},\"device_texture_limits\":{},\"texture_extent_check\":\"{}\",\"g1c_forcing_pass\":{},\"fields\":{{{}}},\"packing_max_abs_error\":{{{}}},\"hourly_texture_bytes\":{},\"psum\":{{\"kept_out_of_texture\":true,\"host_six_way_sum_max_abs_error\":{}}},\"physics_dispatched\":{},\"physics_dispatch\":\"{}\",\"physics_columns\":[{}],\"registered_bound\":\"abs(tex-exact) <= abs(v[h+1]-v[h])/256 + packing_step[field]\"}}\n",
        packed.columns, packed.hours, packed.columns, packed.hours,
        packed.columns * std::mem::size_of::<i16>(), (packed.columns * std::mem::size_of::<i16>() + 3) & !3,
        device_limits, if limits.is_some() { "passed" } else { "pending device query" },
        if tex.is_some() { all_pass.to_string() } else { "null".into() }, fields, packing,
        packed.fields.iter().map(|x| x.len()*2).sum::<usize>(), psum_error, dispatched,
        if dispatched { "pi_snow_advance with FT32 hourly textures" } else if tex.is_some() && !all_pass { "skipped after G1c forcing failure" } else { "pending" }, physics_json)
}

fn value_at(h: &Hourly, i: usize, f: usize) -> f64 {
    match f {
        0 => h.ta[i], 1 => h.rh[i],
        2 => crate::forcing::vector_to_wind(h.u[i], h.v[i]).0,
        3 => crate::forcing::vector_to_wind(h.u[i], h.v[i]).1,
        4 => h.iswr[i], _ => h.ilwr[i],
    }
}

fn row_field(r: &ForcingRow, f: usize) -> f64 {
    [r.ta, r.rh, r.vw, r.dw, r.iswr, r.ilwr][f].unwrap_or(f64::NAN)
}
