//! Registered vector forcing texture probe and device-free runtime compiler.
use std::{fs::{File, OpenOptions}, io::Write, path::Path};
use anyhow::{Context, Result, bail};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use pi_snow_gpu_alternatives_v2::{
    forcing::{self, Hourly, HOURS}, forcing_texture::{self, PackedForcing, TextureGpu},
    gpu, runner, sha256::sha256_hex, stepper::ForcingRow,
};

const KEYS: [&str; 3] = ["px_166_353", "px_324_306", "px_462_47"];
const FIELDS: [&str; 6] = ["TA", "RH", "VW", "DW", "ISWR", "ILWR"];
const STEPS: [f64; 6] = [0.01, 0.0005, 0.01, 0.1, 0.1, 0.05];

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("nvrtc") if args.len() == 1 => nvrtc(),
        Some(command @ ("preflight" | "g1c")) if args.len() == 3 => {
            let out = Path::new(&args[2]);
            if out.exists() { bail!("refusing output overwrite {}", out.display()); }
            let pixels = forcing::read_pixels(&Path::new(&args[1]).join("pixels.tsv"))?;
            let mut file = File::open(Path::new(&args[1]).join("forcing.f32"))?;
            let hourly = KEYS.iter().map(|key| {
                let p = pixels.iter().find(|p| p.key == *key).context("registered pixel absent")?;
                forcing::hourly_forcing(&forcing::read_hourly(&mut file, p.index)?)
            }).collect::<Result<Vec<_>>>()?;
            let packed = PackedForcing::new(&hourly)?;
            let rows: Vec<Vec<ForcingRow>> = hourly.iter().map(|h| forcing::expand(h, 1, HOURS-1)).collect();
            let count = rows[0].len();
            if count != 26059 || rows.iter().any(|r| r.len() != count) { bail!("probe population changed"); }
            let refs: Vec<&[ForcingRow]> = rows.iter().map(Vec::as_slice).collect();
            let (_, fv, _) = runner::pack_rows(&refs);
            let mut psum_residual = 0.0_f64;
            for (c, rs) in rows.iter().enumerate() {
                for (row, r) in rs.iter().enumerate() {
                    if fv[(row*gpu::F_COUNT+6)*rows.len()+c].to_bits() != r.psum.unwrap().to_bits() {
                        bail!("PSUM element identity failed at {c}/{row}");
                    }
                }
                psum_residual = psum_residual.max((rs.iter().map(|r| r.psum.unwrap()).sum::<f64>()
                    - hourly[c].psum.iter().sum::<f64>()).abs());
            }
            let packing = packed.max_pack_error.iter().map(ToString::to_string).collect::<Vec<_>>().join(",");
            let common = format!("\"arm\":\"3.2b\",\"keys\":[\"{}\"],\"rows_per_pixel\":{count},\"hours\":{HOURS},\"hourly_texture_bytes\":{},\"packed_fields\":[\"TA\",\"RH\",\"U\",\"V\",\"ISWR\",\"ILWR\"],\"max_pack_error\":[{packing}],\"psum_element_identity\":true,\"psum_seasonal_sum_max_abs_residual\":{psum_residual}", KEYS.join("\",\""), packed.fields.iter().map(|f| f.len()*2).sum::<usize>());
            let text = if command == "preflight" {
                format!("{{{common},\"state\":\"host-preflight\",\"gpu_created\":false}}\n")
            } else {
                let gpu = TextureGpu::new(&packed)?;
                let probe = gpu.probe_rows(count, rows.len())?;
                let (fields, pass, counts) = score(&hourly, &rows, &probe)?;
                let limits = gpu.device_limits;
                // Retain the measured probe for subsequent diagnostics/animation.
                let probe_path = out.with_extension("probe.f32");
                let mut f = OpenOptions::new().write(true).create_new(true).open(&probe_path)?;
                for v in &probe { f.write_all(&v.to_le_bytes())?; }
                format!("{{{common},\"state\":\"scored\",\"gate_label\":\"3.2b G1c with the U/V-derived DW term\",\"pass\":{pass},\"decision\":\"3.2b {} G1c with the U/V-derived DW term\",\"fields\":{{{fields}}},\"wind_categories\":{{{counts}}},\"physics_dispatched\":false,\"engine\":{},\"device_texture_limits\":{{\"width\":{},\"height\":{},\"layered_width\":{},\"layered_height\":{},\"layers\":{}}}}}\n", if pass {"passes"} else {"fails"}, gpu.gpu.build.to_json(), limits.width_2d, limits.height_2d, limits.width_2d_layered, limits.height_2d_layered, limits.layers_2d)
            };
            let mut f = OpenOptions::new().write(true).create_new(true).open(out)?;
            f.write_all(text.as_bytes())?;
            println!("{}", text.trim());
            Ok(())
        }
        _ => bail!("usage: pi-snow-gpu-32b nvrtc | preflight|g1c FORCING_DIR NEW_RESULT_JSON"),
    }
}

fn compile(source: &str, options: Vec<String>) -> Result<String> {
    let ptx = compile_ptx_with_opts(source, CompileOptions {
        arch: Some("sm_120"), options, ..Default::default()
    }).map_err(|e| anyhow::anyhow!("NVRTC: {e:?}"))?;
    Ok(sha256_hex(ptx.to_src().as_bytes()))
}

fn nvrtc() -> Result<()> {
    let flags: Vec<String> = ["--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false"].iter().map(|s| s.to_string()).collect();
    let baseline = compile(gpu::KERNEL_SOURCE, flags.clone())?;
    let disabled = compile(forcing_texture::KERNEL_SOURCE, flags.clone())?;
    if baseline != disabled { bail!("candidate-disabled F64 PTX identity failed: {baseline} != {disabled}"); }
    let candidate = forcing_texture::compile_nvrtc()?;
    let mut version = (0,0);
    let status = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut version.0, &mut version.1) };
    if status != cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS { bail!("NVRTC version query failed"); }
    println!("{{\"default_f64_ptx_sha256\":\"{baseline}\",\"candidate_disabled_ptx_sha256\":\"{disabled}\",\"candidate_ptx_sha256\":\"{candidate}\",\"identity_pass\":true,\"arch\":\"sm_120\",\"nvrtc_version\":\"{}.{}\",\"default_options\":[\"{}\"],\"candidate_define\":\"PI_FT32\",\"device_created\":false}}", version.0,version.1,flags.join("\",\""));
    Ok(())
}

fn value(h: &Hourly, hour: usize, field: usize) -> f64 {
    match field { 0 => h.ta[hour], 1 => h.rh[hour], 2 => forcing::vector_to_wind(h.u[hour],h.v[hour]).0,
        3 => forcing::vector_to_wind(h.u[hour],h.v[hour]).1, 4 => h.iswr[hour], _ => h.ilwr[hour] }
}

fn score(hourly: &[Hourly], rows: &[Vec<ForcingRow>], probe: &[f32]) -> Result<(String, bool, String)> {
    let (n,b) = (rows[0].len(), rows.len());
    let mut all = true;
    let mut fields = Vec::new();
    let (mut within_radius,mut within_eps,mut zero,mut packed_zero,mut north) = (0usize,0usize,0usize,0usize,0usize);
    for field in 0..6 {
        let (mut violations,mut legacy_violations) = (0usize,0usize);
        let (mut maximum,mut legacy_max,mut max_bound,mut max_excess) = (0.0_f64,0.0_f64,0.0_f64,f64::NEG_INFINITY);
        for (c,rs) in rows.iter().enumerate() {
            for (row,r) in rs.iter().enumerate() {
                let exact = [r.ta,r.rh,r.vw,r.dw,r.iswr,r.ilwr][field].unwrap();
                let tex = f64::from(probe[(row*6+field)*b+c]);
                if !exact.is_finite() || !tex.is_finite() { bail!("nonfinite probe {c}/{row}/{field}"); }
                // Preserve v1's row-to-hour indexing, including exact hour boundaries.
                let h = (row/6).min(HOURS-1);
                let next = (h+1).min(HOURS-1);
                let slack = (value(&hourly[c],next,field)-value(&hourly[c],h,field)).abs()/256.0;
                let ordinary = (tex-exact).abs();
                let (error,bound) = if field == 3 {
                    let speed = r.vw.unwrap();
                    let eps = 0.005*2.0_f64.sqrt();
                    let radius = eps + (hourly[c].u[next]-hourly[c].u[h]).hypot(hourly[c].v[next]-hourly[c].v[h])/256.0;
                    within_radius += usize::from(speed <= radius);
                    within_eps += usize::from(speed <= eps);
                    zero += usize::from(speed == 0.0);
                    packed_zero += usize::from(speed > 0.0 && probe[(row*6+2)*b+c] == 0.0);
                    north += usize::from(ordinary > 180.0);
                    legacy_violations += usize::from(ordinary > slack+0.1);
                    legacy_max = legacy_max.max(ordinary);
                    (ordinary.min(360.0-ordinary), if speed > radius {(radius/speed).min(1.0).asin().to_degrees()} else {180.0})
                } else {(ordinary,slack+STEPS[field])};
                violations += usize::from(error > bound);
                maximum = maximum.max(error);
                max_bound = max_bound.max(bound);
                max_excess = max_excess.max(error-bound);
            }
        }
        all &= violations == 0;
        let legacy = if field == 3 {format!(",\"v1_verbatim_descriptive\":{{\"violations\":{legacy_violations},\"maximum_abs_error\":{legacy_max},\"packing_step_degrees\":0.1}}")} else {String::new()};
        fields.push(format!("\"{}\":{{\"samples\":{},\"maximum_error\":{maximum},\"maximum_bound\":{max_bound},\"maximum_error_minus_bound\":{max_excess},\"violations\":{violations},\"pass\":{}{legacy}}}",FIELDS[field],n*b,violations==0));
    }
    Ok((fields.join(","),all,format!("\"at_or_below_radius\":{within_radius},\"at_or_below_epsilon\":{within_eps},\"exact_host_zero\":{zero},\"quantization_to_zero\":{packed_zero},\"ordinary_error_above_180\":{north}")))
}
