//! Compile candidate and control PTX without creating a CUDA context.
use anyhow::{Result, bail};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
fn main() -> Result<()> {
    let arg = std::env::args().nth(1).unwrap_or_default();
    let source = match arg.as_str() {
        "F64" | "F32" => include_str!("../../../../crates/pi-snow-gpu/kernels/pi_snow.cu"),
        "A5" => include_str!("../../kernels/pi_snow_a5.cu"),
        "A6" => include_str!("../../kernels/pi_snow_a6.cu"),
        "A6-audit" => include_str!("../../kernels/pi_snow_a6_audit.cu"),
        "A1A2" => include_str!("../../kernels/pi_snow_a1a2.cu"),
        _ => bail!("expected F64|F32|A5|A6|A6-audit|A1A2"),
    };
    let f32 = arg == "F32" || arg == "A1A2";
    let mut options = vec![format!("--fmad={f32}"), "--prec-div=true".into(), "--prec-sqrt=true".into(), "--ftz=false".into()];
    if f32 { options.push("-DPI_F32".into()); }
    if arg == "A1A2" { options.push("-DPI_COMPACT".into()); }
    if arg == "A5" { options.push("-DPI_A5".into()); }
    let ptx = compile_ptx_with_opts(source, CompileOptions { arch: Some("sm_120"), options, ..Default::default() })
        .map_err(|e| anyhow::anyhow!("NVRTC: {e:?}"))?.to_src();
    println!("{{\"arm\":\"{}\",\"kernel_source_sha256\":\"{}\",\"ptx_sha256\":\"{}\"}}",arg,
        pi_snow_gpu_alternatives::sha256::sha256_hex(source.as_bytes()), pi_snow_gpu_alternatives::sha256::sha256_hex(ptx.as_bytes()));
    Ok(())
}
