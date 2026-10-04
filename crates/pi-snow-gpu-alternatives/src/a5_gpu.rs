//! A5-only CUDA module loader. The ordinary GPU module and kernel stay untouched.

use std::{fs, path::Path, sync::Arc};

use anyhow::{Context, Result, bail};
use cudarc::driver::{CudaContext, CudaStream};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

use pi_snow_gpu_alternatives::{
    gpu::{Build, Gpu, Precision},
    sha256::sha256_hex,
};

pub const KERNEL_SOURCE: &str = include_str!("../kernels/pi_snow_a5.cu");
const TABLE_LEN: usize = 1026 * 32 * 32 * 2;
const AXES_LEN: usize = 1026 + 32 + 32;

fn read_f64(path: &Path, count: usize) -> Result<Vec<f64>> {
    let bytes = fs::read(path).with_context(|| path.display().to_string())?;
    if bytes.len() != count * 8 {
        bail!(
            "{}: expected {} f64 values, got {} bytes",
            path.display(),
            count,
            bytes.len()
        );
    }
    let values: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().expect("eight-byte chunk")))
        .collect();
    if values.iter().any(|x| !x.is_finite()) {
        bail!("{} contains a non-finite f64", path.display());
    }
    Ok(values)
}

pub fn new_f64(table_path: &Path, axes_path: &Path) -> Result<Gpu> {
    let table = read_f64(table_path, TABLE_LEN)?;
    let axes = read_f64(axes_path, AXES_LEN)?;
    for axis in [&axes[..1026], &axes[1026..1058], &axes[1058..]] {
        if axis.windows(2).any(|w| w[0] >= w[1]) {
            bail!("A5 coordinate axes must be strictly increasing");
        }
    }

    let ctx = CudaContext::new(0).context("CUDA device 0")?;
    let arch = "sm_120";
    let options = vec![
        "--fmad=false".to_string(),
        "--prec-div=true".to_string(),
        "--prec-sqrt=true".to_string(),
        "--ftz=false".to_string(),
        "-DPI_A5".to_string(),
    ];
    let ptx = compile_ptx_with_opts(
        KERNEL_SOURCE,
        CompileOptions {
            arch: Some(arch),
            options: options.clone(),
            ..Default::default()
        },
    )
    .map_err(|e| anyhow::anyhow!("NVRTC A5: {e:?}"))?;
    let ptx_sha256 = sha256_hex(ptx.to_src().as_bytes());
    let module = ctx.load_module(ptx)?;
    let stream = ctx.default_stream();
    upload_global(&module, &stream, "A5_TABLE", &table)?;
    upload_global(&module, &stream, "A5_RI", &axes[..1026])?;
    upload_global(&module, &stream, "A5_ZW", &axes[1026..1058])?;
    upload_global(&module, &stream, "A5_ZS", &axes[1058..])?;
    stream.synchronize()?;
    let advance = module.load_function("pi_snow_advance")?;
    let (mut major, mut minor, mut driver_version) = (0i32, 0i32, 0i32);
    unsafe {
        cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor);
        cudarc::driver::sys::cuDriverGetVersion(&mut driver_version);
    }
    Ok(Gpu {
        ctx,
        stream,
        advance,
        build: Build {
            precision: Precision::F64,
            nvrtc_version: (major, minor),
            driver_version,
            arch,
            options,
            source_sha256: sha256_hex(KERNEL_SOURCE.as_bytes()),
            ptx_sha256,
        },
    })
}

fn upload_global(
    module: &Arc<cudarc::driver::CudaModule>,
    stream: &Arc<CudaStream>,
    name: &str,
    src: &[f64],
) -> Result<()> {
    let mut global = module.get_global(name, stream)?;
    let mut dst = unsafe { global.transmute_mut::<f64>(src.len()) }.context("A5 global size")?;
    stream.memcpy_htod(src, &mut dst)?;
    Ok(())
}
