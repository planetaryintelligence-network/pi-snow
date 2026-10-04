//! Int16 hourly forcing textures for the isolated 3.2 kernel arm.
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaStream, PushKernelArg},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};

use crate::{forcing::Hourly, gpu::{Build, Gpu, Precision}, sha256::sha256_hex};

pub const KERNEL_SOURCE: &str = include_str!("../kernels/pi_snow_32.cu");
pub const FIELDS: [&str; 6] = ["TA", "RH", "VW", "DW", "ISWR", "ILWR"];
pub const SCALE: [f64; 6] = [0.01, 0.0005, 0.01, 0.1, 0.1, 0.05];
pub const OFFSET: [f64; 6] = [273.15, 0.5, 0.0, 0.0, 0.0, 0.0];

pub fn compile_nvrtc() -> Result<String> {
    let options = vec![
        "--fmad=false".to_string(), "--prec-div=true".to_string(),
        "--prec-sqrt=true".to_string(), "--ftz=false".to_string(),
        "-DPI_FT32".to_string(),
    ];
    let ptx = compile_ptx_with_opts(KERNEL_SOURCE, CompileOptions {
        arch: Some("sm_120"), options, ..Default::default()
    }).map_err(|e| anyhow::anyhow!("NVRTC forcing texture: {e:?}"))?;
    Ok(sha256_hex(ptx.to_src().as_bytes()))
}

pub struct PackedForcing {
    pub hours: usize,
    pub columns: usize,
    /// One texture payload per field, each row-major `[hour][column]`.
    pub fields: [Vec<i16>; 6],
    pub max_pack_error: [f64; 6],
}

impl PackedForcing {
    pub fn new(hourly: &[Hourly]) -> Result<Self> {
        if hourly.is_empty() { bail!("texture forcing has no columns"); }
        let columns = hourly.len();
        let hours = hourly[0].ta.len();
        if hourly.iter().any(|h| h.ta.len() != hours || h.rh.len() != hours || h.iswr.len() != hours
            || h.ilwr.len() != hours || h.u.len() != hours || h.v.len() != hours) {
            bail!("hourly texture columns have inconsistent lengths");
        }
        let mut fields: [Vec<i16>; 6] = std::array::from_fn(|_| Vec::with_capacity(hours * columns));
        let mut max_pack_error = [0.0f64; 6];
        for hour in 0..hours {
            for (column, h) in hourly.iter().enumerate() {
                let (vw, dw) = crate::forcing::vector_to_wind(h.u[hour], h.v[hour]);
                let values = [h.ta[hour], h.rh[hour], vw, dw, h.iswr[hour], h.ilwr[hour]];
                for field in 0..6 {
                    let qf = (values[field] - OFFSET[field]) / SCALE[field];
                    if !qf.is_finite() || qf < -32767.0 || qf > 32767.0 {
                        bail!("{} column {column}, hour {hour}: packed value outside the safe int16 range: {qf}", FIELDS[field]);
                    }
                    let q = qf.round() as i16;
                    let reconstructed = OFFSET[field] + SCALE[field] * f64::from(q);
                    max_pack_error[field] = max_pack_error[field].max((reconstructed - values[field]).abs());
                    fields[field].push(q);
                }
            }
        }
        Ok(Self { hours, columns, fields, max_pack_error })
    }
}

pub struct TextureGpu {
    pub gpu: Gpu,
    pub probe: CudaFunction,
    textures: Vec<cudarc::driver::sys::CUtexObject>,
    arrays: Vec<cudarc::driver::sys::CUarray>,
}

impl TextureGpu {
    pub fn new(packed: &PackedForcing) -> Result<Self> {
        let ctx = CudaContext::new(0).context("CUDA device 0")?;
        let mut arrays = Vec::with_capacity(6);
        let mut textures = Vec::with_capacity(6);
        for payload in &packed.fields {
            let desc = cudarc::driver::sys::CUDA_ARRAY_DESCRIPTOR {
                Width: packed.columns,
                Height: packed.hours,
                Format: cudarc::driver::sys::CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT16,
                NumChannels: 1,
            };
            let mut array = std::ptr::null_mut();
            // SAFETY: descriptor contains a 2D, one-channel signed-int16 array.
            let status = unsafe { cudarc::driver::sys::cuArrayCreate_v2(&mut array, &desc) };
            check_driver(status, "cuArrayCreate")?;
            for hour in 0..packed.hours {
                let row = &payload[hour * packed.columns..(hour + 1) * packed.columns];
                // SAFETY: row is live for this synchronous copy and exactly one array row.
                let status = unsafe {
                    cudarc::driver::sys::cuMemcpyHtoA_v2(
                        array,
                        hour * packed.columns * std::mem::size_of::<i16>(),
                        row.as_ptr().cast(),
                        row.len() * std::mem::size_of::<i16>(),
                    )
                };
                if let Err(e) = check_driver(status, "cuMemcpyHtoA") {
                    unsafe { cudarc::driver::sys::cuArrayDestroy(array); }
                    return Err(e);
                }
            }
            let mut resource = cudarc::driver::sys::CUDA_RESOURCE_DESC {
                resType: cudarc::driver::sys::CUresourcetype_enum::CU_RESOURCE_TYPE_ARRAY,
                res: unsafe { std::mem::zeroed() },
                flags: 0,
            };
            resource.res.array.hArray = array;
            let tex_desc = cudarc::driver::sys::CUDA_TEXTURE_DESC {
                addressMode: [
                    cudarc::driver::sys::CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP,
                    cudarc::driver::sys::CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP,
                    cudarc::driver::sys::CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP,
                ],
                filterMode: cudarc::driver::sys::CUfilter_mode_enum::CU_TR_FILTER_MODE_LINEAR,
                flags: 0, // signed int16 is read as normalized float for hardware filtering
                maxAnisotropy: 1,
                mipmapFilterMode: cudarc::driver::sys::CUfilter_mode_enum::CU_TR_FILTER_MODE_POINT,
                mipmapLevelBias: 0.0,
                minMipmapLevelClamp: 0.0,
                maxMipmapLevelClamp: 0.0,
                borderColor: [0.0; 4],
                reserved: [0; 12],
            };
            let mut texture = 0;
            // SAFETY: resource and texture descriptors are initialized and outlive creation.
            let status = unsafe {
                cudarc::driver::sys::cuTexObjectCreate(
                    &mut texture, &resource, &tex_desc, std::ptr::null(),
                )
            };
            if let Err(e) = check_driver(status, "cuTexObjectCreate") {
                unsafe { cudarc::driver::sys::cuArrayDestroy(array); }
                return Err(e);
            }
            arrays.push(array);
            textures.push(texture);
        }

        let arch = "sm_120";
        let options = vec![
            "--fmad=false".to_string(), "--prec-div=true".to_string(),
            "--prec-sqrt=true".to_string(), "--ftz=false".to_string(),
            "-DPI_FT32".to_string(),
        ];
        let ptx = match compile_ptx_with_opts(KERNEL_SOURCE, CompileOptions {
            arch: Some(arch), options: options.clone(), ..Default::default()
        }) {
            Ok(ptx) => ptx,
            Err(e) => {
                cleanup(&textures, &arrays);
                return Err(anyhow::anyhow!("NVRTC forcing texture: {e:?}"));
            }
        };
        let ptx_sha256 = sha256_hex(ptx.to_src().as_bytes());
        let module = ctx.load_module(ptx)?;
        let stream = ctx.default_stream();
        upload_global(&module, &stream, "FT_TEX", &textures.iter().map(|&x| x as u64).collect::<Vec<_>>())?;
        upload_global(&module, &stream, "FT_SCALE", &SCALE)?;
        upload_global(&module, &stream, "FT_OFFSET", &OFFSET)?;
        upload_global(&module, &stream, "FT_START_SECONDS", &[crate::forcing::season_start().timestamp()])?;
        stream.synchronize()?;
        let advance = module.load_function("pi_snow_advance")?;
        let probe = module.load_function("pi_snow_forcing_probe")?;
        let (mut major, mut minor, mut driver_version) = (0, 0, 0);
        // SAFETY: all are initialized out-parameters for CUDA version queries.
        unsafe {
            cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor);
            cudarc::driver::sys::cuDriverGetVersion(&mut driver_version);
        }
        Ok(Self {
            gpu: Gpu {
                ctx, stream, advance,
                build: Build { precision: Precision::F64, nvrtc_version: (major, minor),
                    driver_version, arch, options, source_sha256: sha256_hex(KERNEL_SOURCE.as_bytes()), ptx_sha256 },
            },
            probe, textures, arrays,
        })
    }

    pub fn probe_rows(&self, rows: usize, columns: usize) -> Result<Vec<f32>> {
        let count = rows.checked_mul(6).and_then(|n| n.checked_mul(columns)).context("probe size overflow")?;
        let mut output = self.gpu.stream.alloc_zeros::<f32>(count)?;
        let rows_i = i32::try_from(rows)?;
        let columns_i = i32::try_from(columns)?;
        let block = 128u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((count as u32).div_ceil(block), 1, 1),
            block_dim: (block, 1, 1), shared_mem_bytes: 0,
        };
        let mut launch = self.gpu.stream.launch_builder(&self.probe);
        launch.arg(&mut output).arg(&rows_i).arg(&columns_i);
        // SAFETY: output has `rows*6*columns` elements and the launch bounds are exact.
        unsafe { launch.launch(cfg)?; }
        self.gpu.stream.synchronize()?;
        Ok(self.gpu.stream.clone_dtoh(&output)?)
    }
}

impl Drop for TextureGpu {
    fn drop(&mut self) { cleanup(&self.textures, &self.arrays); }
}

fn upload_global<T: cudarc::driver::DeviceRepr + Copy>(module: &Arc<cudarc::driver::CudaModule>, stream: &Arc<CudaStream>, name: &str, src: &[T]) -> Result<()> {
    let mut global = module.get_global(name, stream)?;
    let mut dst = unsafe { global.transmute_mut::<T>(src.len()) }.context("texture global size")?;
    stream.memcpy_htod(src, &mut dst)?;
    Ok(())
}

fn check_driver(status: cudarc::driver::sys::CUresult, what: &str) -> Result<()> {
    if status == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS { Ok(()) }
    else { bail!("{what} failed with CUDA driver result {status:?}") }
}

fn cleanup(textures: &[cudarc::driver::sys::CUtexObject], arrays: &[cudarc::driver::sys::CUarray]) {
    for &texture in textures { unsafe { cudarc::driver::sys::cuTexObjectDestroy(texture); } }
    for &array in arrays { unsafe { cudarc::driver::sys::cuArrayDestroy(array); } }
}
