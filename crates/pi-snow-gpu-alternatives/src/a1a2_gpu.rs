//! Compile the kernel and drive batches of columns on the GPU.
//!
//! The kernel (`kernels/pi_snow.cu`) is embedded in the binary and compiled by NVRTC at
//! start-up with every option given explicitly; the NVRTC version, the options and the
//! sha256 of the PTX are reported with every run, because a different compiler is a
//! different engine (plan.md section 3.1).

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use pi_snow_core::constants::GRAVITY;
use pi_snow_core::{Column, Element};
use pi_snow_io::ini::RunConfig;
use pi_snow_physics::atmosphere::{CP_AIR, EPSILON, R_DRY};
use pi_snow_physics::new_snow::DensityScheme;
use pi_snow_physics::surface::SNOW_EMISSIVITY;
use pi_snow_physics::turbulence::Scheme;

use crate::sha256::sha256_hex;
use crate::stepper::{Accum, Constants, LastForcing, Site};

pub const KERNEL_SOURCE: &str = include_str!("../kernels/pi_snow_a1a2.cu");

/// Forcing fields per row, in the kernel's `F_*` order.
pub const F_COUNT: usize = 8;
pub const A_COUNT: usize = 20;
pub const L_COUNT: usize = 5;
pub const P_COUNT: usize = 11;
pub const K_COUNT: usize = 15;
pub const D_COUNT: usize = 6;
const K_T_CRAZY_MIN: usize = 7;
const K_T_CRAZY_MAX: usize = 8;
const L_TSG: usize = 4;
pub const ROW_ABSENT: i64 = i64::MIN;
pub const FAIL_CRAZY: i32 = 1;
pub const FAIL_OVERFLOW: i32 = 2;
pub const BLOCK: u32 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F64,
    Mix,
    F32,
    F32K,
}

impl Precision {
    pub fn label(self) -> &'static str {
        match self {
            Precision::F64 => "F64",
            Precision::Mix => "MIX",
            Precision::F32 => "F32",
            Precision::F32K => "F32K",
        }
    }

    fn is_f32(self) -> bool {
        matches!(self, Precision::F32 | Precision::F32K)
    }

    fn accumulator_is_f64(self) -> bool {
        self == Precision::F32K
    }
}

#[derive(Clone, Copy)]
enum DeviceFloat {
    F64,
    F32,
    U16,
    U8,
}

/// A typed CUDA allocation backed by bytes so each precision arm can share the pointer ABI.
struct DeviceValues {
    raw: CudaSlice<u8>,
    len: usize,
    kind: DeviceFloat,
}

impl DeviceValues {
    fn new(stream: &Arc<CudaStream>, len: usize, kind: DeviceFloat) -> Result<Self> {
        let width = match kind { DeviceFloat::F64 => 8, DeviceFloat::F32 => 4, DeviceFloat::U16 => 2, DeviceFloat::U8 => 1 };
        Ok(Self { raw: stream.alloc_zeros::<u8>(len * width)?, len, kind })
    }

    fn ptr(&self, stream: &CudaStream) -> u64 {
        self.raw.device_ptr(stream).0
    }

    fn byte_len(&self) -> usize {
        self.len * match self.kind { DeviceFloat::F64 => 8, DeviceFloat::F32 => 4, DeviceFloat::U16 => 2, DeviceFloat::U8 => 1 }
    }

    fn upload(&mut self, stream: &Arc<CudaStream>, source: &[f64], kelvin_to_celsius: bool) -> Result<()> {
        if source.len() > self.len {
            bail!("device upload has {} values, allocation has {}", source.len(), self.len);
        }
        match self.kind {
            DeviceFloat::F64 => {
                let values: Vec<f64> = source.iter().map(|&v| if kelvin_to_celsius { v - 273.15 } else { v }).collect();
                let bytes = unsafe {
                    std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 8)
                };
                stream.memcpy_htod(bytes, &mut self.raw.slice_mut(0..bytes.len()))?;
            }
            DeviceFloat::F32 => {
                // Convert K to C in f64 before the cast, as registered for A1.
                let values: Vec<f32> = source.iter().map(|&v| {
                    (if kelvin_to_celsius { v - 273.15 } else { v }) as f32
                }).collect();
                let bytes = unsafe {
                    std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4)
                };
                stream.memcpy_htod(bytes, &mut self.raw.slice_mut(0..bytes.len()))?;
            }
            DeviceFloat::U16 => {
                let values: Vec<u16> = source.iter().map(|&v| v.clamp(0.0, 65535.0) as u16).collect();
                let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 2) };
                stream.memcpy_htod(bytes, &mut self.raw.slice_mut(0..bytes.len()))?;
            }
            DeviceFloat::U8 => {
                let values: Vec<u8> = source.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect();
                stream.memcpy_htod(&values, &mut self.raw.slice_mut(0..values.len()))?;
            }
        }
        Ok(())
    }

    fn download(&self, stream: &Arc<CudaStream>, target: &mut [f64], celsius_to_kelvin: bool) -> Result<()> {
        if target.len() > self.len {
            bail!("device download requested {} values, allocation has {}", target.len(), self.len);
        }
        let width = match self.kind { DeviceFloat::F64 => 8, DeviceFloat::F32 => 4, DeviceFloat::U16 => 2, DeviceFloat::U8 => 1 };
        let bytes = stream.clone_dtoh(&self.raw.slice(0..target.len() * width))?;
        match self.kind {
            DeviceFloat::F64 => for (i, chunk) in bytes.chunks_exact(8).enumerate() {
                let mut value = f64::from_ne_bytes(chunk.try_into().unwrap());
                if celsius_to_kelvin { value += 273.15; }
                target[i] = value;
            },
            DeviceFloat::F32 => for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                let mut value = f32::from_ne_bytes(chunk.try_into().unwrap()) as f64;
                if celsius_to_kelvin { value += 273.15; }
                target[i] = value;
            },
            DeviceFloat::U16 => for (i, chunk) in bytes.chunks_exact(2).enumerate() { target[i] = u16::from_ne_bytes(chunk.try_into().unwrap()) as f64; },
            DeviceFloat::U8 => for (i, byte) in bytes.iter().enumerate() { target[i] = *byte as f64; },
        }
        Ok(())
    }
}

/// What compiled the kernel, for every report.
#[derive(Debug, Clone)]
pub struct Build {
    pub precision: Precision,
    pub nvrtc_version: (i32, i32),
    pub driver_version: i32,
    pub arch: &'static str,
    pub options: Vec<String>,
    pub source_sha256: String,
    pub ptx_sha256: String,
}

impl Build {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"precision\": \"{}\", \"nvrtc_version\": \"{}.{}\", \"driver_version\": {}, \"arch\": \"{}\", \"options\": [{}], \"kernel_source_sha256\": \"{}\", \"ptx_sha256\": \"{}\"}}",
            self.precision.label(),
            self.nvrtc_version.0,
            self.nvrtc_version.1,
            self.driver_version,
            self.arch,
            self.options
                .iter()
                .map(|o| format!("\"{o}\""))
                .collect::<Vec<_>>()
                .join(", "),
            self.source_sha256,
            self.ptx_sha256
        )
    }
}

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub advance: CudaFunction,
    pub build: Build,
}

impl Gpu {
    pub fn new(precision: Precision) -> Result<Self> {
        let ctx = CudaContext::new(0).context("CUDA device 0")?;
        let arch = "sm_120";
        let mut options: Vec<String> = [
            "--fmad=false",
            "--prec-div=true",
            "--prec-sqrt=true",
            "--ftz=false",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        match precision {
            Precision::Mix => options.push("-DPI_MIX".into()),
            Precision::F32 => {
                options[0] = "--fmad=true".into();
                options.push("-DPI_F32".into());
                options.push("-DPI_COMPACT".into());
            }
            Precision::F32K => {
                options[0] = "--fmad=true".into();
                options.push("-DPI_F32K".into());
            }
            Precision::F64 => {}
        }
        let opts = CompileOptions {
            arch: Some(arch),
            options: options.clone(),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNEL_SOURCE, opts)
            .map_err(|e| anyhow::anyhow!("NVRTC: {e:?}"))?;
        let ptx_sha256 = sha256_hex(ptx.to_src().as_bytes());
        let module = ctx.load_module(ptx)?;
        let advance = module.load_function("pi_snow_advance")?;
        let stream = ctx.default_stream();
        let (mut major, mut minor, mut driver_version) = (0i32, 0i32, 0i32);
        // SAFETY: plain out-parameter queries; the libraries are loaded above.
        unsafe {
            cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor);
            cudarc::driver::sys::cuDriverGetVersion(&mut driver_version);
        }
        let nvrtc_version = (major, minor);
        Ok(Self {
            ctx,
            stream,
            advance,
            build: Build {
                precision,
                nvrtc_version,
                driver_version,
                arch,
                options,
                source_sha256: sha256_hex(KERNEL_SOURCE.as_bytes()),
                ptx_sha256,
            },
        })
    }

    /// Free and total device memory, bytes.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        Ok(self.ctx.mem_get_info()?)
    }
}

/// The run constants the kernel reads as `k[K_*]`.
fn kernel_scheme_parameters(scheme: Scheme, density_scheme: DensityScheme) -> Result<(f64, f64)> {
    let stability = match scheme {
        Scheme::MoHoltslag => 0.0,
        Scheme::Louis => 1.0,
        Scheme::MoHoltslagScalarHeight => {
            bail!("stability scheme MoHoltslagScalarHeight is not ported to the GPU")
        }
        Scheme::MoHoltslagZetaLimited => {
            bail!("stability scheme MoHoltslagZetaLimited is not ported to the GPU")
        }
    };
    let density = match density_scheme {
        DensityScheme::Pahaut => 0.0,
        DensityScheme::Zwart => 1.0,
        DensityScheme::Zwart2007Eq51 => {
            bail!("density scheme Zwart2007Eq51 is not ported to the GPU")
        }
        DensityScheme::Anderson1976 => {
            bail!("density scheme Anderson1976 is not ported to the GPU")
        }
    };
    Ok((stability, density))
}

pub fn run_constants(cfg: &RunConfig, constants: &Constants) -> Result<[f64; K_COUNT]> {
    let (stability, density) =
        kernel_scheme_parameters(constants.scheme, constants.density_scheme)?;
    Ok([
        cfg.thresh_rain_c,
        cfg.height_new_element,
        0.5e-3 * cfg.new_snow_grain_size_mm,
        cfg.height_of_wind_value,
        cfg.height_of_meteo_values,
        cfg.roughness_length,
        cfg.minimum_element_length,
        cfg.t_crazy_min,
        cfg.t_crazy_max,
        stability,
        density,
        R_DRY,
        CP_AIR,
        EPSILON,
        SNOW_EMISSIVITY,
    ])
}

/// The per-column parameters `par[P_*]`, each computed with the expression the CPU
/// code evaluates, so the kernel receives the CPU's exact values.
pub fn column_params(site: &Site, constants: &Constants) -> [f64; P_COUNT] {
    let b = site.slope_angle.to_radians();
    let phi = site.latitude.to_radians();
    [
        site.latitude,
        site.longitude,
        site.slope_angle,
        site.slope_azimuth,
        site.soil_albedo,
        constants.pressure,
        b.cos(),
        b.sin(),
        phi.sin(),
        phi.cos(),
        GRAVITY * site.slope_angle.to_radians().cos().powi(2),
    ]
}

/// Refuse what the kernel does not carry (plan.md section 3.3).
pub fn check_supported(cfg: &RunConfig, constants: &Constants, start: &Column) -> Result<()> {
    kernel_scheme_parameters(constants.scheme, constants.density_scheme)?;
    if start.soil_elements != 0 {
        bail!("soil elements are not ported");
    }
    for e in &start.elements {
        if e.composition.soil != 0.0 {
            bail!("a nonzero soil fraction is not ported");
        }
        if e.stress_rate != 0.0 || e.metamorphism != 0.0 {
            bail!("nonzero stress_rate or metamorphism in the initial state is not ported");
        }
    }
    let _ = cfg;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_physics_schemes_are_refused() {
        for scheme in [Scheme::MoHoltslagScalarHeight, Scheme::MoHoltslagZetaLimited] {
            let error = kernel_scheme_parameters(scheme, DensityScheme::Pahaut).unwrap_err();
            assert!(error.to_string().contains(&format!("{scheme:?}")));
            assert!(error.to_string().contains("not ported to the GPU"));
        }

        let error = kernel_scheme_parameters(Scheme::MoHoltslag, DensityScheme::Anderson1976)
            .unwrap_err();
        assert!(error.to_string().contains("Anderson1976"));
        assert!(error.to_string().contains("not ported to the GPU"));
    }
}

/// Host copy of a batch's state, in the kernel's layout.
#[derive(Clone)]
pub struct HostState {
    pub b: usize,
    pub nmax: usize,
    pub elem: [Vec<f64>; 11],
    pub marker: Vec<u32>,
    pub t: Vec<f64>,
    pub n: Vec<i32>,
    pub fail: Vec<i32>,
    pub first_out: Vec<i32>,
    pub has_prev: Vec<i32>,
    pub out_ready: Vec<i32>,
    pub cursor: Vec<i32>,
    pub maxn: Vec<i32>,
    pub prev_t: Vec<i64>,
    pub steps: Vec<i64>,
    pub fail_t: Vec<i64>,
    pub column_key: Vec<u32>,
    pub buffer: Vec<f64>,
    pub acc: Vec<f64>,
    pub last: Vec<f64>,
    pub mo_passes: Vec<i32>,
    pub tolerance_flags: Vec<u32>,
    pub count_change_cause: Vec<u32>,
}

/// Element field order of `HostState::elem`, as the kernel's `Arrays`.
pub const ELEMENT_FIELDS: [&str; 11] = [
    "thickness",
    "ice",
    "water",
    "air",
    "dendricity",
    "sphericity",
    "grain_radius",
    "bond_radius",
    "coordination_number",
    "age_days",
    "stress",
];

impl HostState {
    pub fn new(b: usize, nmax: usize) -> Self {
        let e = || vec![0.0; nmax * b];
        Self {
            b,
            nmax,
            elem: [e(), e(), e(), e(), e(), e(), e(), e(), e(), e(), e()],
            marker: vec![0; nmax * b],
            t: vec![0.0; (nmax + 1) * b],
            n: vec![0; b],
            fail: vec![0; b],
            first_out: vec![1; b],
            has_prev: vec![0; b],
            out_ready: vec![0; b],
            cursor: vec![0; b],
            maxn: vec![0; b],
            prev_t: vec![0; b],
            steps: vec![0; b],
            fail_t: vec![0; b],
            column_key: (0..b as u32).collect(),
            buffer: vec![0.0; b],
            acc: vec![0.0; A_COUNT * b],
            last: vec![0.0; L_COUNT * b],
            mo_passes: vec![0; b],
            tolerance_flags: vec![0; b],
            count_change_cause: vec![0; b],
        }
    }

    /// Write a column's state into slot `c`.
    pub fn set_column(&mut self, c: usize, column: &Column) -> Result<()> {
        let n = column.elements.len();
        if n > self.nmax {
            bail!("column has {n} elements, more than N_MAX {}", self.nmax);
        }
        let b = self.b;
        for (i, e) in column.elements.iter().enumerate() {
            let v = element_values(e);
            for (f, x) in v.iter().enumerate() {
                self.elem[f][i * b + c] = *x;
            }
            self.marker[i * b + c] = e.micro.marker as u32;
        }
        for (j, t) in column.node_temperature.iter().enumerate() {
            self.t[j * b + c] = *t;
        }
        self.n[c] = n as i32;
        Ok(())
    }

    pub fn set_column_with_key(&mut self, c: usize, column: &Column, key: u32) -> Result<()> {
        self.set_column(c, column)?;
        self.column_key[c] = key;
        Ok(())
    }

    /// Rebuild slot `c` as a `Column` with the given slope (soil-free, as carried).
    pub fn column(&self, c: usize, slope_angle_deg: f64, slope_azimuth_deg: f64) -> Column {
        let b = self.b;
        let n = self.n[c] as usize;
        let mut col = Column::bare_ground(self.t[c], slope_angle_deg, slope_azimuth_deg);
        col.node_temperature.clear();
        for i in 0..n {
            let v: [f64; 11] = std::array::from_fn(|f| self.elem[f][i * b + c]);
            col.elements.push(element_from(&v, self.marker[i * b + c]));
        }
        for j in 0..=n {
            col.node_temperature.push(self.t[j * b + c]);
        }
        col
    }

    pub fn accum(&self, c: usize) -> Accum {
        let a: [f64; A_COUNT] = std::array::from_fn(|q| self.acc[q * self.b + c]);
        Accum::from_array(&a)
    }

    pub fn set_accum(&mut self, c: usize, acc: &Accum) {
        for (q, x) in acc.to_array().iter().enumerate() {
            self.acc[q * self.b + c] = *x;
        }
    }

    pub fn last(&self, c: usize) -> LastForcing {
        let l = |q: usize| self.last[q * self.b + c];
        LastForcing {
            ta: l(0),
            rh: l(1),
            vw: l(2),
            dw: l(3),
            tsg: l(4),
        }
    }

    pub fn set_last(&mut self, c: usize, last: &LastForcing) {
        let b = self.b;
        self.last[c] = last.ta;
        self.last[b + c] = last.rh;
        self.last[2 * b + c] = last.vw;
        self.last[3 * b + c] = last.dw;
        self.last[4 * b + c] = last.tsg;
    }
}

pub fn element_values(e: &Element) -> [f64; 11] {
    [
        e.thickness,
        e.composition.ice,
        e.composition.water,
        e.composition.air,
        e.micro.dendricity,
        e.micro.sphericity,
        e.micro.grain_radius,
        e.micro.bond_radius,
        e.micro.coordination_number,
        e.age_days,
        e.stress,
    ]
}

fn element_from(v: &[f64; 11], marker: u32) -> Element {
    Element {
        thickness: v[0],
        composition: pi_snow_core::Composition {
            ice: v[1],
            water: v[2],
            air: v[3],
            soil: 0.0,
        },
        micro: pi_snow_core::Microstructure {
            dendricity: v[4],
            sphericity: v[5],
            grain_radius: v[6],
            bond_radius: v[7],
            coordination_number: v[8],
            marker: marker as u16,
        },
        age_days: v[9],
        stress: v[10],
        stress_rate: 0.0,
        metamorphism: 0.0,
        soil_density: 0.0,
        soil_conductivity: 0.0,
        soil_heat_capacity: 0.0,
        grain_history: 0,
    }
}

/// Counter-based initial quantization; each element gets a stable key independent
/// of batch order, matching the kernel's `(column, step, field, element)` key.
fn compact_quantize(values: &[f64], columns: usize, keys: &[u32], scale: f64, field: u32) -> Vec<f64> {
    values.iter().enumerate().map(|(index, &value)| {
        if !value.is_finite() || value <= 0.0 { return 0.0; }
        let scaled = (value as f32) * (scale as f32);
        let floor = scaled.floor();
        let hash = |mut x: u32| { x ^= x >> 16; x = x.wrapping_mul(0x7feb_352d); x ^= x >> 15;
            x = x.wrapping_mul(0x846c_a68b); x ^= x >> 16; x };
        let x = hash(keys[index % columns] ^ hash(0) ^ hash((index / columns) as u32)
            ^ field.wrapping_mul(0x9e37_79b9));
        let uniform = (x as f32) * (1.0f32 / 4_294_967_296.0f32);
        f64::from((floor + if uniform < scaled - floor { 1.0 } else { 0.0 }).clamp(0.0, 65535.0))
    }).collect()
}

pub fn stable_column_key(station_id: &str) -> u32 {
    station_id.as_bytes().iter().fold(0x811c_9dc5u32, |h, b| (h ^ u32::from(*b)).wrapping_mul(0x0100_0193))
}

/// A batch of `b` columns resident on the device.
pub struct Batch {
    pub b: usize,
    pub nmax: usize,
    pub rows: usize,
    elem: Vec<DeviceValues>,
    marker: CudaSlice<u8>,
    t: DeviceValues,
    _scratch: Vec<DeviceValues>,
    ints: Vec<CudaSlice<i32>>,  // n, fail, first_out, has_prev, out_ready, cursor, maxn
    longs: Vec<CudaSlice<i64>>, // prev_t, steps, fail_t
    column_key: CudaSlice<u32>,
    buffer: DeviceValues,
    acc: DeviceValues,
    last: DeviceValues,
    par: DeviceValues,
    pub ft: CudaSlice<i64>,
    fv: DeviceValues,
    daily_values: DeviceValues,
    pub days: usize,
    mo_passes: CudaSlice<i32>,
    tolerance_flags: CudaSlice<u32>,
    count_change_cause: CudaSlice<u32>,
    arrays: CudaSlice<u64>,
    k: DeviceValues,
}

impl Batch {
    /// Allocate a batch with room for `rows` forcing rows and `days` daily slots.
    pub fn new(
        gpu: &Gpu,
        b: usize,
        nmax: usize,
        rows: usize,
        days: usize,
        k: &[f64; K_COUNT],
        par: &[f64],
    ) -> Result<Self> {
        let s = &gpu.stream;
        let ne = nmax * b;
        let nn = (nmax + 1) * b;
        let real_kind = if gpu.build.precision.is_f32() { DeviceFloat::F32 } else { DeviceFloat::F64 };
        let accum_kind = if gpu.build.precision.accumulator_is_f64() { DeviceFloat::F64 } else { real_kind };
        let elem = (0..11)
            .map(|field| DeviceValues::new(s, if field == 3 || field == 8 { 1 } else { ne }, match field { 4..=7 | 9 => DeviceFloat::U16, _ => real_kind }))
            .collect::<Result<Vec<_>, _>>()?;
        let marker = s.alloc_zeros::<u8>(ne)?;
        let t = DeviceValues::new(s, nn, real_kind)?;
        let scratch = (0..8)
            .map(|_| DeviceValues::new(s, nn, real_kind))
            .collect::<Result<Vec<_>, _>>()?;
        let ints = (0..7)
            .map(|_| s.alloc_zeros::<i32>(b))
            .collect::<Result<Vec<_>, _>>()?;
        let longs = (0..3)
            .map(|_| s.alloc_zeros::<i64>(b))
            .collect::<Result<Vec<_>, _>>()?;
        let column_key = s.alloc_zeros::<u32>(b)?;
        let buffer = DeviceValues::new(s, b, accum_kind)?;
        let acc = DeviceValues::new(s, A_COUNT * b, accum_kind)?;
        let last = DeviceValues::new(s, L_COUNT * b, real_kind)?;
        if par.len() != P_COUNT * b {
            bail!("column parameters: {} values for {} columns", par.len(), b);
        }
        let mut par_device = DeviceValues::new(s, par.len(), real_kind)?;
        par_device.upload(s, par, false)?;
        let ft = s.alloc_zeros::<i64>(rows.max(1) * b)?;
        let fv = DeviceValues::new(s, rows.max(1) * F_COUNT * b, real_kind)?;
        let daily_values = DeviceValues::new(s, days.max(1) * D_COUNT * b, real_kind)?;
        let mut constants = k.to_vec();
        if gpu.build.precision.is_f32() {
            constants[K_T_CRAZY_MIN] -= 273.15;
            constants[K_T_CRAZY_MAX] -= 273.15;
        }
        let mut k_device = DeviceValues::new(s, K_COUNT, real_kind)?;
        k_device.upload(s, &constants, false)?;
        let mo_passes = s.alloc_zeros::<i32>(b)?;
        let tolerance_flags = s.alloc_zeros::<u32>(b)?;
        let count_change_cause = s.alloc_zeros::<u32>(b)?;

        // The pointer table, in the order of the kernel's `struct Arrays`.
        let mut ptrs: Vec<u64> = Vec::new();
        {
            let mut push64 = |p: u64| ptrs.push(p);
            // Arrays compact ABI: thick, ice, water, dend/sph/gr/br, age,
            // stress, u8 marker, then node state and the common tail.
            for index in [0usize, 1, 2, 4, 5, 6, 7, 9, 10] { push64(elem[index].ptr(s)); }
            push64(marker.device_ptr(s).0);
            push64(t.ptr(s));
            for x in &scratch {
                push64(x.ptr(s));
            }
            for x in &ints {
                push64(x.device_ptr(s).0);
            }
            for x in &longs {
                push64(x.device_ptr(s).0);
            }
            push64(column_key.device_ptr(s).0);
            push64(buffer.ptr(s));
            push64(acc.ptr(s));
            push64(last.ptr(s));
            push64(par_device.ptr(s));
            push64(ft.device_ptr(s).0);
            push64(fv.ptr(s));
            push64(daily_values.ptr(s));
            push64(mo_passes.device_ptr(s).0);
            push64(tolerance_flags.device_ptr(s).0);
            push64(count_change_cause.device_ptr(s).0);
        }
        let arrays = s.clone_htod(&ptrs)?;
        Ok(Self {
            b,
            nmax,
            rows,
            elem,
            marker,
            t,
            _scratch: scratch,
            ints,
            longs,
            column_key,
            buffer,
            acc,
            last,
            par: par_device,
            ft,
            fv,
            daily_values,
            days,
            mo_passes,
            tolerance_flags,
            count_change_cause,
            arrays,
            k: k_device,
        })
    }

    /// Device bytes held by this batch.
    pub fn device_bytes(&self) -> usize {
        self.elem.iter().map(DeviceValues::byte_len).sum::<usize>() + self.marker.len()
            + self.t.byte_len() + self._scratch.iter().map(DeviceValues::byte_len).sum::<usize>()
            + self.ints.iter().map(|v| v.len() * 4).sum::<usize>()
            + self.longs.iter().map(|v| v.len() * 8).sum::<usize>()
            + self.column_key.len() * 4
            + self.buffer.byte_len() + self.acc.byte_len() + self.last.byte_len() + self.par.byte_len()
            + self.ft.len() * 8 + self.fv.byte_len() + self.daily_values.byte_len() + self.k.byte_len()
            + self.mo_passes.len() * 4 + self.arrays.len() * 8
            + self.tolerance_flags.len() * 4
            + self.count_change_cause.len() * 4
    }

    /// Upload the whole host state.
    pub fn upload(&mut self, gpu: &Gpu, h: &HostState) -> Result<()> {
        let s = &gpu.stream;
        for (field, (d, x)) in self.elem.iter_mut().zip(&h.elem).enumerate() {
            let counter_field = match field { 4 => 1, 5 => 2, 6 => 3, 7 => 4, 9 => 5, _ => 0 };
            if matches!(field, 4 | 5) {
                d.upload(s, &compact_quantize(x, h.b, &h.column_key, 65535.0, counter_field), false)?;
            } else if matches!(field, 6 | 7) {
                d.upload(s, &compact_quantize(x, h.b, &h.column_key, 1.0e7, counter_field), false)?;
            } else if field == 9 {
                d.upload(s, &compact_quantize(x, h.b, &h.column_key, 144.0, counter_field), false)?;
            } else if field != 3 && field != 8 {
                d.upload(s, x, false)?;
            }
        }
        let markers: Vec<u8> = h.marker.iter().map(|&v| v as u8).collect();
        s.memcpy_htod(&markers, &mut self.marker)?;
        self.t.upload(s, &h.t, gpu.build.precision.is_f32())?;
        for (d, x) in self.ints.iter_mut().zip([
            &h.n,
            &h.fail,
            &h.first_out,
            &h.has_prev,
            &h.out_ready,
            &h.cursor,
            &h.maxn,
        ]) {
            s.memcpy_htod(x, d)?;
        }
        for (d, x) in self.longs.iter_mut().zip([&h.prev_t, &h.steps, &h.fail_t]) {
            s.memcpy_htod(x, d)?;
        }
        s.memcpy_htod(&h.column_key, &mut self.column_key)?;
        self.buffer.upload(s, &h.buffer, false)?;
        self.acc.upload(s, &h.acc, false)?;
        if gpu.build.precision.is_f32() {
            let mut last = h.last.clone();
            for c in 0..self.b {
                for field in [0, L_TSG] {
                    let i = field * self.b + c;
                    last[i] -= 273.15;
                }
            }
            self.last.upload(s, &last, false)?;
        } else {
            self.last.upload(s, &h.last, false)?;
        }
        Ok(())
    }

    /// Download the per-column scalars and the element and node arrays up to
    /// `upto` elements (the batch's current maximum count is enough).
    pub fn download(&self, gpu: &Gpu, h: &mut HostState, upto: usize) -> Result<()> {
        let s = &gpu.stream;
        let b = self.b;
        let ne = upto.min(self.nmax) * b;
        let nn = (upto.min(self.nmax) + 1) * b;
        for (field, (d, x)) in self.elem.iter().zip(h.elem.iter_mut()).enumerate() {
            if ne > 0 {
                if field != 3 && field != 8 {
                    d.download(s, &mut x[..ne], false)?;
                    let scale = match field { 4 | 5 => 65535.0, 6 | 7 => 1.0e7, 9 => 144.0, _ => 1.0 };
                    if scale != 1.0 { for v in &mut x[..ne] { *v /= scale; } }
                }
            }
        }
        for i in 0..ne { h.elem[3][i] = 1.0 - h.elem[1][i] - h.elem[2][i]; h.elem[8][i] = 1.0; }
        if ne > 0 {
            let mut markers = vec![0u8; ne];
            s.memcpy_dtoh(&self.marker.slice(0..ne), &mut markers)?;
            for (dst, value) in h.marker[..ne].iter_mut().zip(markers) { *dst = value as u32; }
        }
        self.t.download(s, &mut h.t[..nn], gpu.build.precision.is_f32())?;
        for (d, x) in self.ints.iter().zip([
            &mut h.n,
            &mut h.fail,
            &mut h.first_out,
            &mut h.has_prev,
            &mut h.out_ready,
            &mut h.cursor,
            &mut h.maxn,
        ]) {
            s.memcpy_dtoh(d, x)?;
        }
        for (d, x) in self
            .longs
            .iter()
            .zip([&mut h.prev_t, &mut h.steps, &mut h.fail_t])
        {
            s.memcpy_dtoh(d, x)?;
        }
        self.buffer.download(s, &mut h.buffer, false)?;
        self.acc.download(s, &mut h.acc, false)?;
        self.last.download(s, &mut h.last, false)?;
        if gpu.build.precision.is_f32() {
            for c in 0..b {
                h.last[c] += 273.15;
                h.last[L_TSG * b + c] += 273.15;
            }
        }
        s.memcpy_dtoh(&self.mo_passes, &mut h.mo_passes)?;
        s.memcpy_dtoh(&self.tolerance_flags, &mut h.tolerance_flags)?;
        s.memcpy_dtoh(&self.count_change_cause, &mut h.count_change_cause)?;
        s.synchronize()?;
        Ok(())
    }

    /// Download only the element counts.
    pub fn counts(&self, gpu: &Gpu) -> Result<Vec<i32>> {
        Ok(gpu.stream.clone_dtoh(&self.ints[0])?)
    }

    /// Download cursors, out_ready flags and failure codes.
    pub fn flags(&self, gpu: &Gpu) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>)> {
        let s = &gpu.stream;
        Ok((
            s.clone_dtoh(&self.ints[5])?,
            s.clone_dtoh(&self.ints[4])?,
            s.clone_dtoh(&self.ints[1])?,
        ))
    }

    /// Upload forcing rows: `ft` is `[row * b + c]`, `fv` is `[(row * F + f) * b + c]`.
    pub fn upload_forcing(&mut self, gpu: &Gpu, ft: &[i64], fv: &[f64]) -> Result<()> {
        let s = &gpu.stream;
        s.memcpy_htod(ft, &mut self.ft.slice_mut(0..ft.len()))?;
        let mut values = fv.to_vec();
        if gpu.build.precision.is_f32() {
            let b = self.b;
            let rows = fv.len() / (F_COUNT * b);
            for r in 0..rows {
                for f in [0, 7] {
                    for c in 0..b {
                        let i = (r * F_COUNT + f) * b + c;
                        if values[i].is_finite() {
                            values[i] -= 273.15;
                        }
                    }
                }
            }
        }
        self.fv.upload(s, &values, false)?;
        Ok(())
    }

    /// Set every column's cursor (start of a freshly uploaded forcing chunk).
    pub fn reset_cursors(&mut self, gpu: &Gpu) -> Result<()> {
        gpu.stream.memset_zeros(&mut self.ints[5])?;
        Ok(())
    }

    /// Run the kernel. `rows` forcing rows are valid in the buffer.
    pub fn advance(
        &mut self,
        gpu: &Gpu,
        rows: usize,
        max_rows: usize,
        mode: i32,
        day0: i64,
    ) -> Result<()> {
        let b = self.b as i32;
        let nmax = self.nmax as i32;
        let rows = rows as i32;
        let max_rows = max_rows.min(i32::MAX as usize) as i32;
        let cfg = LaunchConfig {
            grid_dim: ((self.b as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = gpu.stream.launch_builder(&gpu.advance);
        l.arg(&self.arrays)
            .arg(&self.k.raw)
            .arg(&b)
            .arg(&nmax)
            .arg(&rows)
            .arg(&max_rows)
            .arg(&mode)
            .arg(&day0);
        unsafe { l.launch(cfg)? };
        Ok(())
    }

    pub fn synchronize(&self, gpu: &Gpu) -> Result<()> {
        gpu.stream.synchronize()?;
        Ok(())
    }

    /// Download the daily slots.
    pub fn daily(&self, gpu: &Gpu) -> Result<Vec<f64>> {
        let mut values = vec![0.0; self.daily_values.len];
        self.daily_values.download(&gpu.stream, &mut values, false)?;
        Ok(values)
    }

}
