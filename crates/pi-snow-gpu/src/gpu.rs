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

pub const KERNEL_SOURCE: &str = include_str!("../kernels/pi_snow.cu");

/// Forcing fields per row, in the kernel's `F_*` order.
pub const F_COUNT: usize = 8;
pub const A_COUNT: usize = 20;
pub const L_COUNT: usize = 5;
pub const P_COUNT: usize = 11;
pub const K_COUNT: usize = 15;
pub const D_COUNT: usize = 6;
pub const ROW_ABSENT: i64 = i64::MIN;
pub const FAIL_CRAZY: i32 = 1;
pub const FAIL_OVERFLOW: i32 = 2;
pub const BLOCK: u32 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F64,
    Mix,
}

impl Precision {
    pub fn label(self) -> &'static str {
        match self {
            Precision::F64 => "F64",
            Precision::Mix => "MIX",
        }
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
        if precision == Precision::Mix {
            options.push("-DPI_MIX".into());
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
pub fn run_constants(cfg: &RunConfig, constants: &Constants) -> [f64; K_COUNT] {
    [
        cfg.thresh_rain_c,
        cfg.height_new_element,
        0.5e-3 * cfg.new_snow_grain_size_mm,
        cfg.height_of_wind_value,
        cfg.height_of_meteo_values,
        cfg.roughness_length,
        cfg.minimum_element_length,
        cfg.t_crazy_min,
        cfg.t_crazy_max,
        match constants.scheme {
            Scheme::MoHoltslag => 0.0,
            Scheme::Louis => 1.0,
        },
        match constants.density_scheme {
            DensityScheme::Pahaut => 0.0,
            DensityScheme::Zwart => 1.0,
            DensityScheme::Zwart2007Eq51 => -1.0,
        },
        R_DRY,
        CP_AIR,
        EPSILON,
        SNOW_EMISSIVITY,
    ]
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
    if constants.density_scheme == DensityScheme::Zwart2007Eq51 {
        bail!("density scheme Zwart2007Eq51 is not ported");
    }
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
    pub buffer: Vec<f64>,
    pub acc: Vec<f64>,
    pub last: Vec<f64>,
    pub mo_passes: Vec<i32>,
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
            buffer: vec![0.0; b],
            acc: vec![0.0; A_COUNT * b],
            last: vec![0.0; L_COUNT * b],
            mo_passes: vec![0; b],
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

/// A batch of `b` columns resident on the device.
pub struct Batch {
    pub b: usize,
    pub nmax: usize,
    pub rows: usize,
    elem: Vec<CudaSlice<f64>>,
    marker: CudaSlice<u32>,
    t: CudaSlice<f64>,
    _scratch: Vec<CudaSlice<f64>>,
    ints: Vec<CudaSlice<i32>>,  // n, fail, first_out, has_prev, out_ready, cursor, maxn
    longs: Vec<CudaSlice<i64>>, // prev_t, steps, fail_t
    buffer: CudaSlice<f64>,
    acc: CudaSlice<f64>,
    last: CudaSlice<f64>,
    par: CudaSlice<f64>,
    pub ft: CudaSlice<i64>,
    pub fv: CudaSlice<f64>,
    pub daily: CudaSlice<f64>,
    pub days: usize,
    mo_passes: CudaSlice<i32>,
    arrays: CudaSlice<u64>,
    k: CudaSlice<f64>,
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
        let elem = (0..11)
            .map(|_| s.alloc_zeros::<f64>(ne))
            .collect::<Result<Vec<_>, _>>()?;
        let marker = s.alloc_zeros::<u32>(ne)?;
        let t = s.alloc_zeros::<f64>(nn)?;
        let scratch = (0..8)
            .map(|_| s.alloc_zeros::<f64>(nn))
            .collect::<Result<Vec<_>, _>>()?;
        let ints = (0..7)
            .map(|_| s.alloc_zeros::<i32>(b))
            .collect::<Result<Vec<_>, _>>()?;
        let longs = (0..3)
            .map(|_| s.alloc_zeros::<i64>(b))
            .collect::<Result<Vec<_>, _>>()?;
        let buffer = s.alloc_zeros::<f64>(b)?;
        let acc = s.alloc_zeros::<f64>(A_COUNT * b)?;
        let last = s.alloc_zeros::<f64>(L_COUNT * b)?;
        if par.len() != P_COUNT * b {
            bail!("column parameters: {} values for {} columns", par.len(), b);
        }
        let par = s.clone_htod(par)?;
        let ft = s.alloc_zeros::<i64>(rows.max(1) * b)?;
        let fv = s.alloc_zeros::<f64>(rows.max(1) * F_COUNT * b)?;
        let daily = s.alloc_zeros::<f64>(days.max(1) * D_COUNT * b)?;
        let k = s.clone_htod(&k[..])?;
        let mo_passes = s.alloc_zeros::<i32>(b)?;

        // The pointer table, in the order of the kernel's `struct Arrays`.
        let mut ptrs: Vec<u64> = Vec::new();
        {
            let mut push64 = |p: u64| ptrs.push(p);
            for e in &elem {
                push64(e.device_ptr(s).0);
            }
            push64(marker.device_ptr(s).0);
            push64(t.device_ptr(s).0);
            for x in &scratch {
                push64(x.device_ptr(s).0);
            }
            for x in &ints {
                push64(x.device_ptr(s).0);
            }
            for x in &longs {
                push64(x.device_ptr(s).0);
            }
            push64(buffer.device_ptr(s).0);
            push64(acc.device_ptr(s).0);
            push64(last.device_ptr(s).0);
            push64(par.device_ptr(s).0);
            push64(ft.device_ptr(s).0);
            push64(fv.device_ptr(s).0);
            push64(daily.device_ptr(s).0);
            push64(mo_passes.device_ptr(s).0);
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
            buffer,
            acc,
            last,
            par,
            ft,
            fv,
            daily,
            days,
            mo_passes,
            arrays,
            k,
        })
    }

    /// Device bytes held by this batch.
    pub fn device_bytes(&self) -> usize {
        let (b, ne, nn) = (self.b, self.nmax * self.b, (self.nmax + 1) * self.b);
        11 * ne * 8 + ne * 4 + nn * 8 + 8 * nn * 8 + 7 * b * 4 + 3 * b * 8
            + b * 8 * (1 + A_COUNT + L_COUNT + P_COUNT)
            + self.rows.max(1) * b * 8 * (1 + F_COUNT)
            + self.days.max(1) * D_COUNT * b * 8
    }

    /// Upload the whole host state.
    pub fn upload(&mut self, gpu: &Gpu, h: &HostState) -> Result<()> {
        let s = &gpu.stream;
        for (d, x) in self.elem.iter_mut().zip(&h.elem) {
            s.memcpy_htod(x, d)?;
        }
        s.memcpy_htod(&h.marker, &mut self.marker)?;
        s.memcpy_htod(&h.t, &mut self.t)?;
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
        s.memcpy_htod(&h.buffer, &mut self.buffer)?;
        s.memcpy_htod(&h.acc, &mut self.acc)?;
        s.memcpy_htod(&h.last, &mut self.last)?;
        Ok(())
    }

    /// Download the per-column scalars and the element and node arrays up to
    /// `upto` elements (the batch's current maximum count is enough).
    pub fn download(&self, gpu: &Gpu, h: &mut HostState, upto: usize) -> Result<()> {
        let s = &gpu.stream;
        let b = self.b;
        let ne = upto.min(self.nmax) * b;
        let nn = (upto.min(self.nmax) + 1) * b;
        for (d, x) in self.elem.iter().zip(h.elem.iter_mut()) {
            if ne > 0 {
                s.memcpy_dtoh(&d.slice(0..ne), &mut x[..ne])?;
            }
        }
        if ne > 0 {
            s.memcpy_dtoh(&self.marker.slice(0..ne), &mut h.marker[..ne])?;
        }
        s.memcpy_dtoh(&self.t.slice(0..nn), &mut h.t[..nn])?;
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
        s.memcpy_dtoh(&self.buffer, &mut h.buffer)?;
        s.memcpy_dtoh(&self.acc, &mut h.acc)?;
        s.memcpy_dtoh(&self.last, &mut h.last)?;
        s.memcpy_dtoh(&self.mo_passes, &mut h.mo_passes)?;
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
        s.memcpy_htod(fv, &mut self.fv.slice_mut(0..fv.len()))?;
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
            .arg(&self.k)
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
        Ok(gpu.stream.clone_dtoh(&self.daily)?)
    }

    pub fn par(&self) -> &CudaSlice<f64> {
        &self.par
    }
}
