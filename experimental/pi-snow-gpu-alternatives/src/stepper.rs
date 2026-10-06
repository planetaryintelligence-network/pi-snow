//! A CPU port of `pi_snow_cli::engine::run_with_options` that stops after every step.
//!
//! The engine keeps its state inside one function, so nothing outside it can look at
//! the column between steps. The GPU needs exactly that: the lockstep gate (G1a)
//! advances this stepper and the kernel one step from the same state and compares,
//! and the host writes SMET and PRO for GPU columns by rebuilding a [`Column`] and
//! formatting it here. So this file is `engine.rs` with its loop body moved into
//! [`Stepper::step`] and its private output functions copied unchanged. G0 checks that
//! [`run_season`] writes byte-identical SMET and PRO to the `pi-snow` binary.
//!
//! Only the default `--physics` path is ported for the step itself; the output
//! functions keep their options argument so they match `engine.rs` line for line.
//! `MASS_REPORT=LEDGER` is refused: the ledger module is private to the CLI crate.

use chrono::{DateTime, Duration, Timelike, Utc};
use pi_snow_core::constants::{NODATA, RHO_WATER, T_MELT};
use pi_snow_core::options::{GrainClassification, MassReport, S5Index, Turbulence};
use pi_snow_core::{Column, Element, PhysicsOptions};
use pi_snow_io::ini::{NewSnowDensity, RunConfig, Stability};
use pi_snow_io::pro::{self, ElementRecord, ProHeader, ProTimestep};
use pi_snow_io::smet::{OutputRow, Smet};
use pi_snow_physics::albedo;
use pi_snow_physics::atmosphere::pressure_at_altitude;
use pi_snow_physics::grain_classification;
use pi_snow_physics::metamorphism::{self, History, Local};
use pi_snow_physics::new_snow::{self, Conditions, DensityScheme, Deposition};
use pi_snow_physics::settlement::{self, Br92, Viscosity};
use pi_snow_physics::solar;
use pi_snow_physics::stability::{self, GrainClass, Layer};
use pi_snow_physics::surface::{self, Atmosphere, SurfaceParameters};
use pi_snow_physics::thermal::{self, Conductivity, Ground};
use pi_snow_physics::turbulence::Scheme;
use pi_snow_physics::water;

/// Marker bits kept in `Microstructure::marker` (as `engine.rs`).
pub const MARK_REFROZEN: u16 = 1;
const MELTED_OUT_ICE_DEFAULT: f64 = 1e-6;
pub const MARK_SURFACE_HOAR: u16 = 2;
/// Floor on measurement heights above the snow, m (as `engine.rs`).
pub const MIN_MEASUREMENT_HEIGHT: f64 = 0.5;

#[derive(Debug)]
pub enum RunError {
    MissingField(&'static str),
    Crazy {
        time: DateTime<Utc>,
        value: f64,
        min: f64,
        max: f64,
    },
    Unsupported(&'static str),
}

impl std::error::Error for RunError {}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::MissingField(name) => write!(f, "forcing field {name} is missing"),
            RunError::Crazy {
                time,
                value,
                min,
                max,
            } => write!(
                f,
                "temperature {value} K at {time} is outside T_CRAZY [{min}, {max}]"
            ),
            RunError::Unsupported(what) => write!(f, "not supported by the stepper: {what}"),
        }
    }
}

/// Station metadata the output needs (as `engine::Site`).
#[derive(Debug, Clone)]
pub struct Site {
    pub station_id: String,
    pub station_name: String,
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: f64,
    pub slope_angle: f64,
    pub slope_azimuth: f64,
    pub soil_albedo: f64,
}

/// Hourly accumulators for fields the reference averages or sums (as `engine.rs`).
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Accum {
    pub n: f64,
    pub seconds: f64,
    pub qs: f64,
    pub ql: f64,
    pub qr: f64,
    pub qg0: f64,
    pub olwr: f64,
    pub ilwr: f64,
    pub lwr_net: f64,
    pub oswr: f64,
    pub iswr: f64,
    pub iswr_h: f64,
    pub iswr_dir: f64,
    pub iswr_diff: f64,
    pub qw: f64,
    pub albedo: f64,
    pub snow: f64,
    pub rain: f64,
    pub runoff: f64,
    pub sublimation: f64,
}

impl Accum {
    /// The fields in a fixed order, for packing and comparison.
    pub fn to_array(&self) -> [f64; 20] {
        [
            self.n,
            self.seconds,
            self.qs,
            self.ql,
            self.qr,
            self.qg0,
            self.olwr,
            self.ilwr,
            self.lwr_net,
            self.oswr,
            self.iswr,
            self.iswr_h,
            self.iswr_dir,
            self.iswr_diff,
            self.qw,
            self.albedo,
            self.snow,
            self.rain,
            self.runoff,
            self.sublimation,
        ]
    }

    pub fn from_array(a: &[f64; 20]) -> Self {
        Self {
            n: a[0],
            seconds: a[1],
            qs: a[2],
            ql: a[3],
            qr: a[4],
            qg0: a[5],
            olwr: a[6],
            ilwr: a[7],
            lwr_net: a[8],
            oswr: a[9],
            iswr: a[10],
            iswr_h: a[11],
            iswr_dir: a[12],
            iswr_diff: a[13],
            qw: a[14],
            albedo: a[15],
            snow: a[16],
            rain: a[17],
            runoff: a[18],
            sublimation: a[19],
        }
    }
}

/// One forcing row, the values `engine.rs` reads from it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForcingRow {
    pub t: DateTime<Utc>,
    pub ta: Option<f64>,
    pub rh: Option<f64>,
    pub vw: Option<f64>,
    pub dw: Option<f64>,
    pub iswr: Option<f64>,
    pub ilwr: Option<f64>,
    pub psum: Option<f64>,
    /// `None` both when the file has no TSG column and when the value is nodata;
    /// `engine.rs` treats the two alike.
    pub tsg: Option<f64>,
}

/// Field positions of a forcing SMET, resolved once.
#[derive(Debug, Clone, Copy)]
pub struct ForcingIndex {
    ta: usize,
    rh: usize,
    vw: usize,
    dw: usize,
    iswr: usize,
    ilwr: usize,
    psum: usize,
    tsg: Option<usize>,
}

impl ForcingIndex {
    pub fn of(forcing: &Smet) -> Result<Self, RunError> {
        let field = |name: &'static str| {
            forcing
                .field_index(name)
                .ok_or(RunError::MissingField(name))
        };
        Ok(Self {
            ta: field("TA")?,
            rh: field("RH")?,
            vw: field("VW")?,
            dw: field("DW")?,
            iswr: field("ISWR")?,
            ilwr: field("ILWR")?,
            psum: field("PSUM")?,
            tsg: forcing.field_index("TSG"),
        })
    }

    pub fn row(&self, row: &pi_snow_io::smet::Row) -> ForcingRow {
        let v = |i: usize| row.values[i];
        ForcingRow {
            t: row.timestamp,
            ta: v(self.ta),
            rh: v(self.rh),
            vw: v(self.vw),
            dw: v(self.dw),
            iswr: v(self.iswr),
            ilwr: v(self.ilwr),
            psum: v(self.psum),
            tsg: self.tsg.and_then(v),
        }
    }
}

/// The values of the last advanced step that the output row reports.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct LastForcing {
    pub ta: f64,
    pub rh: f64,
    pub vw: f64,
    pub dw: f64,
    pub tsg: f64,
}

/// What one call to [`Stepper::step`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The first row, or a non-increasing timestamp: nothing advanced.
    NoStep,
    /// A step was counted but a required forcing value was missing, so nothing
    /// else happened (`engine.rs`'s `continue` after `steps += 1`).
    Skipped,
    /// The column advanced; `output` is true when this step writes an output row.
    Advanced { output: bool },
}

/// Per-run constants, shared by the CPU stepper and the GPU host.
#[derive(Debug, Clone, Copy)]
pub struct Constants {
    pub scheme: Scheme,
    pub density_scheme: DensityScheme,
    pub conductivity: Conductivity,
    pub pressure: f64,
}

impl Constants {
    pub fn new(cfg: &RunConfig, options: &PhysicsOptions, site: &Site) -> Self {
        let scheme = match (options.turbulence, cfg.stability) {
            (Turbulence::Louis1979, _) => Scheme::Louis,
            (Turbulence::FromIni, Stability::MoHoltslag) => Scheme::MoHoltslag,
            (Turbulence::FromIni, Stability::Richardson) => Scheme::Louis,
        };
        let density_scheme = match cfg.new_snow_density {
            NewSnowDensity::LehningNew => DensityScheme::Pahaut,
            NewSnowDensity::Zwart => DensityScheme::Zwart,
        };
        Self {
            scheme,
            density_scheme,
            conductivity: Conductivity::Yen1981,
            pressure: pressure_at_altitude(site.altitude),
        }
    }
}

/// One column advanced one forcing row at a time.
pub struct Stepper<'a> {
    pub cfg: &'a RunConfig,
    pub options: PhysicsOptions,
    pub site: &'a Site,
    pub constants: Constants,
    pub column: Column,
    pub deposition: Deposition,
    pub acc: Accum,
    pub previous: Option<DateTime<Utc>>,
    pub first_output: bool,
    pub steps: usize,
    pub last: LastForcing,
    /// Monin-Obukhov passes of the last step's final surface budget, recounted by
    /// `mo::fluxes_counted` (the physics crate does not report it).
    pub mo_passes: u32,
    melted_out_ice: f64,
    refrozen_mass_fraction: Option<f64>,
}

impl<'a> Stepper<'a> {
    pub fn new(
        cfg: &'a RunConfig,
        options: &PhysicsOptions,
        site: &'a Site,
        start: Column,
    ) -> Result<Self, RunError> {
        if options.mass_report == MassReport::Ledger {
            return Err(RunError::Unsupported("MASS_REPORT=LEDGER"));
        }
        Ok(Self {
            cfg,
            options: *options,
            site,
            constants: Constants::new(cfg, options, site),
            column: start,
            deposition: Deposition::new(
                cfg.height_new_element,
                0.5e-3 * cfg.new_snow_grain_size_mm,
            ),
            acc: Accum::default(),
            previous: None,
            first_output: true,
            steps: 0,
            last: LastForcing::default(),
            mo_passes: 0,
            melted_out_ice: MELTED_OUT_ICE_DEFAULT,
            refrozen_mass_fraction: None,
        })
    }

    pub fn new_marker_control(
        cfg: &'a RunConfig,
        options: &PhysicsOptions,
        site: &'a Site,
        start: Column,
    ) -> Result<Self, RunError> {
        let mut stepper = Self::new(cfg, options, site, start)?;
        stepper.melted_out_ice = 1e-4;
        stepper.refrozen_mass_fraction = Some(1e-3);
        Ok(stepper)
    }

    /// `engine.rs`'s loop body for one row. The caller stops at `end`.
    pub fn step(&mut self, row: &ForcingRow) -> Result<Outcome, RunError> {
        self.mo_passes = 0;
        let cfg = self.cfg;
        let options = &self.options;
        let site = self.site;
        let Constants {
            scheme,
            density_scheme,
            conductivity,
            pressure,
        } = self.constants;
        let column = &mut self.column;
        let acc = &mut self.acc;

        let t = row.t;
        let Some(t_prev) = self.previous.replace(t) else {
            return Ok(Outcome::NoStep);
        };
        let dt = (t - t_prev).num_seconds() as f64;
        if dt <= 0.0 {
            return Ok(Outcome::NoStep);
        }
        self.steps += 1;
        let (Some(ta), Some(rh), Some(vw), Some(iswr_h), Some(ilwr)) =
            (row.ta, row.rh, row.vw, row.iswr, row.ilwr)
        else {
            return Ok(Outcome::Skipped);
        };
        let dw = row.dw.unwrap_or(NODATA);
        let psum = row.psum.unwrap_or(0.0).max(0.0);
        let tsg = row.tsg.unwrap_or(T_MELT);

        // 1. Radiation on the slope.
        let mid = t - Duration::milliseconds((dt * 500.0) as i64);
        let sun = solar::sun_position(mid, site.latitude, site.longitude);
        let (_, diffuse_h) = solar::split_global(iswr_h, &sun, mid);
        let ground_albedo = if column.elements.is_empty() {
            site.soil_albedo
        } else {
            albedo::shortwave(column, 1.0, pressure).albedo
        };
        let sw = solar::on_slope(
            iswr_h,
            diffuse_h,
            &sun,
            site.slope_angle,
            site.slope_azimuth,
            ground_albedo,
        );
        let iswr = sw.global_on_slope;

        // 2. Precipitation.
        let t_surface = *column.node_temperature.last().unwrap();
        let conditions = Conditions {
            t_air: ta,
            t_surface,
            relative_humidity: rh,
            wind: vw,
        };
        let (solid, liquid) = if ta < T_MELT + cfg.thresh_rain_c {
            (psum, 0.0)
        } else {
            (0.0, psum)
        };
        let deposited = self
            .deposition
            .add(column, solid, &conditions, density_scheme);
        acc.snow += deposited;
        acc.rain += liquid;
        let mut rain_rate = 0.0;
        if liquid > 0.0 {
            if let Some(top) = column.elements.last_mut() {
                top.composition.water += liquid / (RHO_WATER * top.thickness);
                top.composition.air = (1.0 - top.composition.ice - top.composition.water).max(0.0);
                rain_rate = liquid / dt;
            } else {
                acc.runoff += liquid;
            }
        }

        let hs = column.snow_height();
        let params = SurfaceParameters {
            emissivity: surface::SNOW_EMISSIVITY,
            z_wind: (cfg.height_of_wind_value - hs).max(MIN_MEASUREMENT_HEIGHT),
            z_scalar: (cfg.height_of_meteo_values - hs).max(MIN_MEASUREMENT_HEIGHT),
            z0: cfg.roughness_length,
            scheme,
        };

        let (budget, albedo_now, qg0) = if column.elements.is_empty() {
            let atm = Atmosphere {
                t_air: ta,
                relative_humidity: rh,
                wind: vw,
                absorbed_shortwave: 0.0,
                ilwr,
                pressure,
                rain_rate: 0.0,
            };
            column.node_temperature[0] = tsg;
            self.mo_passes = final_budget_passes(&atm, &params, tsg);
            (surface::budget(&atm, &params, tsg), site.soil_albedo, 0.0)
        } else {
            // 3. Shortwave through the pack.
            let absorbed = albedo::shortwave(column, iswr, pressure);
            let atm = Atmosphere {
                t_air: ta,
                relative_humidity: rh,
                wind: vw,
                absorbed_shortwave: absorbed.surface,
                ilwr,
                pressure,
                rain_rate,
            };
            // 4. Energy balance and conduction.
            let before = thermal::internal_energy(column);
            let b = surface::step_general(
                column,
                &atm,
                &params,
                Ground::Temperature(tsg),
                &absorbed.per_element,
                dt,
                conductivity,
            );
            self.mo_passes =
                final_budget_passes(&atm, &params, *column.node_temperature.last().unwrap());
            let sources: f64 = absorbed.per_element.iter().sum();
            let qg0 = (thermal::internal_energy(column) - before) / dt - b.total - sources;

            // 5. Phase change and water.
            let water_before: Vec<(f64, f64)> = column.elements.iter()
                .map(|e| (e.water_mass(), e.mass())).collect();
            let n_before = column.elements.len();
            let pc = water::phase_change_with_melted_out_ice(column, self.melted_out_ice);
            if column.elements.len() == n_before {
                for (e, (w, mass)) in column.elements.iter_mut().zip(water_before) {
                    let threshold = self.refrozen_mass_fraction.map_or(1e-9, |fraction| fraction * mass);
                    if e.water_mass() < w - threshold {
                        e.micro.marker |= MARK_REFROZEN;
                    }
                }
            }
            let runoff = water::cascade_with(column, options.water_retention);
            acc.runoff += pc.drained + runoff;

            // 6. Sublimation or deposition on the top element.
            let vapour = b.vapour_mass * dt;
            if let Some(top) = column.elements.last_mut() {
                let ice = (top.ice_mass() + vapour).max(0.0);
                let scale = if top.ice_mass() > 0.0 {
                    ice / top.ice_mass()
                } else {
                    1.0
                };
                top.thickness *= scale;
                if top.thickness > 0.0 {
                    top.composition.water /= scale;
                    top.composition.air =
                        (1.0 - top.composition.ice - top.composition.water).max(0.0);
                }
            }
            acc.sublimation += (-vapour).max(0.0);

            // 7. Settlement.
            settlement::step(column, dt, Br92);

            // 8. Metamorphism and ageing.
            let temps = column.element_temperatures();
            for (i, e) in column.elements.iter_mut().enumerate() {
                let gradient = (column.node_temperature[i + 1] - column.node_temperature[i]).abs()
                    / e.thickness.max(1e-6);
                let mass = e.mass();
                let lwc = if mass > 0.0 {
                    100.0 * e.water_mass() / mass
                } else {
                    0.0
                };
                let mut h = history(e);
                let local = Local {
                    temperature: temps[i],
                    gradient,
                    density: e.density(),
                    lwc_percent: lwc,
                };
                metamorphism::step(&mut e.micro, &local, &mut h, dt);
                e.grain_history = grain_classification::update_history(
                    e.grain_history,
                    &e.micro,
                    temps[i],
                    lwc / 100.0,
                );
                e.age_days += dt / 86_400.0;
            }

            // 9. Merge thin elements; drop anything melted to nothing.
            new_snow::merge_thin_elements(column, cfg.minimum_element_length);
            (b, absorbed.albedo, qg0)
        };

        for &t_node in &column.node_temperature {
            if t_node < cfg.t_crazy_min || t_node > cfg.t_crazy_max {
                return Err(RunError::Crazy {
                    time: t,
                    value: t_node,
                    min: cfg.t_crazy_min,
                    max: cfg.t_crazy_max,
                });
            }
        }

        // 10. Accumulate; the caller writes on the hour.
        acc.n += 1.0;
        acc.seconds += dt;
        acc.qs += budget.sensible;
        acc.ql += budget.latent;
        acc.qr += budget.rain;
        acc.qg0 += qg0;
        acc.olwr += budget.olwr;
        acc.ilwr += ilwr;
        acc.lwr_net += budget.lwr_net;
        acc.oswr += albedo_now * iswr;
        acc.iswr += iswr;
        acc.iswr_h += iswr_h;
        acc.iswr_dir += sw.direct_on_slope;
        acc.iswr_diff += sw.sky_diffuse_on_slope;
        acc.qw += (1.0 - albedo_now) * iswr;
        acc.albedo += albedo_now;
        self.last = LastForcing {
            ta,
            rh,
            vw,
            dw,
            tsg,
        };

        let on_the_hour = t.minute() == 0 && t.second() == 0;
        let output = self.first_output || on_the_hour;
        if output {
            self.first_output = false;
        }
        Ok(Outcome::Advanced { output })
    }

    /// The SMET row for the current state; resets the accumulators, as the engine
    /// does after writing.
    pub fn take_output_row(&mut self, t: DateTime<Utc>) -> OutputRow {
        let l = self.last;
        let row = output_row(
            t,
            &self.acc,
            &self.column,
            l.ta,
            l.rh,
            l.vw,
            l.dw,
            l.tsg,
            &self.options,
        );
        self.acc = Accum::default();
        row
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skipped_forcing_clears_the_previous_mo_pass_count() {
        let synthetic = crate::synth::generate(1, 1);
        let site = &synthetic.sites[0];
        let cfg = crate::synth::config();
        let mut stepper = Stepper::new(
            &cfg,
            &PhysicsOptions::default(),
            site,
            crate::synth::bare(site),
        )
        .unwrap();
        let rows = &synthetic.rows[0];
        assert!(matches!(stepper.step(&rows[0]).unwrap(), Outcome::NoStep));
        assert!(matches!(
            stepper.step(&rows[1]).unwrap(),
            Outcome::Advanced { .. }
        ));
        assert!(stepper.mo_passes > 0);

        let mut missing = rows[2];
        missing.ta = None;
        assert!(matches!(stepper.step(&missing).unwrap(), Outcome::Skipped));
        assert_eq!(stepper.mo_passes, 0);
    }
}

/// Everything a run writes (SMET and PRO text).
pub struct SeasonOutput {
    pub smet: String,
    pub pro: String,
    pub steps: usize,
}

/// `engine::run_with_options` through the stepper: same files, same bytes.
pub fn run_season(
    cfg: &RunConfig,
    options: &PhysicsOptions,
    site: &Site,
    forcing: &Smet,
    start: Column,
    end: DateTime<Utc>,
) -> Result<SeasonOutput, RunError> {
    let index = ForcingIndex::of(forcing)?;
    let mut s = Stepper::new(cfg, options, site, start)?;
    let mut writer = SeasonWriter::new(site);
    for row in &forcing.rows {
        if row.timestamp > end {
            break;
        }
        let r = index.row(row);
        if let Outcome::Advanced { output: true } = s.step(&r)? {
            writer.push(&mut s, r.t);
        }
    }
    let steps = s.steps;
    Ok(writer.finish(site, steps))
}

/// Collects output rows and profiles the way `engine.rs` does.
pub struct SeasonWriter {
    rows: Vec<OutputRow>,
    pro_text: String,
}

impl SeasonWriter {
    pub fn new(site: &Site) -> Self {
        let header = ProHeader {
            station_name: site.station_name.clone(),
            latitude: site.latitude,
            longitude: site.longitude,
            altitude: site.altitude,
            slope_angle_deg: site.slope_angle,
            slope_azimuth_deg: site.slope_azimuth,
        };
        Self {
            rows: Vec::new(),
            pro_text: pro::write_header(&header),
        }
    }

    pub fn push(&mut self, s: &mut Stepper, t: DateTime<Utc>) {
        self.rows.push(s.take_output_row(t));
        self.pro_text
            .push_str(&pro::write_timestep(&profile(t, &s.column, &s.options)));
    }

    /// Append an already-built row and, when `with_profile`, the profile of `column`.
    pub fn push_parts(
        &mut self,
        row: OutputRow,
        t: DateTime<Utc>,
        column: &Column,
        options: &PhysicsOptions,
        with_profile: bool,
    ) {
        self.rows.push(row);
        if with_profile {
            self.pro_text
                .push_str(&pro::write_timestep(&profile(t, column, options)));
        }
    }

    pub fn finish(self, site: &Site, steps: usize) -> SeasonOutput {
        let smet = pi_snow_io::smet::write_output_smet(
            &site.station_id,
            &site.station_name,
            site.latitude,
            site.longitude,
            site.altitude,
            site.slope_angle,
            site.slope_azimuth,
            &self.rows,
        );
        SeasonOutput {
            smet,
            pro: self.pro_text,
            steps,
        }
    }
}

fn final_budget_passes(atm: &Atmosphere, p: &SurfaceParameters, t_surface: f64) -> u32 {
    let layer = crate::mo::budget_layer(
        atm.t_air,
        atm.relative_humidity,
        atm.wind,
        atm.pressure,
        p.z_wind,
        p.z_scalar,
        p.z0,
        t_surface,
    );
    crate::mo::fluxes_counted(&layer, p.scheme).2
}

pub fn history(e: &Element) -> History {
    History {
        refrozen: e.micro.marker & MARK_REFROZEN != 0,
        surface_hoar: e.micro.marker & MARK_SURFACE_HOAR != 0,
        wet: e.composition.water > 0.0,
    }
}

// ---- Output, copied from engine.rs (private there) ------------------------------

#[allow(clippy::too_many_arguments)]
pub fn output_row(
    t: DateTime<Utc>,
    a: &Accum,
    c: &Column,
    ta: f64,
    rh: f64,
    vw: f64,
    dw: f64,
    tsg: f64,
    options: &PhysicsOptions,
) -> OutputRow {
    let n = a.n.max(1.0);
    let mut r = OutputRow::nodata(t);
    let hours = (a.seconds / 3600.0).max(1e-9);
    for (name, value) in [
        ("Qs", a.qs / n),
        ("Ql", a.ql / n),
        ("Qg0", a.qg0 / n),
        ("Qr", a.qr / n),
        ("OLWR", a.olwr / n),
        ("ILWR", a.ilwr / n),
        ("LWR_net", a.lwr_net / n),
        ("OSWR", a.oswr / n),
        ("ISWR", a.iswr / n),
        ("ISWR_h", a.iswr_h / n),
        ("ISWR_dir", a.iswr_dir / n),
        ("ISWR_diff", a.iswr_diff / n),
        ("Qw", a.qw / n),
        ("pAlbedo", a.albedo / n),
        ("TSG", tsg - T_MELT),
        ("TA", ta - T_MELT),
        ("TSS_mod", c.node_temperature.last().unwrap() - T_MELT),
        ("T_bottom", c.node_temperature[0] - T_MELT),
        ("RH", 100.0 * rh),
        ("VW", vw),
        ("VW_drift", vw),
        ("DW", dw),
        ("MS_Snow", a.snow / hours),
        ("MS_Rain", a.rain / hours),
        ("MS_SN_Runoff", a.runoff),
        ("MS_Sublimation", a.sublimation),
        ("HS_mod", 100.0 * c.snow_height()),
        ("SWE", c.swe()),
        ("MS_Water", c.liquid_water()),
        ("hoar_size", 0.0),
        ("wind_trans24", 0.0),
        ("HN3", 100.0 * new_snow_height(c, 3.0 / 24.0)),
        ("HN6", 100.0 * new_snow_height(c, 6.0 / 24.0)),
        ("HN12", 100.0 * new_snow_height(c, 0.5)),
        ("HN24", 100.0 * new_snow_height(c, 1.0)),
        ("HN72_24", 100.0 * new_snow_height(c, 3.0)),
        ("Sclass1", -1.0),
        ("Sclass2", -1.0),
    ] {
        r.set(name, value);
    }
    let layers = layers(c, options);
    let idx = stability::indices(&layers, c.snow_height());
    let tops = c.element_top_heights();
    let minimum = |f: fn(&stability::Indices) -> Option<f64>| {
        idx.iter()
            .enumerate()
            .filter_map(|(i, x)| f(x).map(|v| (v, tops[i])))
            .fold((stability::INDEX_CAP, 0.0), |best, cur| {
                if cur.0 < best.0 { cur } else { best }
            })
    };
    for (z, s, f) in [
        (
            "zSn",
            "Sn",
            (|x: &stability::Indices| x.sn38) as fn(&stability::Indices) -> Option<f64>,
        ),
        ("zSs", "Ss", |x| x.sk38),
        ("zS4", "S4", |x| x.ssi),
    ] {
        let (value, height) = minimum(f);
        r.set(s, value);
        r.set(z, 100.0 * height);
    }
    match options.s5 {
        S5Index::None => {}
        S5Index::LwcIndexMitterer2013 => {
            r.set("zS5", 0.0);
            if let Some(index) = lwc_index_s5(c) {
                r.set("S5", index);
            }
        }
    }
    r
}

pub fn lwc_index_s5(c: &Column) -> Option<f64> {
    let snow = &c.elements[c.soil_elements..];
    stability::reported_lwc_index(
        snow.iter().map(|e| (e.thickness, e.composition.water)),
        c.snow_height(),
    )
}

fn new_snow_height(c: &Column, days: f64) -> f64 {
    c.elements
        .iter()
        .filter(|e| e.age_days < days)
        .map(|e| e.thickness)
        .sum()
}

fn layers(c: &Column, options: &PhysicsOptions) -> Vec<Layer> {
    let hs = c.snow_height();
    let tops = c.element_top_heights();
    let temps = c.element_temperatures();
    c.elements
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let code = match options.grain_classification {
                GrainClassification::PiSnow2026 => {
                    metamorphism::classify(&e.micro, &history(e), e.density())
                }
                GrainClassification::Lafaysse2026 => {
                    grain_classification::lafaysse2026(&e.micro, e.grain_history)
                }
            };
            Layer {
                depth_top: hs - tops[i],
                thickness: e.thickness,
                density: e.density(),
                grain_type: code,
                grain_size_mm: 2e3 * e.micro.grain_radius,
                hand_hardness: hardness(code, e.density()),
                temperature_c: temps[i] - T_MELT,
                age_days: e.age_days,
            }
        })
        .collect()
}

pub fn grain_hardness_values(c: &Column, options: &PhysicsOptions) -> Vec<(u16, f64)> {
    layers(c, options).into_iter().map(|layer| (layer.grain_type, layer.hand_hardness)).collect()
}

fn hardness(code: u16, density: f64) -> f64 {
    GrainClass::from_code(code)
        .and_then(|c| stability::monti_hardness(c, density))
        .unwrap_or(NODATA)
}

pub fn profile(t: DateTime<Utc>, c: &Column, options: &PhysicsOptions) -> ProTimestep {
    let layers = layers(c, options);
    let idx = stability::indices(&layers, c.snow_height());
    let tops = c.element_top_heights();
    let temps = c.element_temperatures();
    let elements = c
        .elements
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let eta = Br92.viscosity(e, temps[i]);
            let stress = e.stress;
            ElementRecord {
                top_height_cm: 100.0 * tops[i],
                density: e.density(),
                temperature_c: temps[i] - T_MELT,
                id: i as u32 + 1,
                age_days: e.age_days,
                lwc_percent: 100.0 * e.composition.water,
                dendricity: e.micro.dendricity,
                sphericity: e.micro.sphericity,
                coordination_number: e.micro.coordination_number,
                bond_size_mm: 2e3 * e.micro.bond_radius,
                grain_size_mm: 2e3 * e.micro.grain_radius,
                grain_type: layers[i].grain_type,
                ice_percent: 100.0 * e.composition.ice,
                air_percent: 100.0 * e.composition.air,
                stress_kpa: stress / 1e3,
                viscosity_gpa_s: eta / 1e9,
                temperature_gradient: (c.node_temperature[i + 1] - c.node_temperature[i])
                    / e.thickness,
                conductivity: Conductivity::Yen1981.of_density(e.density()),
                absorbed_shortwave: 0.0,
                deformation_rate: 1e6 * stress / eta,
                hand_hardness: layers[i].hand_hardness,
                shear_strength_kpa: idx[i].strength_kpa.unwrap_or(NODATA),
                sn38: idx[i].sn38.unwrap_or(NODATA),
                sk38: idx[i].sk38.unwrap_or(NODATA),
                ssi: idx[i].ssi.unwrap_or(NODATA),
                critical_cut_length_m: idx[i].critical_cut_length_m.unwrap_or(NODATA),
                optical_grain_size_mm: 1e3 * albedo::optical_diameter(&e.micro),
                ..ElementRecord::default()
            }
        })
        .collect();
    ProTimestep {
        timestamp: t,
        elements,
        surface_hoar: None,
    }
}
