//! The season driver: one column, one slope, forcing step by step.
//!
//! Order within a step, following the round-1 critique's coupling order:
//!
//! 1. forcing at the step's end; measurement heights reduced by the snow depth;
//!    sun at the step's midpoint; global split into direct and diffuse and
//!    projected onto the slope;
//! 2. precipitation: a step at `THRESH_RAIN` (the reference's behaviour,
//!    lab notes); snow to the deposition buffer, rain into the top element;
//! 3. albedo and shortwave absorbed through the pack;
//! 4. surface energy balance and conduction, the ground held at `TSG`;
//! 5. melting and refreezing, then the bucket cascade;
//! 6. sublimation or deposition on the top element;
//! 7. settlement;
//! 8. metamorphism and ageing;
//! 9. merging of thin elements;
//! 10. diagnostics and output on the hour.
//!
//! Nothing here is physics: it wires modules that each carry their own sources.

use chrono::{DateTime, Duration, Timelike, Utc};
use pi_snow_core::constants::{LH_FUSION, NODATA, RHO_WATER, T_MELT};
use pi_snow_core::options::{
    DrySettlement, FreshSnowDensity, GrainClassification, GroundBoundary, MassReport, S5Index,
    SnowAlbedo, SublimationMeltOut, SurfaceLayer, SurfaceLinearisation, TotalMelt, Turbulence,
    WetGrainGrowth,
};
use pi_snow_core::{Column, Element, PhysicsOptions};
use pi_snow_io::ini::{NewSnowDensity, RunConfig, Stability};
use pi_snow_io::pro::{self, ElementRecord, ProHeader, ProTimestep};
use pi_snow_io::smet::{OutputRow, Smet};
use pi_snow_physics::albedo;
use pi_snow_physics::aoki_albedo::{self, AokiAlbedo};
use pi_snow_physics::atmosphere::pressure_at_altitude;
use pi_snow_physics::chan_albedo::{self, DepositionSeries, Slair};
use pi_snow_physics::density_settlement::{
    DensitySettlementViscosity, Scheme as DensitySettlementScheme,
};
use pi_snow_physics::grain_classification;
use pi_snow_physics::metamorphism::{self, History, Local};
use pi_snow_physics::new_snow::{self, Conditions, DensityScheme, Deposition};
use pi_snow_physics::settlement::{self, Viscosity};
use pi_snow_physics::solar;
use pi_snow_physics::stability::{self, GrainClass, Layer};
use pi_snow_physics::surface::{self, Atmosphere, Budget, SurfaceParameters};
use pi_snow_physics::surface_diagnostic;
use pi_snow_physics::surface_numerics;
use pi_snow_physics::thermal::{self, Conductivity, Ground};
use pi_snow_physics::thin_top_merge;
use pi_snow_physics::turbulence::Scheme;
use pi_snow_physics::water;
use pi_snow_physics::wet_growth;
use std::cell::RefCell;

use crate::diagnostic::{Collector, DiagnosticMode, Refusal, RunDiagnostics, StepTrace};
use crate::dirichlet_storage_diagnostic::DirichletStorageDiagnostic;
use crate::ledger::{Interval, Ledger};
use crate::meltout_candidate::{self, MeltoutCandidateDiagnostic, PackState};
use crate::meltout_diagnostic::MeltoutDiagnostic;

mod trace_substep;

/// Marker bits kept in `Microstructure::marker`.
const MARK_REFROZEN: u16 = 1;
const MARK_SURFACE_HOAR: u16 = 2;
/// Floor on measurement heights above the snow, m: ours, so a deep pack cannot
/// bury the sensors in the stability calculation.
const MIN_MEASUREMENT_HEIGHT: f64 = 0.5;

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("forcing field {0} is missing")]
    MissingField(&'static str),
    #[error("temperature {value} K at node {node}, time {time} is outside T_CRAZY [{min}, {max}]")]
    Crazy {
        node: usize,
        time: DateTime<Utc>,
        value: f64,
        min: f64,
        max: f64,
        partial: Box<RunOutput>,
    },
    #[error("{0}")]
    IncompatiblePhysics(&'static str),
    #[error("invalid SOIL_COLUMN initialization: {0}")]
    SoilInitialization(String),
    #[error("MERRA-2 deposition is missing for snow-covered step at {time}")]
    MissingDeposition { time: DateTime<Utc> },
    /// Soil numerical failure is a separate registered stop class.
    #[error("soil solver failure at time step {step} ({time}): {message}")]
    SoilSolver {
        step: usize,
        time: DateTime<Utc>,
        message: String,
        partial: Box<RunOutput>,
    },
    /// A NaN or infinity in the state or checked surface budget while the
    /// column holds snow. `partial` contains output through the last completed
    /// output interval; `step` counts forcing steps from 1.
    #[error("non-finite state at time step {step} ({time}): {field}")]
    NonFinite {
        step: usize,
        time: DateTime<Utc>,
        field: String,
        partial: Box<RunOutput>,
    },
}

/// Station metadata the output needs.
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

/// Everything a run writes.
#[derive(Debug)]
pub struct RunOutput {
    pub smet: String,
    pub pro: String,
    pub steps: usize,
    /// The mass ledger, under `MASS_REPORT=LEDGER` only.
    pub mass: Option<String>,
    pub diagnostics: Option<RunDiagnostics>,
    /// Per-step Chan branches and daily SLAIR mass reconciliation, candidate only.
    pub chan_csv: Option<String>,
    /// Initial density and forcing conditions for newly deposited elements.
    pub density_events: Option<String>,
    /// Dirichlet lower-boundary reconstruction, enabled by
    /// `PI_SNOW_MELTOUT_DIAGNOSTICS=1`.
    pub meltout_diagnostic: Option<String>,
    /// Pinned-node storage accounting for the amended A4 control proof.
    pub dirichlet_storage_diagnostic: Option<String>,
    /// Full candidate physics ledger, enabled only in registered diagnostic runs.
    pub meltout_candidate_diagnostic: Option<String>,
    /// Keyed hourly soil state and aggregate diagnostics.
    pub soil: Option<crate::soil_column::output::Data>,
}

/// Hourly accumulators for fields the reference averages or sums.
#[derive(Default)]
struct Accum {
    n: f64,
    seconds: f64,
    qs: f64,
    ql: f64,
    qr: f64,
    qg0: f64,
    olwr: f64,
    ilwr: f64,
    lwr_net: f64,
    oswr: f64,
    iswr: f64,
    iswr_h: f64,
    iswr_dir: f64,
    iswr_diff: f64,
    qw: f64,
    albedo: f64,
    snow: f64,
    rain: f64,
    runoff: f64,
    sublimation: f64,
}

pub fn run(
    cfg: &RunConfig,
    site: &Site,
    forcing: &Smet,
    start: Column,
    end: DateTime<Utc>,
) -> Result<RunOutput, RunError> {
    run_with_options(cfg, &PhysicsOptions::default(), site, forcing, start, end)
}

/// `run` with physics chosen by `--physics`; the default options are Phase 2.
pub fn run_with_options(
    cfg: &RunConfig,
    options: &PhysicsOptions,
    site: &Site,
    forcing: &Smet,
    start: Column,
    end: DateTime<Utc>,
) -> Result<RunOutput, RunError> {
    run_with_options_and_diagnostics_and_deposition(
        cfg, options, site, forcing, start, end, None, None, None, None,
    )
}

/// The default entry point plus an explicit report-only collection mode.
pub fn run_with_diagnostics(
    cfg: &RunConfig,
    options: &PhysicsOptions,
    site: &Site,
    forcing: &Smet,
    start: Column,
    end: DateTime<Utc>,
    diagnostic_mode: Option<DiagnosticMode>,
) -> Result<RunOutput, RunError> {
    run_with_options_and_diagnostics_and_deposition(
        cfg,
        options,
        site,
        forcing,
        start,
        end,
        diagnostic_mode,
        None,
        None,
        None,
    )
}

/// Run with the hourly independent MERRA-2 record required by the Chan arm.
pub fn run_with_options_and_deposition(
    cfg: &RunConfig,
    options: &PhysicsOptions,
    site: &Site,
    forcing: &Smet,
    start: Column,
    end: DateTime<Utc>,
    merra2: Option<&DepositionSeries>,
) -> Result<RunOutput, RunError> {
    run_with_options_and_diagnostics_and_deposition(
        cfg, options, site, forcing, start, end, None, merra2, None, None,
    )
}

/// Replay a bound warm-up soil state beside the rig's original snow state.
pub fn run_with_soil_state(
    cfg: &RunConfig,
    options: &PhysicsOptions,
    site: &Site,
    forcing: &Smet,
    start: Column,
    end: DateTime<Utc>,
    diagnostic_mode: Option<DiagnosticMode>,
    soil: crate::soil_column::SoilColumn,
) -> Result<RunOutput, RunError> {
    run_with_options_and_diagnostics_and_deposition(
        cfg,
        options,
        site,
        forcing,
        start,
        end,
        diagnostic_mode,
        None,
        Some(soil),
        None,
    )
}

/// A5.3: the bound runner shares one predicate cache among a unit-season's arms.
pub fn run_with_soil_cache(
    cfg: &RunConfig,
    options: &PhysicsOptions,
    site: &Site,
    forcing: &Smet,
    start: Column,
    end: DateTime<Utc>,
    diagnostic_mode: Option<DiagnosticMode>,
    initial_soil: Option<crate::soil_column::SoilColumn>,
    cache: crate::soil_column::bare_cache::Shared,
) -> Result<RunOutput, RunError> {
    run_with_options_and_diagnostics_and_deposition(
        cfg,
        options,
        site,
        forcing,
        start,
        end,
        diagnostic_mode,
        None,
        initial_soil,
        Some(cache),
    )
}

fn run_with_options_and_diagnostics_and_deposition(
    cfg: &RunConfig,
    options: &PhysicsOptions,
    site: &Site,
    forcing: &Smet,
    start: Column,
    end: DateTime<Utc>,
    diagnostic_mode: Option<DiagnosticMode>,
    merra2: Option<&DepositionSeries>,
    initial_soil: Option<crate::soil_column::SoilColumn>,
    predicate_cache: Option<crate::soil_column::bare_cache::Shared>,
) -> Result<RunOutput, RunError> {
    options
        .soil
        .validate(
            options.ground_boundary == GroundBoundary::SoilColumn,
            options.total_melt == TotalMelt::CrocusEq116,
        )
        .map_err(RunError::IncompatiblePhysics)?;
    if (initial_soil.is_some() || predicate_cache.is_some())
        && options.ground_boundary != GroundBoundary::SoilColumn
    {
        return Err(RunError::IncompatiblePhysics(
            "replayed soil state requires GROUND_BOUNDARY=SOIL_COLUMN",
        ));
    }
    if options.ground_boundary == GroundBoundary::SoilColumn
        && [
            "PI_SNOW_MELTOUT_DIAGNOSTICS",
            "PI_SNOW_MELTOUT_CANDIDATE_DIAGNOSTICS",
        ]
        .iter()
        .any(|key| std::env::var_os(key).is_some_and(|v| v == "1"))
    {
        return Err(RunError::IncompatiblePhysics(
            "native melt-out boundary diagnostics do not describe SOIL_COLUMN",
        ));
    }
    let field = |name: &'static str| {
        forcing
            .field_index(name)
            .ok_or(RunError::MissingField(name))
    };
    let (i_ta, i_rh, i_vw, i_dw) = (field("TA")?, field("RH")?, field("VW")?, field("DW")?);
    let (i_iswr, i_ilwr, i_psum) = (field("ISWR")?, field("ILWR")?, field("PSUM")?);
    let i_tsg = forcing.field_index("TSG");

    let scheme = match (options.turbulence, cfg.stability) {
        (Turbulence::Louis1979, _) => Scheme::Louis,
        (Turbulence::FromIni, Stability::MoHoltslag) => Scheme::MoHoltslag,
        (Turbulence::FromIni, Stability::Richardson) => Scheme::Louis,
    };
    let scheme = match (options.surface_layer, scheme) {
        (SurfaceLayer::Phase2, s) => s,
        (SurfaceLayer::ScalarHeightNeutralWind, Scheme::MoHoltslag) => {
            Scheme::MoHoltslagScalarHeight
        }
        (SurfaceLayer::ZetaLimitClm5, Scheme::MoHoltslag) => Scheme::MoHoltslagZetaLimited,
        _ => {
            return Err(RunError::IncompatiblePhysics(
                "SURFACE_LAYER other than PHASE2 needs ATMOSPHERIC_STABILITY = MO_HOLTSLAG and no TURBULENCE key",
            ));
        }
    };
    if options.snow_albedo == SnowAlbedo::Aoki2003TempRate
        && options.surface_layer != SurfaceLayer::ZetaLimitClm5
    {
        return Err(RunError::IncompatiblePhysics(
            "AOKI2003_TEMP_RATE requires SURFACE_LAYER=ZETA_LIMIT_CLM5",
        ));
    }
    if options.snow_albedo == SnowAlbedo::Chan2022Merra2
        && options.surface_layer != SurfaceLayer::ZetaLimitClm5
    {
        return Err(RunError::IncompatiblePhysics(
            "CHAN2022_MERRA2 requires SURFACE_LAYER=ZETA_LIMIT_CLM5",
        ));
    }
    if options.snow_albedo == SnowAlbedo::Chan2022Merra2 && merra2.is_none() {
        return Err(RunError::IncompatiblePhysics(
            "CHAN2022_MERRA2 requires --merra2-deposition CSV",
        ));
    }
    let density_scheme = match options.fresh_snow_density {
        FreshSnowDensity::A76Anderson1976 => DensityScheme::Anderson1976,
        FreshSnowDensity::Phase2 => match cfg.new_snow_density {
            NewSnowDensity::LehningNew => DensityScheme::Pahaut,
            NewSnowDensity::Zwart => DensityScheme::Zwart,
        },
    };
    let conductivity = Conductivity::Yen1981;
    let pressure = pressure_at_altitude(site.altitude);

    let mut column = start;
    let soil = RefCell::new(if options.ground_boundary == GroundBoundary::SoilColumn {
        use crate::soil_column::{
            output::Data,
            runtime::{InitializationContext, Runtime},
        };
        let mut runtime = if let Some(state) = initial_soil {
            Runtime::from_column(options.soil, state, &mut column)
        } else {
            let context = InitializationContext::from_forcing(options.soil, forcing)
                .map_err(RunError::IncompatiblePhysics)?;
            Runtime::new(options.soil, context, &mut column)
        }
        .map_err(RunError::SoilInitialization)?;
        runtime.predicate_cache = predicate_cache;
        Some(Data::new(runtime))
    } else {
        None
    });
    let mut aoki_state = AokiAlbedo::default();
    if options.snow_albedo == SnowAlbedo::Aoki2003TempRate && !column.elements.is_empty() {
        aoki_state.start_pack();
    }
    let chan_active = options.snow_albedo == SnowAlbedo::Chan2022Merra2;
    let mut slair = Slair::default();
    if chan_active && !column.elements.is_empty() {
        slair.initialize_clean(column.swe());
    }
    let (
        mut daily_snowfall,
        mut daily_clean_deposition,
        mut daily_bc,
        mut daily_dust,
        mut daily_sublimation,
        mut daily_runoff,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut daily_branch_counts = [0usize; 3];
    let mut chan_csv = chan_active.then(|| {
        String::from("kind,timestamp_utc,HS_m,r_um,BC_ng_g,dust_ng_g,c_eqv_ng_g,branch,alpha_vis,layer_count,slair_mass_kg_m2,engine_swe_kg_m2,mass_drift_pre_reconcile_kg_m2,modal_branch\n")
    });
    let mut deposition =
        Deposition::new(cfg.height_new_element, 0.5e-3 * cfg.new_snow_grain_size_mm);
    let header = ProHeader {
        station_name: site.station_name.clone(),
        latitude: site.latitude,
        longitude: site.longitude,
        altitude: site.altitude,
        slope_angle_deg: site.slope_angle,
        slope_azimuth_deg: site.slope_azimuth,
    };
    let mut pro_text = pro::write_header(&header);
    let mut rows: Vec<OutputRow> = Vec::new();
    let mut acc = Accum::default();
    let mut previous: Option<DateTime<Utc>> = None;
    let mut first_output = true;
    let mut steps = 0usize;
    let mut ledger = (options.mass_report == MassReport::Ledger).then(Ledger::new);
    let mut density_events = (options.fresh_snow_density != FreshSnowDensity::Phase2
        || options.dry_settlement != DrySettlement::Br92)
        .then(|| String::from("timestamp_utc,event,element,initial_density_kg_m3,element_mass_kg_m2,element_height_m,air_temp_c,surface_temp_c,relative_humidity,wind_m_s\n"));
    let mut density_event_number = 0usize;
    let mut interval = Interval::default();
    let initial_time = forcing.rows.first().map_or(end, |r| r.timestamp);
    let mut collector =
        diagnostic_mode.map(|mode| Collector::new(mode, &column.node_temperature, initial_time));
    let mut meltout_diagnostic = std::env::var_os("PI_SNOW_MELTOUT_DIAGNOSTICS")
        .is_some_and(|value| value == "1")
        .then(MeltoutDiagnostic::new);
    let mut dirichlet_storage_diagnostic = std::env::var_os("PI_SNOW_MELTOUT_DIAGNOSTICS")
        .is_some_and(|value| value == "1")
        .then(DirichletStorageDiagnostic::new);
    let mut meltout_candidate_diagnostic =
        std::env::var_os("PI_SNOW_MELTOUT_CANDIDATE_DIAGNOSTICS")
            .is_some_and(|value| value == "1")
            .then(MeltoutCandidateDiagnostic::new);
    let make_output = |rows: Vec<OutputRow>,
                       pro_text: String,
                       steps: usize,
                       ledger: Option<Ledger>,
                       diagnostics: Option<RunDiagnostics>,
                       density_events: Option<String>,
                       chan_csv: Option<String>,
                       meltout_diagnostic: Option<String>,
                       dirichlet_storage_diagnostic: Option<String>,
                       meltout_candidate_diagnostic: Option<String>| {
        let soil = soil.borrow().clone();
        let smet = pi_snow_io::smet::write_output_smet_with_acdd(
            &site.station_id,
            &site.station_name,
            site.latitude,
            site.longitude,
            site.altitude,
            site.slope_angle,
            site.slope_azimuth,
            &rows,
            cfg.acdd_write,
        );
        RunOutput {
            smet: match soil.as_ref() {
                Some(data) => data.extend_smet(&smet),
                None => smet,
            },
            soil,
            pro: pro_text,
            steps,
            mass: ledger.map(Ledger::into_text),
            diagnostics,
            density_events,
            chan_csv,
            meltout_diagnostic,
            dirichlet_storage_diagnostic,
            meltout_candidate_diagnostic,
        }
    };

    // A failed keyed solve keeps completed intervals and its distinct stop class.
    macro_rules! soil_try {
        ($result:expr, $time:expr) => {
            match $result {
                Ok(value) => value,
                Err(message) => {
                    let partial = make_output(
                        rows.clone(),
                        pro_text.clone(),
                        steps,
                        ledger.take(),
                        collector.as_ref().map(|c| c.report.clone()),
                        density_events.clone(),
                        chan_csv.clone(),
                        meltout_diagnostic.as_ref().map(|d| d.clone().into_text()),
                        dirichlet_storage_diagnostic
                            .as_ref()
                            .map(|d| d.clone().into_text()),
                        meltout_candidate_diagnostic
                            .as_ref()
                            .map(|d| d.clone().into_text()),
                    );
                    return Err(RunError::SoilSolver {
                        step: steps,
                        time: $time,
                        message,
                        partial: Box::new(partial),
                    });
                }
            }
        };
    }

    for row in &forcing.rows {
        let t = row.timestamp;
        if t > end {
            break;
        }
        let Some(t_prev) = previous.replace(t) else {
            continue;
        };
        let dt = (t - t_prev).num_seconds() as f64;
        if dt <= 0.0 {
            continue;
        }
        steps += 1;
        let step_mass_before = column.swe() + deposition.buffered() + column.surface_hoar_mass;
        let interval_before = interval;
        let v = |i: usize| row.values[i];
        let (Some(ta), Some(rh), Some(vw), Some(iswr_h), Some(ilwr)) =
            (v(i_ta), v(i_rh), v(i_vw), v(i_iswr), v(i_ilwr))
        else {
            continue;
        };
        let dw = v(i_dw).unwrap_or(NODATA);
        let psum = v(i_psum).unwrap_or(0.0).max(0.0);
        let tsg = i_tsg.and_then(v).unwrap_or(T_MELT);
        let soil_before = soil.borrow().as_ref().map(|data| data.snapshot());

        // 1. Radiation on the slope.
        let mid = t - Duration::milliseconds((dt * 500.0) as i64);
        let sun = solar::sun_position(mid, site.latitude, site.longitude);
        let (_, diffuse_h) = solar::split_global(iswr_h, &sun, mid);
        let ground_albedo = if column.elements.is_empty() {
            site.soil_albedo
        } else if chan_active {
            let (bc, dust) = slair.near_surface_concentrations();
            let radius_um = 0.5 * albedo::surface_optical_diameter(&column) * 1e6;
            let (visible, _) =
                chan_albedo::visible_albedo(column.snow_height(), radius_um, bc, dust);
            albedo::shortwave_with_visible_band(&column, 1.0, visible).albedo
        } else {
            albedo::shortwave(&column, 1.0, pressure).albedo
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
        let had_snow = !column.elements.is_empty();
        let new_snow_density = (options.snow_albedo == SnowAlbedo::Aoki2003TempRate)
            .then(|| density_scheme.density(&conditions));
        let deposited = deposition.add(&mut column, solid, &conditions, density_scheme);
        if deposited > 0.0 {
            if let Some(events) = density_events.as_mut() {
                let density = density_scheme.density(&conditions);
                let element_mass = cfg.height_new_element * density;
                let count = (deposited / element_mass).round() as usize;
                density_event_number += 1;
                for element in 1..=count {
                    events.push_str(&format!(
                        "{},{density_event_number},{element},{density:.9},{element_mass:.9},{:.9},{:.6},{:.6},{:.9},{:.6}\n",
                        t.to_rfc3339(),
                        cfg.height_new_element,
                        ta - T_MELT,
                        t_surface - T_MELT,
                        rh,
                        vw,
                    ));
                }
            }
        }
        let deposited_height = new_snow_density
            .filter(|density| *density > 0.0)
            .map_or(0.0, |density| deposited / density);
        let new_pack = !had_snow && !column.elements.is_empty();
        let (bc_mass, dust_mass) = if chan_active && had_snow {
            merra2
                .and_then(|series| series.step_mass(mid, dt))
                .ok_or(RunError::MissingDeposition { time: mid })?
        } else {
            (0.0, 0.0)
        };
        if chan_active {
            daily_snowfall += deposited;
            daily_bc += bc_mass;
            daily_dust += dust_mass;
        }
        acc.snow += deposited;
        acc.rain += liquid;
        interval.solid_precipitation += solid;
        interval.deposited += deposited;
        let mut rain_rate = 0.0;
        if liquid > 0.0 {
            if let Some(top) = column.elements.last_mut() {
                top.composition.water += liquid / (RHO_WATER * top.thickness);
                top.composition.air = (1.0 - top.composition.ice - top.composition.water).max(0.0);
                rain_rate = liquid / dt;
                interval.rain_to_pack += liquid;
            } else {
                acc.runoff += liquid;
                interval.rain_to_ground += liquid;
            }
        }

        if options.thin_top_merge == pi_snow_core::options::ThinTopMerge::Clm4_010Mm {
            thin_top_merge::merge_top_once(&mut column, 0.010);
        }

        let mut trace_first_nonfinite_substep = None;
        let mut trace_pre_thickness = Vec::new();
        let mut trace_pre_grain_type = Vec::new();
        let mut trace_pre_density = Vec::new();
        let mut trace_pre_nodes = Vec::new();
        let mut trace_post_solve_nodes = Vec::new();
        let mut trace_sources = Vec::new();
        let mut trace_surface_iterations = Vec::new();
        let mut trace_shortwave = 0.0;

        if soil.borrow().is_some() {
            let result = soil
                .borrow_mut()
                .as_mut()
                .unwrap()
                .runtime
                .reconcile(&mut column);
            soil_try!(result, t);
        }
        let hs = column.snow_height();
        let params = SurfaceParameters {
            emissivity: surface::SNOW_EMISSIVITY,
            z_wind: (cfg.height_of_wind_value - hs).max(MIN_MEASUREMENT_HEIGHT),
            z_scalar: (cfg.height_of_meteo_values - hs).max(MIN_MEASUREMENT_HEIGHT),
            z0: cfg.roughness_length,
            scheme,
        };
        let trace_budget: Budget;

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
            if soil.borrow().is_some() {
                let atm = Atmosphere {
                    absorbed_shortwave: (1.0 - site.soil_albedo) * iswr,
                    rain_rate: liquid / dt,
                    ..atm
                };
                let result = soil
                    .borrow_mut()
                    .as_mut()
                    .unwrap()
                    .runtime
                    .bare_step_for_row(&mut column, dt, &atm, &params, t.timestamp());
                (soil_try!(result, t), site.soil_albedo, 0.0)
            } else {
                column.node_temperature[0] = tsg;
                (surface::budget(&atm, &params, tsg), site.soil_albedo, 0.0)
            }
        } else {
            // 3. Shortwave through the pack.
            let absorbed = if options.snow_albedo == SnowAlbedo::Aoki2003TempRate {
                let bands = aoki_state.advance(
                    cfg.step_seconds,
                    t_surface - T_MELT,
                    deposited_height,
                    new_pack,
                );
                aoki_albedo::shortwave(&column, iswr, bands)
            } else if chan_active {
                let (bc, dust) = slair.near_surface_concentrations();
                let radius_um = 0.5 * albedo::surface_optical_diameter(&column) * 1e6;
                let (visible, branch) = chan_albedo::visible_albedo(hs, radius_um, bc, dust);
                let branch_index = match branch {
                    chan_albedo::Branch::Malinka => 0,
                    chan_albedo::Branch::Rozenberg => 1,
                    chan_albedo::Branch::He => 2,
                };
                daily_branch_counts[branch_index] += 1;
                if let Some(csv) = chan_csv.as_mut() {
                    csv.push_str(&format!(
                        "step,{},{:.8},{:.6},{:.6},{:.6},{:.6},{},{:.8},{},{:.8},{:.8},,\n",
                        t.to_rfc3339(),
                        hs,
                        radius_um,
                        bc,
                        dust,
                        chan_albedo::equivalent_bc_concentration(bc, dust),
                        chan_albedo::branch_name(branch),
                        visible,
                        slair.layers.len(),
                        slair.total_mass(),
                        column.swe()
                    ));
                }
                albedo::shortwave_with_visible_band(&column, iswr, visible)
            } else {
                albedo::shortwave(&column, iswr, pressure)
            };
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
            if matches!(diagnostic_mode, Some(DiagnosticMode::Trace)) {
                trace_pre_thickness = column.elements.iter().map(|e| e.thickness).collect();
                trace_pre_grain_type = layers(&column, options)
                    .iter()
                    .map(|l| l.grain_type)
                    .collect();
                trace_pre_density = column.elements.iter().map(Element::density).collect();
                trace_pre_nodes = column.node_temperature.clone();
                trace_sources = absorbed.per_element.clone();
            }
            let before = thermal::internal_energy(&column);
            let pack_before = PackState::capture(&column);
            let t0_pre = column.node_temperature[0];
            let bottom_before = column.elements.first().map(|bottom| {
                (
                    bottom.thickness,
                    bottom.density(),
                    0.5 * thermal::volumetric_heat_capacity(bottom) * bottom.thickness,
                )
            });
            let ground = match options.ground_boundary {
                GroundBoundary::Phase2 => Ground::Temperature(tsg),
                GroundBoundary::GeoHeatFlux => Ground::Flux(cfg.geo_heat),
                GroundBoundary::SoilColumn => Ground::Flux(0.0), // Unused by the coupled branch.
            };
            let record_linearisation = options.ground_boundary != GroundBoundary::Phase2
                || options.total_melt != TotalMelt::Phase2
                || meltout_candidate_diagnostic.is_some();
            let soil_top_before = soil
                .borrow()
                .as_ref()
                .map(|data| data.runtime.top_energy_j_m2);
            let (b, linear_a, linear_b, solved_surface_temperature) = if soil.borrow().is_some() {
                let mut observer = |iteration| trace_surface_iterations.push(iteration);
                let tracing = matches!(diagnostic_mode, Some(DiagnosticMode::Trace));
                let result = soil
                    .borrow_mut()
                    .as_mut()
                    .unwrap()
                    .runtime
                    .snow_step_observed(
                        &mut column,
                        dt,
                        &atm,
                        &params,
                        &absorbed.per_element,
                        absorbed.to_ground,
                        conductivity,
                        options.surface_linearisation,
                        if tracing { Some(&mut observer) } else { None },
                    );
                let b = soil_try!(result, t);
                if matches!(diagnostic_mode, Some(DiagnosticMode::Trace)) {
                    trace_post_solve_nodes = column.node_temperature.clone();
                    trace_shortwave = atm.absorbed_shortwave;
                }
                let (a, linear_b) = soil
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .runtime
                    .last_snow_linearisation
                    .unwrap();
                (b, a, linear_b, *column.node_temperature.last().unwrap())
            } else if record_linearisation {
                if options.surface_linearisation != SurfaceLinearisation::Phase2 {
                    return Err(RunError::IncompatiblePhysics(
                        "engine-meltout diagnostics require SURFACE_LINEARISATION=PHASE2",
                    ));
                }
                let step = meltout_candidate::step_general_recorded(
                    &mut column,
                    &atm,
                    &params,
                    ground,
                    &absorbed.per_element,
                    dt,
                    conductivity,
                );
                (step.budget, step.a, step.b, step.solved_surface_temperature)
            } else {
                let step_surface = match options.surface_linearisation {
                    SurfaceLinearisation::Phase2 => surface::step_general,
                    SurfaceLinearisation::RadiativeFloor => surface_numerics::step_general,
                };
                let b = if matches!(diagnostic_mode, Some(DiagnosticMode::Trace)) {
                    let linearise = match options.surface_linearisation {
                        SurfaceLinearisation::Phase2 => surface::linearise,
                        SurfaceLinearisation::RadiativeFloor => surface_numerics::linearise,
                    };
                    surface_diagnostic::step_general(
                        &mut column,
                        &atm,
                        &params,
                        ground,
                        &absorbed.per_element,
                        dt,
                        conductivity,
                        linearise,
                        |iteration| trace_surface_iterations.push(iteration),
                    )
                } else {
                    step_surface(
                        &mut column,
                        &atm,
                        &params,
                        ground,
                        &absorbed.per_element,
                        dt,
                        conductivity,
                    )
                };
                if matches!(diagnostic_mode, Some(DiagnosticMode::Trace)) {
                    trace_post_solve_nodes = column.node_temperature.clone();
                    trace_shortwave = atm.absorbed_shortwave;
                }
                let surface_k = column.node_temperature.last().copied().unwrap_or(tsg);
                (b, f64::NAN, f64::NAN, surface_k)
            };
            trace_budget = b;
            trace_substep::capture_first_nonfinite(
                &mut trace_first_nonfinite_substep,
                diagnostic_mode,
                "surface and conduction solve",
                &column,
                &trace_budget,
                deposition.buffered(),
            );
            let sources: f64 = absorbed.per_element.iter().sum();
            let qg0 = if let Some(top_before) = soil_top_before {
                -(soil.borrow().as_ref().unwrap().runtime.top_energy_j_m2 - top_before) / dt
                    + absorbed.to_ground
            } else {
                (thermal::internal_energy(&column) - before) / dt - b.total - sources
            };
            let thermal_residual =
                (thermal::internal_energy(&column) - before) - (b.total + sources + qg0) * dt;
            let nonlinear_surface =
                surface::budget(&atm, &params, solved_surface_temperature).total;
            let source0 = absorbed.per_element.first().copied().unwrap_or(0.0);
            let half_source0 = 0.5 * source0;
            let discrete_flux = match (
                options.ground_boundary,
                bottom_before,
                column.node_temperature.get(1),
            ) {
                (GroundBoundary::Phase2, Some((thickness, density, _)), Some(&t1)) => {
                    -(conductivity.of_density(density) / thickness * (t1 - tsg) + half_source0)
                }
                (GroundBoundary::GeoHeatFlux, _, _) => cfg.geo_heat,
                _ => f64::NAN,
            };
            let ground_reset_storage = if options.ground_boundary == GroundBoundary::Phase2 {
                bottom_before.map_or(0.0, |(_, _, c0)| c0 * (column.node_temperature[0] - t0_pre))
            } else {
                0.0
            };
            let available_melt_energy = b.total + sources + discrete_flux;
            let energy_criterion =
                meltout_candidate::decide(pack_before, available_melt_energy, dt);
            let decision = if options.total_melt == TotalMelt::CrocusEq116 {
                energy_criterion
            } else {
                meltout_candidate::TotalMeltDecision {
                    available_j_m2: energy_criterion.available_j_m2,
                    requirement_j_m2: energy_criterion.requirement_j_m2,
                    ..Default::default()
                }
            };
            let mut total_melt_removed = (0.0, 0.0);
            if decision.fired {
                total_melt_removed = meltout_candidate::clear_snow(&mut column);
                let exported_water = total_melt_removed.0 + total_melt_removed.1;
                acc.runoff += exported_water;
                interval.runoff += exported_water;
                column.node_temperature[0] = tsg;
            }
            if let (Some(diagnostic), Some((thickness, density, _)), Some(&t1_post)) = (
                meltout_diagnostic.as_mut(),
                bottom_before,
                column.node_temperature.get(1),
            ) {
                let conductance = conductivity.of_density(density) / thickness;
                diagnostic.push(
                    t,
                    column.elements.len(),
                    thickness,
                    density,
                    source0,
                    tsg,
                    t1_post,
                    conductance,
                    qg0,
                );
            }
            if let (
                Some(diagnostic),
                Some((thickness, density, c0)),
                Some(&t1_post),
                Some(&t0_post),
            ) = (
                dirichlet_storage_diagnostic.as_mut(),
                bottom_before,
                column.node_temperature.get(1),
                column.node_temperature.first(),
            ) {
                let conductance0 = conductivity.of_density(density) / thickness;
                diagnostic.push(
                    t,
                    t0_pre,
                    t0_post,
                    c0,
                    dt,
                    conductance0,
                    t1_post,
                    source0,
                    qg0,
                );
            }

            // 5. Phase change and water.
            let water_before: Vec<f64> = column.elements.iter().map(Element::water_mass).collect();
            let n_before = column.elements.len();
            let pc = if soil.borrow().is_some() {
                let result = soil
                    .borrow_mut()
                    .as_mut()
                    .unwrap()
                    .runtime
                    .phase_change(&mut column);
                soil_try!(result, t)
            } else {
                water::phase_change(&mut column)
            };
            if column.elements.len() == n_before {
                for (e, w) in column.elements.iter_mut().zip(water_before) {
                    if e.water_mass() < w - 1e-9 {
                        e.micro.marker |= MARK_REFROZEN;
                    }
                }
            }
            let runoff = if soil.borrow().is_some() {
                let result = soil
                    .borrow_mut()
                    .as_mut()
                    .unwrap()
                    .runtime
                    .cascade(&mut column, options.water_retention);
                soil_try!(result, t)
            } else {
                water::cascade_with(&mut column, options.water_retention)
            };
            acc.runoff += pc.drained + runoff;
            interval.drained += pc.drained;
            interval.runoff += runoff;
            trace_substep::capture_first_nonfinite(
                &mut trace_first_nonfinite_substep,
                diagnostic_mode,
                "phase change and water",
                &column,
                &trace_budget,
                deposition.buffered(),
            );
            if chan_active {
                daily_runoff += pc.drained + runoff;
            }

            // 6. Sublimation or deposition on the top element.
            let vapour = b.vapour_mass * dt;
            interval.vapour_flux += vapour;
            let sublimated_away = options.sublimation_melt_out == SublimationMeltOut::RemoveElement
                && column.elements.len() > column.soil_elements
                && column
                    .elements
                    .last()
                    .is_some_and(|top| top.ice_mass() + vapour <= 0.0);
            if sublimated_away {
                // engine-finite-guard-v1: the element goes with its top node (its
                // sensible energy is that of a vanishing thickness), its liquid
                // water is runoff, and no zero-thickness element reaches the
                // next conduction solve.
                let gone = column.elements.pop().unwrap();
                column.node_temperature.pop();
                interval.sublimation_applied += gone.ice_mass();
                let water = gone.water_mass();
                acc.runoff += water;
                interval.drained += water;
            } else if let Some(top) = column.elements.last_mut() {
                let ice = (top.ice_mass() + vapour).max(0.0);
                let applied = ice - top.ice_mass();
                if applied > 0.0 {
                    interval.deposition_applied += applied;
                } else {
                    interval.sublimation_applied -= applied;
                }
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
            if chan_active {
                daily_sublimation += (-vapour).max(0.0);
                if vapour < 0.0 {
                    daily_clean_deposition += -vapour;
                }
            }
            trace_substep::capture_first_nonfinite(
                &mut trace_first_nonfinite_substep,
                diagnostic_mode,
                "sublimation or deposition",
                &column,
                &trace_budget,
                deposition.buffered(),
            );

            // 7. Settlement.
            settlement::step(&mut column, dt, density_settlement_viscosity(options));
            trace_substep::capture_first_nonfinite(
                &mut trace_first_nonfinite_substep,
                diagnostic_mode,
                "settlement",
                &column,
                &trace_budget,
                deposition.buffered(),
            );

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
                if options.wet_grain_growth == WetGrainGrowth::Brun1989 {
                    wet_growth::step(&mut e.micro, lwc, dt);
                }
                e.grain_history = grain_classification::update_history(
                    e.grain_history,
                    &e.micro,
                    temps[i],
                    lwc / 100.0,
                );
                e.age_days += dt / 86_400.0;
            }
            trace_substep::capture_first_nonfinite(
                &mut trace_first_nonfinite_substep,
                diagnostic_mode,
                "metamorphism",
                &column,
                &trace_budget,
                deposition.buffered(),
            );

            // 9. Merge thin elements; drop anything melted to nothing.
            new_snow::merge_thin_elements(&mut column, cfg.minimum_element_length);
            trace_substep::capture_first_nonfinite(
                &mut trace_first_nonfinite_substep,
                diagnostic_mode,
                "merge",
                &column,
                &trace_budget,
                deposition.buffered(),
            );
            if let Some(diagnostic) = meltout_candidate_diagnostic.as_mut() {
                let end_mass = column.swe() + deposition.buffered() + column.surface_hoar_mass;
                let step_runoff = (interval.runoff - interval_before.runoff)
                    + (interval.drained - interval_before.drained)
                    + (interval.rain_to_ground - interval_before.rain_to_ground);
                let deposited_vapour =
                    interval.deposition_applied - interval_before.deposition_applied;
                let sublimated = interval.sublimation_applied - interval_before.sublimation_applied;
                let mass_residual = end_mass
                    - (step_mass_before + solid + liquid + deposited_vapour
                        - sublimated
                        - step_runoff);
                let pack_after = PackState::capture(&column);
                let snow_runoff = (interval.runoff - interval_before.runoff)
                    + (interval.drained - interval_before.drained);
                let energy_residual = meltout_candidate::energy_residual(
                    pack_before,
                    pack_after,
                    available_melt_energy,
                    ground_reset_storage,
                    snow_runoff,
                    deposited_vapour - sublimated,
                    solved_surface_temperature,
                    decision,
                    dt,
                );
                diagnostic.push(
                    t,
                    match options.ground_boundary {
                        GroundBoundary::Phase2 => "DIRICHLET",
                        GroundBoundary::GeoHeatFlux => "GEO_HEAT_FLUX",
                        GroundBoundary::SoilColumn => {
                            unreachable!("soil dispatch precedes native solve")
                        }
                    },
                    if options.ground_boundary == GroundBoundary::GeoHeatFlux {
                        cfg.geo_heat
                    } else {
                        0.0
                    },
                    linear_a,
                    linear_b,
                    b.total,
                    nonlinear_surface,
                    absorbed.surface,
                    absorbed.to_ground,
                    b.olwr,
                    b.lwr_net,
                    b.sensible,
                    b.latent,
                    b.rain,
                    source0,
                    half_source0,
                    discrete_flux,
                    qg0,
                    ground_reset_storage,
                    pack_before,
                    decision,
                    total_melt_removed.0 + pc.melted,
                    pc.refrozen,
                    LH_FUSION * (total_melt_removed.0 + pc.melted - pc.refrozen),
                    step_runoff,
                    sublimated,
                    step_mass_before,
                    end_mass,
                    mass_residual,
                    energy_residual,
                    thermal_residual,
                    solved_surface_temperature,
                    column.node_temperature.first().copied().unwrap_or(tsg),
                );
            }
            if options.snow_albedo == SnowAlbedo::Aoki2003TempRate && column.elements.is_empty() {
                aoki_state.end_pack();
            }
            (b, absorbed.albedo, qg0)
        };

        if soil.borrow().is_some() {
            let result = soil
                .borrow_mut()
                .as_mut()
                .unwrap()
                .runtime
                .reconcile(&mut column);
            soil_try!(result, t);
        }
        let current_min = column
            .node_temperature
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(node, value)| (node, *value));
        let crazy = column
            .node_temperature
            .iter()
            .enumerate()
            .find(|(_, value)| **value < cfg.t_crazy_min || **value > cfg.t_crazy_max)
            .map(|(node, value)| (node, *value))
            .or_else(|| {
                soil.borrow()
                    .as_ref()
                    .and_then(|data| data.crazy_temperature(cfg.t_crazy_min, cfg.t_crazy_max))
            });

        if let Some(collector) = collector.as_mut() {
            if crazy.is_none() {
                collector.observe_minimum(t, steps, &column.node_temperature);
            }
            if matches!(diagnostic_mode, Some(DiagnosticMode::Trace)) {
                collector.observe_trace(StepTrace {
                    step: steps,
                    time: t,
                    dt_seconds: dt,
                    air_temperature_k: ta,
                    relative_humidity: rh,
                    wind_m_s: vw,
                    incoming_shortwave_horizontal_w_m2: iswr,
                    incoming_longwave_w_m2: ilwr,
                    pressure_pa: pressure,
                    ground_temperature_k: tsg,
                    z_wind_m: params.z_wind,
                    z_scalar_m: params.z_scalar,
                    roughness_length_m: params.z0,
                    min_node: current_min.map(|(node, _)| node),
                    min_node_k: current_min.map(|(_, kelvin)| kelvin),
                    snowfall_m: solid,
                    rain_m: liquid,
                    deposited_kg_m2: deposited,
                    deposition_buffer_kg_m2: deposition.buffered(),
                    shortwave_absorbed_w_m2: trace_shortwave,
                    internal_sources_w_m2: trace_sources.clone(),
                    budget: [
                        budget.olwr,
                        budget.lwr_net,
                        budget.sensible,
                        budget.latent,
                        budget.rain,
                        budget.vapour_mass,
                        budget.total,
                    ],
                    inferred_ground_flux_w_m2: qg0,
                    pre_solve_element_thickness_m: trace_pre_thickness.clone(),
                    pre_solve_grain_type: trace_pre_grain_type.clone(),
                    pre_solve_density_kg_m3: trace_pre_density.clone(),
                    pre_solve_node_temperature_k: trace_pre_nodes.clone(),
                    post_solve_node_temperature_k: trace_post_solve_nodes.clone(),
                    post_step_element_thickness_m: column
                        .elements
                        .iter()
                        .map(|e| e.thickness)
                        .collect(),
                    post_step_node_temperature_k: column.node_temperature.clone(),
                    surface_iterations: trace_surface_iterations.clone(),
                    first_nonfinite_substep: trace_first_nonfinite_substep.clone(),
                });
            }
        }
        if let Some((node, t_node)) = crazy {
            let diagnostics = if let Some(c) = collector.as_mut() {
                c.report.refusal = Some(Refusal {
                    kind: "T_CRAZY",
                    kelvin: t_node,
                    time: t,
                    step: steps,
                    node,
                    detail: format!("outside [{}, {}] K", cfg.t_crazy_min, cfg.t_crazy_max),
                });
                c.observe_refusal();
                Some(c.report.clone())
            } else {
                None
            };
            let partial = make_output(
                rows.clone(),
                pro_text.clone(),
                steps,
                ledger.take(),
                diagnostics,
                density_events.clone(),
                chan_csv.clone(),
                meltout_diagnostic.as_ref().map(|d| d.clone().into_text()),
                dirichlet_storage_diagnostic
                    .as_ref()
                    .map(|d| d.clone().into_text()),
                meltout_candidate_diagnostic
                    .as_ref()
                    .map(|d| d.clone().into_text()),
            );
            return Err(RunError::Crazy {
                time: t,
                value: t_node,
                min: cfg.t_crazy_min,
                max: cfg.t_crazy_max,
                node,
                partial: Box::new(partial),
            });
        }

        if chan_active {
            if column.swe() <= 0.0 {
                slair.clear();
            }
            if t.hour() == 7 && t.minute() == 0 && t.second() == 0 {
                slair.snowfall(daily_snowfall, daily_bc, daily_dust);
                slair.add_clean_surface_mass(daily_clean_deposition);
                slair.sublimate(daily_sublimation);
                slair.melt(daily_runoff);
                let engine_swe = column.swe();
                let slair_mass = slair.total_mass();
                let drift = slair_mass - engine_swe;
                let max_count = daily_branch_counts.iter().copied().max().unwrap_or(0);
                let modal = ["malinka", "rozenberg", "he"]
                    .iter()
                    .zip(daily_branch_counts)
                    .filter_map(|(name, count)| (count == max_count && count > 0).then_some(*name))
                    .collect::<Vec<_>>()
                    .join("|");
                if let Some(csv) = chan_csv.as_mut() {
                    csv.push_str(&format!(
                        "daily,{},{:.8},,,,,,,{},{:.8},{:.8},{:.8},{}\n",
                        t.to_rfc3339(),
                        column.snow_height(),
                        slair.layers.len(),
                        slair_mass,
                        engine_swe,
                        drift,
                        modal
                    ));
                }
                slair.reconcile(column.swe());
                daily_branch_counts = [0; 3];
                (
                    daily_snowfall,
                    daily_clean_deposition,
                    daily_bc,
                    daily_dust,
                    daily_sublimation,
                    daily_runoff,
                ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            }
        }
        // The comparisons above let a NaN through. While the column holds snow,
        // stop on a non-finite state or checked budget term. Qg0 is excluded:
        // it reads pre-solve energy and can be NaN for one step while the ground
        // boundary resets the node and the state remains finite. Empty columns
        // remain exempt to preserve Phase 2's vanishing-pack output.
        // Validated configurations have zero soil elements; if soil elements
        // are introduced, revisit this empty-column scope.
        if let Some(field) = (!column.elements.is_empty() || soil.borrow().is_some())
            .then(|| {
                crate::finite::column(&column)
                    .or_else(|| crate::finite::budget(&budget, deposition.buffered()))
            })
            .flatten()
            .or_else(|| soil.borrow().as_ref().and_then(|data| data.nonfinite()))
        {
            if let Some(collector) = collector.as_mut() {
                collector.report.refusal = Some(Refusal {
                    kind: "NONFINITE",
                    kelvin: f64::NAN,
                    time: t,
                    step: steps,
                    node: usize::MAX,
                    detail: field.clone(),
                });
                collector.observe_refusal();
            }
            let partial = make_output(
                rows,
                pro_text,
                steps,
                ledger.take(),
                collector.as_ref().map(|c| c.report.clone()),
                density_events.clone(),
                chan_csv.clone(),
                meltout_diagnostic.as_ref().map(|d| d.clone().into_text()),
                dirichlet_storage_diagnostic
                    .as_ref()
                    .map(|d| d.clone().into_text()),
                meltout_candidate_diagnostic
                    .as_ref()
                    .map(|d| d.clone().into_text()),
            );
            return Err(RunError::NonFinite {
                step: steps,
                time: t,
                field,
                partial: Box::new(partial),
            });
        }

        if let Some(data) = soil.borrow_mut().as_mut() {
            data.finish_step(dt, soil_before.unwrap());
            data.assess_snow_mass(
                step_mass_before,
                column.swe() + deposition.buffered() + column.surface_hoar_mass,
                interval_before,
                interval,
            );
        }
        // 10. Accumulate, and write on the hour.
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

        let on_the_hour = t.minute() == 0 && t.second() == 0;
        if first_output || on_the_hour {
            first_output = false;
            rows.push(output_row(t, &acc, &column, ta, rh, vw, dw, tsg, options));
            if let Some(data) = soil.borrow_mut().as_mut() {
                data.output.push(t, &data.runtime.column);
            }
            pro_text.push_str(&pro::write_timestep(&profile(t, &column, options)));
            acc = Accum::default();
            if let Some(l) = ledger.as_mut() {
                l.push(t, column.swe(), deposition.buffered(), &interval);
            }
            interval = Interval::default();
        }
    }
    Ok(make_output(
        rows,
        pro_text,
        steps,
        ledger,
        collector.map(|c| c.report),
        density_events,
        chan_csv,
        meltout_diagnostic.map(MeltoutDiagnostic::into_text),
        dirichlet_storage_diagnostic.map(DirichletStorageDiagnostic::into_text),
        meltout_candidate_diagnostic.map(MeltoutCandidateDiagnostic::into_text),
    ))
}

fn history(e: &Element) -> History {
    History {
        refrozen: e.micro.marker & MARK_REFROZEN != 0,
        surface_hoar: e.micro.marker & MARK_SURFACE_HOAR != 0,
        wet: e.composition.water > 0.0,
    }
}

#[allow(clippy::too_many_arguments)]
fn output_row(
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
        // The reference writes VW_drift = VW and wind_trans24 = 0 in every row
        // of this configuration (lab notes); copied.
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
    // Profile minima of the indices, with their heights, as the reference
    // reports them; the ceiling and zero height on bare ground.
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
            // Nodata on bare ground and zS5 = 0 on every row are the
            // reference's output conventions, copied, not derived.
            r.set("zS5", 0.0);
            if let Some(index) = lwc_index_s5(c) {
                r.set("S5", index);
            }
        }
    }
    r
}

/// SMET `S5` under `S5=LWC_INDEX_MITTERER2013`: the LWC index over the snow
/// elements only (soil excluded), nodata where the written snow height is 0.
pub fn lwc_index_s5(c: &Column) -> Option<f64> {
    let snow = &c.elements[c.soil_elements..];
    stability::reported_lwc_index(
        snow.iter().map(|e| (e.thickness, e.composition.water)),
        c.snow_height(),
    )
}

/// Settled thickness of elements younger than `days`, m.
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

fn hardness(code: u16, density: f64) -> f64 {
    GrainClass::from_code(code)
        .and_then(|c| stability::monti_hardness(c, density))
        .unwrap_or(NODATA)
}

fn density_settlement_viscosity(options: &PhysicsOptions) -> DensitySettlementViscosity {
    let scheme = match options.dry_settlement {
        DrySettlement::Br92 => DensitySettlementScheme::Br92,
        DrySettlement::T11 => DensitySettlementScheme::T11,
        DrySettlement::K75 => DensitySettlementScheme::K75,
        DrySettlement::T11K75 => DensitySettlementScheme::T11K75,
    };
    DensitySettlementViscosity(scheme)
}

fn profile(t: DateTime<Utc>, c: &Column, options: &PhysicsOptions) -> ProTimestep {
    let layers = layers(c, options);
    let idx = stability::indices(&layers, c.snow_height());
    let tops = c.element_top_heights();
    let temps = c.element_temperatures();
    let elements = c
        .elements
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let eta = density_settlement_viscosity(options).viscosity(e, temps[i]);
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
