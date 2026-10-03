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
use pi_snow_core::constants::{NODATA, RHO_WATER, T_MELT};
use pi_snow_core::options::{
    GrainClassification, MassReport, S5Index, SnowAlbedo, SublimationMeltOut, SurfaceLayer,
    SurfaceLinearisation, Turbulence, WetGrainGrowth,
};
use pi_snow_core::{Column, Element, PhysicsOptions};
use pi_snow_io::ini::{NewSnowDensity, RunConfig, Stability};
use pi_snow_io::pro::{self, ElementRecord, ProHeader, ProTimestep};
use pi_snow_io::smet::{OutputRow, Smet};
use pi_snow_physics::albedo;
use pi_snow_physics::aoki_albedo::{self, AokiAlbedo};
use pi_snow_physics::atmosphere::pressure_at_altitude;
use pi_snow_physics::chan_albedo::{self, DepositionSeries, Slair};
use pi_snow_physics::grain_classification;
use pi_snow_physics::metamorphism::{self, History, Local};
use pi_snow_physics::new_snow::{self, Conditions, DensityScheme, Deposition};
use pi_snow_physics::settlement::{self, Br92, Viscosity};
use pi_snow_physics::solar;
use pi_snow_physics::stability::{self, GrainClass, Layer};
use pi_snow_physics::surface::{self, Atmosphere, SurfaceParameters};
use pi_snow_physics::surface_numerics;
use pi_snow_physics::thermal::{self, Conductivity, Ground};
use pi_snow_physics::turbulence::Scheme;
use pi_snow_physics::water;
use pi_snow_physics::wet_growth;

use crate::ledger::{Interval, Ledger};

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
    #[error("temperature {value} K at {time} is outside T_CRAZY [{min}, {max}]")]
    Crazy {
        time: DateTime<Utc>,
        value: f64,
        min: f64,
        max: f64,
    },
    #[error("{0}")]
    IncompatiblePhysics(&'static str),
    #[error("MERRA-2 deposition is missing for snow-covered step at {time}")]
    MissingDeposition { time: DateTime<Utc> },
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
    /// Per-step Chan branches and daily SLAIR mass reconciliation, candidate only.
    pub chan_csv: Option<String>,
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
    run_with_options_and_deposition(cfg, options, site, forcing, start, end, None)
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
    let density_scheme = match cfg.new_snow_density {
        NewSnowDensity::LehningNew => DensityScheme::Pahaut,
        NewSnowDensity::Zwart => DensityScheme::Zwart,
    };
    let conductivity = Conductivity::Yen1981;
    let pressure = pressure_at_altitude(site.altitude);

    let mut column = start;
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
    let mut interval = Interval::default();
    let make_output = |rows: Vec<OutputRow>,
                       pro_text: String,
                       steps: usize,
                       ledger: Option<Ledger>,
                       chan_csv: Option<String>| RunOutput {
        smet: pi_snow_io::smet::write_output_smet(
            &site.station_id,
            &site.station_name,
            site.latitude,
            site.longitude,
            site.altitude,
            site.slope_angle,
            site.slope_azimuth,
            &rows,
        ),
        pro: pro_text,
        steps,
        mass: ledger.map(Ledger::into_text),
        chan_csv,
    };

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
        let v = |i: usize| row.values[i];
        let (Some(ta), Some(rh), Some(vw), Some(iswr_h), Some(ilwr)) =
            (v(i_ta), v(i_rh), v(i_vw), v(i_iswr), v(i_ilwr))
        else {
            continue;
        };
        let dw = v(i_dw).unwrap_or(NODATA);
        let psum = v(i_psum).unwrap_or(0.0).max(0.0);
        let tsg = i_tsg.and_then(v).unwrap_or(T_MELT);

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
            (surface::budget(&atm, &params, tsg), site.soil_albedo, 0.0)
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
            let before = thermal::internal_energy(&column);
            let step_surface = match options.surface_linearisation {
                SurfaceLinearisation::Phase2 => surface::step_general,
                SurfaceLinearisation::RadiativeFloor => surface_numerics::step_general,
            };
            let b = step_surface(
                &mut column,
                &atm,
                &params,
                Ground::Temperature(tsg),
                &absorbed.per_element,
                dt,
                conductivity,
            );
            let sources: f64 = absorbed.per_element.iter().sum();
            let qg0 = (thermal::internal_energy(&column) - before) / dt - b.total - sources;

            // 5. Phase change and water.
            let water_before: Vec<f64> = column.elements.iter().map(Element::water_mass).collect();
            let n_before = column.elements.len();
            let pc = water::phase_change(&mut column);
            if column.elements.len() == n_before {
                for (e, w) in column.elements.iter_mut().zip(water_before) {
                    if e.water_mass() < w - 1e-9 {
                        e.micro.marker |= MARK_REFROZEN;
                    }
                }
            }
            let runoff = water::cascade_with(&mut column, options.water_retention);
            acc.runoff += pc.drained + runoff;
            interval.drained += pc.drained;
            interval.runoff += runoff;
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
                    // Clean vapor deposition is applied as snowfall in Eq. 1 order.
                    daily_clean_deposition += -vapour;
                }
            }

            // 7. Settlement.
            settlement::step(&mut column, dt, Br92);

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

            // 9. Merge thin elements; drop anything melted to nothing.
            new_snow::merge_thin_elements(&mut column, cfg.minimum_element_length);
            if options.snow_albedo == SnowAlbedo::Aoki2003TempRate && column.elements.is_empty() {
                aoki_state.end_pack();
            }
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
        if let Some(field) = (!column.elements.is_empty())
            .then(|| {
                crate::finite::column(&column)
                    .or_else(|| crate::finite::budget(&budget, deposition.buffered()))
            })
            .flatten()
        {
            return Err(RunError::NonFinite {
                step: steps,
                time: t,
                field,
                partial: Box::new(make_output(rows, pro_text, steps, ledger, chan_csv.clone())),
            });
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
            pro_text.push_str(&pro::write_timestep(&profile(t, &column, options)));
            acc = Accum::default();
            if let Some(l) = ledger.as_mut() {
                l.push(t, column.swe(), deposition.buffered(), &interval);
            }
            interval = Interval::default();
        }
    }
    Ok(make_output(rows, pro_text, steps, ledger, chan_csv))
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
