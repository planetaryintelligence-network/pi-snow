//! Soil state, initialization context and cumulative counters for keyed runs.
use super::{Bottom, FREEZING_K, Material, SoilColumn, bare, coupled, phase};
use pi_snow_core::{
    Column,
    options::soil::{self, Settings},
};
use pi_snow_physics::{
    surface::{Atmosphere, Budget, SurfaceParameters},
    thermal::{self, Conductivity},
};

/// Derived by the registered runner from the complete hash-bound forcing.
/// A sliced arm forcing alone cannot recover the 336 preceding hourly rows.
#[derive(Debug, Clone, Copy)]
pub struct InitializationContext {
    pub mean_336h_k: Option<f64>,
    pub annual_mean_k: f64,
}

#[derive(Debug, Default, Clone)]
pub struct Counters {
    pub bare_seconds: f64,
    pub snow_seconds: f64,
    pub bare_rows: usize,
    pub snow_rows: usize,
    pub certified_rows: usize,
    pub uncertified_rows: usize,
    pub predicate_not_computed_rows: usize,
    pub multiple_roots: usize,
    pub nonconvergences: usize,
    pub newton_iterations: usize,
    pub column_solves: usize,
    /// Bare budget calls, including the final output budget.
    pub budget_evaluations: usize,
    pub bare_turbulence_calls: usize,
    pub scan_resolve_solves: usize,
}

#[derive(Debug, Clone)]
pub struct Runtime {
    pub column: SoilColumn,
    pub bottom: Bottom,
    pub counters: Counters,
    pub initial_graft_j_m2: f64,
    pub top_energy_j_m2: f64,
    pub bottom_energy_j_m2: f64,
    initial_energy_j_m2: f64,
}

impl Runtime {
    pub fn new(
        settings: Settings,
        context: InitializationContext,
        snow: &mut Column,
    ) -> Result<Self, String> {
        settings.validate(true, false)?;
        let material = match settings.properties.unwrap() {
            soil::Properties::MoistLoam => Material::MoistLoam,
            soil::Properties::DryLoam => Material::DryLoam,
        };
        let tbot = settings.tbot_k().unwrap_or(context.annual_mean_k);
        if !tbot.is_finite() {
            return Err("nonfinite initialization deep temperature".into());
        }
        let bottom = match settings.bottom.unwrap() {
            soil::Bottom::DeepTbot8m => Bottom::Deep {
                temperature_k: tbot,
            },
            soil::Bottom::Geothermal005 => Bottom::Geothermal005,
        };
        let column = match settings.initialization.unwrap() {
            soil::Initialization::UniformTmelt => SoilColumn::uniform(material, FREEZING_K),
            soil::Initialization::AirProfile => {
                let mean = context
                    .mean_336h_k
                    .ok_or("AIR_PROFILE needs the bound 336-hour mean")?;
                if !mean.is_finite() {
                    return Err("nonfinite 336-hour initialization mean".into());
                }
                SoilColumn::air_profile(material, tbot, mean, settings.offset_k())
            }
        };
        // Existing SNO soil rows do not define this registered column.
        let n = snow.soil_elements;
        if n > snow.elements.len() || !snow.is_well_formed() {
            return Err("invalid initial snow column".into());
        }
        snow.elements.drain(..n);
        snow.node_temperature.drain(..n);
        snow.soil_elements = 0;
        let before = thermal::internal_energy(snow);
        snow.node_temperature[0] = column.phase(0).temperature_k;
        let initial_graft_j_m2 = thermal::internal_energy(snow) - before;
        let initial_energy_j_m2 = column.energy_j_m2();
        Ok(Self {
            column,
            bottom,
            counters: Counters::default(),
            initial_graft_j_m2,
            top_energy_j_m2: 0.0,
            bottom_energy_j_m2: 0.0,
            initial_energy_j_m2,
        })
    }

    fn account_bare(&mut self, diagnostics: &bare::Diagnostics) {
        if diagnostics.certification.is_some() {
            self.counters.certified_rows += usize::from(diagnostics.certified);
            self.counters.uncertified_rows += usize::from(!diagnostics.certified);
        } else {
            self.counters.predicate_not_computed_rows += 1;
        }
        self.counters.multiple_roots += usize::from(diagnostics.multiple_roots());
        self.counters.nonconvergences += diagnostics.nonconvergences;
        self.counters.newton_iterations += diagnostics.newton_iterations;
        self.counters.column_solves += diagnostics.column_solves;
        self.counters.budget_evaluations += diagnostics.budget_evaluations;
        self.counters.bare_turbulence_calls += 2 * diagnostics.budget_evaluations;
        self.counters.scan_resolve_solves +=
            diagnostics.scan.as_ref().map_or(0, |s| s.resolve_solves);
    }

    pub fn bare_step(
        &mut self,
        snow: &mut Column,
        dt: f64,
        atmosphere: &Atmosphere,
        parameters: &SurfaceParameters,
    ) -> Result<Budget, String> {
        if !snow.elements.is_empty() {
            return Err("bare solve called with snow".into());
        }
        let parameters = SurfaceParameters {
            emissivity: 0.97,
            ..*parameters
        };
        self.counters.bare_rows += 1;
        let report = match bare::step(&mut self.column, dt, self.bottom, atmosphere, &parameters) {
            Ok(report) => report,
            Err(error) => {
                self.account_bare(&error.diagnostics);
                return Err(error.message);
            }
        };
        self.account_bare(&report.diagnostics);
        self.counters.bare_seconds += dt;
        self.top_energy_j_m2 += dt * report.soil.top_flux_w_m2;
        self.bottom_energy_j_m2 += dt * report.soil.bottom_flux_w_m2;
        snow.node_temperature[0] = self.column.phase(0).temperature_k;
        self.counters.budget_evaluations += 1;
        self.counters.bare_turbulence_calls += 2;
        let mut budget = bare::budget(atmosphere, &parameters, self.column.phase(0));
        budget.total = report.surface_flux_w_m2;
        Ok(budget)
    }

    pub fn snow_step(
        &mut self,
        snow: &mut Column,
        dt: f64,
        atmosphere: &Atmosphere,
        parameters: &SurfaceParameters,
        sources: &[f64],
        ground_shortwave: f64,
        conductivity: Conductivity,
    ) -> Result<Budget, String> {
        let (budget, report) = coupled::step(
            snow,
            &mut self.column,
            dt,
            self.bottom,
            atmosphere,
            parameters,
            sources,
            ground_shortwave,
            conductivity,
        )?;
        self.counters.snow_rows += 1;
        self.counters.snow_seconds += dt;
        self.top_energy_j_m2 += dt * report.soil.top_flux_w_m2;
        self.bottom_energy_j_m2 += dt * report.soil.bottom_flux_w_m2;
        Ok(budget)
    }

    pub fn phase_change(
        &mut self,
        snow: &mut Column,
    ) -> Result<pi_snow_physics::water::PhaseChange, String> {
        let report = phase::phase_change(snow, &mut self.column)?;
        self.top_energy_j_m2 += report.soil_exchange_j_m2;
        Ok(report.snow)
    }

    pub fn cascade(
        &mut self,
        snow: &mut Column,
        retention: pi_snow_core::options::WaterRetention,
    ) -> Result<f64, String> {
        let (runoff, exchange) = phase::cascade(snow, &mut self.column, retention)?;
        self.top_energy_j_m2 += exchange;
        Ok(runoff)
    }

    pub fn reconcile(&mut self, snow: &mut Column) -> Result<(), String> {
        self.top_energy_j_m2 += phase::reconcile(snow, &mut self.column)?;
        Ok(())
    }

    pub fn ledger_residual_j_m2(&self) -> f64 {
        self.column.energy_j_m2()
            - self.initial_energy_j_m2
            - (self.top_energy_j_m2 - self.bottom_energy_j_m2)
    }
}
