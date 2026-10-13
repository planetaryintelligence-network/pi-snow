//! Soil state, initialization context and cumulative counters for keyed runs.
use super::{bare, coupled, phase, Bottom, Material, SoilColumn, FREEZING_K};
use pi_snow_core::{
    options::soil::{self, Settings},
    Column,
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

impl InitializationContext {
    pub fn from_forcing(
        settings: Settings,
        forcing: &pi_snow_io::smet::Smet,
    ) -> Result<Self, &'static str> {
        let air = settings.initialization == Some(soil::Initialization::AirProfile);
        let mean = if air {
            Some(
                forcing
                    .header
                    .get("soil_mean_336h_k")
                    .and_then(|v| v.parse::<f64>().ok())
                    .filter(|v| v.is_finite())
                    .ok_or(
                        "AIR_PROFILE requires the runner-bound soil_mean_336h_k forcing header",
                    )?,
            )
        } else {
            None
        };
        let annual_mean_k = match settings.tbot_k() {
            Some(t) => t,
            None if air => {
                return Err("AIR_PROFILE requires SOIL_TBOT_K for its annual-mean profile");
            }
            None => FREEZING_K, // Unused by the uniform geothermal initialization.
        };
        Ok(Self {
            mean_336h_k: mean,
            annual_mean_k,
        })
    }
}

#[derive(Debug, Default, Clone)]
pub struct Counters {
    pub bare_seconds: f64,
    pub snow_seconds: f64,
    pub bare_rows: usize,
    pub snow_rows: usize,
    pub certified_rows: usize,
    pub uncertified_rows: usize,
    pub uncertified_a: usize,
    pub uncertified_b: usize,
    pub uncertified_c: usize,
    pub no_root: usize,
    pub jump_crossing_rows: usize,
    pub disagreement_outside_root_cells: usize,
    pub disagreement_failed_with_roots: usize,
    pub disagreement_converged_without_roots: usize,
    pub new_pack_equilibrations: usize,
    pub new_pack_entry_correction_j_m2: f64,
    pub predicate_not_computed_rows: usize,
    pub multiple_roots: usize,
    pub nonconvergences: usize,
    pub newton_iterations: usize,
    pub bare_fallback_rows: usize,
    pub bare_fallback_iterations: usize,
    pub bare_fallback_failures: usize,
    pub column_solves: usize,
    /// Bare budget calls, including the final output budget.
    pub budget_evaluations: usize,
    pub computed_budget_evaluations: usize,
    pub bare_turbulence_calls: usize,
    pub computed_bare_turbulence_calls: usize,
    pub scan_resolve_solves: usize,
    pub maximum_bare_consistency_w_m2: f64,
    pub bare_consistency_failures: usize,
    pub maximum_coupled_ledger_residual_j_m2: f64,
    pub maximum_phase_ledger_residual_j_m2: f64,
}

#[derive(Debug, Clone)]
pub struct Runtime {
    pub column: SoilColumn,
    pub bottom: Bottom,
    pub counters: Counters,
    pub initial_graft_j_m2: f64,
    pub top_energy_j_m2: f64,
    pub bottom_energy_j_m2: f64,
    pub predicate_cache: Option<super::bare_cache::Shared>,
    pub last_snow_linearisation: Option<(f64, f64)>,
    initial_energy_j_m2: f64,
    snow_was_empty: bool,
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
        Self::graft(column, bottom, snow)
    }

    pub fn from_column(
        settings: Settings,
        column: SoilColumn,
        snow: &mut Column,
    ) -> Result<Self, String> {
        settings.validate(true, false)?;
        let expected = match settings.properties.unwrap() {
            soil::Properties::MoistLoam => Material::MoistLoam,
            soil::Properties::DryLoam => Material::DryLoam,
        };
        if column.material != expected
            || column.temperatures().iter().any(|v| !v.is_finite())
            || !column.energy_j_m2().is_finite()
        {
            return Err("invalid or incompatible replayed soil state".into());
        }
        let bottom = match settings.bottom.unwrap() {
            soil::Bottom::DeepTbot8m => Bottom::Deep {
                temperature_k: settings.tbot_k().unwrap(),
            },
            soil::Bottom::Geothermal005 => Bottom::Geothermal005,
        };
        Self::graft(column, bottom, snow)
    }

    fn graft(column: SoilColumn, bottom: Bottom, snow: &mut Column) -> Result<Self, String> {
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
            predicate_cache: None,
            last_snow_linearisation: None,
            initial_energy_j_m2,
            snow_was_empty: snow.elements.is_empty(),
        })
    }

    pub(super) fn account_bare(&mut self, diagnostics: &bare::Diagnostics) {
        if diagnostics.certification.is_some() {
            self.counters.certified_rows += usize::from(diagnostics.certified);
            self.counters.uncertified_rows += usize::from(!diagnostics.certified);
        } else {
            self.counters.predicate_not_computed_rows += 1;
        }
        self.counters.multiple_roots += usize::from(diagnostics.multiple_roots());
        if let Some(c) = &diagnostics.certification {
            self.counters.uncertified_a += usize::from(c.failed[0]);
            self.counters.uncertified_b += usize::from(c.failed[1]);
            self.counters.uncertified_c += usize::from(c.failed[2]);
        }
        if let Some(scan) = &diagnostics.scan {
            self.counters.no_root += usize::from(scan.roots.is_empty());
            self.counters.jump_crossing_rows += usize::from(scan.jump_crossings.iter().any(|x| *x));
            self.counters.disagreement_outside_root_cells +=
                usize::from(scan.agreement == Some(bare::Agreement::OutsideRootCells));
            self.counters.disagreement_failed_with_roots +=
                usize::from(scan.agreement == Some(bare::Agreement::FailedWithRoots));
            self.counters.disagreement_converged_without_roots +=
                usize::from(scan.agreement == Some(bare::Agreement::ConvergedWithoutRoots));
        }
        self.counters.nonconvergences += diagnostics.nonconvergences;
        self.counters.newton_iterations += diagnostics.newton_iterations;
        self.counters.bare_fallback_rows += diagnostics.fallback_entries;
        self.counters.bare_fallback_iterations += diagnostics.fallback_iterations;
        self.counters.bare_fallback_failures += diagnostics.fallback_failures;
        self.counters.column_solves += diagnostics.column_solves;
        self.counters.budget_evaluations += diagnostics.budget_evaluations;
        self.counters.computed_budget_evaluations += diagnostics.computed_budget_evaluations;
        self.counters.bare_turbulence_calls += 2 * diagnostics.budget_evaluations;
        self.counters.computed_bare_turbulence_calls += 2 * diagnostics.computed_budget_evaluations;
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
        self.bare_step_inner(snow, dt, atmosphere, parameters, None)
    }

    pub fn bare_step_for_row(
        &mut self,
        snow: &mut Column,
        dt: f64,
        atmosphere: &Atmosphere,
        parameters: &SurfaceParameters,
        stamp: i64,
    ) -> Result<Budget, String> {
        self.bare_step_inner(snow, dt, atmosphere, parameters, Some(stamp))
    }

    fn bare_step_inner(
        &mut self,
        snow: &mut Column,
        dt: f64,
        atmosphere: &Atmosphere,
        parameters: &SurfaceParameters,
        stamp: Option<i64>,
    ) -> Result<Budget, String> {
        if !snow.elements.is_empty() {
            return Err("bare solve called with snow".into());
        }
        let parameters = SurfaceParameters {
            emissivity: 0.97,
            ..*parameters
        };
        self.counters.bare_rows += 1;
        let result = if let Some(cache) = &self.predicate_cache {
            let stamp =
                stamp.ok_or("shared predicate cache requires a bound forcing-row timestamp")?;
            let mut cache = cache.lock().map_err(|_| "predicate cache lock poisoned")?;
            bare::solve_with_samples(
                &mut self.column,
                dt,
                self.bottom,
                |p| bare::budget(atmosphere, &parameters, p).total,
                Some(cache.row(stamp, atmosphere, &parameters)),
            )
        } else {
            bare::step(&mut self.column, dt, self.bottom, atmosphere, &parameters)
        };
        let report = match result {
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
        self.counters.computed_budget_evaluations += 1;
        self.counters.bare_turbulence_calls += 2;
        self.counters.computed_bare_turbulence_calls += 2;
        let mut budget = bare::budget(atmosphere, &parameters, self.column.phase(0));
        let difference = (budget.total - report.surface_flux_w_m2).abs();
        self.counters.maximum_bare_consistency_w_m2 =
            self.counters.maximum_bare_consistency_w_m2.max(difference);
        self.counters.bare_consistency_failures +=
            usize::from(difference > 20.0_f64.max(0.05 * budget.total.abs()));
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
        linearisation: pi_snow_core::options::SurfaceLinearisation,
    ) -> Result<Budget, String> {
        self.snow_step_observed(
            snow,
            dt,
            atmosphere,
            parameters,
            sources,
            ground_shortwave,
            conductivity,
            linearisation,
            None,
        )
    }

    pub fn snow_step_observed(
        &mut self,
        snow: &mut Column,
        dt: f64,
        atmosphere: &Atmosphere,
        parameters: &SurfaceParameters,
        sources: &[f64],
        ground_shortwave: f64,
        conductivity: Conductivity,
        linearisation: pi_snow_core::options::SurfaceLinearisation,
        observe: Option<&mut dyn FnMut(pi_snow_physics::surface_diagnostic::SurfaceIteration)>,
    ) -> Result<Budget, String> {
        let linearise = match linearisation {
            pi_snow_core::options::SurfaceLinearisation::Phase2 => {
                pi_snow_physics::surface::linearise
            }
            pi_snow_core::options::SurfaceLinearisation::RadiativeFloor => {
                pi_snow_physics::surface_numerics::linearise
            }
        };
        let (budget, report) = coupled::step_observed(
            snow,
            &mut self.column,
            dt,
            self.bottom,
            atmosphere,
            parameters,
            sources,
            ground_shortwave,
            conductivity,
            linearise,
            observe,
        )?;
        self.last_snow_linearisation = Some((report.linear_a_w_m2, report.linear_b_w_m2_k));
        self.counters.maximum_coupled_ledger_residual_j_m2 = self
            .counters
            .maximum_coupled_ledger_residual_j_m2
            .max(report.combined_ledger_residual_j_m2.abs());
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
        self.counters.maximum_phase_ledger_residual_j_m2 = self
            .counters
            .maximum_phase_ledger_residual_j_m2
            .max(report.combined_ledger_residual_j_m2.abs());
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
        if self.snow_was_empty && !snow.elements.is_empty() {
            let before = thermal::internal_energy(snow);
            // The deposit carries its own bottom half-cell temperature. The
            // bare node's old temperature was soil storage, not snow storage.
            snow.node_temperature[0] = snow.node_temperature[1];
            self.counters.new_pack_entry_correction_j_m2 += thermal::internal_energy(snow) - before;
            self.counters.new_pack_equilibrations += 1;
        }
        self.top_energy_j_m2 += phase::reconcile(snow, &mut self.column)?;
        self.snow_was_empty = snow.elements.is_empty();
        Ok(())
    }

    pub fn ledger_residual_j_m2(&self) -> f64 {
        self.column.energy_j_m2()
            - self.initial_energy_j_m2
            - (self.top_energy_j_m2 - self.bottom_energy_j_m2)
    }
}
