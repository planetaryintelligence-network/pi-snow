//! Keyed, registered whole-pack melt arm and its step diagnostics.

use std::fmt::Write;

use chrono::{DateTime, Utc};
use pi_snow_core::Column;
use pi_snow_core::constants::{C_ICE, LH_FUSION, T_MELT};
use pi_snow_physics::surface::{self, Atmosphere, Budget, SurfaceParameters};
use pi_snow_physics::thermal::{self, Conductivity, Ground, volumetric_heat_capacity};

#[derive(Debug, Clone, Copy, Default)]
pub struct PackState {
    pub cold_content_j_m2: f64,
    pub enthalpy_j_m2: f64,
    pub ice_kg_m2: f64,
    pub liquid_kg_m2: f64,
}

impl PackState {
    pub fn capture(column: &Column) -> Self {
        let start = column.soil_elements;
        let mut sensible_enthalpy_j_m2 = 0.0;
        let mut ice_kg_m2 = 0.0;
        let mut liquid_kg_m2 = 0.0;
        for (index, element) in column.elements.iter().enumerate().skip(start) {
            let mean_t =
                0.5 * (column.node_temperature[index] + column.node_temperature[index + 1]);
            sensible_enthalpy_j_m2 +=
                volumetric_heat_capacity(element) * element.thickness * (mean_t - T_MELT);
            ice_kg_m2 += element.ice_mass();
            liquid_kg_m2 += element.water_mass();
        }
        Self {
            cold_content_j_m2: (-sensible_enthalpy_j_m2).max(0.0),
            enthalpy_j_m2: sensible_enthalpy_j_m2 + LH_FUSION * liquid_kg_m2,
            ice_kg_m2,
            liquid_kg_m2,
        }
    }

    pub fn melt_requirement_j_m2(self) -> f64 {
        self.cold_content_j_m2 + LH_FUSION * self.ice_kg_m2
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TotalMeltDecision {
    pub available_j_m2: f64,
    pub requirement_j_m2: f64,
    pub fired: bool,
    pub excess_j_m2: f64,
}

pub fn decide(before: PackState, net_flux_w_m2: f64, dt_s: f64) -> TotalMeltDecision {
    let available_j_m2 = net_flux_w_m2 * dt_s;
    let requirement_j_m2 = before.melt_requirement_j_m2();
    let fired = before.ice_kg_m2 > 0.0 && available_j_m2 >= requirement_j_m2;
    TotalMeltDecision {
        available_j_m2,
        requirement_j_m2,
        fired,
        excess_j_m2: if fired {
            available_j_m2 - requirement_j_m2
        } else {
            0.0
        },
    }
}

pub fn energy_residual(
    before: PackState,
    after: PackState,
    net_flux_w_m2: f64,
    ground_reset_j_m2: f64,
    snow_runoff_kg_m2: f64,
    vapor_mass_change_kg_m2: f64,
    surface_temperature_k: f64,
    decision: TotalMeltDecision,
    dt_s: f64,
) -> f64 {
    after.enthalpy_j_m2 - before.enthalpy_j_m2
        + LH_FUSION * snow_runoff_kg_m2
        + decision.excess_j_m2
        - net_flux_w_m2 * dt_s
        - ground_reset_j_m2
        - C_ICE * (surface_temperature_k - T_MELT) * vapor_mass_change_kg_m2
}

pub fn clear_snow(column: &mut Column) -> (f64, f64) {
    let start = column.soil_elements;
    let ice = column.elements[start..].iter().map(|e| e.ice_mass()).sum();
    let liquid = column.elements[start..]
        .iter()
        .map(|e| e.water_mass())
        .sum();
    column.elements.truncate(start);
    column.node_temperature.truncate(start + 1);
    (ice, liquid)
}

#[derive(Debug, Clone, Copy)]
pub struct LinearizedStep {
    pub budget: Budget,
    pub a: f64,
    pub b: f64,
    pub solved_surface_temperature: f64,
}

/// The Phase 2 three-pass surface solve, exposed here for registered A/B output.
/// Candidate behavior remains in this new module; no-key runs keep using surface.rs.
pub fn step_general_recorded(
    column: &mut Column,
    atm: &Atmosphere,
    params: &SurfaceParameters,
    ground: Ground,
    sources: &[f64],
    dt: f64,
    conductivity: Conductivity,
) -> LinearizedStep {
    let start = column.node_temperature.clone();
    let mut t_lin = *start.last().expect("column has a surface node");
    let mut used = (0.0, 0.0);
    for _ in 0..3 {
        column.node_temperature.clone_from(&start);
        used = linearise(atm, params, t_lin);
        thermal::step_general(column, ground, used.0, used.1, sources, dt, conductivity);
        t_lin = *column
            .node_temperature
            .last()
            .expect("column has a surface node");
    }
    let solved_surface_temperature = t_lin;
    let mut budget = surface::budget(atm, params, t_lin);
    budget.total = used.0 + used.1 * t_lin;
    LinearizedStep {
        budget,
        a: used.0,
        b: used.1,
        solved_surface_temperature,
    }
}

fn linearise(atm: &Atmosphere, params: &SurfaceParameters, t0: f64) -> (f64, f64) {
    const DELTA: f64 = 0.01;
    let t = t0.min(T_MELT - DELTA);
    let f0 = surface::budget(atm, params, t).total;
    let f1 = surface::budget(atm, params, t + DELTA).total;
    let b = ((f1 - f0) / DELTA).min(0.0);
    (f0 - b * t, b)
}

/// Optional detailed per-step record. Enabled only in registered diagnostic runs.
#[derive(Debug, Default, Clone)]
pub struct MeltoutCandidateDiagnostic {
    text: String,
}

impl MeltoutCandidateDiagnostic {
    pub fn new() -> Self {
        Self {
            text: String::from(concat!(
                "timestamp_utc,ground_boundary,ground_flux_W_m2,a_W_m2,b_W_m2_K,surface_flux_W_m2,",
                "surface_nonlinear_W_m2,absorbed_surface_sw_W_m2,to_ground_W_m2,",
                "olwr_W_m2,lwr_net_W_m2,sensible_W_m2,latent_W_m2,rain_heat_W_m2,",
                "source0_W_m2,half_source0_W_m2,discrete_base_flux_W_m2,",
                "residual_qg0_W_m2,ground_reset_storage_J_m2,cold_content_J_m2,ice_before_kg_m2,liquid_before_kg_m2,",
                "available_energy_J_m2,melt_requirement_J_m2,total_melt_fired,",
                "total_melt_excess_J_m2,phase_change_energy_J_m2,melted_kg_m2,refrozen_kg_m2,runoff_kg_m2,",
                "sublimation_kg_m2,mass_before_kg_m2,mass_after_kg_m2,mass_residual_kg_m2,",
                "energy_residual_J_m2,thermal_solve_residual_J_m2,surface_temperature_K,bottom_temperature_K\n",
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push(
        &mut self,
        t: DateTime<Utc>,
        ground_boundary: &str,
        ground_flux: f64,
        a: f64,
        b: f64,
        surface_flux: f64,
        surface_nonlinear: f64,
        absorbed_surface_sw: f64,
        to_ground: f64,
        olwr: f64,
        lwr_net: f64,
        sensible: f64,
        latent: f64,
        rain_heat: f64,
        source0: f64,
        half_source0: f64,
        discrete_base_flux: f64,
        qg0: f64,
        ground_reset_storage: f64,
        before: PackState,
        decision: TotalMeltDecision,
        melted: f64,
        refrozen: f64,
        phase_change_energy: f64,
        runoff: f64,
        sublimation: f64,
        mass_before: f64,
        mass_after: f64,
        mass_residual: f64,
        energy_residual: f64,
        thermal_solve_residual: f64,
        surface_temperature: f64,
        bottom_temperature: f64,
    ) {
        writeln!(
            self.text,
            "{},{ground_boundary},{ground_flux:.9},{a:.9},{b:.9},{surface_flux:.9},{surface_nonlinear:.9},{absorbed_surface_sw:.9},{to_ground:.9},{olwr:.9},{lwr_net:.9},{sensible:.9},{latent:.9},{rain_heat:.9},{source0:.9},{half_source0:.9},{discrete_base_flux:.9},{qg0:.9},{ground_reset_storage:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{:.9},{phase_change_energy:.9},{melted:.9},{refrozen:.9},{runoff:.9},{sublimation:.9},{mass_before:.9},{mass_after:.9},{mass_residual:.9},{energy_residual:.9},{thermal_solve_residual:.9},{surface_temperature:.9},{bottom_temperature:.9}",
            t.to_rfc3339(),
            before.cold_content_j_m2,
            before.ice_kg_m2,
            before.liquid_kg_m2,
            decision.available_j_m2,
            decision.requirement_j_m2,
            decision.fired,
            decision.excess_j_m2,
        )
        .expect("writing to a String cannot fail");
    }

    pub fn into_text(self) -> String {
        self.text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_snow_core::{Composition, Element, Microstructure};
    use pi_snow_physics::surface::{Atmosphere, SurfaceParameters};
    use pi_snow_physics::thermal::Conductivity;
    use pi_snow_physics::turbulence::Scheme;

    #[test]
    fn recorded_surface_step_matches_phase2() {
        let mut reference = Column::bare_ground(T_MELT, 0.0, 0.0);
        reference.elements.push(Element {
            thickness: 0.08,
            composition: Composition {
                ice: 0.4,
                water: 0.01,
                air: 0.59,
                soil: 0.0,
            },
            micro: Microstructure {
                dendricity: 0.0,
                sphericity: 1.0,
                grain_radius: 0.15e-3,
                bond_radius: 0.05e-3,
                coordination_number: 3.0,
                marker: 0,
            },
            age_days: 10.0,
            stress: 0.0,
            stress_rate: 0.0,
            metamorphism: 0.0,
            soil_density: 0.0,
            soil_conductivity: 0.0,
            soil_heat_capacity: 0.0,
            grain_history: 0,
        });
        reference.node_temperature = vec![T_MELT, 271.0];
        let mut recorded = reference.clone();
        let atm = Atmosphere {
            t_air: 276.0,
            relative_humidity: 0.65,
            wind: 2.0,
            absorbed_shortwave: 150.0,
            ilwr: 300.0,
            pressure: 72_000.0,
            rain_rate: 0.0,
        };
        let params = SurfaceParameters {
            emissivity: surface::SNOW_EMISSIVITY,
            z_wind: 2.0,
            z_scalar: 2.0,
            z0: 0.001,
            scheme: Scheme::Louis,
        };
        let expected = surface::step_general(
            &mut reference,
            &atm,
            &params,
            Ground::Temperature(T_MELT),
            &[10.0],
            600.0,
            Conductivity::Yen1981,
        );
        let actual = step_general_recorded(
            &mut recorded,
            &atm,
            &params,
            Ground::Temperature(T_MELT),
            &[10.0],
            600.0,
            Conductivity::Yen1981,
        );
        assert_eq!(recorded, reference);
        assert_eq!(actual.budget, expected);
        assert!(actual.a.is_finite() && actual.b.is_finite());
    }

    #[test]
    fn total_melt_fires_at_registered_energy_threshold() {
        let pack = PackState {
            cold_content_j_m2: 2_000.0,
            enthalpy_j_m2: -2_000.0,
            ice_kg_m2: 0.01,
            liquid_kg_m2: 0.0,
        };
        let just_above = decide(pack, pack.melt_requirement_j_m2() / 600.0 + 1e-8, 600.0);
        assert!(just_above.fired);
        assert!((just_above.excess_j_m2 - 6e-6).abs() < 1e-8);
        assert!(!decide(pack, 0.0, 600.0).fired);
        let decision = decide(pack, pack.melt_requirement_j_m2() / 600.0, 600.0);
        let residual = energy_residual(
            pack,
            PackState::default(),
            decision.available_j_m2 / 600.0,
            0.0,
            pack.ice_kg_m2,
            0.0,
            T_MELT,
            decision,
            600.0,
        );
        assert!(decision.fired);
        assert!(residual.abs() < 1e-9);
    }

    #[test]
    fn diagnostic_header_and_row_have_matching_columns() {
        let mut diagnostic = MeltoutCandidateDiagnostic::new();
        diagnostic.push(
            Utc::now(),
            "DIRICHLET",
            0.0,
            1.0,
            -1.0,
            2.0,
            2.0,
            1.0,
            0.0,
            -10.0,
            -8.0,
            1.0,
            -2.0,
            0.0,
            1.0,
            0.5,
            1.0,
            1.0,
            0.0,
            PackState::default(),
            TotalMeltDecision::default(),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            273.15,
            273.15,
        );
        let text = diagnostic.into_text();
        let mut lines = text.lines();
        assert_eq!(
            lines.next().unwrap().split(',').count(),
            lines.next().unwrap().split(',').count()
        );
    }
}
