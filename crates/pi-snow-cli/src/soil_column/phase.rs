//! Snow phase and transport bookkeeping at the soil's shared enthalpy node.
use super::{FREEZING_K, LATENT_J_KG, SoilColumn};
use pi_snow_core::Column;
use pi_snow_core::constants::{LH_FUSION, RHO_ICE, RHO_WATER};
use pi_snow_core::options::WaterRetention;
use pi_snow_physics::{thermal, water};

const MELTED_OUT_ICE: f64 = 1e-6;

#[derive(Debug, Clone, Copy)]
pub struct PhaseReport {
    pub snow: water::PhaseChange,
    /// Explicit shared-node transfer into soil, additional to conduction.
    pub soil_exchange_j_m2: f64,
    pub combined_ledger_residual_j_m2: f64,
}

fn capacities(column: &Column) -> Vec<f64> {
    let mut values = vec![0.0; column.elements.len() + 1];
    for (i, e) in column.elements.iter().enumerate() {
        let half = 0.5 * thermal::volumetric_heat_capacity(e) * e.thickness;
        values[i] += half;
        values[i + 1] += half;
    }
    values
}

fn joint_temperature(soil: &mut SoilColumn, energy: f64, snow_capacity: f64) -> f64 {
    let volume = SoilColumn::node_volume(0);
    let latent = soil.material.water_kg_m3() * LATENT_J_KG * volume;
    let enthalpy = if energy <= 0.0 {
        let c = soil.material.capacity(0.0) * volume;
        let delta = energy / (c + snow_capacity);
        c * delta
    } else if energy >= latent {
        let c = soil.material.capacity(1.0) * volume;
        let delta = (energy - latent) / (c + snow_capacity);
        latent + c * delta
    } else {
        energy
    };
    soil.enthalpy[0] = enthalpy;
    // Use the enthalpy inversion as the single shared-node temperature.
    // A separate inversion through lumped capacity can differ by one ULP.
    soil.phase(0).temperature_k
}

/// Snow melting/thinning uses the native published operator; its energy half
/// at node zero is debited from soil plus snow storage, rather than snow alone.
pub fn phase_change(column: &mut Column, soil: &mut SoilColumn) -> Result<PhaseReport, String> {
    if column.soil_elements != 0 || !column.is_well_formed() {
        return Err("phase coupling requires separate soil".into());
    }
    let before = water::energy_relative_to_melt(column) + soil.energy_j_m2();
    let soil_before = soil.energy_j_m2();
    let mut energies: Vec<f64> = capacities(column)
        .iter()
        .zip(&column.node_temperature)
        .map(|(c, t)| c * (t - FREEZING_K))
        .collect();
    energies[0] += soil.enthalpy[0];
    let temperatures = column.element_temperatures();
    let mut report = water::PhaseChange::default();
    for (i, element) in column.elements.iter_mut().enumerate() {
        let sensible = thermal::volumetric_heat_capacity(element)
            * element.thickness
            * (temperatures[i] - FREEZING_K);
        let (ice, liquid) = (element.ice_mass(), element.water_mass());
        let melt = if sensible > 0.0 && ice > 0.0 {
            (sensible / LH_FUSION).min(ice)
        } else if sensible < 0.0 && liquid > 0.0 {
            -((-sensible / LH_FUSION).min(liquid))
        } else {
            0.0
        };
        if melt == 0.0 {
            continue;
        }
        energies[i] -= 0.5 * LH_FUSION * melt;
        energies[i + 1] -= 0.5 * LH_FUSION * melt;
        let (new_ice, new_liquid) = (ice - melt, liquid + melt);
        let soil_volume = element.composition.soil * element.thickness;
        let mut thickness = if melt > 0.0 && ice > 0.0 {
            element.thickness * new_ice / ice
        } else {
            element.thickness
        };
        thickness = thickness.max(new_ice / RHO_ICE + new_liquid / RHO_WATER + soil_volume);
        element.thickness = thickness;
        if thickness > 0.0 {
            element.composition.ice = new_ice / (RHO_ICE * thickness);
            element.composition.water = new_liquid / (RHO_WATER * thickness);
            element.composition.soil = soil_volume / thickness;
            element.composition.air = (1.0
                - element.composition.ice
                - element.composition.water
                - element.composition.soil)
                .max(0.0);
        }
        if melt > 0.0 {
            report.melted += melt;
        } else {
            report.refrozen -= melt;
        }
    }
    for i in (0..column.elements.len()).rev() {
        if column.elements[i].ice_mass() >= MELTED_OUT_ICE {
            continue;
        }
        let gone = column.elements.remove(i);
        let (ice, liquid) = (gone.ice_mass(), gone.water_mass());
        let top_energy = energies.remove(i + 1);
        energies[i] += top_energy;
        column.node_temperature.remove(i + 1);
        if i == 0 {
            report.drained += liquid + ice;
            // The native mass convention exports its sub-threshold ice as water.
            // Charge that latent conversion to the surviving shared storage.
            energies[0] -= LH_FUSION * ice;
        } else {
            let below = &mut column.elements[i - 1];
            below.composition.water += liquid / (RHO_WATER * below.thickness);
            below.composition.ice += ice / (RHO_ICE * below.thickness);
            below.composition.air =
                (1.0 - below.composition.ice - below.composition.water - below.composition.soil)
                    .max(0.0);
        }
    }
    let new_capacity = capacities(column);
    column.node_temperature[0] = joint_temperature(soil, energies[0], new_capacity[0]);
    for i in 1..column.node_temperature.len() {
        if new_capacity[i] > 0.0 {
            column.node_temperature[i] = FREEZING_K + energies[i] / new_capacity[i];
        }
    }
    let after = water::energy_relative_to_melt(column) + soil.energy_j_m2();
    let residual = after - before + LH_FUSION * report.drained;
    if !residual.is_finite() || column.node_temperature.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite phase coupling".into());
    }
    Ok(PhaseReport {
        snow: report,
        soil_exchange_j_m2: soil.energy_j_m2() - soil_before,
        combined_ledger_residual_j_m2: residual,
    })
}

/// Native bucket transport holds snow nodal energy. Re-equilibrate its shared
/// node with the soil storage after the bottom snow capacity changes.
pub fn cascade(
    column: &mut Column,
    soil: &mut SoilColumn,
    retention: WaterRetention,
) -> Result<(f64, f64), String> {
    let capacity = capacities(column)[0];
    let joint_energy = soil.enthalpy[0] + capacity * (column.node_temperature[0] - FREEZING_K);
    let before = soil.energy_j_m2();
    let runoff = water::cascade_with(column, retention);
    let temperature = joint_temperature(soil, joint_energy, capacities(column)[0]);
    if !temperature.is_finite() {
        return Err("nonfinite transport interface".into());
    }
    column.node_temperature[0] = temperature;
    Ok((runoff, soil.energy_j_m2() - before))
}
