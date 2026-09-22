//! Phase change, bucket water transport and refreeze.
//!
//! The configuration selects `WATERTRANSPORTMODEL_SNOW = BUCKET` and
//! `COUPLEDPHASECHANGES = false`. Wever et al. (2014, *Solving Richards
//! Equation for snow improves snowpack meltwater runoff estimations in detailed
//! multi-layer snowpack model*, The Cryosphere 8, 257-274) describe the
//! resulting scheme: the thermal solve runs first; phase changes are then
//! executed from the new temperature profile; and liquid water above a layer's
//! holding capacity moves downward unconditionally each step, whatever the
//! capacity of the layer receiving it.
//!
//! # Energy bookkeeping
//!
//! This module is where the node/element split in the state has to be settled,
//! and the round-1 critique flagged it as unsettled. The rule used:
//!
//! * Node `n` has lumped heat capacity `K_n = sum over its elements of C h / 2`,
//!   the same lumping the conduction solver uses, and holds sensible energy
//!   `U_n = K_n (T_n - T_m)`, measured from the melting point.
//! * An element's sensible energy is the two halves it contributes to its nodes,
//!   `S_e = C h (Tbar_e - T_m)`, and it is `S_e` that decides melting or
//!   freezing.
//! * The latent energy an element takes up or releases is removed from or added
//!   to its two nodes equally. After the composition changes, node temperatures
//!   are recomputed from the conserved `U_n` and the new `K_n`.
//!
//! Measured from the melting point, water at the melting point carries no
//! sensible energy, so moving or draining it changes only the latent term, and
//! `sum U_n + L_f * liquid mass` is conserved exactly by construction. The
//! price is that an element does not end exactly at the melting point when its
//! neighbours differ, because the nodes it shares carry their energy too. That
//! smearing is a splitting error of one element width, stated here rather than
//! discovered later.
//!
//! The refreeze rule itself needs no parameterisation: frozen mass is the lesser
//! of the available liquid and the cold content divided by the latent heat,
//! which is conservation of energy with the constants already cited in
//! `pi_snow_core::constants`.

use pi_snow_core::Column;
use pi_snow_core::constants::{LH_FUSION, RHO_ICE, RHO_WATER, T_MELT};

use crate::thermal::volumetric_heat_capacity;

/// Ice mass per unit area below which an element is treated as melted out,
/// kg m^-2. At 1e-6 kg m^-2 an element is well under a micrometre of ice, so
/// the choice affects nothing measurable; it exists to stop an element from
/// lingering at near-zero thickness in front of the conduction solver.
const MELTED_OUT_ICE: f64 = 1e-6;

/// Irreducible (holding) water content as a volumetric fraction of the element,
/// as a function of the volumetric ice fraction.
///
/// Coleou and Lesaffre (1998), as given by Wever et al. (2014) Eq. 1:
///
/// ```text
/// theta_h = 0.0264 + 0.0099 (1 - theta_i) / theta_i     theta_i <= 0.23
///         = 0.08 - 0.1023 (theta_i - 0.03)             0.23 < theta_i <= 0.812
///         = 0                                           theta_i > 0.812
/// ```
///
/// Read from Wever, not from Coleou and Lesaffre directly, so a secondary
/// citation. The pieces join continuously at both breakpoints, which the tests
/// use as a transcription check.
///
/// The first piece diverges as `theta_i` tends to zero and exceeds the whole
/// pore volume below `theta_i` of about 0.01. Capacity is therefore clamped to
/// the pore volume `1 - theta_i`; that clamp is ours, not the paper's.
pub fn holding_capacity(ice_fraction: f64) -> f64 {
    let theta = if ice_fraction <= 0.23 {
        0.0264 + 0.0099 * (1.0 - ice_fraction) / ice_fraction
    } else if ice_fraction <= 0.812 {
        0.08 - 0.1023 * (ice_fraction - 0.03)
    } else {
        0.0
    };
    theta.clamp(0.0, (1.0 - ice_fraction).max(0.0))
}

/// What one call to [`phase_change`] did, in kg m^-2.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PhaseChange {
    pub melted: f64,
    pub refrozen: f64,
    /// Liquid water that left the bottom of the column because the bottom
    /// element itself melted out. Normally zero.
    pub drained: f64,
}

/// Lumped heat capacity of each node, J m^-2 K^-1.
fn node_capacities(column: &Column) -> Vec<f64> {
    let mut k = vec![0.0; column.elements.len() + 1];
    for (index, element) in column.elements.iter().enumerate() {
        let half = 0.5 * volumetric_heat_capacity(element) * element.thickness;
        k[index] += half;
        k[index + 1] += half;
    }
    k
}

/// Sensible energy held by each node relative to the melting point, J m^-2.
pub(crate) fn node_energies(column: &Column) -> Vec<f64> {
    node_capacities(column)
        .iter()
        .zip(&column.node_temperature)
        .map(|(k, t)| k * (t - T_MELT))
        .collect()
}

/// Set node temperatures from energies and the current capacities.
pub(crate) fn set_temperatures_from_energies(column: &mut Column, energies: &[f64]) {
    let capacities = node_capacities(column);
    for ((t, u), k) in column
        .node_temperature
        .iter_mut()
        .zip(energies)
        .zip(capacities)
    {
        if k > 0.0 {
            *t = T_MELT + u / k;
        }
    }
}

/// Total energy relative to ice at the melting point, J m^-2: nodal sensible
/// energy plus the latent energy held in the liquid water. The conservation
/// tests assert on this.
pub fn energy_relative_to_melt(column: &Column) -> f64 {
    node_energies(column).iter().sum::<f64>() + LH_FUSION * column.liquid_water()
}

/// Melt or refreeze each element according to its sensible energy relative to
/// the melting point, then remove any element that has melted out.
///
/// Melting follows Vionnet et al. (2012, GMD 5, 773-791) section 3.9: dry
/// density is conserved and the element thins in proportion to its lost ice.
/// Refreezing keeps the thickness and fills pore space, since ice is 9 % less
/// dense than the water it forms from. Either way, if the new composition would
/// need negative air, the element is widened to exactly fit its contents.
///
/// Water may exceed the holding capacity afterwards; [`cascade`] runs next and
/// moves the excess down.
pub fn phase_change(column: &mut Column) -> PhaseChange {
    let mut report = PhaseChange::default();
    if column.elements.is_empty() {
        return report;
    }
    let mut energies = node_energies(column);
    let temperatures = column.element_temperatures();

    for (index, element) in column.elements.iter_mut().enumerate() {
        let sensible =
            volumetric_heat_capacity(element) * element.thickness * (temperatures[index] - T_MELT);
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

        let latent = LH_FUSION * melt;
        energies[index] -= 0.5 * latent;
        energies[index + 1] -= 0.5 * latent;

        let (new_ice, new_liquid) = (ice - melt, liquid + melt);
        let c = element.composition;
        let soil_volume = c.soil * element.thickness;
        let mut thickness = if melt > 0.0 && ice > 0.0 {
            element.thickness * new_ice / ice
        } else {
            element.thickness
        };
        let solid_volume = new_ice / RHO_ICE + new_liquid / RHO_WATER + soil_volume;
        thickness = thickness.max(solid_volume);

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

    report.drained = remove_melted_out(column, &mut energies);
    set_temperatures_from_energies(column, &energies);
    report
}

/// Remove elements whose ice has melted out, moving their remaining ice and
/// water into the element below and their top node's energy into the node
/// below it. Returns liquid that left the column because the bottom element
/// itself melted out.
fn remove_melted_out(column: &mut Column, energies: &mut Vec<f64>) -> f64 {
    let mut drained = 0.0;
    let mut index = column.elements.len();
    while index > 0 {
        index -= 1;
        if column.elements[index].ice_mass() >= MELTED_OUT_ICE {
            continue;
        }
        let gone = column.elements.remove(index);
        let (ice, liquid) = (gone.ice_mass(), gone.water_mass());
        // Its top node disappears and the element above (if any) now rests on
        // its bottom node. Energy moves with the node so nothing is lost.
        let node_energy = energies.remove(index + 1);
        energies[index] += node_energy;
        column.node_temperature.remove(index + 1);

        if index == 0 {
            // Nothing below: the water leaves the column. The trace of ice goes
            // with it as water, which would need L_f per kilogram; at under
            // MELTED_OUT_ICE per step that is below anything measurable, and it
            // is charged to the drained mass so the mass budget still closes.
            drained += liquid + ice;
            continue;
        }
        let below = &mut column.elements[index - 1];
        let thickness = below.thickness;
        below.composition.water += liquid / (RHO_WATER * thickness);
        below.composition.ice += ice / (RHO_ICE * thickness);
        below.composition.air =
            (1.0 - below.composition.ice - below.composition.water - below.composition.soil)
                .max(0.0);
    }
    if column.elements.is_empty() {
        column.node_temperature.truncate(1);
    }
    drained
}

/// Move liquid water above each element's holding capacity down the column,
/// top to bottom. Returns the runoff leaving the bottom, kg m^-2.
///
/// Water in excess of capacity passes on regardless of whether the element below
/// can hold it, as Wever et al. (2014) describe for the bucket scheme; the
/// element below keeps what it can and passes on the rest in the same sweep.
/// Nodal sensible energy is preserved, so a node that gains water moves toward
/// the melting point, which is what adding water at the melting point should do.
pub fn cascade(column: &mut Column) -> f64 {
    if column.elements.is_empty() {
        return 0.0;
    }
    let energies = node_energies(column);
    let mut carry = 0.0; // kg m^-2 entering the current element from above
    for element in column.elements.iter_mut().rev() {
        let thickness = element.thickness;
        let c = &mut element.composition;
        c.water += carry / (RHO_WATER * thickness);
        let capacity = holding_capacity(c.ice);
        carry = if c.water > capacity {
            let excess = (c.water - capacity) * RHO_WATER * thickness;
            c.water = capacity;
            excess
        } else {
            0.0
        };
        c.air = (1.0 - c.ice - c.water - c.soil).max(0.0);
    }
    set_temperatures_from_energies(column, &energies);
    carry
}
