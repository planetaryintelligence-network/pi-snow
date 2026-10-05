//! Registered engine-cold-crash candidate: merge a thin top snow element once
//! after precipitation and before the surface solve.
//!
//! The 10 mm threshold follows the Jordan (1991) top-layer minimum as adopted
//! by CLM4 Technical Note §7.2.6, Table 7.2. Applying it to fresh snow is the
//! registered adaptation in engine-cold-crash-v1 Amendment 2.

use pi_snow_core::{Column, Composition, Element, Microstructure};

use crate::water::{node_energies, set_temperatures_from_energies};

/// Merge the top snow element into the snow element immediately below it.
/// Returns true when one merge occurred. Soil elements and lone snow elements
/// are left untouched.
pub fn merge_top_once(column: &mut Column, minimum_thickness: f64) -> bool {
    let snow_count = column.elements.len().saturating_sub(column.soil_elements);
    if snow_count < 2 || !(column.elements.last().unwrap().thickness < minimum_thickness) {
        return false;
    }

    let lower = column.elements.len() - 2;
    if lower < column.soil_elements {
        return false;
    }

    let mut energies = node_energies(column);
    let shared_node = lower + 1;
    let removed_node_energy = energies.remove(shared_node);
    column.node_temperature.remove(shared_node);

    let top = column.elements.pop().unwrap();
    let lower_element = column.elements.pop().unwrap();
    let merged = combine(&lower_element, &top);
    column.elements.insert(lower, merged);

    // Match the existing mesh-merge convention: distribute the removed
    // interface node's sensible energy equally to its surviving neighbours.
    energies[lower] += 0.5 * removed_node_energy;
    energies[lower + 1] += 0.5 * removed_node_energy;
    set_temperatures_from_energies(column, &energies);
    true
}

fn combine(a: &Element, b: &Element) -> Element {
    let thickness = a.thickness + b.thickness;
    let volume_mean = |x: f64, y: f64| (x * a.thickness + y * b.thickness) / thickness;
    let mass_a = a.mass();
    let mass_b = b.mass();
    let mass_mean = |x: f64, y: f64| {
        if mass_a + mass_b > 0.0 {
            (x * mass_a + y * mass_b) / (mass_a + mass_b)
        } else {
            0.5 * (x + y)
        }
    };
    Element {
        thickness,
        composition: Composition {
            ice: volume_mean(a.composition.ice, b.composition.ice),
            water: volume_mean(a.composition.water, b.composition.water),
            air: volume_mean(a.composition.air, b.composition.air),
            soil: volume_mean(a.composition.soil, b.composition.soil),
        },
        micro: Microstructure {
            dendricity: mass_mean(a.micro.dendricity, b.micro.dendricity),
            sphericity: mass_mean(a.micro.sphericity, b.micro.sphericity),
            grain_radius: mass_mean(a.micro.grain_radius, b.micro.grain_radius),
            bond_radius: mass_mean(a.micro.bond_radius, b.micro.bond_radius),
            coordination_number: mass_mean(
                a.micro.coordination_number,
                b.micro.coordination_number,
            ),
            marker: a.micro.marker.max(b.micro.marker),
        },
        age_days: mass_mean(a.age_days, b.age_days),
        stress: a.stress,
        stress_rate: mass_mean(a.stress_rate, b.stress_rate),
        metamorphism: mass_mean(a.metamorphism, b.metamorphism),
        soil_density: a.soil_density,
        soil_conductivity: a.soil_conductivity,
        soil_heat_capacity: a.soil_heat_capacity,
        grain_history: crate::grain_classification::merge_history(a.grain_history, b.grain_history),
    }
}
