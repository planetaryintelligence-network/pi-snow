//! Stage 4 gate: phase change, bucket water transport and refreeze.
//!
//! Energy is measured relative to the melting point, as
//! `E = sum K_n (T_n - T_m) + L_f * liquid mass`, where `K_n` is the lumped
//! heat capacity of node `n`.
//! With that datum, water at the melting point carries no sensible energy, so
//! moving it or draining it changes only the latent term, and conservation can
//! be asserted exactly rather than to within a heat-capacity drift.

use approx::assert_relative_eq;
use pi_snow_core::constants::*;
use pi_snow_core::{Column, Composition, Element, Microstructure};
use pi_snow_physics::water;

fn element(thickness: f64, ice: f64, liquid: f64) -> Element {
    Element {
        thickness,
        composition: Composition {
            ice,
            water: liquid,
            air: 1.0 - ice - liquid,
            soil: 0.0,
        },
        micro: Microstructure {
            dendricity: 0.0,
            sphericity: 1.0,
            grain_radius: 0.5e-3,
            bond_radius: 0.2e-3,
            coordination_number: 3.0,
            marker: 0,
        },
        age_days: 0.0,
        stress: 0.0,
        stress_rate: 0.0,
        metamorphism: 0.0,
        soil_density: 0.0,
        soil_conductivity: 0.0,
        soil_heat_capacity: 0.0,
        grain_history: 0,
    }
}

fn column(elements: Vec<Element>, temperature: f64) -> Column {
    let n = elements.len();
    let mut c = Column::bare_ground(temperature, 0.0, 0.0);
    c.elements = elements;
    c.node_temperature = vec![temperature; n + 1];
    c
}

#[test]
fn holding_capacity_pieces_join_continuously() {
    // Coleou and Lesaffre (1998) via Wever et al. (2014) Eq. 1. The breakpoints
    // are where transcription errors show. The published pieces meet at
    // theta_i = 0.23 to within 3.5e-6 (0.0595435 against 0.0595400), which is
    // the rounding of coefficients printed to three or four figures. A real
    // transcription slip, 0.0099 read as 0.099 say, would open a gap near 0.01,
    // so 1e-5 separates the two cases cleanly. An earlier version asserted 1e-9
    // and failed on the published rounding, not on our code.
    let below = water::holding_capacity(0.23);
    let above = water::holding_capacity(0.23 + 1e-12);
    assert_relative_eq!(
        below,
        0.0264 + 0.0099 * (1.0 - 0.23) / 0.23,
        max_relative = 1e-12
    );
    assert_relative_eq!(below, above, epsilon = 1e-5);
    // Same published rounding at the upper breakpoint: 0.08 - 0.1023 * 0.782
    // is 1.4e-6, not zero.
    assert_relative_eq!(water::holding_capacity(0.812 - 1e-12), 0.0, epsilon = 1e-5);
    assert_eq!(water::holding_capacity(0.85), 0.0);
    // Capacity falls as ice fraction rises: dense snow holds less water.
    assert!(water::holding_capacity(0.1) > water::holding_capacity(0.3));
}

#[test]
fn holding_capacity_never_exceeds_the_pore_volume() {
    // The first piece diverges as theta_i -> 0 (0.0099 (1 - theta_i) / theta_i),
    // and at theta_i = 0.01 exceeds the whole pore volume. A capacity bigger
    // than the pores is unphysical, so it is clamped, and this pins the clamp.
    for ice in [0.001, 0.005, 0.01, 0.02] {
        assert!(
            water::holding_capacity(ice) <= 1.0 - ice + 1e-15,
            "ice {ice}"
        );
    }
}

#[test]
fn cascade_retains_up_to_capacity_and_drains_the_rest() {
    // Three elements, 0.1 m each, ice fraction 0.3. All liquid starts in the top
    // element; each retains its capacity and passes the excess down, and what
    // leaves the bottom is runoff. Closed form: runoff = total - 3 * retained.
    let ice = 0.3;
    let capacity = water::holding_capacity(ice);
    // Capacity at theta_i = 0.3 is 0.0524. The start must exceed three
    // capacities or the bottom element never fills; 0.15 (an earlier value)
    // did not, and made the closed form below negative.
    let start = 0.25; // volumetric, top element only
    assert!(start > 3.0 * capacity);
    let mut c = column(
        vec![
            element(0.1, ice, 0.0),
            element(0.1, ice, 0.0),
            element(0.1, ice, start),
        ],
        T_MELT,
    );
    let runoff = water::cascade(&mut c);
    for e in &c.elements {
        assert_relative_eq!(e.composition.water, capacity, max_relative = 1e-12);
        assert!(e.composition.is_consistent(1e-12));
    }
    let expected = RHO_WATER * 0.1 * (start - 3.0 * capacity);
    assert_relative_eq!(runoff, expected, max_relative = 1e-12);
}

#[test]
fn cascade_conserves_mass_exactly() {
    let mut c = column(
        vec![
            element(0.05, 0.45, 0.02),
            element(0.08, 0.20, 0.11),
            element(0.03, 0.60, 0.00),
            element(0.12, 0.15, 0.25),
        ],
        T_MELT,
    );
    let before = c.liquid_water();
    let runoff = water::cascade(&mut c);
    assert_relative_eq!(c.liquid_water() + runoff, before, max_relative = 1e-12);
}

#[test]
fn a_warm_single_element_melts_exactly_its_excess_energy() {
    // One element at T > T_m. Its excess sensible energy relative to the melting
    // point, C h (T - T_m), becomes latent: melted mass = C h (T - T_m) / L_f,
    // and the element ends at the melting point.
    let mut c = column(vec![element(0.1, 0.4, 0.0)], T_MELT + 0.5);
    let capacity = pi_snow_physics::thermal::volumetric_heat_capacity(&c.elements[0]);
    let ice_before = c.elements[0].ice_mass();
    let report = water::phase_change(&mut c);
    let expected = capacity * 0.1 * 0.5 / LH_FUSION;
    assert_relative_eq!(report.melted, expected, max_relative = 1e-12);
    assert_relative_eq!(
        c.elements[0].ice_mass(),
        ice_before - expected,
        max_relative = 1e-12
    );
    assert_relative_eq!(c.elements[0].water_mass(), expected, max_relative = 1e-12);
    for &t in &c.node_temperature {
        assert_relative_eq!(t, T_MELT, epsilon = 1e-12);
    }
}

#[test]
fn melting_conserves_dry_density_and_thins_the_element() {
    // Vionnet et al. (2012) section 3.9: the dry density of a melting layer is
    // conserved and its thickness decreases accordingly. So the ice volume
    // fraction is unchanged and the thickness scales with the ice mass.
    let mut c = column(vec![element(0.1, 0.4, 0.0)], T_MELT + 2.0);
    water::phase_change(&mut c);
    let e = &c.elements[0];
    assert_relative_eq!(e.composition.ice, 0.4, max_relative = 1e-12);
    assert!(e.thickness < 0.1);
    assert_relative_eq!(
        e.thickness,
        e.ice_mass() / (RHO_ICE * 0.4),
        max_relative = 1e-12
    );
}

#[test]
fn refreeze_raises_temperature_by_exactly_the_latent_heat_released() {
    // Liquid water in a cold element refreezes until either the cold content or
    // the water runs out. Here the water runs out first, so all of it freezes
    // and the latent heat L_f m warms the element: first-principles energy
    // conservation, needing no parameterisation.
    let liquid = 0.002; // volumetric, 0.2 kg m^-2 in a 0.1 m element
    let mut c = column(vec![element(0.1, 0.3, liquid)], T_MELT - 5.0);
    let energy_before = water::energy_relative_to_melt(&c);
    let report = water::phase_change(&mut c);
    assert_relative_eq!(
        report.refrozen,
        RHO_WATER * liquid * 0.1,
        max_relative = 1e-12
    );
    assert_eq!(c.elements[0].composition.water, 0.0);
    assert!(c.node_temperature[0] > T_MELT - 5.0 && c.node_temperature[0] < T_MELT);
    assert_relative_eq!(
        water::energy_relative_to_melt(&c),
        energy_before,
        max_relative = 1e-12
    );
}

#[test]
fn refreeze_stops_at_the_melting_point_when_cold_content_runs_out() {
    // Now the cold content runs out first: a slightly cold element with a lot of
    // water freezes only as much as its cold content allows and ends at T_m.
    let mut c = column(vec![element(0.1, 0.3, 0.05)], T_MELT - 0.2);
    let capacity = pi_snow_physics::thermal::volumetric_heat_capacity(&c.elements[0]);
    let report = water::phase_change(&mut c);
    assert_relative_eq!(
        report.refrozen,
        capacity * 0.1 * 0.2 / LH_FUSION,
        max_relative = 1e-12
    );
    for &t in &c.node_temperature {
        assert_relative_eq!(t, T_MELT, epsilon = 1e-12);
    }
}

#[test]
fn phase_change_conserves_energy_on_a_mixed_column() {
    // Warm, cold, wet and dry elements side by side, sharing nodes. Individual
    // elements do not end exactly at T_m, because shared nodes smear the
    // correction across neighbours; the total must still be exact.
    let mut c = column(
        vec![
            element(0.04, 0.35, 0.03),
            element(0.06, 0.25, 0.00),
            element(0.02, 0.50, 0.08),
            element(0.05, 0.30, 0.01),
        ],
        T_MELT,
    );
    c.node_temperature = vec![
        T_MELT - 3.0,
        T_MELT - 1.0,
        T_MELT + 0.4,
        T_MELT - 0.2,
        T_MELT + 1.1,
    ];
    let before = water::energy_relative_to_melt(&c);
    let mass_before: f64 = c.elements.iter().map(Element::mass).sum();
    water::phase_change(&mut c);
    assert_relative_eq!(
        water::energy_relative_to_melt(&c),
        before,
        max_relative = 1e-12
    );
    let mass_after: f64 = c.elements.iter().map(Element::mass).sum();
    assert_relative_eq!(mass_after, mass_before, max_relative = 1e-12);
    for e in &c.elements {
        assert!(e.composition.is_consistent(1e-9), "{:?}", e.composition);
    }
}

#[test]
fn a_melted_out_element_is_removed_and_its_water_passed_down() {
    // An element whose ice is down to a trace (melted out over earlier steps)
    // must disappear rather than linger near zero thickness, which would hand
    // the conduction solver an effectively infinite conductance. Its water goes
    // to the element below and its node's energy to the surviving node.
    //
    // A single step's own overheat cannot produce this state: melting all the
    // ice from sensible heat alone needs roughly L_f / c_i = 167 K of excess.
    // Real melt-out accumulates over many steps of surface flux, so the test
    // starts from the melted-out state rather than trying to reach it.
    let mut c = column(
        vec![element(0.1, 0.3, 0.0), element(0.001, 1e-9, 0.02)],
        T_MELT,
    );
    *c.node_temperature.last_mut().unwrap() = T_MELT + 0.3;
    let mass_before: f64 = c.elements.iter().map(Element::mass).sum();
    let energy_before = water::energy_relative_to_melt(&c);
    water::phase_change(&mut c);
    assert_eq!(c.elements.len(), 1, "melted-out element should be removed");
    assert!(c.is_well_formed());
    let mass_after: f64 = c.elements.iter().map(Element::mass).sum();
    assert_relative_eq!(mass_after, mass_before, max_relative = 1e-12);
    assert_relative_eq!(
        water::energy_relative_to_melt(&c),
        energy_before,
        max_relative = 1e-12
    );
}
