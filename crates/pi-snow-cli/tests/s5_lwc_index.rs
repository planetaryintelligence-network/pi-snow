//! The SMET `S5` value the engine writes under `S5=LWC_INDEX_MITTERER2013`,
//! tested at the seam `output_row` uses rather than at the physics helper.

use chrono::{TimeZone, Utc};
use pi_snow_cli::engine::lwc_index_s5;
use pi_snow_core::{Column, Composition, Element, Microstructure};
use pi_snow_io::smet::{OutputRow, Smet, write_output_smet};
use pi_snow_physics::stability::HS_REPORTED_MIN_M;

fn element(thickness: f64, water: f64, soil: f64) -> Element {
    let ice = 0.3;
    Element {
        thickness,
        composition: Composition {
            ice,
            water,
            air: 1.0 - ice - water - soil,
            soil,
        },
        micro: Microstructure {
            dendricity: 0.0,
            sphericity: 0.5,
            grain_radius: 5e-4,
            bond_radius: 2e-4,
            coordination_number: 3.0,
            marker: 0,
        },
        age_days: 1.0,
        stress: 0.0,
        stress_rate: 0.0,
        metamorphism: 0.0,
        soil_density: 2600.0,
        soil_conductivity: 2.0,
        soil_heat_capacity: 800.0,
        grain_history: 0,
    }
}

fn column(elements: Vec<Element>, soil_elements: usize) -> Column {
    let mut c = Column::bare_ground(270.0, 0.0, 0.0);
    c.node_temperature = vec![270.0; elements.len() + 1];
    c.elements = elements;
    c.soil_elements = soil_elements;
    c
}

#[test]
fn soil_elements_do_not_count() {
    // 0.5 m of wet soil below 0.1 m of snow at theta_w 0.03: the snow's index
    // is 1; counting the soil would give (0.5 * 0.2 + 0.1 * 0.03) / 0.6 / 0.03.
    let c = column(vec![element(0.5, 0.2, 0.4), element(0.1, 0.03, 0.0)], 1);
    assert!((lwc_index_s5(&c).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn bare_ground_and_a_trace_the_output_shows_as_zero_have_none() {
    assert_eq!(lwc_index_s5(&column(vec![], 0)), None);
    // The case amendment 1 found: 4 micrometres of wet snow, HS written 0.000.
    assert_eq!(
        lwc_index_s5(&column(vec![element(4e-6, 0.3, 0.0)], 0)),
        None
    );
    assert!(lwc_index_s5(&column(vec![element(6e-6, 0.03, 0.0)], 0)).is_some());
}

/// The threshold must be where the SMET writer's HS_mod turns from 0.000 to
/// 0.001; if either the writer's precision or the constant changes, this fails.
#[test]
fn threshold_is_where_the_written_snow_height_leaves_zero() {
    let written = |h: f64| {
        let mut row = OutputRow::nodata(Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap());
        row.set("HS_mod", 100.0 * h);
        let text = write_output_smet("X", "X", 0.0, 0.0, 0.0, 0.0, 0.0, &[row]);
        Smet::parse(&text).unwrap().value(0, "HS_mod").unwrap()
    };
    assert_eq!(written(HS_REPORTED_MIN_M * 0.999), 0.0);
    assert!(written(HS_REPORTED_MIN_M * 1.001) > 0.0);
}
