//! Plausibility probe: how fast does BR92 settle a fresh 1 m snowfall?
use pi_snow_core::{Column, Composition, Element, Microstructure};
use pi_snow_physics::settlement::{self, Br92, Viscosity};

fn main() {
    let mut c = Column::bare_ground(265.15, 0.0, 0.0);
    for _ in 0..20 {
        c.elements.push(Element {
            thickness: 0.05,
            composition: Composition::dry_snow(90.0),
            micro: Microstructure {
                dendricity: 1.0,
                sphericity: 0.5,
                grain_radius: 0.15e-3,
                bond_radius: 0.04e-3,
                coordination_number: 1.5,
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
        });
    }
    c.node_temperature = vec![265.15; 21];
    println!("day  HS[cm]  rho_bottom  rho_top  eta_bottom[Pa s]");
    for day in 0..=10 {
        if day > 0 {
            for _ in 0..144 {
                settlement::step(&mut c, 600.0, Br92);
            }
        }
        let b = &c.elements[0];
        let t = c.elements.last().unwrap();
        println!(
            "{day:>3}  {:>6.1}  {:>10.1}  {:>7.1}  {:>12.3e}",
            100.0 * c.snow_height(),
            b.density(),
            t.density(),
            Br92.viscosity(b, 265.15)
        );
    }
}
