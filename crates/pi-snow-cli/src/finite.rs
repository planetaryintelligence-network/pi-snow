//! The finite-state guard (engine-finite-guard-v1).
//!
//! The `T_CRAZY` check compares node temperatures with `<` and `>`, which a NaN
//! passes, so a non-finite state could run to the end of a season and write
//! NaN to the output with exit status 0. These checks read the state after
//! each step, while the column holds snow, and name the first non-finite
//! quantity found. They change nothing: a run whose snow state stays finite
//! writes exactly the bytes it wrote before.
//!
//! Not checked, by decision (engine-finite-guard-v1, option D): an empty
//! column, and the SMET rows. Phase 2 leaves one NaN node for a step when the
//! last element vanishes, which shows as NaN in that hour's Qg0, TSS_mod and
//! T_bottom and is reset on bare ground; 40 of the 65 stored Phase 2 units do
//! this, and stopping them would move every hash-bound artifact. The fix is a
//! physics key, `SUBLIMATION_MELT_OUT=REMOVE_ELEMENT`.

use pi_snow_core::Column;
use pi_snow_physics::surface::Budget;

fn bad(value: f64) -> bool {
    !value.is_finite()
}

/// The first non-finite quantity of the column, described for the message,
/// with the column's size so a reader can tell a vanishing pack from a deep one.
pub fn column(c: &Column) -> Option<String> {
    column_field(c).map(|f| {
        let hs: f64 = c.elements.iter().map(|e| e.thickness).sum();
        format!(
            "{f} (column: {} elements, {} nodes, total thickness {hs} m)",
            c.elements.len(),
            c.node_temperature.len()
        )
    })
}

fn column_field(c: &Column) -> Option<String> {
    for (i, &t) in c.node_temperature.iter().enumerate() {
        if bad(t) {
            return Some(format!("node_temperature[{i}] = {t}"));
        }
    }
    if bad(c.surface_hoar_mass) {
        return Some(format!("surface_hoar_mass = {}", c.surface_hoar_mass));
    }
    for (i, e) in c.elements.iter().enumerate() {
        let fields = [
            ("thickness", e.thickness),
            ("ice", e.composition.ice),
            ("water", e.composition.water),
            ("air", e.composition.air),
            ("soil", e.composition.soil),
            ("dendricity", e.micro.dendricity),
            ("sphericity", e.micro.sphericity),
            ("grain_radius", e.micro.grain_radius),
            ("bond_radius", e.micro.bond_radius),
            ("coordination_number", e.micro.coordination_number),
            ("age_days", e.age_days),
            ("stress", e.stress),
            ("stress_rate", e.stress_rate),
            ("metamorphism", e.metamorphism),
        ];
        if let Some((name, v)) = fields.iter().find(|(_, v)| bad(*v)) {
            return Some(format!("element {i} (from the bottom) {name} = {v}"));
        }
    }
    None
}

/// The first non-finite term of the surface budget or the ground flux.
pub fn budget(b: &Budget, qg0: f64, buffered: f64) -> Option<String> {
    [
        ("Qs", b.sensible),
        ("Ql", b.latent),
        ("Qr", b.rain),
        ("OLWR", b.olwr),
        ("LWR_net", b.lwr_net),
        ("vapour_mass", b.vapour_mass),
        ("surface flux total", b.total),
        ("Qg0", qg0),
        ("deposition buffer", buffered),
    ]
    .iter()
    .find(|(_, v)| bad(*v))
    .map(|(name, v)| format!("{name} = {v}"))
}
