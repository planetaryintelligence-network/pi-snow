//! A characterisation, not a specification: with wind measured well above the
//! scalars, the Monin-Obukhov equations with Holtslag and De Bruin's stable
//! functions have no solution above a bulk Richardson number that, at z_w 9.5
//! m and z_s 1.5 m, lies between about 0.05 and 0.07 (above the large-zeta
//! bound (z_s/z_w)^2 / 0.7 = 0.036 a second, weaker root appears first), and
//! the iteration returns the zero-flux limit. Found by
//! engine-mass-v1 (a third of both-snow hours on the v3b units); pinned here so
//! a change to it is deliberate.

use pi_snow_physics::turbulence::{self, Scheme, SurfaceLayer};

fn layer(wind: f64, t_air: f64, t_surface: f64) -> SurfaceLayer {
    SurfaceLayer {
        wind,
        z_wind: 9.5,
        z_scalar: 1.5,
        z0: 0.002,
        t_air,
        t_surface,
        q_air: 2.0e-3,
        q_surface: 1.5e-3,
        pressure: 73_000.0,
    }
}

#[test]
fn below_the_critical_richardson_number_the_solve_converges() {
    // U 3 m/s, 3 K inversion: bulk Ri 0.018 at the scalar height.
    let f = turbulence::fluxes(&layer(3.0, 268.0, 265.0), Scheme::MoHoltslag);
    assert!(f.obukhov_length.is_finite() && f.obukhov_length > 1.0);
    assert!(f.sensible > 5.0, "{}", f.sensible);
}

#[test]
fn above_it_the_mo_solve_decouples_and_louis_does_not() {
    // U 2 m/s, 5 K inversion: bulk Ri 0.069, where no root is left.
    let l = layer(2.0, 268.0, 263.0);
    let mo = turbulence::fluxes(&l, Scheme::MoHoltslag);
    assert!(
        mo.obukhov_length.abs() < 1e-20,
        "1/L ran away: L = {}",
        mo.obukhov_length
    );
    assert!(mo.sensible.abs() < 1e-6, "{}", mo.sensible);
    let louis = turbulence::fluxes(&l, Scheme::Louis);
    assert!(louis.sensible > 1.0, "{}", louis.sensible);
}
