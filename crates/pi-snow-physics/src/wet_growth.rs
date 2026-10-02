//! Growth of wet rounded grains (engine-surface-v1,
//! `--physics WET_GRAIN_GROWTH=BRUN1989`).
//!
//! The Phase 2 metamorphism holds wet rounded grains at their size because
//! the coefficients of the law were not in hand (`metamorphism`). They are
//! printed in the primary source, read in full for this experiment: Brun
//! (1989), Investigation on wet-snow metamorphism in respect of liquid-water
//! content, Ann. Glaciol. 13, 22-26, p. 26. The mean volume of the rounded
//! grains grows linearly in time, `v(t) = v0 + v' t`, with
//! `v' = v0' + v1' L^3`, `v0' = 1.28e-8 mm^3 s^-1`, `v1' = 4.22e-10 mm^3 s^-1`,
//! `L` the liquid water content in per cent of total mass, "with a maximum
//! value of 9 to 10"; beyond it (their experiments 5 and 6, Fig. 6) water
//! percolates and the rate stops rising. The volume is that of the sphere of
//! the mean convex radius of curvature.
//!
//! Which grains: Vionnet et al. (2012), GMD 5, 773-791, Table 2 (p. 777) applies
//! the law to non-dendritic snow of sphericity 1 only; below that, wet snow
//! rounds first (the Phase 2 `metamorphism::step`, Brun et al. 1992 eq. 5).
//!
//! Choices that are ours: `L` is capped at 10, the upper end of Brun's "9 to
//! 10", where his Fig. 6 levels off (at 10 the law gives 4.35e-7 mm^3 s^-1,
//! the plateau's level); the step is applied after the shape update of the
//! same step, so a grain that reaches sphericity 1 starts growing in that step.

use pi_snow_core::Microstructure;

const V0: f64 = 1.28e-8;
const V1: f64 = 4.22e-10;
const L_MAX: f64 = 10.0;

/// Brun (1989)'s volume growth rate, mm^3 s^-1, at `lwc_percent` (of mass).
pub fn rate_mm3_per_s(lwc_percent: f64) -> f64 {
    let l = lwc_percent.clamp(0.0, L_MAX);
    V0 + V1 * l.powi(3)
}

/// Grow `m.grain_radius` (m) over `dt` seconds if the grain is wet,
/// non-dendritic and fully rounded; otherwise leave it.
pub fn step(m: &mut Microstructure, lwc_percent: f64, dt: f64) {
    if lwc_percent <= 0.0 || m.dendricity > 0.0 || m.sphericity < 1.0 {
        return;
    }
    let r_mm = 1e3 * m.grain_radius;
    let volume = 4.0 / 3.0 * std::f64::consts::PI * r_mm.powi(3) + rate_mm3_per_s(lwc_percent) * dt;
    m.grain_radius = 1e-3 * (3.0 * volume / (4.0 * std::f64::consts::PI)).cbrt();
}
