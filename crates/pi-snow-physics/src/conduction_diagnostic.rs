//! Read-only diagnostics for the existing tridiagonal heat solve.

use pi_snow_core::Column;

use crate::thermal::{self, Conductivity, Ground};

#[derive(Debug, Clone, Copy, Default)]
pub struct ConductionStats {
    pub layers: usize,
    pub min_thickness_m: f64,
    pub max_thickness_m: f64,
    pub min_conductance_w_m2k: f64,
    pub max_conductance_w_m2k: f64,
    pub min_half_capacity_w_m2k: f64,
    pub max_half_capacity_w_m2k: f64,
    pub min_diagonal: f64,
    pub max_diagonal: f64,
    pub min_abs_off_diagonal: f64,
    pub max_abs_off_diagonal: f64,
    pub residual_linf: f64,
}

/// Reassemble the production backward-Euler matrix and evaluate its residual
/// against the state produced by `thermal::step_general`.
pub fn summarize(
    before: &Column,
    after: &Column,
    ground: Ground,
    a: f64,
    b: f64,
    sources: &[f64],
    dt: f64,
    conductivity: Conductivity,
) -> Option<ConductionStats> {
    let n = before.elements.len();
    if n == 0 || after.node_temperature.len() != n + 1 {
        return None;
    }
    let nodes = n + 1;
    let mut sub = vec![0.0; nodes];
    let mut diag = vec![0.0; nodes];
    let mut sup = vec![0.0; nodes];
    let mut rhs = vec![0.0; nodes];
    let mut stats = ConductionStats {
        layers: n,
        min_thickness_m: f64::INFINITY,
        max_thickness_m: f64::NEG_INFINITY,
        min_conductance_w_m2k: f64::INFINITY,
        max_conductance_w_m2k: f64::NEG_INFINITY,
        min_half_capacity_w_m2k: f64::INFINITY,
        max_half_capacity_w_m2k: f64::NEG_INFINITY,
        min_diagonal: f64::INFINITY,
        max_diagonal: f64::NEG_INFINITY,
        min_abs_off_diagonal: f64::INFINITY,
        max_abs_off_diagonal: 0.0,
        residual_linf: 0.0,
    };
    for (i, element) in before.elements.iter().enumerate() {
        let k = conductivity.of_density(element.density());
        let g = k / element.thickness;
        let cap = 0.5 * thermal::volumetric_heat_capacity(element) * element.thickness / dt;
        stats.min_thickness_m = stats.min_thickness_m.min(element.thickness);
        stats.max_thickness_m = stats.max_thickness_m.max(element.thickness);
        stats.min_conductance_w_m2k = stats.min_conductance_w_m2k.min(g);
        stats.max_conductance_w_m2k = stats.max_conductance_w_m2k.max(g);
        stats.min_half_capacity_w_m2k = stats.min_half_capacity_w_m2k.min(cap);
        stats.max_half_capacity_w_m2k = stats.max_half_capacity_w_m2k.max(cap);
        diag[i] += g + cap;
        sup[i] -= g;
        rhs[i] += cap * before.node_temperature[i];
        diag[i + 1] += g + cap;
        sub[i + 1] -= g;
        rhs[i + 1] += cap * before.node_temperature[i + 1];
    }
    for (i, source) in sources.iter().enumerate().take(n) {
        rhs[i] += 0.5 * source;
        rhs[i + 1] += 0.5 * source;
    }
    match ground {
        Ground::Flux(flux) => rhs[0] += flux,
        Ground::Temperature(temp) => {
            sub[0] = 0.0;
            sup[0] = 0.0;
            diag[0] = 1.0;
            rhs[0] = temp;
        }
    }
    rhs[n] += a;
    diag[n] -= b;

    for i in 0..nodes {
        stats.min_diagonal = stats.min_diagonal.min(diag[i]);
        stats.max_diagonal = stats.max_diagonal.max(diag[i]);
        for off in [sub[i], sup[i]] {
            if off != 0.0 {
                stats.min_abs_off_diagonal = stats.min_abs_off_diagonal.min(off.abs());
                stats.max_abs_off_diagonal = stats.max_abs_off_diagonal.max(off.abs());
            }
        }
        let mut lhs = diag[i] * after.node_temperature[i];
        if i > 0 {
            lhs += sub[i] * after.node_temperature[i - 1];
        }
        if i + 1 < nodes {
            lhs += sup[i] * after.node_temperature[i + 1];
        }
        stats.residual_linf = stats.residual_linf.max((lhs - rhs[i]).abs());
    }
    Some(stats)
}
