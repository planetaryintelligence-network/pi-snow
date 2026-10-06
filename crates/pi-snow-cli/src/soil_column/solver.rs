//! Enthalpy backward Euler: sharp phase change and a conservative nodal ledger.

use super::{DEPTHS_M, SoilColumn};

#[derive(Debug, Clone, Copy)]
pub enum Surface {
    Flux(f64),
    /// Inward flux h*(air-temperature), W m-2.
    Robin {
        transfer_w_m2_k: f64,
        air_k: f64,
    },
    Temperature(f64),
}

#[derive(Debug, Clone, Copy)]
pub enum Bottom {
    Deep {
        temperature_k: f64,
    },
    Geothermal005,
    /// Unit-test conservation and analytic limits.
    Insulated,
}

#[derive(Debug, Clone, Copy)]
pub struct StepReport {
    pub top_flux_w_m2: f64,
    /// Positive leaving the soil; geothermal is negative.
    pub bottom_flux_w_m2: f64,
    pub ledger_residual_j_m2: f64,
    pub iterations: usize,
}

fn conductances(soil: &SoilColumn, h: &[f64; 10]) -> ([f64; 9], f64) {
    let conductivity: [f64; 10] = std::array::from_fn(|i| {
        let phase = soil.material.phase(h[i] / SoilColumn::node_volume(i));
        soil.material.conductivity(phase.liquid_fraction)
    });
    let links = std::array::from_fn(|i| {
        2.0 / (1.0 / conductivity[i] + 1.0 / conductivity[i + 1]) / (DEPTHS_M[i + 1] - DEPTHS_M[i])
    });
    (links, conductivity[9] / 7.0)
}

fn residual(
    soil: &SoilColumn,
    old: &[f64; 10],
    h: &[f64; 10],
    dt: f64,
    top: Surface,
    bottom: Bottom,
    g: &[f64; 9],
    deep: f64,
) -> ([f64; 10], [f64; 10], f64, f64) {
    let t = std::array::from_fn(|i| {
        soil.material
            .phase(h[i] / SoilColumn::node_volume(i))
            .temperature_k
    });
    let mut flux = [0.0; 10];
    for i in 0..9 {
        let q = g[i] * (t[i] - t[i + 1]);
        flux[i] -= q;
        flux[i + 1] += q;
    }
    let qtop = match top {
        Surface::Flux(q) => q,
        Surface::Robin {
            transfer_w_m2_k: a,
            air_k,
        } => a * (air_k - t[0]),
        Surface::Temperature(_) => (h[0] - old[0]) / dt - flux[0],
    };
    let qbottom = match bottom {
        Bottom::Deep { temperature_k } => deep * (t[9] - temperature_k),
        Bottom::Geothermal005 => -0.05,
        Bottom::Insulated => 0.0,
    };
    flux[0] += qtop;
    flux[9] -= qbottom;
    let mut r = std::array::from_fn(|i| h[i] - old[i] - dt * flux[i]);
    if let Surface::Temperature(value) = top {
        r[0] = h[0] - soil.material.enthalpy_at_temperature(value) * SoilColumn::node_volume(0);
    }
    (r, t, qtop, qbottom)
}

fn solve_tridiagonal(
    lower: &[f64; 10],
    diagonal: &[f64; 10],
    upper: &[f64; 10],
    rhs: &[f64; 10],
) -> [f64; 10] {
    let mut c = [0.0; 10];
    let mut d = [0.0; 10];
    c[0] = upper[0] / diagonal[0];
    d[0] = rhs[0] / diagonal[0];
    for i in 1..10 {
        let denominator = diagonal[i] - lower[i] * c[i - 1];
        c[i] = upper[i] / denominator;
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / denominator;
    }
    let mut answer = d;
    for i in (0..9).rev() {
        answer[i] -= c[i] * answer[i + 1];
    }
    answer
}

fn coordinate_sweep(
    soil: &SoilColumn,
    old: &[f64; 10],
    h: &mut [f64; 10],
    dt: f64,
    top: Surface,
    bottom: Bottom,
    g: &[f64; 9],
    deep: f64,
) {
    for i in 0..10 {
        if i == 0 {
            if let Surface::Temperature(t) = top {
                h[i] = soil.material.enthalpy_at_temperature(t) * SoilColumn::node_volume(i);
                continue;
            }
        }
        let mut a = 0.0;
        let mut rhs = old[i];
        for neighbor in [i.checked_sub(1), (i < 9).then_some(i + 1)]
            .into_iter()
            .flatten()
        {
            let conductance = g[i.min(neighbor)];
            a += dt * conductance;
            rhs += dt
                * conductance
                * soil
                    .material
                    .phase(h[neighbor] / SoilColumn::node_volume(neighbor))
                    .temperature_k;
        }
        if i == 0 {
            match top {
                Surface::Flux(q) => rhs += dt * q,
                Surface::Robin {
                    transfer_w_m2_k,
                    air_k,
                } => {
                    a += dt * transfer_w_m2_k;
                    rhs += dt * transfer_w_m2_k * air_k;
                }
                Surface::Temperature(_) => unreachable!(),
            }
        }
        if i == 9 {
            match bottom {
                Bottom::Deep { temperature_k } => {
                    a += dt * deep;
                    rhs += dt * deep * temperature_k;
                }
                Bottom::Geothermal005 => rhs += dt * 0.05,
                Bottom::Insulated => {}
            }
        }
        // Exact inverse of H + a*T(H), including the latent interval. This
        // scalar monotone solve is the phase-corner fallback, with no smoothing.
        let volume = SoilColumn::node_volume(i);
        let latent = soil.material.water_kg_m3() * super::LATENT_J_KG * volume;
        let reduced = rhs - a * super::FREEZING_K;
        h[i] = if reduced < 0.0 {
            reduced / (1.0 + a / (soil.material.capacity(0.0) * volume))
        } else if reduced > latent {
            latent + (reduced - latent) / (1.0 + a / (soil.material.capacity(1.0) * volume))
        } else {
            reduced
        };
    }
}

pub(super) fn step(
    soil: &mut SoilColumn,
    dt: f64,
    top: Surface,
    bottom: Bottom,
) -> Result<StepReport, String> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err("invalid soil time step".into());
    }
    let old = soil.enthalpy;
    let mut h = old;
    let before: f64 = old.iter().sum();
    // As in the native thermal assembly, material coefficients belong to the
    // composition at step entry. Temperatures and latent enthalpy are implicit.
    // Iterating Johansen's frozen/unfrozen branch inside Newton would make the
    // residual discontinuous at disappearance of the last soil ice.
    let (g, deep) = conductances(soil, &old);
    for iteration in 0..240 {
        let (r, _, qtop, qbottom) = residual(soil, &old, &h, dt, top, bottom, &g, deep);
        let norm = r.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
        if norm < 1e-5 {
            let after: f64 = h.iter().sum();
            let ledger = after - before - dt * (qtop - qbottom);
            if !ledger.is_finite() || ledger.abs() > 1e-3 {
                return Err(format!("soil ledger residual {ledger} J m-2"));
            }
            soil.enthalpy = h;
            return Ok(StepReport {
                top_flux_w_m2: qtop,
                bottom_flux_w_m2: qbottom,
                ledger_residual_j_m2: ledger,
                iterations: iteration + 1,
            });
        }
        let derivative: [f64; 10] = std::array::from_fn(|i| {
            soil.material
                .phase(h[i] / SoilColumn::node_volume(i))
                .dtemperature_denthalpy
                / SoilColumn::node_volume(i)
        });
        let mut diagonal = [1.0; 10];
        let mut lower = [0.0; 10];
        let mut upper = [0.0; 10];
        for i in 0..9 {
            diagonal[i] += dt * g[i] * derivative[i];
            upper[i] -= dt * g[i] * derivative[i + 1];
            diagonal[i + 1] += dt * g[i] * derivative[i + 1];
            lower[i + 1] -= dt * g[i] * derivative[i];
        }
        if let Surface::Robin {
            transfer_w_m2_k, ..
        } = top
        {
            diagonal[0] += dt * transfer_w_m2_k * derivative[0];
        }
        if let Bottom::Deep { .. } = bottom {
            diagonal[9] += dt * deep * derivative[9];
        }
        if let Surface::Temperature(_) = top {
            diagonal[0] = 1.0;
            upper[0] = 0.0;
        }
        let direction = solve_tridiagonal(&lower, &diagonal, &upper, &r.map(|v| -v));
        // A residual line search safeguards crossings of the phase corners.
        let mut scale = 1.0;
        let mut accepted = false;
        for _ in 0..32 {
            let trial = std::array::from_fn(|i| h[i] + scale * direction[i]);
            let (rt, _, _, _) = residual(soil, &old, &trial, dt, top, bottom, &g, deep);
            let next = rt.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
            if next.is_finite() && next < norm {
                h = trial;
                accepted = true;
                break;
            }
            scale *= 0.5;
        }
        if !accepted {
            coordinate_sweep(soil, &old, &mut h, dt, top, bottom, &g, deep);
        }
    }
    Err("soil nonlinear solve failed after 240 iterations".into())
}
