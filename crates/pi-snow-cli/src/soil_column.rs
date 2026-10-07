//! Registered engine-soil-spinup-v1 soil state and conservative thermal solver.
//! Soil stays separate from the native snow composition and bucket transport.

pub mod bare;
pub mod bare_scan;
pub mod coupled;
mod properties;
mod solver;

pub use properties::{FREEZING_K, LATENT_J_KG, Material, Phase};
pub use solver::{Bottom, StepReport, Surface};

pub const DEPTHS_M: [f64; 10] = [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00];

#[derive(Debug, Clone)]
pub struct SoilColumn {
    pub material: Material,
    /// J m-2 at each lumped node, relative to fully frozen soil at freezing.
    enthalpy: [f64; 10],
}

impl SoilColumn {
    pub fn uniform(material: Material, temperature_k: f64) -> Self {
        let volumetric_h = material.enthalpy_at_temperature(temperature_k);
        Self {
            material,
            enthalpy: std::array::from_fn(|i| volumetric_h * Self::node_volume(i)),
        }
    }

    pub fn air_profile(material: Material, tbot_k: f64, mean_336h_k: f64, offset_k: f64) -> Self {
        let damping_m =
            (material.conductivity(FREEZING_K, 1.0) / material.capacity(1.0) * 30.0 * 86400.0
                / std::f64::consts::PI)
                .sqrt();
        Self {
            material,
            enthalpy: std::array::from_fn(|i| {
                let t =
                    tbot_k + (mean_336h_k - tbot_k) * (-DEPTHS_M[i] / damping_m).exp() + offset_k;
                material.enthalpy_at_temperature(t) * Self::node_volume(i)
            }),
        }
    }

    pub fn node_volume(i: usize) -> f64 {
        let above = if i > 0 {
            DEPTHS_M[i] - DEPTHS_M[i - 1]
        } else {
            0.0
        };
        let below = if i < 9 {
            DEPTHS_M[i + 1] - DEPTHS_M[i]
        } else {
            0.0
        };
        0.5 * (above + below)
    }

    pub fn phase(&self, i: usize) -> Phase {
        self.material.phase(self.enthalpy[i] / Self::node_volume(i))
    }

    pub fn temperatures(&self) -> [f64; 10] {
        std::array::from_fn(|i| self.phase(i).temperature_k)
    }

    pub fn energy_j_m2(&self) -> f64 {
        self.enthalpy.iter().sum()
    }

    pub fn layer_water_kg_m2(&self) -> [f64; 9] {
        std::array::from_fn(|i| self.material.water_kg_m3() * (DEPTHS_M[i + 1] - DEPTHS_M[i]))
    }

    pub fn temperature_at_depth(&self, depth_m: f64) -> f64 {
        assert!((0.0..=1.0).contains(&depth_m));
        let i = (0..9).find(|&i| depth_m <= DEPTHS_M[i + 1]).unwrap();
        let fraction = (depth_m - DEPTHS_M[i]) / (DEPTHS_M[i + 1] - DEPTHS_M[i]);
        self.phase(i).temperature_k * (1.0 - fraction) + self.phase(i + 1).temperature_k * fraction
    }

    pub fn step(&mut self, dt_s: f64, top: Surface, bottom: Bottom) -> Result<StepReport, String> {
        solver::step(self, dt_s, top, bottom)
    }
}

#[cfg(test)]
mod tests;
