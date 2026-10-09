//! CLM5 mineral-loam equations registered in plan sections 1.1 and 1.6.

pub const FREEZING_K: f64 = 273.15;
pub const LATENT_J_KG: f64 = 333_700.0;
const POROSITY: f64 = 0.45;
const C_LIQUID: f64 = 4188.0;
const C_ICE: f64 = 2117.27;
const C_SOLID: f64 = (2.128 * 40.0 + 2.385 * 20.0) / 60.0 * 1e6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Material {
    MoistLoam,
    DryLoam,
}

#[derive(Debug, Clone, Copy)]
pub struct Phase {
    pub temperature_k: f64,
    pub liquid_fraction: f64,
    /// dT/dH, H in J m-3. Zero inside the sharp latent interval.
    pub dtemperature_denthalpy: f64,
}

impl Material {
    pub fn water_kg_m3(self) -> f64 {
        match self {
            Self::MoistLoam => 250.0,
            Self::DryLoam => 50.0,
        }
    }

    pub fn capacity(self, liquid_fraction: f64) -> f64 {
        C_SOLID * (1.0 - POROSITY)
            + self.water_kg_m3() * (C_ICE * (1.0 - liquid_fraction) + C_LIQUID * liquid_fraction)
    }

    pub fn conductivity(self, temperature_k: f64, liquid_fraction: f64) -> f64 {
        let rho_dry = 2700.0 * (1.0 - POROSITY);
        let dry = (0.135 * rho_dry + 64.7) / (2700.0 - 0.947 * rho_dry);
        let solid: f64 = (8.80 * 40.0 + 2.92 * 20.0) / 60.0;
        let saturated = solid.powf(1.0 - POROSITY)
            * 0.57_f64.powf(POROSITY * liquid_fraction)
            * 2.29_f64.powf(POROSITY * (1.0 - liquid_fraction));
        let saturation = self.water_kg_m3() / 1000.0 / POROSITY;
        let kersten = if temperature_k < FREEZING_K {
            saturation
        } else {
            (saturation.log10() + 1.0).max(0.0)
        };
        kersten * saturated + (1.0 - kersten) * dry
    }

    pub fn enthalpy_at_temperature(self, temperature_k: f64) -> f64 {
        if temperature_k < FREEZING_K {
            self.capacity(0.0) * (temperature_k - FREEZING_K)
        } else {
            self.water_kg_m3() * LATENT_J_KG + self.capacity(1.0) * (temperature_k - FREEZING_K)
        }
    }

    pub fn phase(self, h: f64) -> Phase {
        let latent = self.water_kg_m3() * LATENT_J_KG;
        if h <= 0.0 {
            Phase {
                temperature_k: FREEZING_K + h / self.capacity(0.0),
                liquid_fraction: 0.0,
                dtemperature_denthalpy: 1.0 / self.capacity(0.0),
            }
        } else if h >= latent {
            Phase {
                temperature_k: FREEZING_K + (h - latent) / self.capacity(1.0),
                liquid_fraction: 1.0,
                dtemperature_denthalpy: 1.0 / self.capacity(1.0),
            }
        } else {
            Phase {
                temperature_k: FREEZING_K,
                liquid_fraction: h / latent,
                dtemperature_denthalpy: 0.0,
            }
        }
    }
}
