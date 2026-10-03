//! Chan, Ginoux, Malyshev, and Kapnick (2022), GFDL202203, SLAIR.
//!
//! Implements the registered external-mixing BC/dust visible albedo forms
//! (Eqs. 8-18) and the daily surface-layer mass bookkeeping (Eqs. 1-6).
//! Grain radius is supplied by pi-snow; SLAIR's Eq. 7 is intentionally absent.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};

const RHO_SNOW: f64 = 300.0;
const H_NEAR_SURFACE: f64 = 0.03;
const MIN_LAYER_MASS: f64 = 0.75;
const MAX_LAYERS: usize = 365;
const THETA: f64 = 48.19_f64.to_radians();
const DIFFUSE_FRACTION: f64 = 0.0;
const G: f64 = 0.895;
const OMEGA_BC: f64 = 0.209;
const SIGMA_EXT_BC: f64 = 9_267.0;
const SIGMA_ABS_BC: f64 = 7_330.0;
const SIGMA_ABS_DUST: f64 = 67.8;
const SIGMA_ABS_ICE: f64 = 5.795e-14;
const RHO_ICE: f64 = 917.0;
const REAL_REFRACTIVE_INDEX_ICE: f64 = 1.313;
const T_DIFF: f64 = 0.9368;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Layer {
    /// Snow-water mass per unit area, kg m-2.
    pub mass: f64,
    /// BC mass per unit area, kg m-2.
    pub bc_mass: f64,
    /// Dust mass per unit area, kg m-2.
    pub dust_mass: f64,
}

impl Layer {
    fn merge(&mut self, other: Layer) {
        self.mass += other.mass;
        self.bc_mass += other.bc_mass;
        self.dust_mass += other.dust_mass;
    }
}

/// Daily state for SLAIR. Layers are ordered bottom to top.
#[derive(Debug, Default)]
pub struct Slair {
    pub layers: Vec<Layer>,
    pending: Layer,
}

/// Hourly MERRA-2 dry+sedimentation+wet deposition rates, keyed by the
/// collection's HH:30 UTC midpoint. Values are kg m-2 s-1.
#[derive(Debug, Default)]
pub struct DepositionSeries {
    hourly: BTreeMap<i64, (f64, f64)>,
}

impl DepositionSeries {
    /// Strict CSV: `timestamp_utc,bc_kg_m2_s,dust_kg_m2_s`; timestamps use RFC3339.
    pub fn from_csv(text: &str) -> Result<Self, String> {
        let mut lines = text.lines();
        let header = lines.next().ok_or("empty MERRA-2 deposition CSV")?;
        if header.trim() != "timestamp_utc,bc_kg_m2_s,dust_kg_m2_s" {
            return Err("MERRA-2 CSV header must be timestamp_utc,bc_kg_m2_s,dust_kg_m2_s".into());
        }
        let mut hourly = BTreeMap::new();
        for (index, line) in lines.enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let fields: Vec<_> = line.split(',').collect();
            if fields.len() != 3 {
                return Err(format!("MERRA-2 CSV row {} must have 3 fields", index + 2));
            }
            let timestamp = DateTime::parse_from_rfc3339(fields[0])
                .map_err(|error| format!("MERRA-2 CSV row {} timestamp: {error}", index + 2))?
                .with_timezone(&Utc)
                .timestamp();
            if timestamp.rem_euclid(3600) != 1800 {
                return Err(format!(
                    "MERRA-2 CSV row {} timestamp is not HH:30 UTC",
                    index + 2
                ));
            }
            let bc = fields[1]
                .parse::<f64>()
                .map_err(|error| format!("MERRA-2 CSV row {} BC rate: {error}", index + 2))?;
            let dust = fields[2]
                .parse::<f64>()
                .map_err(|error| format!("MERRA-2 CSV row {} dust rate: {error}", index + 2))?;
            if !bc.is_finite() || !dust.is_finite() || bc < 0.0 || dust < 0.0 {
                return Err(format!(
                    "MERRA-2 CSV row {} has invalid deposition rate",
                    index + 2
                ));
            }
            if hourly.insert(timestamp, (bc, dust)).is_some() {
                return Err(format!(
                    "duplicate MERRA-2 timestamp at CSV row {}",
                    index + 2
                ));
            }
        }
        Ok(Self { hourly })
    }

    /// Convert the step midpoint to its containing HH:30 collection record.
    pub fn step_mass(&self, midpoint: DateTime<Utc>, seconds: f64) -> Option<(f64, f64)> {
        let record_time = midpoint.timestamp().div_euclid(3600) * 3600 + 1800;
        self.hourly
            .get(&record_time)
            .map(|(bc, dust)| (bc * seconds, dust * seconds))
    }
}

impl Slair {
    /// Seed a pre-existing snowpack as one clean layer, as registered.
    pub fn initialize_clean(&mut self, swe: f64) {
        self.clear();
        if swe > 0.0 {
            self.layers.push(Layer {
                mass: swe,
                bc_mass: 0.0,
                dust_mass: 0.0,
            });
        }
    }

    /// Apply one local-standard day's fresh snow and accumulated LAI deposition.
    /// Flux inputs are daily masses per unit area in kg m-2.
    pub fn snowfall(&mut self, fresh_snow: f64, bc_deposition: f64, dust_deposition: f64) {
        let fresh = Layer {
            mass: fresh_snow,
            bc_mass: bc_deposition,
            dust_mass: dust_deposition,
        };
        if self.layers.is_empty() {
            self.pending.merge(fresh);
            if self.pending.mass > MIN_LAYER_MASS {
                self.layers.push(self.pending);
                self.pending = Layer {
                    mass: 0.0,
                    bc_mass: 0.0,
                    dust_mass: 0.0,
                };
            }
        } else if self.pending.mass > 0.0 {
            self.pending.merge(fresh);
            if self.pending.mass > MIN_LAYER_MASS {
                self.layers.push(self.pending);
                self.pending = Layer::default();
            }
        } else if fresh_snow > MIN_LAYER_MASS {
            self.layers.push(fresh);
            if self.layers.len() > MAX_LAYERS {
                let bottom = self.layers.remove(0);
                self.layers[0].merge(bottom);
            }
        } else if let Some(top) = self.layers.last_mut() {
            top.merge(fresh);
        }
    }

    /// Apply clean vapor deposition to the current surface without creating a
    /// new layer under the fresh-snow threshold.
    pub fn add_clean_surface_mass(&mut self, mass: f64) {
        if mass <= 0.0 {
            return;
        }
        let clean = Layer {
            mass,
            bc_mass: 0.0,
            dust_mass: 0.0,
        };
        if let Some(top) = self.layers.last_mut() {
            top.merge(clean);
        } else {
            self.pending.merge(clean);
        }
    }

    /// Surface sublimation removes snow and its impurities at layer concentration.
    pub fn sublimate(&mut self, mut mass: f64) {
        mass = mass.max(0.0);
        while mass > 0.0 {
            let Some(top) = self.layers.last_mut() else {
                break;
            };
            let removed = mass.min(top.mass);
            top.mass -= removed;
            mass -= removed;
            if top.mass <= 1e-12 {
                let gone = self.layers.pop().unwrap();
                if let Some(new_top) = self.layers.last_mut() {
                    new_top.bc_mass += gone.bc_mass;
                    new_top.dust_mass += gone.dust_mass;
                }
            }
        }
    }

    /// Surface-melt mode, eta_LAI=0: retain impurities in the remaining snow.
    pub fn melt(&mut self, mut mass: f64) {
        mass = mass.max(0.0);
        while mass > 0.0 {
            let Some(top) = self.layers.last_mut() else {
                break;
            };
            let removed = mass.min(top.mass);
            top.mass -= removed;
            mass -= removed;
            if top.mass <= 1e-12 {
                let gone = self.layers.pop().unwrap();
                if let Some(new_top) = self.layers.last_mut() {
                    new_top.bc_mass += gone.bc_mass;
                    new_top.dust_mass += gone.dust_mass;
                }
            }
        }
    }

    /// Reconcile total model SWE, preserving layer concentrations on removal.
    pub fn reconcile(&mut self, swe: f64) {
        let current: f64 =
            self.layers.iter().map(|layer| layer.mass).sum::<f64>() + self.pending.mass;
        if swe > current {
            self.layers.insert(
                0,
                Layer {
                    mass: swe - current,
                    bc_mass: 0.0,
                    dust_mass: 0.0,
                },
            );
            if self.layers.len() > MAX_LAYERS {
                let bottom = self.layers.remove(0);
                self.layers[0].merge(bottom);
            }
        } else {
            let mut excess = current - swe;
            while excess > 1e-12 {
                let Some(bottom) = self.layers.first_mut() else {
                    break;
                };
                let removed = excess.min(bottom.mass);
                let fraction = if bottom.mass > 0.0 {
                    removed / bottom.mass
                } else {
                    0.0
                };
                bottom.mass -= removed;
                bottom.bc_mass *= 1.0 - fraction;
                bottom.dust_mass *= 1.0 - fraction;
                excess -= removed;
                if bottom.mass <= 1e-12 {
                    self.layers.remove(0);
                }
            }
        }
        if swe <= 0.0 {
            self.clear();
        }
    }

    /// Eq. 6 concentrations averaged through the top 3 cm (ng g-1).
    pub fn near_surface_concentrations(&self) -> (f64, f64) {
        let mut depth = 0.0;
        let mut bc = 0.0;
        let mut dust = 0.0;
        for layer in std::iter::once(&self.pending).chain(self.layers.iter().rev()) {
            if layer.mass <= 0.0 {
                continue;
            }
            let thickness = layer.mass / RHO_SNOW;
            let take = thickness.min(H_NEAR_SURFACE - depth);
            let share = take / thickness;
            bc += layer.bc_mass * share;
            dust += layer.dust_mass * share;
            depth += take;
            if depth >= H_NEAR_SURFACE {
                break;
            }
        }
        if depth < H_NEAR_SURFACE {
            // Eq. 6 extends the uppermost layer concentration over the missing
            // depth; the actual mass-weighted concentration is unchanged.
            if let Some(top) = (self.pending.mass > 0.0)
                .then_some(&self.pending)
                .or_else(|| self.layers.last())
            {
                let thickness = top.mass / RHO_SNOW;
                if thickness > 0.0 {
                    let extra = H_NEAR_SURFACE - depth;
                    bc += top.bc_mass / thickness * extra;
                    dust += top.dust_mass / thickness * extra;
                    depth = H_NEAR_SURFACE;
                }
            }
        }
        if depth == 0.0 {
            (0.0, 0.0)
        } else {
            let mass = depth * RHO_SNOW;
            (bc / mass * 1e9, dust / mass * 1e9)
        }
    }

    pub fn clear(&mut self) {
        self.layers.clear();
        self.pending = Layer {
            mass: 0.0,
            bc_mass: 0.0,
            dust_mass: 0.0,
        };
    }

    pub fn total_mass(&self) -> f64 {
        self.layers.iter().map(|layer| layer.mass).sum::<f64>() + self.pending.mass
    }
}

/// Chan et al. Eq. 10 He et al. clean-snow value; radius is in micrometres.
pub fn clean_snow_albedo(radius_um: f64) -> f64 {
    let r = effective_radius(radius_um);
    let log_r = r.log10();
    (0.980_508 - 0.013_610_4 * log_r - 0.019_541_6 * log_r * log_r).clamp(0.0, 1.0)
}

/// Chan et al. Eq. 8 and Eqs. 9-18; concentrations are ng g-1, radius µm,
/// snow depth m. Returns the visible albedo and the branch used.
pub fn visible_albedo(hs_m: f64, radius_um: f64, bc_ng_g: f64, dust_ng_g: f64) -> (f64, Branch) {
    let ceqv = equivalent_bc_concentration(bc_ng_g, dust_ng_g);
    let radius = effective_radius(radius_um);
    if ceqv < 500.0 && hs_m <= 0.04 {
        return (malinka(hs_m, ceqv, radius), Branch::Malinka);
    }
    if (ceqv >= 7_500.0 && radius_um >= 900.0)
        || (ceqv >= 10_000.0 && (500.0..900.0).contains(&radius_um))
    {
        return (rozenberg(hs_m, ceqv, radius), Branch::Rozenberg);
    }
    let log_r = radius.log10();
    let clean = 0.980_508 - 0.013_610_4 * log_r - 0.019_541_6 * log_r * log_r;
    let darkening = 0.003_014_70 * ceqv.powf(0.468_312 * radius.powf(0.127_961));
    ((clean - darkening).clamp(0.0, 1.0), Branch::He)
}

pub fn equivalent_bc_concentration(bc_ng_g: f64, dust_ng_g: f64) -> f64 {
    bc_ng_g + dust_ng_g * SIGMA_ABS_DUST / SIGMA_ABS_BC
}

pub fn branch_name(branch: Branch) -> &'static str {
    match branch {
        Branch::Malinka => "malinka",
        Branch::Rozenberg => "rozenberg",
        Branch::He => "he",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Branch {
    Malinka,
    Rozenberg,
    He,
}

fn effective_radius(radius_um: f64) -> f64 {
    let delta_cos = THETA.cos() - 49.5_f64.to_radians().cos();
    let phi = (1.0 + 0.781 * delta_cos).powi(2);
    radius_um * phi / 100.0
}

fn malinka(hs_m: f64, ceqv: f64, radius_um: f64) -> f64 {
    let tau = optical_depth(hs_m, ceqv, radius_um);
    let cos = THETA.cos();
    ((1.0 - DIFFUSE_FRACTION) * (1.0 - (12.0 / 7.0) * (1.0 + 2.0 * cos) / (tau + 4.0))
        + DIFFUSE_FRACTION * tau / (tau + 4.0))
        .clamp(0.0, 1.0)
}

fn rozenberg(hs_m: f64, ceqv: f64, radius: f64) -> f64 {
    let tau = optical_depth(hs_m, ceqv, radius * 100.0);
    let omega0 = single_scattering_albedo(ceqv, radius * 100.0);
    let gamma = (3.0 * (1.0 - omega0) * (1.0 - omega0 * G)).sqrt();
    let y = (4.0 * (1.0 - omega0) / (3.0 * (1.0 - omega0 * G))).sqrt();
    let denom = (gamma * tau + y).sinh();
    let direct = (gamma * tau + y * (1.0 - (3.0 / 7.0) * (1.0 + 2.0 * THETA.cos()))).sinh() / denom;
    let diffuse = (gamma * tau).sinh() / denom;
    ((1.0 - DIFFUSE_FRACTION) * direct + DIFFUSE_FRACTION * diffuse).clamp(0.0, 1.0)
}

fn optical_depth(hs_m: f64, ceqv: f64, radius_um: f64) -> f64 {
    let radius_m = radius_um * 1e-6;
    let x = REAL_REFRACTIVE_INDEX_ICE.powi(2) * radius_m * SIGMA_ABS_ICE * RHO_ICE;
    let omega_ice = 1.0 - x * T_DIFF / (x + T_DIFF);
    let sigma_ext_ice = SIGMA_ABS_ICE / (1.0 - omega_ice);
    let sigma_ext_total = sigma_ext_ice + ceqv * 1e-9 * SIGMA_EXT_BC;
    hs_m.max(0.0) * sigma_ext_total * RHO_SNOW
}

fn single_scattering_albedo(ceqv: f64, radius_um: f64) -> f64 {
    let radius_m = radius_um * 1e-6;
    let x = REAL_REFRACTIVE_INDEX_ICE.powi(2) * radius_m * SIGMA_ABS_ICE * RHO_ICE;
    let omega_ice = 1.0 - x * T_DIFF / (x + T_DIFF);
    let sigma_ext_ice = SIGMA_ABS_ICE / (1.0 - omega_ice);
    let impurity = ceqv * 1e-9;
    let ext_bc = impurity * SIGMA_EXT_BC;
    (sigma_ext_ice * omega_ice + ext_bc * OMEGA_BC) / (sigma_ext_ice + ext_bc).max(1e-30)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_deposition_returns_chan_clean_snow_values() {
        for (radius, expected) in [(50.0, 0.983), (500.0, 0.961), (1_150.0, 0.944)] {
            let (albedo, branch) = visible_albedo(0.20, radius, 0.0, 0.0);
            assert_eq!(branch, Branch::He);
            assert!((albedo - expected).abs() < 0.001, "{radius}: {albedo}");
            assert!((albedo - clean_snow_albedo(radius)).abs() < 1e-12);
        }
    }

    #[test]
    fn daily_snowfall_threshold_and_zero_deposition_are_mass_conserving() {
        let mut state = Slair::default();
        state.snowfall(0.5, 0.0, 0.0);
        assert!(state.layers.is_empty());
        state.snowfall(0.3, 0.0, 0.0);
        assert_eq!(state.layers.len(), 1);
        assert_eq!(state.layers[0].mass, 0.8);
        assert_eq!(state.near_surface_concentrations(), (0.0, 0.0));
    }

    #[test]
    fn surface_melt_and_sublimation_retain_surface_impurity() {
        let mut state = Slair::default();
        state.initialize_clean(2.0);
        state.layers[0].bc_mass = 2e-6;
        state.melt(1.0);
        assert_eq!(state.layers[0].mass, 1.0);
        assert_eq!(state.layers[0].bc_mass, 2e-6);
        state.sublimate(0.5);
        assert_eq!(state.layers[0].mass, 0.5);
        assert_eq!(state.layers[0].bc_mass, 2e-6);
    }

    #[test]
    fn zero_swe_clears_layers() {
        let mut state = Slair::default();
        state.initialize_clean(1.0);
        state.reconcile(0.0);
        assert!(state.layers.is_empty());
    }

    #[test]
    fn hourly_rates_map_to_six_steps_by_their_midpoint_hour() {
        let data = DepositionSeries::from_csv(
            "timestamp_utc,bc_kg_m2_s,dust_kg_m2_s\n2026-03-01T12:30:00Z,1e-9,2e-9\n",
        )
        .unwrap();
        let midpoint = DateTime::parse_from_rfc3339("2026-03-01T12:55:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let (bc, dust) = data.step_mass(midpoint, 600.0).unwrap();
        assert!((bc - 6e-7).abs() < 1e-20);
        assert!((dust - 1.2e-6).abs() < 1e-20);
        assert!(
            DepositionSeries::from_csv(
                "timestamp_utc,bc_kg_m2_s,dust_kg_m2_s\n2026-03-01T12:00:00Z,1e-9,2e-9\n"
            )
            .is_err()
        );
    }
}
