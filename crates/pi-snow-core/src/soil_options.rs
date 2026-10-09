//! Registered soil keys; validation waits until all settings have been parsed.
use super::OptionError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Properties {
    MoistLoam,
    DryLoam,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottom {
    DeepTbot8m,
    Geothermal005,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Initialization {
    AirProfile,
    UniformTmelt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Settings {
    pub properties: Option<Properties>,
    pub bottom: Option<Bottom>,
    pub initialization: Option<Initialization>,
    // Bits preserve PhysicsOptions' exact Eq semantics without a float wrapper.
    tbot_bits: Option<u64>,
    offset_k: Option<i8>,
}

impl Settings {
    pub fn tbot_k(self) -> Option<f64> {
        self.tbot_bits.map(f64::from_bits)
    }

    pub fn offset_k(self) -> f64 {
        f64::from(self.offset_k.unwrap_or(0))
    }

    pub fn set(&mut self, key: &str, value: &str) -> Result<(), OptionError> {
        let bad = || OptionError(format!("unknown value {value:?} for physics option {key}"));
        match key {
            "SOIL_PROPERTIES" => {
                self.properties = Some(match value {
                    "MOIST_LOAM" => Properties::MoistLoam,
                    "DRY_LOAM" => Properties::DryLoam,
                    _ => return Err(bad()),
                })
            }
            "SOIL_BOTTOM" => {
                self.bottom = Some(match value {
                    "DEEP_TBOT_8M" => Bottom::DeepTbot8m,
                    "GEOTHERMAL_005" => Bottom::Geothermal005,
                    _ => return Err(bad()),
                })
            }
            "SOIL_INIT" => {
                self.initialization = Some(match value {
                    "AIR_PROFILE" => Initialization::AirProfile,
                    "UNIFORM_TMELT" => Initialization::UniformTmelt,
                    _ => return Err(bad()),
                })
            }
            "SOIL_TBOT_K" => {
                let temperature: f64 = value.parse().map_err(|_| bad())?;
                if !temperature.is_finite() {
                    return Err(bad());
                }
                self.tbot_bits = Some(temperature.to_bits());
            }
            "SOIL_INIT_OFFSET_K" => {
                self.offset_k = Some(match value {
                    "0" => 0,
                    "+3" | "3" => 3,
                    "-3" => -3,
                    _ => return Err(bad()),
                })
            }
            _ => return Err(OptionError(format!("unknown physics option {key:?}"))),
        }
        Ok(())
    }

    pub fn validate(self, column: bool, eq116: bool) -> Result<(), &'static str> {
        if !column {
            return if self == Self::default() {
                Ok(())
            } else {
                Err("SOIL_* keys require GROUND_BOUNDARY=SOIL_COLUMN")
            };
        }
        if eq116 {
            return Err("SOIL_COLUMN is incompatible with TOTAL_MELT=CROCUS_EQ116");
        }
        if self.properties.is_none() || self.bottom.is_none() || self.initialization.is_none() {
            return Err("SOIL_COLUMN requires SOIL_PROPERTIES, SOIL_BOTTOM and SOIL_INIT");
        }
        if self.bottom == Some(Bottom::DeepTbot8m) && self.tbot_bits.is_none() {
            return Err("DEEP_TBOT_8M requires SOIL_TBOT_K");
        }
        if self.initialization != Some(Initialization::AirProfile) && self.offset_k.is_some() {
            return Err("SOIL_INIT_OFFSET_K requires SOIL_INIT=AIR_PROFILE");
        }
        Ok(())
    }
}
