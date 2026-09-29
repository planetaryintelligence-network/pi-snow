//! Physics options chosen on the command line (`--physics KEY=VALUE`).
//!
//! Phase 2 results are bound to the engine's behaviour by binary hash, so new
//! physics never replaces an existing form in place. Each alternative is a
//! labelled variant here, and `PhysicsOptions::default()` is exactly the
//! Phase 2 engine: a run given no `--physics` flag must write byte-identical
//! output to the Phase 2 binary. Unknown keys and values are refused, the same
//! way the INI reader refuses physics it does not implement.

use std::fmt;

/// Irreducible water content of the bucket scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WaterRetention {
    /// Wever et al. (2014) eq. 1 as printed, clamped at the pore volume (Phase 2).
    #[default]
    Wever2014Printed,
    /// Coleou and Lesaffre (1998): their regression over the porosity range
    /// their samples span, a constant share of pore volume outside it.
    ColeouLesaffre1998,
    /// The same regression over the narrower range P/(1-P) 0.9 to 1.8 once
    /// proposed for it (sensitivity arm of engine-mass-v1).
    ColeouLesaffre1998Narrow,
}

/// Mapping of microstructure onto the Swiss grain code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GrainClassification {
    /// The Phase 2 engine's own, unsourced rule.
    #[default]
    PiSnow2026,
}

/// The SMET `S5` stability column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum S5Index {
    /// Not computed; written as nodata (Phase 2).
    #[default]
    None,
    /// The liquid water content index of Mitterer et al. (2013) eq. 1, which is
    /// what SNOWPACK 3.7.0 writes as `S5` here (identified from its output);
    /// see `pi_snow_physics::stability::lwc_index`.
    LwcIndexMitterer2013,
}

/// Extra mass bookkeeping output. Changes no physics and no byte of PRO or SMET.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MassReport {
    /// SMET and PRO only (Phase 2).
    #[default]
    Phase2,
    /// Also write `<station>_<experiment>.mass.csv`, the engine's own mass
    /// terms per output row, since the SMET columns omit deposition.
    Ledger,
}

/// Turbulent exchange scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Turbulence {
    /// Whatever the INI's `ATMOSPHERIC_STABILITY` selects (Phase 2).
    #[default]
    FromIni,
    /// The engine's Louis (1979) bulk-Richardson scheme whatever the INI
    /// says: a diagnostic arm of engine-mass-v1 (amendment 1), not a fix.
    Louis1979,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PhysicsOptions {
    pub water_retention: WaterRetention,
    pub grain_classification: GrainClassification,
    pub s5: S5Index,
    pub mass_report: MassReport,
    pub turbulence: Turbulence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionError(pub String);

impl fmt::Display for OptionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for OptionError {}

impl PhysicsOptions {
    /// Apply one `KEY=VALUE` setting. Keys and values are upper-case names.
    pub fn set(&mut self, setting: &str) -> Result<(), OptionError> {
        let (key, value) = setting
            .split_once('=')
            .ok_or_else(|| OptionError(format!("--physics expects KEY=VALUE, got {setting:?}")))?;
        let bad = || OptionError(format!("unknown value {value:?} for physics option {key}"));
        match key {
            "WATER_RETENTION" => {
                self.water_retention = match value {
                    "WEVER2014_PRINTED" => WaterRetention::Wever2014Printed,
                    "COLEOU_LESAFFRE_1998" => WaterRetention::ColeouLesaffre1998,
                    "COLEOU_LESAFFRE_1998_NARROW" => WaterRetention::ColeouLesaffre1998Narrow,
                    _ => return Err(bad()),
                }
            }
            "GRAIN_CLASSIFICATION" => {
                self.grain_classification = match value {
                    "PI_SNOW_2026" => GrainClassification::PiSnow2026,
                    _ => return Err(bad()),
                }
            }
            "S5" => {
                self.s5 = match value {
                    "NONE" => S5Index::None,
                    "LWC_INDEX_MITTERER2013" => S5Index::LwcIndexMitterer2013,
                    _ => return Err(bad()),
                }
            }
            "MASS_REPORT" => {
                self.mass_report = match value {
                    "PHASE2" => MassReport::Phase2,
                    "LEDGER" => MassReport::Ledger,
                    _ => return Err(bad()),
                }
            }
            "TURBULENCE" => {
                self.turbulence = match value {
                    "FROM_INI" => Turbulence::FromIni,
                    "LOUIS1979" => Turbulence::Louis1979,
                    _ => return Err(bad()),
                }
            }
            _ => return Err(OptionError(format!("unknown physics option {key:?}"))),
        }
        Ok(())
    }

    pub fn from_settings<'a>(
        settings: impl IntoIterator<Item = &'a str>,
    ) -> Result<Self, OptionError> {
        let mut options = Self::default();
        for s in settings {
            options.set(s)?;
        }
        Ok(options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_phase_2() {
        let o = PhysicsOptions::from_settings([]).unwrap();
        assert_eq!(o.water_retention, WaterRetention::Wever2014Printed);
        assert_eq!(o.grain_classification, GrainClassification::PiSnow2026);
        assert_eq!(o.s5, S5Index::None);
        assert_eq!(o.mass_report, MassReport::Phase2);
        assert_eq!(o.turbulence, Turbulence::FromIni);
    }

    #[test]
    fn engine_mass_values_parse() {
        let o = PhysicsOptions::from_settings([
            "WATER_RETENTION=COLEOU_LESAFFRE_1998",
            "MASS_REPORT=LEDGER",
        ])
        .unwrap();
        assert_eq!(o.water_retention, WaterRetention::ColeouLesaffre1998);
        assert_eq!(o.mass_report, MassReport::Ledger);
        let o =
            PhysicsOptions::from_settings(["WATER_RETENTION=COLEOU_LESAFFRE_1998_NARROW"]).unwrap();
        assert_eq!(o.water_retention, WaterRetention::ColeouLesaffre1998Narrow);
        assert!(PhysicsOptions::from_settings(["MASS_REPORT=YES"]).is_err());
        let o = PhysicsOptions::from_settings(["TURBULENCE=LOUIS1979"]).unwrap();
        assert_eq!(o.turbulence, Turbulence::Louis1979);
        assert_eq!(
            PhysicsOptions::from_settings(["TURBULENCE=FROM_INI"]).unwrap(),
            PhysicsOptions::default()
        );
    }

    #[test]
    fn explicit_phase_2_values_parse() {
        let o = PhysicsOptions::from_settings([
            "WATER_RETENTION=WEVER2014_PRINTED",
            "GRAIN_CLASSIFICATION=PI_SNOW_2026",
            "S5=NONE",
        ])
        .unwrap();
        assert_eq!(o, PhysicsOptions::default());
    }

    #[test]
    fn unknown_keys_and_values_are_refused() {
        assert!(PhysicsOptions::from_settings(["NOPE=1"]).is_err());
        assert!(PhysicsOptions::from_settings(["S5=MAYBE"]).is_err());
        assert!(PhysicsOptions::from_settings(["S5"]).is_err());
    }

    #[test]
    fn s5_lwc_index_parses() {
        let o = PhysicsOptions::from_settings(["S5=LWC_INDEX_MITTERER2013"]).unwrap();
        assert_eq!(o.s5, S5Index::LwcIndexMitterer2013);
        assert_eq!(
            PhysicsOptions {
                s5: S5Index::None,
                ..o
            },
            PhysicsOptions::default(),
            "only the S5 key changes"
        );
    }
}
