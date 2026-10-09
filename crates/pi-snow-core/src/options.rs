//! Physics options chosen on the command line (`--physics KEY=VALUE`).
//!
//! Phase 2 results are bound to the engine's behaviour by binary hash, so new
//! physics never replaces an existing form in place. Each alternative is a
//! labelled variant here, and `PhysicsOptions::default()` is exactly the
//! Phase 2 engine: a run given no `--physics` flag must write byte-identical
//! output to the Phase 2 binary. Unknown keys and values are refused, the same
//! way the INI reader refuses physics it does not implement.

use std::fmt;

#[path = "soil_options.rs"]
pub mod soil;

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
    /// Crocus v3.0.2's Eq. 144 with its history tracker (Lafaysse et al. 2026).
    Lafaysse2026,
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

/// How the Monin-Obukhov surface layer is solved when the INI selects
/// `MO_HOLTSLAG` (engine-surface-v1). Only valid with that scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SurfaceLayer {
    /// Wind at its own height, scalars at theirs, no limit on z/L (Phase 2).
    #[default]
    Phase2,
    /// Wind carried to the scalar height along the neutral log profile, then
    /// the same solve at that one height (after Schlogl et al. 2017's
    /// single-height form); drops the momentum stability correction above
    /// the scalar height.
    ScalarHeightNeutralWind,
    /// Wind and scalars at their own heights, with z/L at the wind height held
    /// to 0.01..2 (stable) and -100..-0.01 (unstable), CLM5 eq. 2.5.49.
    ZetaLimitClm5,
}

/// Snow-surface reflectance and its aging law.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SnowAlbedo {
    /// Vionnet et al. (2012), Table 4 (Phase 2).
    #[default]
    Vionnet2012,
    /// Aoki et al. (2003), Figure 6 and Table 2, temperature-rate aging.
    Aoki2003TempRate,
    /// Chan et al. (2022) SLAIR with independent MERRA-2 impurity deposition.
    Chan2022Merra2,
}

/// Growth of wet rounded grains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WetGrainGrowth {
    /// Wet rounded grains keep their size (Phase 2).
    #[default]
    None,
    /// Brun (1989): grain volume grows at 1.28e-8 + 4.22e-10 L^3 mm^3 s^-1,
    /// L the liquid water in per cent of mass, for non-dendritic grains of
    /// sphericity 1 (Vionnet et al. 2012, Table 2).
    Brun1989,
}

/// What happens when sublimation takes the top element's last ice
/// (engine-finite-guard-v1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SublimationMeltOut {
    /// The element's thickness is scaled to zero and it stays until phase
    /// change removes it; its liquid water leaves with no term, and the next
    /// conduction solve meets a zero-thickness element, which leaves one NaN
    /// node when the column empties (Phase 2).
    #[default]
    Phase2,
    /// The element is removed at once with its top node; its liquid water is
    /// counted as runoff.
    RemoveElement,
}

/// How the surface energy balance is linearised for the implicit solve
/// (engine-finite-guard-v1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SurfaceLinearisation {
    /// Finite-difference slope, capped at zero (Phase 2).
    #[default]
    Phase2,
    /// The finite-difference slope, but never shallower than the emitted
    /// longwave's own, -4 eps sigma T^3, which the budget always contains.
    RadiativeFloor,
}

/// Optional pre-solve merge of a thin top snow element (engine-cold-crash-v1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinTopMerge {
    /// Phase 2 element ordering, unchanged.
    #[default]
    Phase2,
    /// Merge one top snow element thinner than 10 mm into the element below.
    Clm4_010Mm,
}

/// Lower boundary treatment registered by engine-meltout-v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GroundBoundary {
    /// Prescribed forcing temperature (Phase 2).
    #[default]
    Phase2,
    /// Prescribed `GEO_HEAT` flux into snow, W m-2.
    GeoHeatFlux,
    /// Prognostic registered one-metre column beneath the snow.
    SoilColumn,
}

/// Whole-pack melt trigger registered by engine-meltout-v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TotalMelt {
    /// Layer phase change (Phase 2).
    #[default]
    Phase2,
    /// Crocus v3.0.2 §2.4.15 Eq. 116 whole-pack energy condition.
    CrocusEq116,
}

/// Fresh-snow density assigned to deposited solid precipitation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FreshSnowDensity {
    /// Phase 2 Pahaut/Vionnet form.
    #[default]
    Phase2,
    /// Anderson (1976), reproduced by Lafaysse et al. (2026), Eq. 35.
    A76Anderson1976,
}

/// Dry-layer viscosity selection for the density experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DrySettlement {
    /// Phase 2 BR92 viscosity for every layer.
    #[default]
    Br92,
    /// Teufelsbauer's dry-snow form, with wet layers retained at BR92.
    T11,
    /// Kojima's dry dendritic form, with other layers retained at BR92.
    K75,
    /// K75 on dry dendritic layers and T11 on other dry layers.
    T11K75,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PhysicsOptions {
    pub water_retention: WaterRetention,
    pub grain_classification: GrainClassification,
    pub s5: S5Index,
    pub mass_report: MassReport,
    pub turbulence: Turbulence,
    pub surface_layer: SurfaceLayer,
    pub snow_albedo: SnowAlbedo,
    pub wet_grain_growth: WetGrainGrowth,
    pub sublimation_melt_out: SublimationMeltOut,
    pub surface_linearisation: SurfaceLinearisation,
    pub thin_top_merge: ThinTopMerge,
    pub ground_boundary: GroundBoundary,
    pub soil: soil::Settings,
    pub total_melt: TotalMelt,
    pub fresh_snow_density: FreshSnowDensity,
    pub dry_settlement: DrySettlement,
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
                    "LAFAYSSE2026" => GrainClassification::Lafaysse2026,
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
            "SURFACE_LAYER" => {
                self.surface_layer = match value {
                    "PHASE2" => SurfaceLayer::Phase2,
                    "SCALAR_HEIGHT_NEUTRAL_WIND" => SurfaceLayer::ScalarHeightNeutralWind,
                    "ZETA_LIMIT_CLM5" => SurfaceLayer::ZetaLimitClm5,
                    _ => return Err(bad()),
                }
            }
            "SNOW_ALBEDO" => {
                self.snow_albedo = match value {
                    "VIONNET2012" => SnowAlbedo::Vionnet2012,
                    "AOKI2003_TEMP_RATE" => SnowAlbedo::Aoki2003TempRate,
                    "CHAN2022_MERRA2" => SnowAlbedo::Chan2022Merra2,
                    _ => return Err(bad()),
                }
            }
            "SUBLIMATION_MELT_OUT" => {
                self.sublimation_melt_out = match value {
                    "PHASE2" => SublimationMeltOut::Phase2,
                    "REMOVE_ELEMENT" => SublimationMeltOut::RemoveElement,
                    _ => return Err(bad()),
                }
            }
            "SURFACE_LINEARISATION" => {
                self.surface_linearisation = match value {
                    "PHASE2" => SurfaceLinearisation::Phase2,
                    "RADIATIVE_FLOOR" => SurfaceLinearisation::RadiativeFloor,
                    _ => return Err(bad()),
                }
            }
            "GROUND_BOUNDARY" => {
                self.ground_boundary = match value {
                    "PHASE2" => GroundBoundary::Phase2,
                    "GEO_HEAT_FLUX" => GroundBoundary::GeoHeatFlux,
                    "SOIL_COLUMN" => GroundBoundary::SoilColumn,
                    _ => return Err(bad()),
                }
            }
            "TOTAL_MELT" => {
                self.total_melt = match value {
                    "PHASE2" => TotalMelt::Phase2,
                    "CROCUS_EQ116" => TotalMelt::CrocusEq116,
                    _ => return Err(bad()),
                }
            }
            "WET_GRAIN_GROWTH" => {
                self.wet_grain_growth = match value {
                    "NONE" => WetGrainGrowth::None,
                    "BRUN1989" => WetGrainGrowth::Brun1989,
                    _ => return Err(bad()),
                }
            }
            "THIN_TOP_MERGE" => {
                self.thin_top_merge = match value {
                    "PHASE2" => ThinTopMerge::Phase2,
                    "CLM4_010MM" => ThinTopMerge::Clm4_010Mm,
                    _ => return Err(bad()),
                }
            }
            "NEW_SNOW_DENSITY" => {
                self.fresh_snow_density = match value {
                    "PHASE2" => FreshSnowDensity::Phase2,
                    "A76_ANDERSON1976" => FreshSnowDensity::A76Anderson1976,
                    _ => return Err(bad()),
                }
            }
            "DRY_SETTLEMENT" => {
                self.dry_settlement = match value {
                    "BR92" => DrySettlement::Br92,
                    "T11" => DrySettlement::T11,
                    "K75" => DrySettlement::K75,
                    "T11_K75" => DrySettlement::T11K75,
                    _ => return Err(bad()),
                }
            }
            key if key.starts_with("SOIL_") => self.soil.set(key, value)?,
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
        assert_eq!(o.surface_layer, SurfaceLayer::Phase2);
        assert_eq!(o.snow_albedo, SnowAlbedo::Vionnet2012);
        assert_eq!(o.wet_grain_growth, WetGrainGrowth::None);
        assert_eq!(o.sublimation_melt_out, SublimationMeltOut::Phase2);
        assert_eq!(o.surface_linearisation, SurfaceLinearisation::Phase2);
    }

    #[test]
    fn engine_finite_guard_values_parse() {
        let o = PhysicsOptions::from_settings([
            "SUBLIMATION_MELT_OUT=REMOVE_ELEMENT",
            "SURFACE_LINEARISATION=RADIATIVE_FLOOR",
        ])
        .unwrap();
        assert_eq!(o.sublimation_melt_out, SublimationMeltOut::RemoveElement);
        assert_eq!(
            o.surface_linearisation,
            SurfaceLinearisation::RadiativeFloor
        );
        assert_eq!(
            PhysicsOptions::from_settings([
                "SUBLIMATION_MELT_OUT=PHASE2",
                "SURFACE_LINEARISATION=PHASE2"
            ])
            .unwrap(),
            PhysicsOptions::default()
        );
        assert!(PhysicsOptions::from_settings(["SUBLIMATION_MELT_OUT=DROP"]).is_err());
        assert!(PhysicsOptions::from_settings(["SURFACE_LINEARISATION=CLAMP"]).is_err());
    }

    #[test]
    fn engine_surface_values_parse() {
        let o = PhysicsOptions::from_settings([
            "SURFACE_LAYER=SCALAR_HEIGHT_NEUTRAL_WIND",
            "WET_GRAIN_GROWTH=BRUN1989",
        ])
        .unwrap();
        assert_eq!(o.surface_layer, SurfaceLayer::ScalarHeightNeutralWind);
        assert_eq!(o.wet_grain_growth, WetGrainGrowth::Brun1989);
        let o = PhysicsOptions::from_settings(["SURFACE_LAYER=ZETA_LIMIT_CLM5"]).unwrap();
        assert_eq!(o.surface_layer, SurfaceLayer::ZetaLimitClm5);
        assert_eq!(
            PhysicsOptions::from_settings(["SURFACE_LAYER=PHASE2", "WET_GRAIN_GROWTH=NONE"])
                .unwrap(),
            PhysicsOptions::default()
        );
        assert!(PhysicsOptions::from_settings(["SURFACE_LAYER=CAP"]).is_err());
        assert!(PhysicsOptions::from_settings(["WET_GRAIN_GROWTH=YES"]).is_err());
        let o = PhysicsOptions::from_settings(["SNOW_ALBEDO=AOKI2003_TEMP_RATE"]).unwrap();
        assert_eq!(o.snow_albedo, SnowAlbedo::Aoki2003TempRate);
        let o = PhysicsOptions::from_settings(["SNOW_ALBEDO=CHAN2022_MERRA2"]).unwrap();
        assert_eq!(o.snow_albedo, SnowAlbedo::Chan2022Merra2);
        assert!(PhysicsOptions::from_settings(["SNOW_ALBEDO=UNKNOWN"]).is_err());
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
    fn meltout_candidate_keys_parse_without_changing_other_options() {
        let base = PhysicsOptions::default();
        assert_eq!(
            PhysicsOptions::from_settings(["GROUND_BOUNDARY=PHASE2", "TOTAL_MELT=PHASE2"]).unwrap(),
            base
        );
        let flux = PhysicsOptions::from_settings(["GROUND_BOUNDARY=GEO_HEAT_FLUX"]).unwrap();
        assert_eq!(flux.ground_boundary, GroundBoundary::GeoHeatFlux);
        assert_eq!(flux.total_melt, TotalMelt::Phase2);
        let melt = PhysicsOptions::from_settings(["TOTAL_MELT=CROCUS_EQ116"]).unwrap();
        assert_eq!(melt.total_melt, TotalMelt::CrocusEq116);
        assert!(PhysicsOptions::from_settings(["GROUND_BOUNDARY=FLUX"]).is_err());
        assert!(PhysicsOptions::from_settings(["TOTAL_MELT=EQ116"]).is_err());
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
    fn lafaysse2026_grain_classification_parses() {
        let o = PhysicsOptions::from_settings(["GRAIN_CLASSIFICATION=LAFAYSSE2026"]).unwrap();
        assert_eq!(o.grain_classification, GrainClassification::Lafaysse2026);
        assert_eq!(o.water_retention, WaterRetention::default());
        assert_eq!(o.s5, S5Index::default());
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
