//! Shear strength, hand hardness and the stability indices.
//!
//! These take grain type as an input, so they can be tested and used before
//! the engine's own metamorphism exists. Sources, all read in full:
//!
//! * Shear strength by grain type: Jamieson and Johnston (2001), *Evaluation
//!   of the shear frame test for weak snowpack layers*, Annals of Glaciology 32,
//!   59-69, Table 8. Lehning et al. (2004, Annals of Glaciology 38, 331-338)
//!   state that SNOWPACK's default strength follows this paper, on density and
//!   grain type only, but do not reproduce the table.
//! * Buried surface hoar strength: Lehning et al. (2004) Table 1, their
//!   re-analysis without minimum grain size.
//! * Normal-load adjustment, the natural and skier indices, the 38-degree skier
//!   stress and ski penetration: Jamieson and Johnston (1998), *Refinements to
//!   the stability index for skier-triggered dry-slab avalanches*, Annals of
//!   Glaciology 26, 296-302.
//! * Sk38 in SNOWPACK's form, SSI and ski penetration: Schweizer et al.
//!   (2006), CRST 46, 52-59, eqs. 1-5 and p. 54. Bellaire et al. (2006, ISSW)
//!   restate the same and were read first.
//! * Critical cut length: Richter et al. (2019), *Validating modeled critical
//!   crack length for crack propagation in the snow cover model SNOWPACK*, The
//!   Cryosphere 13, 3353-3366, eqs. 1-3.
//! * Hand hardness: Monti, Schweizer and Fierz (2014), *Hardness estimation and
//!   weak layer detection in simulated snow stratigraphy*, CRST 103, 82-90,
//!   Table 2.
//!
//! Choices that are ours, not the sources', and that a reader should not
//! mistake for published physics:
//!
//! * Jamieson and Johnston (2001) fit only precipitation particles, decomposing
//!   particles and rounds (Group I) and facets and depth hoar (Group II). Melt
//!   forms and ice are put in Group I and rounding facets in Group II here.
//! * The normal-load term follows Jamieson and Johnston (1998) for both
//!   indices, zero for persistent classes. Lehning et al. (2004) instead extend
//!   the rounded-grain term to every class but buried surface hoar; not used.
//! * SSI's structural contrast is taken across the interface with the layer
//!   above.
//! * The shallow-depth convention, the ceilings (6 for the indices, 3 m for
//!   critical cut length) and writing the ceiling rather than 0 above the ski
//!   penetration depth are copied from the reference engine's output, not
//!   derived.

/// Primary grain classes in the Swiss code's first digit.
///
/// The mapping is established from interface evidence alone: the reference
/// writes 660 for surface hoar and 772 for a melt-freeze crust, and the frozen
/// feature extractor's hardness rows match Monti et al. (2014) Table 2 in
/// exactly this order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum GrainClass {
    PP,
    DF,
    RG,
    FC,
    DH,
    SH,
    MF,
    IF,
    FCxr,
}

impl GrainClass {
    pub fn from_primary_digit(digit: u16) -> Option<Self> {
        use GrainClass::*;
        Some(match digit {
            1 => PP,
            2 => DF,
            3 => RG,
            4 => FC,
            5 => DH,
            6 => SH,
            7 => MF,
            8 => IF,
            9 => FCxr,
            _ => return None,
        })
    }

    /// Class of a three-digit Swiss code, from its first digit.
    pub fn from_code(code: u16) -> Option<Self> {
        Self::from_primary_digit(code / 100)
    }

    /// A pure-class Swiss code: the digit repeated in the first two places and
    /// a zero marker, as the reference writes for fresh snow (110).
    pub fn code(self) -> u16 {
        let d = self as u16 + 1;
        d * 100 + d * 10
    }

    /// Persistent weak-layer classes, for which Jamieson and Johnston (1998)
    /// set the normal-load adjustment to zero.
    pub fn is_persistent(self) -> bool {
        matches!(
            self,
            GrainClass::FC | GrainClass::DH | GrainClass::SH | GrainClass::FCxr
        )
    }
}

/// One layer as the stability calculation sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    /// Vertical depth of the layer's top below the snow surface, m.
    pub depth_top: f64,
    /// Vertical thickness, m.
    pub thickness: f64,
    /// Bulk density, kg m^-3.
    pub density: f64,
    /// Swiss three-digit grain code.
    pub grain_type: u16,
    /// Grain size, mm.
    pub grain_size_mm: f64,
    /// Hand hardness index, 1 to 6.
    pub hand_hardness: f64,
    /// Layer temperature, degrees C.
    pub temperature_c: f64,
    /// Time since deposition, days.
    pub age_days: f64,
}

const RHO_ICE: f64 = 917.0;
const GRAVITY: f64 = 9.80665;
/// Reference slope of every index here, regardless of the run's own slope.
const PSI_38: f64 = 38.0 * std::f64::consts::PI / 180.0;
/// Ceiling the reference engine writes for Sn38, Sk38 and SSI.
pub const INDEX_CAP: f64 = 6.0;
/// Ceiling the reference engine writes for critical cut length, m. Observed in
/// its output (0606 never exceeds 3.00), not derived.
pub const CUT_LENGTH_CAP: f64 = 3.0;
/// `MIN_DEPTH_SUBSURF` from the target INI: layers whose top is shallower than
/// this get the ceiling, which is what the reference output shows.
pub const MIN_DEPTH_SUBSURF: f64 = 0.07;
/// Floor on the buried surface hoar regression, kPa. The published linear fit
/// goes negative for deep, thick, cold layers outside its data; a negative
/// strength has no meaning. Ours, not the paper's.
const MIN_STRENGTH_KPA: f64 = 0.01;

/// Daniels shear strength of a layer, kPa.
///
/// `sigma_zz_kpa` is the slope-normal load on it and `snow_height_m` the total
/// snow depth; both enter only the buried surface hoar regression.
pub fn shear_strength_kpa(layer: &Layer, sigma_zz_kpa: f64, snow_height_m: f64) -> f64 {
    let x = layer.density / RHO_ICE;
    match GrainClass::from_code(layer.grain_type) {
        Some(GrainClass::SH) => (0.429 + 0.0138 * layer.age_days + 1.12 * sigma_zz_kpa
            - 0.00596 * (100.0 * layer.depth_top)
            + 0.000785 * (100.0 * snow_height_m)
            - 0.271 * (100.0 * layer.thickness)
            + 0.0202 * layer.temperature_c)
            .max(MIN_STRENGTH_KPA),
        // Group II, Model B: the paper's better fit for facets and depth hoar.
        Some(GrainClass::FC | GrainClass::DH | GrainClass::FCxr) => 18.5 * x.powf(2.11),
        // Group I, Model A, and by our choice melt forms, ice and unknown codes.
        _ => 14.5 * x.powf(1.73),
    }
}

/// Normal-load adjustment `phi`, dimensionless, from Jamieson and Johnston
/// (1998) eqs. 5a-5b in their Daniels-strength form.
fn normal_load_factor(class: Option<GrainClass>, strength_kpa: f64, sigma_zz_kpa: f64) -> f64 {
    match class {
        Some(c) if c.is_persistent() => 0.0,
        Some(GrainClass::PP) => 0.08 * strength_kpa + 0.056 + 0.022 * sigma_zz_kpa,
        _ => 0.08 * strength_kpa + 0.224,
    }
}

/// Which published form of the skier index to use: it fixes both the skier's
/// shear stress at 38 degrees and the ski penetration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SkierForm {
    /// Schweizer, Bellaire, Fierz, Lehning and Pielmeier (2006), *Evaluating
    /// and improving the stability predictions of the snow cover model
    /// SNOWPACK*, CRST 46, 52-59, read in full. Skier stress `155 / (h - P)` Pa
    /// at 38 degrees (their eq. 1), and `P = 34.6 / rho_30` with `rho_30` the
    /// mean density of the upper 30 cm of the snowpack (p. 54). The default: it
    /// is the paper that describes SNOWPACK's own stability module.
    ///
    /// `34.6 = 0.8 * 43.3` to three figures, which is where the frozen RF1.1
    /// extractor's otherwise unexplained 0.8 comes from.
    #[default]
    Schweizer2006,
    /// Jamieson and Johnston (1998): skier stress `0.14 / (h - P)` kPa at 38
    /// degrees and `P = 42.4 / rho_30` with `rho_30 = rho_bar + 150 (0.3 - h/2)`
    /// estimated from the mean slab density, their r = 0.72 fit.
    JamiesonJohnston1998,
}

/// Skier-induced shear stress on a 38-degree slope, kPa, for slab thickness
/// `h` and ski penetration `p` (both m). The closed form at 38 degrees is what
/// removes the need for Fohn's (1987) alpha_max table, which was not obtained.
pub fn skier_shear_stress_38_kpa(form: SkierForm, h: f64, penetration: f64) -> f64 {
    let numerator = match form {
        SkierForm::Schweizer2006 => 0.155,
        SkierForm::JamiesonJohnston1998 => 0.14,
    };
    numerator / (h - penetration)
}

/// Ski penetration, m.
///
/// `upper_30_density` is the thickness-weighted mean density of the top 0.30
/// m of the snowpack (Schweizer 2006); `slab_density` and `h` are the mean
/// density and thickness of the slab above the layer (J&J 1998).
pub fn ski_penetration(form: SkierForm, upper_30_density: f64, slab_density: f64, h: f64) -> f64 {
    match form {
        SkierForm::Schweizer2006 => 34.6 / upper_30_density.max(1.0),
        SkierForm::JamiesonJohnston1998 => {
            let rho_30 = (slab_density + 150.0 * (0.3 - h / 2.0)).max(1.0);
            42.4 / rho_30
        }
    }
}

/// Thickness-weighted mean density of the top `depth` metres of a profile,
/// bottom layer first.
pub fn upper_mean_density(layers: &[Layer], depth: f64) -> f64 {
    let (mut mass, mut thickness) = (0.0, 0.0);
    for l in layers.iter().rev() {
        let take = l.thickness.min((depth - thickness).max(0.0));
        if take <= 0.0 {
            break;
        }
        mass += l.density * take;
        thickness += take;
    }
    if thickness > 0.0 {
        mass / thickness
    } else {
        0.0
    }
}

/// Structural penalty added to Sk38 to make SSI: one for a hardness contrast
/// below 1.5 steps, one for a grain size contrast below 0.5 mm.
pub fn structural_penalty(
    hardness: f64,
    hardness_above: f64,
    grain_mm: f64,
    grain_above_mm: f64,
) -> f64 {
    let dr = if (hardness_above - hardness).abs() < 1.5 {
        1.0
    } else {
        0.0
    };
    let de = if (grain_above_mm - grain_mm).abs() < 0.5 {
        1.0
    } else {
        0.0
    };
    dr + de
}

/// Critical cut length, m, Richter et al. (2019):
/// `r_c = sqrt(E' D F 2 tau / sigma_n)`, `E' = E / (1 - 0.2^2)`,
/// `E = 5.07e9 (rho_sl / rho_ice)^5.13` Pa, `sigma_n = rho_sl g D`,
/// `F = 4.6e-9 ((rho_wl / rho_ice) (gs_wl / 1.25 mm))^-2` m/Pa.
pub fn critical_cut_length_m(
    rho_slab: f64,
    slab_depth: f64,
    rho_weak: f64,
    grain_weak_mm: f64,
    strength_kpa: f64,
) -> f64 {
    let modulus = 5.07e9 * (rho_slab / RHO_ICE).powf(5.13);
    let plane_strain = modulus / (1.0 - 0.2_f64.powi(2));
    let compliance = 4.6e-9 * ((rho_weak / RHO_ICE) * (grain_weak_mm / 1.25)).powf(-2.0);
    let sigma_n = rho_slab * GRAVITY * slab_depth;
    (plane_strain * slab_depth * compliance * 2.0 * strength_kpa * 1e3 / sigma_n).sqrt()
}

/// Hand hardness index from Monti et al. (2014) Table 2, or `None` for surface
/// hoar and ice, which the table does not cover.
pub fn monti_hardness(class: GrainClass, density: f64) -> Option<f64> {
    use GrainClass::*;
    let lower_bounds: &[f64] = match class {
        PP => &[144.0],
        DF => &[214.0, 269.0],
        RG => &[190.0, 278.0, 369.0, 444.0],
        FC => &[248.0, 320.0, 401.0, 519.0],
        DH => &[288.0, 345.0],
        MF => &[214.0, 318.0, 408.0, 740.0],
        FCxr => &[260.0, 327.0, 397.0, 485.0],
        SH | IF => return None,
    };
    Some(1.0 + lower_bounds.iter().filter(|&&b| density >= b).count() as f64)
}

/// Stability indices of one layer, each `None` where the reference writes
/// nodata.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Indices {
    pub strength_kpa: Option<f64>,
    pub sn38: Option<f64>,
    pub sk38: Option<f64>,
    pub ssi: Option<f64>,
    pub critical_cut_length_m: Option<f64>,
}

/// Indices for every layer of a profile, bottom layer first.
pub fn indices(layers: &[Layer], snow_height_m: f64) -> Vec<Indices> {
    indices_with(layers, snow_height_m, SkierForm::default())
}

/// Indices for every layer, with a chosen skier-index form.
///
/// Where the ski penetrates past a layer, Jamieson and Johnston (1998) define
/// its skier index as 0. The reference engine writes the ceiling there instead,
/// treating the layer as not skier-relevant. That convention is copied: the
/// `Ss` time series feeding an RF1.1 feature is a profile minimum, and a 0 in
/// every shallow layer would pull it to zero for a bookkeeping reason.
pub fn indices_with(layers: &[Layer], snow_height_m: f64, form: SkierForm) -> Vec<Indices> {
    let upper_30 = upper_mean_density(layers, 0.30);
    let n = layers.len();
    let (sin, cos) = (PSI_38.sin(), PSI_38.cos());
    let mut out = Vec::with_capacity(n);
    for (i, layer) in layers.iter().enumerate() {
        let h = layer.depth_top;
        if h < MIN_DEPTH_SUBSURF {
            out.push(Indices {
                strength_kpa: None,
                sn38: Some(INDEX_CAP),
                sk38: Some(INDEX_CAP),
                ssi: Some(INDEX_CAP),
                critical_cut_length_m: None,
            });
            continue;
        }
        // Slab weight per horizontal area, kPa.
        let weight = layers[i + 1..]
            .iter()
            .map(|l| l.density * l.thickness)
            .sum::<f64>()
            * GRAVITY
            / 1e3;
        let sigma_zz = weight * cos * cos;
        let sigma_xz = weight * sin * cos;
        let strength = shear_strength_kpa(layer, sigma_zz, snow_height_m);
        let class = GrainClass::from_code(layer.grain_type);
        let resisting = strength + sigma_zz * normal_load_factor(class, strength, sigma_zz);
        let sn38 = (resisting / sigma_xz).min(INDEX_CAP);

        let rho_bar = weight * 1e3 / (GRAVITY * h);
        let penetration = ski_penetration(form, upper_30, rho_bar, h);
        let sk38 = if penetration >= h {
            INDEX_CAP
        } else {
            (resisting / (sigma_xz + skier_shear_stress_38_kpa(form, h, penetration)))
                .min(INDEX_CAP)
        };
        let ssi = layers.get(i + 1).map(|above| {
            (sk38
                + structural_penalty(
                    layer.hand_hardness,
                    above.hand_hardness,
                    layer.grain_size_mm,
                    above.grain_size_mm,
                ))
            .min(INDEX_CAP)
        });
        // The reference writes nodata for the bottom element's critical cut
        // length, as for the top; copied, not derived.
        let rc = (i > 0).then(|| {
            critical_cut_length_m(rho_bar, h, layer.density, layer.grain_size_mm, strength)
                .min(CUT_LENGTH_CAP)
        });
        out.push(Indices {
            strength_kpa: Some(strength),
            sn38: Some(sn38),
            sk38: Some(sk38),
            ssi,
            critical_cut_length_m: rc,
        });
    }
    out
}
