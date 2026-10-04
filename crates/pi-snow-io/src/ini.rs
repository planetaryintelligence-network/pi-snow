//! The run configuration, read from the INI the reference engine is invoked
//! with.
//!
//! Two layers: [`Ini`] is the file format (sections, keys, values, all
//! case-insensitive, because the input writes `[Snowpack]` and the expanded copy
//! the reference writes back says `[SNOWPACK]`), and [`RunConfig`] is what this
//! engine understands of it.
//!
//! Defaults for keys the input leaves out are the values in the expanded INI the
//! reference writes beside its output, which is interface documentation of what
//! it ran with. Options this engine does not implement are **refused** with the
//! key's name rather than ignored: silently running the bucket scheme under a
//! configuration that asked for Richards' equation would produce numbers that
//! look like an answer to a question nobody asked.

use std::collections::BTreeMap;
use std::fmt;

/// A parsed INI file: section to key to value, all uppercased.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Ini {
    sections: BTreeMap<String, BTreeMap<String, String>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConfigError {
    Syntax {
        line: usize,
        text: String,
    },
    Missing {
        section: &'static str,
        key: &'static str,
    },
    BadValue {
        key: String,
        value: String,
    },
    Unsupported {
        key: String,
        value: String,
        supported: &'static str,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax { line, text } => write!(f, "INI line {line}: cannot parse {text:?}"),
            Self::Missing { section, key } => write!(f, "INI is missing [{section}] {key}"),
            Self::BadValue { key, value } => {
                write!(f, "INI {key} = {value:?} is not a valid value")
            }
            Self::Unsupported {
                key,
                value,
                supported,
            } => write!(
                f,
                "INI {key} = {value} is not implemented by this engine (supported: {supported})"
            ),
        }
    }
}

impl std::error::Error for ConfigError {}

impl Ini {
    pub fn parse(text: &str) -> Result<Self, ConfigError> {
        let mut ini = Self::default();
        let mut section = String::new();
        for (index, raw) in text.lines().enumerate() {
            let line = raw.split(['#', ';']).next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
                section = name.trim().to_ascii_uppercase();
                continue;
            }
            let (key, value) = line.split_once('=').ok_or_else(|| ConfigError::Syntax {
                line: index + 1,
                text: raw.to_string(),
            })?;
            ini.sections
                .entry(section.clone())
                .or_default()
                .insert(key.trim().to_ascii_uppercase(), value.trim().to_string());
        }
        Ok(ini)
    }

    pub fn get(&self, section: &str, key: &str) -> Option<&str> {
        self.sections
            .get(&section.to_ascii_uppercase())?
            .get(&key.to_ascii_uppercase())
            .map(String::as_str)
    }
}

/// Surface-layer stability correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stability {
    /// `MO_HOLTSLAG`, the "ours" arm.
    MoHoltslag,
    /// `RICHARDSON`, the "uac" arm.
    Richardson,
}

/// New-snow density parameterisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewSnowDensity {
    /// `LEHNING_NEW`, the "ours" arm.
    LehningNew,
    /// `ZWART`, the "uac" arm.
    Zwart,
}

/// What this engine understands of a run's INI.
#[derive(Debug, Clone, PartialEq)]
pub struct RunConfig {
    pub station_id: String,
    pub experiment: String,
    pub meteo_path: String,
    pub output_path: String,
    pub prof_write: bool,
    pub ts_write: bool,
    pub snow_write: bool,
    pub acdd_write: bool,
    /// `CALCULATION_STEP_LENGTH`, converted from minutes.
    pub step_seconds: f64,
    /// `TS_DAYS_BETWEEN`, converted from days. `PROF_DAYS_BETWEEN` must match.
    pub output_interval_seconds: f64,
    pub stability: Stability,
    pub new_snow_density: NewSnowDensity,
    pub roughness_length: f64,
    pub height_of_meteo_values: f64,
    pub height_of_wind_value: f64,
    pub geo_heat: f64,
    /// MeteoIO `PRECSPLITTING` bounds, K: all snow below, all rain above,
    /// linear between.
    pub precip_snow_below_k: f64,
    pub precip_rain_above_k: f64,
    pub height_new_element: f64,
    pub minimum_element_length: f64,
    pub new_snow_grain_size_mm: f64,
    pub hoar_thresh_rh: f64,
    pub hoar_thresh_ta_c: f64,
    pub hoar_thresh_vw: f64,
    pub hoar_density_surface: f64,
    pub hoar_density_buried: f64,
    pub thresh_rh: f64,
    pub thresh_dtemp_air_snow: f64,
    pub thresh_rain_c: f64,
    pub t_crazy_min: f64,
    pub t_crazy_max: f64,
}

/// Where to look up a key: the section the input file uses, plus the default
/// the reference's expanded INI shows for it.
struct Key {
    section: &'static str,
    name: &'static str,
    default: Option<&'static str>,
}

const fn key(section: &'static str, name: &'static str, default: Option<&'static str>) -> Key {
    Key {
        section,
        name,
        default,
    }
}

/// Keys whose value this engine must be told, or must be able to refuse. Each
/// entry lists the only values it implements.
const REFUSALS: &[(Key, &[&str], &str)] = &[
    (
        key("SNOWPACK", "SNP_SOIL", Some("false")),
        &["false"],
        "false (no soil column)",
    ),
    (
        key("SNOWPACK", "CANOPY", Some("false")),
        &["false"],
        "false",
    ),
    (
        key("SNOWPACK", "CHANGE_BC", Some("false")),
        &["false"],
        "false (energy-balance upper boundary)",
    ),
    (
        key("SNOWPACK", "MEAS_TSS", Some("false")),
        &["false"],
        "false",
    ),
    (
        key("SNOWPACK", "ENFORCE_MEASURED_SNOW_HEIGHTS", Some("false")),
        &["false"],
        "false (mass-driven accumulation)",
    ),
    (
        key("SNOWPACK", "SW_MODE", Some("INCOMING")),
        &["INCOMING"],
        "INCOMING",
    ),
    (
        key("SNOWPACKADVANCED", "NUMBER_SLOPES", Some("1")),
        &["1"],
        "1 (one slope per run)",
    ),
    (
        key("SNOWPACKADVANCED", "SNOW_EROSION", Some("false")),
        &["false"],
        "false",
    ),
    (
        key("SNOWPACKADVANCED", "SNOW_REDISTRIBUTION", Some("false")),
        &["false"],
        "false",
    ),
    (
        key(
            "SNOWPACKADVANCED",
            "WATERTRANSPORTMODEL_SNOW",
            Some("BUCKET"),
        ),
        &["BUCKET"],
        "BUCKET",
    ),
    (
        key("SNOWPACKADVANCED", "PREF_FLOW", Some("false")),
        &["false"],
        "false",
    ),
    (
        key("SNOWPACKADVANCED", "ENABLE_VAPOUR_TRANSPORT", Some("false")),
        &["false"],
        "false",
    ),
    (
        key("SNOWPACKADVANCED", "COUPLEDPHASECHANGES", Some("false")),
        &["false"],
        "false",
    ),
    (
        key("SNOWPACKADVANCED", "VARIANT", Some("DEFAULT")),
        &["DEFAULT"],
        "DEFAULT",
    ),
    (
        key("SNOWPACKADVANCED", "PERP_TO_SLOPE", Some("false")),
        &["false"],
        "false (vertical thickness)",
    ),
];

impl RunConfig {
    pub fn from_ini(ini: &Ini) -> Result<Self, ConfigError> {
        for (k, allowed, supported) in REFUSALS {
            let value = lookup(ini, k)?;
            if !allowed.iter().any(|a| a.eq_ignore_ascii_case(&value)) {
                return Err(ConfigError::Unsupported {
                    key: k.name.into(),
                    value,
                    supported,
                });
            }
        }

        let stability = match lookup(
            ini,
            &key("SNOWPACK", "ATMOSPHERIC_STABILITY", Some("MO_HOLTSLAG")),
        )?
        .to_ascii_uppercase()
        .as_str()
        {
            "MO_HOLTSLAG" => Stability::MoHoltslag,
            "RICHARDSON" => Stability::Richardson,
            other => {
                return Err(ConfigError::Unsupported {
                    key: "ATMOSPHERIC_STABILITY".into(),
                    value: other.into(),
                    supported: "MO_HOLTSLAG, RICHARDSON",
                });
            }
        };

        // The uac arm writes HN_DENSITY_MODEL, the reference's expanded INI
        // writes HN_DENSITY_PARAMETERIZATION; both name the same choice.
        let hn = ini
            .get("SNOWPACKADVANCED", "HN_DENSITY_MODEL")
            .or_else(|| ini.get("SNOWPACKADVANCED", "HN_DENSITY_PARAMETERIZATION"))
            .unwrap_or("LEHNING_NEW")
            .to_ascii_uppercase();
        let new_snow_density = match hn.as_str() {
            "LEHNING_NEW" => NewSnowDensity::LehningNew,
            "ZWART" => NewSnowDensity::Zwart,
            other => {
                return Err(ConfigError::Unsupported {
                    key: "HN_DENSITY_PARAMETERIZATION".into(),
                    value: other.into(),
                    supported: "LEHNING_NEW, ZWART",
                });
            }
        };

        let ts = number(ini, &key("OUTPUT", "TS_DAYS_BETWEEN", None))?;
        let prof = number(ini, &key("OUTPUT", "PROF_DAYS_BETWEEN", None))?;
        if (ts - prof).abs() > 1e-12 {
            return Err(ConfigError::Unsupported {
                key: "PROF_DAYS_BETWEEN".into(),
                value: format!("{prof}"),
                supported: "equal to TS_DAYS_BETWEEN",
            });
        }
        let n = |k: Key| number(ini, &k);
        Ok(Self {
            station_id: lookup(ini, &key("INPUT", "STATION1", None))?,
            experiment: lookup(ini, &key("OUTPUT", "EXPERIMENT", Some("NO_EXP")))?,
            meteo_path: lookup(ini, &key("INPUT", "METEOPATH", None))?,
            output_path: lookup(ini, &key("OUTPUT", "METEOPATH", None))?,
            prof_write: boolean(ini, &key("OUTPUT", "PROF_WRITE", None))?,
            ts_write: boolean(ini, &key("OUTPUT", "TS_WRITE", None))?,
            snow_write: boolean(ini, &key("OUTPUT", "SNOW_WRITE", Some("true")))?,
            acdd_write: boolean(ini, &key("OUTPUT", "ACDD_WRITE", Some("false")))?,
            step_seconds: 60.0 * n(key("SNOWPACK", "CALCULATION_STEP_LENGTH", None))?,
            output_interval_seconds: (86_400.0 * ts).round(),
            stability,
            new_snow_density,
            roughness_length: n(key("SNOWPACK", "ROUGHNESS_LENGTH", None))?,
            height_of_meteo_values: n(key("SNOWPACK", "HEIGHT_OF_METEO_VALUES", None))?,
            height_of_wind_value: n(key("SNOWPACK", "HEIGHT_OF_WIND_VALUE", None))?,
            geo_heat: n(key("SNOWPACK", "GEO_HEAT", Some("0.06")))?,
            precip_snow_below_k: n(key("INPUT", "PSUM_PH::PRECSPLITTING::SNOW", None))?,
            precip_rain_above_k: n(key("INPUT", "PSUM_PH::PRECSPLITTING::RAIN", None))?,
            height_new_element: n(key("SNOWPACKADVANCED", "HEIGHT_NEW_ELEM", Some("0.005")))?,
            minimum_element_length: n(key(
                "SNOWPACKADVANCED",
                "MINIMUM_L_ELEMENT",
                Some("0.0025"),
            ))?,
            new_snow_grain_size_mm: n(key("SNOWPACKADVANCED", "NEW_SNOW_GRAIN_SIZE", Some("0.3")))?,
            hoar_thresh_rh: n(key("SNOWPACKADVANCED", "HOAR_THRESH_RH", Some("0.97")))?,
            hoar_thresh_ta_c: n(key("SNOWPACKADVANCED", "HOAR_THRESH_TA", Some("1.2")))?,
            hoar_thresh_vw: n(key("SNOWPACKADVANCED", "HOAR_THRESH_VW", Some("3.5")))?,
            hoar_density_surface: n(key("SNOWPACKADVANCED", "HOAR_DENSITY_SURF", Some("100")))?,
            hoar_density_buried: n(key("SNOWPACKADVANCED", "HOAR_DENSITY_BURIED", Some("125")))?,
            thresh_rh: n(key("SNOWPACKADVANCED", "THRESH_RH", Some("0.5")))?,
            thresh_dtemp_air_snow: n(key(
                "SNOWPACKADVANCED",
                "THRESH_DTEMP_AIR_SNOW",
                Some("3.0"),
            ))?,
            thresh_rain_c: n(key("SNOWPACKADVANCED", "THRESH_RAIN", Some("1.2")))?,
            t_crazy_min: n(key("SNOWPACKADVANCED", "T_CRAZY_MIN", Some("210")))?,
            t_crazy_max: n(key("SNOWPACKADVANCED", "T_CRAZY_MAX", Some("340")))?,
        })
    }
}

fn lookup(ini: &Ini, k: &Key) -> Result<String, ConfigError> {
    ini.get(k.section, k.name)
        .or(k.default)
        .map(str::to_string)
        .ok_or(ConfigError::Missing {
            section: k.section,
            key: k.name,
        })
}

fn number(ini: &Ini, k: &Key) -> Result<f64, ConfigError> {
    let value = lookup(ini, k)?;
    value
        .trim_end_matches('.')
        .parse()
        .map_err(|_| ConfigError::BadValue {
            key: k.name.into(),
            value,
        })
}

fn boolean(ini: &Ini, k: &Key) -> Result<bool, ConfigError> {
    let value = lookup(ini, k)?;
    if value.eq_ignore_ascii_case("true") {
        Ok(true)
    } else if value.eq_ignore_ascii_case("false") {
        Ok(false)
    } else {
        Err(ConfigError::BadValue {
            key: k.name.into(),
            value,
        })
    }
}
