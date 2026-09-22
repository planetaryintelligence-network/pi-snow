//! Solar geometry and shortwave radiation on a slope.
//!
//! The forcing supplies only global shortwave on a horizontal surface. Four of
//! the five simulated products are inclined, and the output reports shortwave
//! on the slope (`ISWR`), on the horizontal (`ISWR_h`), direct (`ISWR_dir`) and
//! diffuse (`ISWR_diff`), the last of which feeds an RF1.1 feature directly. So
//! the engine needs the sun's position, a split of global into direct and
//! diffuse, and a projection of each onto the slope.
//!
//! Sources, and how far each was read:
//!
//! * Zenith and azimuth from declination and hour angle, and the incidence
//!   angle on a tilted surface: Reda and Andreas, *Solar Position Algorithm for
//!   Solar Radiation Applications*, NREL/TP-560-34302, eqs. 41 and 45-47, read
//!   in full, with eq. 47 attributed there to Iqbal (1983).
//! * Declination and equation of time: Spencer (1971), *Search* 2(5), 172,
//!   via its implementation and citation in the pvlib-python library, which is
//!   a secondary source. The full SPA (VSOP87 series, nutation, parallax) is
//!   deliberately not used; it buys 0.0003 degrees of accuracy that shortwave
//!   forcing at hourly resolution cannot use.
//! * Diffuse fraction: Erbs, Klein and Duffie (1982), *Solar Energy* 28,
//!   293-302, Eq. 1, via pvlib, secondary.
//! * Isotropic sky diffuse and ground-reflected view factors: standard, via
//!   pvlib's `isotropic` and `get_ground_diffuse`, secondary.
//!
//! Azimuths are compass bearings: degrees clockwise from north, so a
//! south-facing slope has aspect 180. That is the convention of the SNO
//! `SlopeAzi` header. SPA's own eq. 47 uses a south-based frame, and the
//! conversion is where a north/south swap hides; the test against SPA's worked
//! example is what pins it.

use chrono::{DateTime, Datelike, Timelike, Utc};

/// Position of the sun as seen from a point on the ground.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunPosition {
    /// Angle from the zenith, degrees. Above 90 the sun is below the horizon.
    pub zenith_deg: f64,
    /// Compass bearing, degrees clockwise from north.
    pub azimuth_deg: f64,
}

/// Declination (radians) and equation of time (minutes) from Spencer (1971).
///
/// The day angle is `2 pi (n - 1 + h / 24) / 365`, with `n` the 1-based day of
/// the year and `h` the UTC hour. pvlib, where the series was read, evaluates it
/// at the integer day, i.e. at midnight. Evaluating it at the actual time
/// changes no coefficient, only the argument, and on SPA's worked example it
/// cuts the zenith error from -0.33 to -0.026 degrees: in mid-October the
/// declination moves about 0.4 degrees a day, so a 19:30 UTC sun evaluated at
/// 00:00 is 0.3 degrees off before any other error enters.
///
/// pvlib's docstring records a typographical history in the equation-of-time
/// constant (0.000075 printed in the original, 0.0000075 correct); the
/// corrected value is used.
fn spencer(fractional_day_of_year: f64) -> (f64, f64) {
    let g = 2.0 * std::f64::consts::PI * (fractional_day_of_year - 1.0) / 365.0;
    let declination = 0.006918 - 0.399912 * g.cos() + 0.070257 * g.sin()
        - 0.006758 * (2.0 * g).cos()
        + 0.000907 * (2.0 * g).sin()
        - 0.002697 * (3.0 * g).cos()
        + 0.00148 * (3.0 * g).sin();
    let equation_of_time = (1440.0 / (2.0 * std::f64::consts::PI))
        * (0.0000075 + 0.001868 * g.cos()
            - 0.032077 * g.sin()
            - 0.014615 * (2.0 * g).cos()
            - 0.040849 * (2.0 * g).sin());
    (declination, equation_of_time)
}

/// Sun position at `time` for a site at `latitude` and `longitude` (degrees,
/// north and east positive).
///
/// Geocentric: no refraction and no topocentric parallax. Both are below 0.03
/// degrees for a sun more than about 10 degrees above the horizon, which is
/// where nearly all the shortwave energy arrives.
pub fn sun_position(time: DateTime<Utc>, latitude: f64, longitude: f64) -> SunPosition {
    let utc_hours =
        time.hour() as f64 + time.minute() as f64 / 60.0 + time.second() as f64 / 3600.0;
    let (declination, equation_of_time) = spencer(time.ordinal() as f64 + utc_hours / 24.0);
    let solar_hours = utc_hours + longitude / 15.0 + equation_of_time / 60.0;
    let hour_angle = (15.0 * (solar_hours - 12.0)).to_radians();
    let phi = latitude.to_radians();

    // SPA eq. 41 with geocentric declination and hour angle.
    let cos_zenith =
        phi.sin() * declination.sin() + phi.cos() * declination.cos() * hour_angle.cos();
    let zenith = cos_zenith.clamp(-1.0, 1.0).acos();

    // SPA eqs. 45-46: azimuth west of south, then shifted to east of north.
    let west_of_south = hour_angle
        .sin()
        .atan2(hour_angle.cos() * phi.sin() - declination.tan() * phi.cos());
    let azimuth = (west_of_south.to_degrees() + 180.0).rem_euclid(360.0);

    SunPosition {
        zenith_deg: zenith.to_degrees(),
        azimuth_deg: azimuth,
    }
}

/// Cosine of the angle between the sun and the normal of a surface tilted by
/// `slope_deg` with compass aspect `aspect_deg`.
///
/// SPA eq. 47, `cos I = cos(Z) cos(b) + sin(b) sin(Z) cos(A_sun - A_surface)`,
/// written with both azimuths as compass bearings. Negative means the sun is
/// behind the surface.
pub fn cos_incidence(sun: &SunPosition, slope_deg: f64, aspect_deg: f64) -> f64 {
    let (z, b) = (sun.zenith_deg.to_radians(), slope_deg.to_radians());
    let relative = (sun.azimuth_deg - aspect_deg).to_radians();
    z.cos() * b.cos() + b.sin() * z.sin() * relative.cos()
}

/// Diffuse fraction of global horizontal shortwave from the hourly clearness
/// index `k_t` (global over extraterrestrial horizontal).
///
/// Erbs, Klein and Duffie (1982) Eq. 1, secondary via pvlib:
/// `1 - 0.09 k` for `k <= 0.22`; a quartic for `0.22 < k <= 0.8`; `0.165` above.
pub fn erbs_diffuse_fraction(clearness_index: f64) -> f64 {
    let k = clearness_index.max(0.0);
    if k <= 0.22 {
        1.0 - 0.09 * k
    } else if k <= 0.8 {
        0.9511 - 0.1604 * k + 4.388 * k.powi(2) - 16.638 * k.powi(3) + 12.336 * k.powi(4)
    } else {
        0.165
    }
}

/// Shortwave components on an inclined surface, W m^-2.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SlopeShortwave {
    /// Direct beam on the horizontal, `GHI - DHI`.
    pub direct_horizontal: f64,
    /// Direct beam on the slope.
    pub direct_on_slope: f64,
    /// Sky diffuse on the slope, isotropic.
    pub sky_diffuse_on_slope: f64,
    /// Reflected from the surrounding ground onto the slope.
    pub ground_reflected: f64,
    /// Sum of the three slope components.
    pub global_on_slope: f64,
}

/// Floor on `cos(zenith)` when converting horizontal beam to beam on the slope,
/// equal to `cos(85 degrees)`. Without it the ratio diverges as the sun reaches
/// the horizon and a few W m^-2 of low-sun forcing become hundreds on a slope
/// facing the sunrise. The same floor appears as `b = max(cos 85, cos Z)` in the
/// Perez et al. (1990) model as pvlib implements it, which is where it was read.
const MIN_COS_ZENITH: f64 = 0.087_155_742_747_658_17;

/// Project global (`ghi`) and diffuse (`dhi`) horizontal shortwave onto a slope.
///
/// `ground_albedo` should be the albedo of the snow surface around the slope,
/// not a generic ground constant: under a snow cover a fixed 0.25 would badly
/// understate the reflected term. That coupling is our choice; no source read
/// discusses it.
pub fn on_slope(
    ghi: f64,
    dhi: f64,
    sun: &SunPosition,
    slope_deg: f64,
    aspect_deg: f64,
    ground_albedo: f64,
) -> SlopeShortwave {
    let dhi = dhi.clamp(0.0, ghi.max(0.0));
    let direct_horizontal = (ghi - dhi).max(0.0);
    let cos_zenith = sun.zenith_deg.to_radians().cos();
    let direct_on_slope = if cos_zenith > 0.0 && direct_horizontal > 0.0 {
        direct_horizontal * cos_incidence(sun, slope_deg, aspect_deg).max(0.0)
            / cos_zenith.max(MIN_COS_ZENITH)
    } else {
        0.0
    };
    let c = slope_deg.to_radians().cos();
    let sky_diffuse_on_slope = dhi * (1.0 + c) / 2.0;
    let ground_reflected = ghi.max(0.0) * ground_albedo * (1.0 - c) / 2.0;
    SlopeShortwave {
        direct_horizontal,
        direct_on_slope,
        sky_diffuse_on_slope,
        ground_reflected,
        global_on_slope: direct_on_slope + sky_diffuse_on_slope + ground_reflected,
    }
}
