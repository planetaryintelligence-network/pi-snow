//! Stage 1c gate: solar geometry and shortwave on a slope.
//!
//! The oracle is the worked example in Reda and Andreas, *Solar Position
//! Algorithm for Solar Radiation Applications*, NREL/TP-560-34302, Appendix
//! A.5, Table A5.1, read in full. Its algorithm is accurate to 0.0003 degrees;
//! ours is Spencer's (1971) three-harmonic series, which is far simpler, so the
//! position test carries a stated tolerance, while the pure-geometry tests
//! (incidence angle from a given sun position) are exact.

use approx::assert_relative_eq;
use chrono::{TimeZone, Utc};
use pi_snow_physics::solar::{self, SunPosition};

/// Table A5.1 inputs: 17 October 2003, 12:30:30 local standard time at
/// UTC-7, so 19:30:30 UTC; 39.742476 N, 105.1786 W.
const LATITUDE: f64 = 39.742_476;
const LONGITUDE: f64 = -105.1786;
/// Table A5.1 outputs: topocentric zenith, azimuth east of north, and the
/// incidence angle on a 30-degree slope with surface azimuth rotation -10.
const SPA_ZENITH: f64 = 50.111_62;
const SPA_AZIMUTH: f64 = 194.340_24;
const SPA_INCIDENCE: f64 = 25.187_00;

#[test]
fn incidence_angle_reproduces_the_spa_worked_example_exactly() {
    // Pure geometry from the example's own sun position, so this is exact to
    // the printed precision. It also settles the surface-azimuth convention.
    //
    // The example's slope has "surface azimuth rotation" -10 degrees in SPA's
    // south-based frame. The reader who fetched the report transcribed that
    // frame as positive toward the east; the printed answer says otherwise.
    // Compass 170 (10 degrees EAST of south) gives 25.187; compass 190 gives
    // 20.3. A sign error here swaps every north slope with a south one, so the
    // worked example is the arbiter, not anybody's paraphrase of the text.
    let sun = SunPosition {
        zenith_deg: SPA_ZENITH,
        azimuth_deg: SPA_AZIMUTH,
    };
    let incidence = solar::cos_incidence(&sun, 30.0, 170.0).acos().to_degrees();
    assert_relative_eq!(incidence, SPA_INCIDENCE, epsilon = 2e-3);
    let wrong = solar::cos_incidence(&sun, 30.0, 190.0).acos().to_degrees();
    assert!(
        (wrong - SPA_INCIDENCE).abs() > 4.0,
        "compass 190 must NOT match"
    );
}

#[test]
fn spencer_sun_position_is_close_to_spa_on_the_worked_example() {
    // Spencer's truncated series is not SPA. Measured on this example: zenith
    // -0.026, azimuth +0.115, incidence within 0.1 degrees. Evaluating the series
    // at the integer day instead of the actual time made the zenith error
    // -0.33, which is why the day angle carries the hour; the tolerance below
    // would catch a regression to that.
    let t = Utc.with_ymd_and_hms(2003, 10, 17, 19, 30, 30).unwrap();
    let sun = solar::sun_position(t, LATITUDE, LONGITUDE);
    assert!(
        (sun.zenith_deg - SPA_ZENITH).abs() < 0.1,
        "zenith {}",
        sun.zenith_deg
    );
    assert!(
        (sun.azimuth_deg - SPA_AZIMUTH).abs() < 0.2,
        "azimuth {}",
        sun.azimuth_deg
    );
    let incidence = solar::cos_incidence(&sun, 30.0, 170.0).acos().to_degrees();
    assert!(
        (incidence - SPA_INCIDENCE).abs() < 0.15,
        "incidence {incidence}"
    );
}

#[test]
fn a_flat_surface_sees_exactly_the_horizontal_irradiance() {
    let t = Utc.with_ymd_and_hms(2026, 1, 20, 19, 0, 0).unwrap();
    let sun = solar::sun_position(t, 40.6, -111.6);
    for aspect in [0.0, 90.0, 180.0, 270.0] {
        assert_relative_eq!(
            solar::cos_incidence(&sun, 0.0, aspect),
            sun.zenith_deg.to_radians().cos(),
            epsilon = 1e-12
        );
    }
    let sw = solar::on_slope(500.0, 120.0, &sun, 0.0, 0.0, 0.8);
    assert_relative_eq!(sw.global_on_slope, 500.0, epsilon = 1e-9);
}

#[test]
fn a_winter_noon_south_slope_gets_more_beam_than_north() {
    // Utah, late January, near local solar noon: the sun is low in the south,
    // so a south-facing 34-degree slope faces it and a north-facing one turns
    // away. Direct beam must scale with the cosine ratio in both directions.
    let t = Utc.with_ymd_and_hms(2026, 1, 20, 19, 20, 0).unwrap();
    let sun = solar::sun_position(t, 40.6, -111.6);
    assert!(
        sun.azimuth_deg > 170.0 && sun.azimuth_deg < 190.0,
        "noon azimuth {}",
        sun.azimuth_deg
    );
    let south = solar::on_slope(600.0, 100.0, &sun, 34.0, 180.0, 0.8);
    let north = solar::on_slope(600.0, 100.0, &sun, 34.0, 0.0, 0.8);
    let flat = solar::on_slope(600.0, 100.0, &sun, 0.0, 0.0, 0.8);
    assert!(south.direct_on_slope > flat.direct_on_slope);
    assert!(north.direct_on_slope < flat.direct_on_slope);
    let beam_normal = (600.0 - 100.0) / sun.zenith_deg.to_radians().cos();
    assert_relative_eq!(
        south.direct_on_slope,
        beam_normal * solar::cos_incidence(&sun, 34.0, 180.0),
        max_relative = 1e-12
    );
}

#[test]
fn no_beam_on_a_slope_facing_away_from_the_sun_or_at_night() {
    let t = Utc.with_ymd_and_hms(2026, 1, 20, 19, 20, 0).unwrap();
    let sun = solar::sun_position(t, 40.6, -111.6);
    // A 70-degree north face in January is self-shaded at noon.
    let steep_north = solar::on_slope(600.0, 100.0, &sun, 70.0, 0.0, 0.8);
    assert_eq!(steep_north.direct_on_slope, 0.0);
    // Local midnight: no sun, nothing but whatever diffuse the forcing claims.
    let night = Utc.with_ymd_and_hms(2026, 1, 20, 7, 0, 0).unwrap();
    let dark = solar::sun_position(night, 40.6, -111.6);
    assert!(dark.zenith_deg > 90.0);
    let sw = solar::on_slope(0.0, 0.0, &dark, 34.0, 180.0, 0.8);
    assert_eq!(sw.global_on_slope, 0.0);
}

#[test]
fn diffuse_and_ground_reflected_follow_the_view_factors() {
    // Isotropic sky diffuse DHI (1 + cos b) / 2 and ground-reflected
    // GHI rho (1 - cos b) / 2: they sum to DHI on the flat and redistribute
    // between sky and ground as the slope steepens.
    let t = Utc.with_ymd_and_hms(2026, 1, 20, 19, 20, 0).unwrap();
    let sun = solar::sun_position(t, 40.6, -111.6);
    let (ghi, dhi, rho, beta) = (600.0, 150.0, 0.8, 34.0_f64);
    let sw = solar::on_slope(ghi, dhi, &sun, beta, 180.0, rho);
    let c = beta.to_radians().cos();
    assert_relative_eq!(
        sw.sky_diffuse_on_slope,
        dhi * (1.0 + c) / 2.0,
        max_relative = 1e-12
    );
    assert_relative_eq!(
        sw.ground_reflected,
        ghi * rho * (1.0 - c) / 2.0,
        max_relative = 1e-12
    );
}

#[test]
fn erbs_diffuse_fraction_matches_its_piecewise_form() {
    // Erbs, Klein and Duffie (1982) Eq. 1, as implemented and cited in pvlib.
    assert_relative_eq!(solar::erbs_diffuse_fraction(0.1), 1.0 - 0.09 * 0.1);
    assert_relative_eq!(solar::erbs_diffuse_fraction(0.9), 0.165);
    let k: f64 = 0.5;
    let expected =
        0.9511 - 0.1604 * k + 4.388 * k.powi(2) - 16.638 * k.powi(3) + 12.336 * k.powi(4);
    assert_relative_eq!(
        solar::erbs_diffuse_fraction(k),
        expected,
        max_relative = 1e-12
    );
    // Near-continuity at the breakpoints, to the published rounding.
    assert!(
        (solar::erbs_diffuse_fraction(0.22) - solar::erbs_diffuse_fraction(0.22 + 1e-9)).abs()
            < 5e-3
    );
    assert!(
        (solar::erbs_diffuse_fraction(0.8) - solar::erbs_diffuse_fraction(0.8 + 1e-9)).abs() < 5e-3
    );
}
