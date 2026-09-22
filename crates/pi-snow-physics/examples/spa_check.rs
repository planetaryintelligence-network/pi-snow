//! How far Spencer (1971) lands from SPA's worked example (NREL/TP-560-34302, Table A5.1).
use chrono::{TimeZone, Utc};
use pi_snow_physics::solar;
fn main() {
    let t = Utc.with_ymd_and_hms(2003, 10, 17, 19, 30, 30).unwrap();
    let sun = solar::sun_position(t, 39.742_476, -105.1786);
    let inc = solar::cos_incidence(&sun, 30.0, 170.0).acos().to_degrees();
    println!(
        "zenith    ours {:.5}  SPA 50.11162  diff {:+.4} deg",
        sun.zenith_deg,
        sun.zenith_deg - 50.11162
    );
    println!(
        "azimuth   ours {:.5}  SPA 194.34024 diff {:+.4} deg",
        sun.azimuth_deg,
        sun.azimuth_deg - 194.34024
    );
    println!(
        "incidence ours {:.5}  SPA 25.18700  diff {:+.4} deg",
        inc,
        inc - 25.187
    );
}
