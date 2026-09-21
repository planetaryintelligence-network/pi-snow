//! Stage 0 gate: the file formats, before any physics exists.
//!
//! Fixtures are real extracts from a completed v3b run under
//! `/geo/weather/runs/pi-pod/hrrr-terrain-network/v1/v3b/`. They are committed so
//! the test does not depend on an NFS mount being up.

use pi_snow_io::smet::Smet;
use pi_snow_io::pro::Profiles;
use pi_snow_io::sno::SnoFile;

fn fixture(name: &str) -> String {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");
    std::fs::read_to_string(format!("{path}{name}")).expect("fixture readable")
}

// The TA and RH literals below carry more digits than an f64 holds, because the
// pipeline writes forcing with `.17g` and these are copied character for
// character out of the file. Truncating them to what clippy considers
// representable would make the test assert something the file does not say.
#[allow(clippy::excessive_precision)]
#[test]
fn forcing_smet_parses_with_the_fields_the_engine_needs() {
    let smet = Smet::parse(&fixture("forcing_ULAMB.smet")).unwrap();
    assert_eq!(smet.header.get("station_id").map(String::as_str), Some("ULAMB"));
    assert_eq!(smet.fields, ["timestamp", "TA", "RH", "TSG", "VW", "DW", "ISWR", "ILWR", "PSUM"]);
    assert_eq!(smet.rows.len(), 144, "24 h of 10-minute forcing");
    assert_eq!(smet.nodata, -999.0);

    // First row, verbatim from the file. TA is in kelvin in the *forcing*, unlike
    // the output time series where it is in degrees Celsius.
    let first = &smet.rows[0];
    assert_eq!(first.timestamp.to_string(), "2026-01-30 23:00:00 UTC");
    assert!((smet.value(0, "TA").unwrap() - 274.401_277_255_242_12).abs() < 1e-12);
    assert!((smet.value(0, "RH").unwrap() - 0.502_712_989_779_147_91).abs() < 1e-12);
    assert_eq!(smet.value(0, "PSUM").unwrap(), 0.0);
}

#[test]
fn output_smet_carries_all_68_fields_in_order() {
    let smet = Smet::parse(&fixture("ts_ULAMB.smet")).unwrap();
    assert_eq!(smet.fields.len(), 69, "timestamp plus 68 data fields");
    // The four stability fields the RF1.1 features average over, by name, because
    // the names are not self-describing: Sn is Sn38, Ss is Sk38, S4 is SSI.
    for name in ["Sn", "Ss", "S4", "S5", "Qw", "ISWR_diff", "Qg0", "zSn", "wind_trans24", "HN24", "HS_mod"] {
        assert!(smet.fields.contains(&name.to_string()), "missing output field {name}");
    }
    // S5 is not computed in this configuration and is written as nodata, so the
    // RF1.1 feature S5_mean is null. Asserting it so a future change is visible.
    assert!(smet.value(0, "S5").is_none(), "S5 should read back as nodata");
    // Qw is net shortwave, not rain energy: at the first row it equals ISWR - OSWR.
    let (iswr, oswr, qw) = (
        smet.value(0, "ISWR").unwrap(),
        smet.value(0, "OSWR").unwrap(),
        smet.value(0, "Qw").unwrap(),
    );
    assert!((qw - (iswr - oswr)).abs() < 1e-3, "Qw={qw} should be ISWR-OSWR={}", iswr - oswr);
}

#[test]
fn smet_round_trips_byte_identically() {
    for name in ["forcing_ULAMB.smet", "ts_ULAMB.smet"] {
        let text = fixture(name);
        let smet = Smet::parse(&text).unwrap();
        assert_eq!(smet.to_string(), text, "round trip changed {name}");
    }
}

#[test]
fn empty_sno_is_a_bare_ground_column() {
    let sno = SnoFile::parse(&fixture("init_ULAMB.sno")).unwrap();
    assert_eq!(sno.layers.len(), 0);
    assert_eq!(sno.slope_angle_deg, 0.0);
    assert_eq!(sno.hs_last, 0.0);
    let column = sno.to_column();
    assert!(column.elements.is_empty());
    assert!(column.is_well_formed());
    assert_eq!(column.snow_height(), 0.0);
}

#[test]
fn layered_sno_parses_and_round_trips() {
    let text = fixture("layered_CLN2.sno");
    let sno = SnoFile::parse(&text).unwrap();
    assert_eq!(sno.layers.len(), 12);
    let first = &sno.layers[0];
    assert!((first.thickness - 0.003_535).abs() < 1e-9);
    assert!((first.temperature - 273.15).abs() < 1e-9);
    assert!((first.volume_fraction_ice - 0.514_402).abs() < 1e-9);
    assert!((first.grain_radius_mm - 0.800_151).abs() < 1e-9);
    assert!((first.bond_radius_mm - 0.418_470).abs() < 1e-9);
    assert_eq!(first.marker, 12);
    assert_eq!(sno.to_string(), text, "SNO round trip must be byte identical");

    let column = sno.to_column();
    assert!(column.is_well_formed());
    assert_eq!(column.elements.len(), 12);
    // Grain radius is millimetres in the file and metres in the state.
    assert!((column.elements[0].micro.grain_radius - 0.000_800_151).abs() < 1e-12);
}

#[test]
fn pro_parses_every_code_the_feature_extractor_reads() {
    let profiles = Profiles::parse(&fixture("profile_ULAMB.pro")).unwrap();
    assert_eq!(profiles.timesteps.len(), 4, "three thin timesteps and one deep one");
    let step = &profiles.timesteps[0];
    assert_eq!(step.timestamp.to_string(), "2026-03-15 21:00:00 UTC");
    assert_eq!(step.element_count(), 23);
    // The deep timestep carries 69 elements, which exercises a realistic stack.
    assert_eq!(profiles.timesteps[3].element_count(), 69);

    // These are the codes uac_reference_features._PROFILE_DATA reads and that this
    // configuration actually writes. 0607 and 0540 are absent by configuration.
    for code in ["0501", "0502", "0503", "0504", "0506", "0508", "0509", "0511", "0512",
                 "0513", "0515", "0516", "0517", "0518", "0520", "0523", "0531",
                 "0532", "0533", "0534", "0601", "0604", "0606"] {
        assert!(step.field(code).is_some(), "missing PRO code {code}");
    }
    assert!(step.field("0607").is_none(), "0607 is not written by this configuration");

    // Heights are cumulative and increasing, in centimetres.
    let heights = step.field("0501").unwrap();
    assert_eq!(heights.len(), 23);
    assert!(heights.windows(2).all(|w| w[1] > w[0]), "heights must increase");
    assert!((heights[0] - 0.68).abs() < 1e-9);

    // 0513 is the one per-element code that writes nElems+1 values. Every other
    // per-element code writes exactly nElems. `element_field` drops the extra.
    assert_eq!(step.field("0513").unwrap().len(), 24, "0513 carries nElems+1 raw values");
    assert_eq!(step.element_field("0513").unwrap().len(), 23);
    assert_eq!(step.element_field("0502").unwrap().len(), 23, "0502 is not special-cased");

    // Grain type is the Swiss 3-digit code, so every element value is a 3-digit
    // integer. This is the assertion that caught the nElems+1 quirk.
    for value in step.element_field("0513").unwrap() {
        let code = *value as i32;
        assert!((100..=999).contains(&code), "grain type {code} is not a 3-digit Swiss code");
    }
}

#[test]
fn pro_round_trips_byte_identically() {
    let text = fixture("profile_ULAMB.pro");
    let profiles = Profiles::parse(&text).unwrap();
    assert_eq!(profiles.to_string(), text);
}
