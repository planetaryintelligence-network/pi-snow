//! The from-scratch writing gate.
//!
//! Round-trip (smet_roundtrip.rs) proves we reproduce a file we read. This
//! proves we can emit a correct file we constructed, which is a different
//! claim: every per-element row must have the right arity, `0513` must carry
//! its extra surface-hoar value, bare ground must be the single `0501,1,0`
//! line, and anything the engine does not compute must be written as -999
//! rather than as a plausible-looking zero.

use pi_snow_io::pro::{ElementRecord, ProHeader, ProTimestep, Profiles, SurfaceHoar};
use pi_snow_io::smet::{OUTPUT_FIELDS, OutputRow, Smet, write_output_smet};

fn header() -> ProHeader {
    ProHeader {
        station_name: "Synthetic".into(),
        latitude: 40.5,
        longitude: -111.6,
        altitude: 2900.0,
        slope_angle_deg: 34.0,
        slope_azimuth_deg: 180.0,
    }
}

fn element(top_cm: f64, grain_type: u16) -> ElementRecord {
    ElementRecord {
        top_height_cm: top_cm,
        density: 180.0,
        temperature_c: -4.5,
        id: 1,
        age_days: 2.5,
        lwc_percent: 0.0,
        dendricity: 0.3,
        sphericity: 0.6,
        coordination_number: 2.5,
        bond_size_mm: 0.2,
        grain_size_mm: 0.6,
        grain_type,
        ice_percent: 19.6,
        air_percent: 80.4,
        stress_kpa: -0.3,
        viscosity_gpa_s: 2.1,
        temperature_gradient: -12.0,
        conductivity: 0.09,
        absorbed_shortwave: 0.0,
        deformation_rate: -0.4,
        ..ElementRecord::default()
    }
}

#[test]
fn a_written_profile_parses_with_consistent_arity() {
    let mut text = pi_snow_io::pro::write_header(&header());
    let step = ProTimestep {
        timestamp: "2026-01-15T12:00:00Z".parse().unwrap(),
        elements: vec![element(10.0, 330), element(25.0, 220), element(31.5, 110)],
        surface_hoar: None,
    };
    text.push_str(&pi_snow_io::pro::write_timestep(&step));

    let parsed = Profiles::parse(&text).unwrap();
    let t = &parsed.timesteps[0];
    assert_eq!(t.element_count(), 3);
    for code in [
        "0502", "0503", "0504", "0506", "0508", "0509", "0511", "0512", "0515", "0516", "0517",
        "0518", "0520", "0523", "0531", "0532", "0533", "0534", "0601", "0604", "0606",
    ] {
        assert_eq!(t.field(code).map(<[f64]>::len), Some(3), "code {code}");
    }
    // 0513 carries nElems + 1 values; the last is the surface-hoar grain type,
    // 0 when there is none.
    assert_eq!(t.field("0513").unwrap(), &[330.0, 220.0, 110.0, 0.0]);
    assert_eq!(t.element_field("0513").unwrap(), &[330.0, 220.0, 110.0]);
    assert_eq!(t.field("0501").unwrap(), &[10.0, 25.0, 31.5]);
}

#[test]
fn quantities_the_engine_does_not_compute_are_written_as_nodata() {
    // A missing stability index must be -999, which the extractor reads as NaN.
    // Writing 0.0 instead would be a silent, very unstable-looking layer.
    let step = ProTimestep {
        timestamp: "2026-01-15T12:00:00Z".parse().unwrap(),
        elements: vec![element(12.0, 440)],
        surface_hoar: None,
    };
    let text = pi_snow_io::pro::write_header(&header()) + &pi_snow_io::pro::write_timestep(&step);
    let t = &Profiles::parse(&text).unwrap().timesteps[0];
    for code in ["0532", "0533", "0604", "0606", "0601"] {
        assert_eq!(t.field(code).unwrap(), &[-999.0], "code {code}");
    }
}

#[test]
fn surface_hoar_is_carried_in_0513_and_0514() {
    // In the reference output the extra 0513 value and 0514's grain type agree:
    // 0 and -999 with no surface hoar, 660 and 660 with it.
    let step = ProTimestep {
        timestamp: "2026-01-15T12:00:00Z".parse().unwrap(),
        elements: vec![element(8.0, 330)],
        surface_hoar: Some(SurfaceHoar {
            grain_size_mm: 0.7,
            density: 100.0,
        }),
    };
    let text = pi_snow_io::pro::write_header(&header()) + &pi_snow_io::pro::write_timestep(&step);
    let t = &Profiles::parse(&text).unwrap().timesteps[0];
    assert_eq!(t.field("0513").unwrap(), &[330.0, 660.0]);
    assert_eq!(t.field("0514").unwrap(), &[660.0, 0.7, 100.0]);
}

#[test]
fn bare_ground_is_the_single_zero_height_line() {
    let step = ProTimestep {
        timestamp: "2025-11-01T00:00:00Z".parse().unwrap(),
        elements: vec![],
        surface_hoar: None,
    };
    let body = pi_snow_io::pro::write_timestep(&step);
    assert_eq!(body, "0500,01.11.2025 00:00:00\n0501,1,0\n");
    let text = pi_snow_io::pro::write_header(&header()) + &body;
    assert!(Profiles::parse(&text).unwrap().timesteps[0].is_bare_ground());
}

#[test]
fn the_header_carries_what_the_scorers_read() {
    // The sibling validation harness's elevation filter reads the Altitude line,
    // and the chain's own validator checks SlopeAngle and SlopeAzi.
    let text = pi_snow_io::pro::write_header(&header());
    let parsed = Profiles::parse(&(text + "0500,01.11.2025 00:00:00\n0501,1,0\n")).unwrap();
    assert_eq!(
        parsed.station.get("Altitude").map(String::as_str),
        Some("2900")
    );
    assert_eq!(
        parsed.station.get("SlopeAngle").map(String::as_str),
        Some("34.00")
    );
    assert_eq!(
        parsed.station.get("SlopeAzi").map(String::as_str),
        Some("180.00")
    );
    let codes: Vec<&str> = parsed
        .declared_codes
        .iter()
        .map(|(c, _)| c.as_str())
        .collect();
    assert!(codes.contains(&"0513") && codes.contains(&"0606"));
}

#[test]
fn output_smet_has_the_reference_field_order_and_display_units() {
    let mut row = OutputRow::nodata("2026-01-15T12:00:00Z".parse().unwrap());
    row.set("TA", -6.25); // degrees Celsius in this file, not kelvin
    row.set("HS_mod", 31.5); // centimetres
    row.set("Qw", 88.0);
    let text = write_output_smet(
        "SYN",
        "Synthetic",
        40.5,
        -111.6,
        2900.0,
        34.0,
        180.0,
        &[row],
    );
    let smet = Smet::parse(&text).unwrap();
    assert_eq!(smet.fields.len(), 69);
    assert_eq!(&smet.fields[1..], OUTPUT_FIELDS);
    assert_eq!(smet.value(0, "TA"), Some(-6.25));
    assert_eq!(smet.value(0, "HS_mod"), Some(31.5));
    assert_eq!(smet.value(0, "S5"), None, "unset fields are nodata");
    // The header's unit transforms must match the reference file, because they
    // are what tell any unit-aware reader that TA is Celsius and HS is cm.
    let reference = Smet::parse(include_str!("fixtures/ts_ULAMB.smet")).unwrap();
    for key in ["units_offset", "units_multiplier", "fields"] {
        assert_eq!(
            smet.header
                .get(key)
                .map(|s| s.split_whitespace().collect::<Vec<_>>()),
            reference
                .header
                .get(key)
                .map(|s| s.split_whitespace().collect::<Vec<_>>()),
            "header {key}"
        );
    }
}
