//! Write a synthetic five-slope station directory with the engine's writers.
//!
//! Used by `tests/rust_engine/test_io_contract.py` to check that files written
//! from scratch, not round-tripped, pass through the frozen RF1.1 extractor's
//! whole path: `native_output_paths`, `read_smet`, `read_profiles`,
//! `extract_feature_row`. The values are synthetic and mean nothing physically;
//! only the structure is under test.
//!
//! Usage: `cargo run -p pi-snow-io --example write_synthetic -- <out_dir>`

use chrono::{Duration, TimeZone, Utc};
use pi_snow_io::pro::{self, ElementRecord, ProHeader, ProTimestep};
use pi_snow_io::smet::{OutputRow, write_output_smet};

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_synthetic <out_dir>");
    std::fs::create_dir_all(&out).unwrap();
    let start = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
    let slopes = [
        ("", 0.0, 0.0),
        ("1", 34.0, 0.0),
        ("2", 34.0, 90.0),
        ("3", 34.0, 180.0),
        ("4", 34.0, 270.0),
    ];
    for (suffix, angle, azimuth) in slopes {
        let stem = format!("{out}/SYN{suffix}_synthetic");
        let header = ProHeader {
            station_name: "Synthetic".into(),
            latitude: 40.5,
            longitude: -111.6,
            altitude: 2900.0,
            slope_angle_deg: angle,
            slope_azimuth_deg: azimuth,
        };
        let mut pro_text = pro::write_header(&header);
        let mut rows = Vec::new();
        for hour in 0..(24 * 10) {
            let t = start + Duration::hours(hour);
            let depth = 60.0 + hour as f64 * 0.05;
            let elements = (0..6)
                .map(|i| ElementRecord {
                    top_height_cm: depth * (i + 1) as f64 / 6.0,
                    density: 320.0 - 30.0 * i as f64,
                    temperature_c: -1.0 - i as f64,
                    id: i as u32 + 1,
                    age_days: 10.0 - i as f64,
                    lwc_percent: 0.0,
                    dendricity: 0.0,
                    sphericity: 0.5,
                    coordination_number: 3.0,
                    bond_size_mm: 0.3,
                    grain_size_mm: if i == 2 { 1.5 } else { 0.5 },
                    // A faceted layer (4xx) under rounds and new snow, so the
                    // Monti weak-layer rules have something to find.
                    grain_type: [330, 330, 440, 330, 220, 110][i],
                    ice_percent: 30.0,
                    air_percent: 70.0,
                    stress_kpa: -1.0,
                    viscosity_gpa_s: 10.0,
                    temperature_gradient: -10.0,
                    conductivity: 0.2,
                    absorbed_shortwave: 0.0,
                    deformation_rate: -0.1,
                    sn38: 1.5 + i as f64 * 0.3,
                    sk38: 1.0 + i as f64 * 0.2,
                    ssi: 1.2,
                    critical_cut_length_m: 0.4 + i as f64 * 0.05,
                    hand_hardness: 2.0,
                    ..ElementRecord::default()
                })
                .collect();
            pro_text.push_str(&pro::write_timestep(&ProTimestep {
                timestamp: t,
                elements,
                surface_hoar: None,
            }));

            let mut row = OutputRow::nodata(t);
            for (name, v) in [
                ("HN24", 5.0),
                ("HN72_24", 12.0),
                ("HS_mod", depth),
                ("ILWR", 230.0),
                ("ISWR_diff", 40.0),
                ("LWR_net", -60.0),
                ("MS_Snow", 0.4),
                ("MS_Water", 0.0),
                ("OLWR", 290.0),
                ("OSWR", 80.0),
                ("Qg0", 0.06),
                ("Qs", -20.0),
                ("Qw", 15.0),
                ("RH", 70.0),
                ("S4", 1.2),
                ("Sn", 1.5),
                ("Ss", 1.0),
                ("TA", -6.0),
                ("VW_drift", 3.0),
                ("VW", 3.0),
                ("pAlbedo", 0.85),
                ("wind_trans24", 2.0),
                ("zSn", 20.0),
            ] {
                row.set(name, v);
            }
            rows.push(row);
        }
        std::fs::write(format!("{stem}.pro"), pro_text).unwrap();
        std::fs::write(
            format!("{stem}.smet"),
            write_output_smet(
                &format!("SYN{suffix}"),
                "Synthetic",
                40.5,
                -111.6,
                2900.0,
                angle,
                azimuth,
                &rows,
            ),
        )
        .unwrap();
    }
}
