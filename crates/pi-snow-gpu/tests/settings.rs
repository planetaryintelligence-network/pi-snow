//! CPU-only tests of the `grid` settings (gpu-alternatives-v4 Amendment 2, A2.5): the parser, the `auto` selector on
//! fake budgets, the run-JSON blocks and the kernel files each setting selects. No device is touched.

use chrono::{DateTime, TimeZone, Utc};
use pi_snow_gpu::gpu::{Kernel, Precision};
use pi_snow_gpu::settings::*;

const STATUS: &str = "\
# MACHINE STATUS: jason-AI10

state: NORMAL
health: 30
updated: 2026-10-18T05:00:00Z

## Directives for agents
- gpu: budget 23 GiB of 31.8 GiB total, 30.9 GiB free now. Holders: termius-app(1) 0.1 GiB.
- cpu: 48 threads, 6 reserved for interactive use. load1 5.83, 8.3% busy. Cap build jobs at `-j36`.
- ram: 61 GiB available of 123 GiB. Post feedback before allocating more than 10 GiB.
";

fn now() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 10, 18, 5, 4, 0).unwrap()
}

fn fresh(gpu_budget: f64, ram: f64) -> BrokerState {
    BrokerState::Fresh(BrokerReading { updated: now(), gpu_budget_gib: gpu_budget, ram_available_gib: ram })
}

fn live(gpu_gib: f64, ram_gib: f64) -> Live {
    Live { gpu_free_bytes: (gpu_gib * 1_073_741_824.0) as u64, ram_available_kib: (ram_gib * 1_048_576.0) as u64 }
}

fn auto(precision: Precision, broker: &BrokerState, l: &Live) -> anyhow::Result<AutoChoice> {
    select_auto(precision, broker, l, &VALIDATED_WIDTHS)
}

// ---- the non-finite scan setting ---------------------------------------------------------------------------

#[test]
fn scan_default_and_values() {
    assert_eq!(NonfiniteScan::parse("every-step", false).unwrap(), NonfiniteScan::EveryStep);
    assert_eq!(NonfiniteScan::parse("hourly", true).unwrap(), NonfiniteScan::Hourly);
    assert_eq!(NonfiniteScan::EveryStep.label(), "every-step");
    assert_eq!(NonfiniteScan::Hourly.label(), "hourly");
}

#[test]
fn scan_has_no_off_and_refuses_unknown_values() {
    for offered in [false, true] {
        let e = NonfiniteScan::parse("off", offered).unwrap_err().to_string();
        assert!(e.contains("does not exist"), "{e}");
        for bad in ["", "Off", "none", "0", "false", "every_step", "every-hour", "hourly "] {
            assert!(NonfiniteScan::parse(bad, offered).is_err(), "{bad:?} must be refused");
        }
    }
}

#[test]
fn hourly_is_refused_while_the_phase_f_record_does_not_offer_it() {
    let offered = std::hint::black_box(HOURLY_OFFERED);
    assert!(!offered, "the final Phase F record says hourly_offered false (K2 longer than K1h)");
    let e = NonfiniteScan::parse("hourly", HOURLY_OFFERED).unwrap_err().to_string();
    assert!(e.contains("not offered") && e.contains(PHASE_F_RECORD), "{e}");
}

#[test]
fn scan_selects_the_kernel_by_setting_and_precision() {
    assert_eq!(NonfiniteScan::EveryStep.kernel(Precision::F32).unwrap(), Kernel::K1h);
    // K1h was validated at F32 only: every other precision keeps the pinned kernel.
    for p in [Precision::F64, Precision::Mix, Precision::F32K] {
        assert_eq!(NonfiniteScan::EveryStep.kernel(p).unwrap(), Kernel::Pinned, "{p:?}");
    }
    assert_eq!(NonfiniteScan::Hourly.kernel(Precision::F32).unwrap(), Kernel::K2);
    for p in [Precision::F64, Precision::Mix, Precision::F32K] {
        assert!(NonfiniteScan::Hourly.kernel(p).is_err(), "{p:?}");
    }
}

#[test]
fn kernel_files_are_the_ones_phase_f_measured() {
    // sha256 of the three whole kernel files: the pinned guarded kernel (mode 1), K1h (mode 5), K2 (mode 3).
    assert_eq!(Kernel::Pinned.source_sha256(), "fe12cefab34b69813befab1508c0c0a1eb976794217ff7d4232d48347fe3e74b");
    assert_eq!(Kernel::K1h.source_sha256(), "3cfaf22b99e8e7899558de22f4e1d8b9f8184f58f241d72328210348e58863fe");
    assert_eq!(Kernel::K2.source_sha256(), "0e7d79550cf2950f869cfbe34e61efb7898e9c53b96f98ea03b365533525b870");
    assert_eq!(pi_snow_gpu::gpu::KERNEL_SOURCE, Kernel::Pinned.source());
    for k in [Kernel::Pinned, Kernel::K1h, Kernel::K2] {
        assert!(k.source().contains("pi_snow_advance"), "{k:?}");
    }
}

#[test]
fn scan_run_json_carries_the_a2_2_fields() {
    let j = NonfiniteScan::EveryStep.run_json(Kernel::K1h, &Kernel::K1h.source_sha256());
    for part in ["\"setting\": \"every-step\"", "\"kernel\": \"k1h\"", "\"kernel_source_sha256\": \"3cfaf22b", "\"equivalent_to_pinned\": true"] {
        assert!(j.contains(part), "{j}");
    }
    let j = NonfiniteScan::Hourly.run_json(Kernel::K2, &Kernel::K2.source_sha256());
    assert!(j.contains("\"equivalent_to_pinned\": false") && j.contains("\"setting\": \"hourly\""), "{j}");
    // At F64 every-step runs the pinned kernel and says so.
    let j = NonfiniteScan::EveryStep.run_json(Kernel::Pinned, &Kernel::Pinned.source_sha256());
    assert!(j.contains("\"kernel\": \"pinned\"") && j.contains("fe12cefa"), "{j}");
}

#[test]
fn help_labels_hourly_non_equivalent_and_says_there_is_no_off() {
    assert!(GRID_HELP.contains("NON-EQUIVALENT"));
    assert!(GRID_HELP.contains("no `off`"));
    assert!(GRID_HELP.contains("--batch N|auto") && GRID_HELP.contains("--nonfinite-scan"));
}

// ---- the batch setting ---------------------------------------------------------------------------------------

#[test]
fn batch_parses_numbers_and_auto_only() {
    assert_eq!(BatchSetting::parse("auto").unwrap(), BatchSetting::Auto);
    assert_eq!(BatchSetting::parse("16384").unwrap(), BatchSetting::Fixed(16384));
    for bad in ["0", "-1", "Auto", "", "1e4", "16384.0", "wide"] {
        assert!(BatchSetting::parse(bad).is_err(), "{bad:?}");
    }
}

// ---- the broker file -----------------------------------------------------------------------------------------

#[test]
fn status_file_parses_the_directive_figures() {
    let r = parse_status(STATUS).unwrap();
    assert_eq!(r.gpu_budget_gib, 23.0);
    assert_eq!(r.ram_available_gib, 61.0);
    assert_eq!(r.updated, Utc.with_ymd_and_hms(2026, 10, 18, 5, 0, 0).unwrap());
    assert!(parse_status("state: NORMAL\n").is_err());
    assert!(parse_status(&STATUS.replace("updated:", "refreshed:")).is_err());
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    let d = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("settings-{name}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn broker_state_absent_stale_and_fresh() {
    let none = temp_dir("none").join("no-such-machine-dir");
    assert_eq!(broker_state(&none, &none.join("MACHINE_STATUS.md"), now()), BrokerState::Absent);

    let dir = temp_dir("present");
    let file = dir.join("MACHINE_STATUS.md");
    // Directory present, file missing: stale (the house rule: proceed with the conservative defaults).
    assert!(matches!(broker_state(&dir, &file, now()), BrokerState::Stale { .. }));
    std::fs::write(&file, STATUS).unwrap();
    // Four minutes old: fresh.
    assert!(matches!(broker_state(&dir, &file, now()), BrokerState::Fresh(_)));
    // Ten minutes exactly (240 s + 360 s) is not yet stale; 601 s is.
    assert!(matches!(broker_state(&dir, &file, now() + chrono::Duration::seconds(360)), BrokerState::Fresh(_)));
    assert!(matches!(broker_state(&dir, &file, now() + chrono::Duration::seconds(361)), BrokerState::Stale { .. }));
    std::fs::write(&file, "garbage\n").unwrap();
    assert!(matches!(broker_state(&dir, &file, now()), BrokerState::Stale { .. }));
}

#[test]
fn meminfo_available_is_read_in_kib() {
    assert_eq!(parse_meminfo_available_kib("MemTotal: 100 kB\nMemAvailable:   64000000 kB\n"), Some(64_000_000));
    assert_eq!(parse_meminfo_available_kib("MemTotal: 100 kB\n"), None);
}

// ---- the auto selector ---------------------------------------------------------------------------------------

#[test]
fn table_is_best_first_and_ends_with_16384() {
    let w: Vec<usize> = VALIDATED_WIDTHS.iter().map(|v| v.width).collect();
    assert_eq!(w, vec![83_232, 32_768, 16_384]);
    assert!(VALIDATED_WIDTHS.windows(2).all(|p| p[0].median_gpu_ratio < p[1].median_gpu_ratio));
    assert_eq!(VALIDATED_WIDTHS[2].median_gpu_ratio, 1.0);
}

#[test]
fn auto_takes_the_widest_validated_width_when_everything_fits() {
    let c = auto(Precision::F32, &fresh(23.0, 61.0), &live(30.9, 62.0)).unwrap();
    assert_eq!(c.chosen, 83_232);
    assert_eq!(c.source, "broker");
    assert_eq!(c.candidates.len(), 3, "every validated width is listed");
    assert!(c.candidates.iter().all(|x| x.fits));
    assert_eq!(c.device_bytes_per_column, 182_424);
    assert!(c.reason.contains("smallest median GPU-line ratio"), "{}", c.reason);
    // The reserve to post: 83,232 columns at 182,424 B plus 1.5 GiB, and 26.7 GiB of RSS plus the 4 GiB margin.
    assert!((c.reserve_gpu_gib - 15.64).abs() < 0.01, "{}", c.reserve_gpu_gib);
    assert!((c.reserve_ram_gib - 30.70).abs() < 0.01, "{}", c.reserve_ram_gib);
}

#[test]
fn auto_directive_below_the_live_reading_binds_on_the_gpu() {
    // The broker budget (12 GiB) is below what the device has free (30 GiB): the directive binds.
    let c = auto(Precision::F32, &fresh(12.0, 61.0), &live(30.0, 62.0)).unwrap();
    assert_eq!(c.chosen, 32_768);
    assert!(c.reason.contains("83232 does not fit"), "{}", c.reason);
}

#[test]
fn auto_live_gpu_reading_below_the_directive_binds() {
    // The broker says 23 GiB but only 10 GiB is free right now: the live reading binds.
    let c = auto(Precision::F32, &fresh(23.0, 61.0), &live(10.0, 62.0)).unwrap();
    assert_eq!(c.chosen, 32_768);
}

#[test]
fn auto_ram_binds() {
    // 20 GiB of host RAM: 83,232 needs 30.7, 32,768 needs 14.7.
    let c = auto(Precision::F32, &fresh(23.0, 20.0), &live(30.0, 62.0)).unwrap();
    assert_eq!(c.chosen, 32_768);
    // Live RAM below the directive binds too.
    let c = auto(Precision::F32, &fresh(23.0, 61.0), &live(30.0, 20.0)).unwrap();
    assert_eq!(c.chosen, 32_768);
}

#[test]
fn auto_falls_to_16384_last_and_refuses_when_that_does_not_fit() {
    // 6.5 GiB device: 32,768 needs 7.07, so only 16,384 (4.28 GiB device, 9.46 GiB host) fits.
    assert_eq!(auto(Precision::F32, &fresh(6.5, 61.0), &live(30.0, 62.0)).unwrap().chosen, 16_384);
    assert_eq!(auto(Precision::F32, &fresh(23.0, 12.0), &live(30.0, 62.0)).unwrap().chosen, 16_384);
    // Not even 16,384: an error that names the shortfall, never a narrower width.
    let e = auto(Precision::F32, &fresh(3.0, 61.0), &live(30.0, 62.0)).unwrap_err().to_string();
    assert!(e.contains("not even 16384 fits") && e.contains("short"), "{e}");
    let e = auto(Precision::F32, &fresh(23.0, 61.0), &live(30.0, 5.0)).unwrap_err().to_string();
    assert!(e.contains("host RAM") && e.contains("short"), "{e}");
}

#[test]
fn auto_chooses_the_smallest_median_ratio_not_the_widest() {
    let table = [
        ValidatedWidth { width: 60_000, median_gpu_ratio: 0.5, median_wall_ratio: 0.5, peak_rss_kib: Some(10_000_000) },
        ValidatedWidth { width: 40_000, median_gpu_ratio: 0.2, median_wall_ratio: 0.2, peak_rss_kib: Some(8_000_000) },
        ValidatedWidth { width: BASE_WIDTH, median_gpu_ratio: 1.0, median_wall_ratio: 1.0, peak_rss_kib: Some(5_728_136) },
    ];
    // The table is deliberately not sorted by ratio: the rule must not rely on its order.
    let c = select_auto(Precision::F32, &fresh(23.0, 61.0), &live(30.0, 62.0), &table).unwrap();
    assert_eq!(c.chosen, 40_000);
    // When the best does not fit, the next one down the ratio order is taken.
    let c = select_auto(Precision::F32, &fresh(8.0, 61.0), &live(30.0, 62.0), &table).unwrap();
    assert_eq!(c.chosen, BASE_WIDTH, "40,000 needs 8.3 GiB and 60,000 needs 11.7 GiB of device memory");
}

#[test]
fn a_width_with_no_recorded_peak_rss_is_not_offered() {
    let table = [
        ValidatedWidth { width: 50_000, median_gpu_ratio: 0.1, median_wall_ratio: 0.1, peak_rss_kib: None },
        ValidatedWidth { width: BASE_WIDTH, median_gpu_ratio: 1.0, median_wall_ratio: 1.0, peak_rss_kib: Some(5_728_136) },
    ];
    let c = select_auto(Precision::F32, &fresh(23.0, 61.0), &live(30.0, 62.0), &table).unwrap();
    assert_eq!(c.chosen, BASE_WIDTH);
    assert_eq!(c.candidates.len(), 1);
}

#[test]
fn a_stale_broker_returns_16384_with_the_conservative_source() {
    let stale = BrokerState::Stale { reason: "status file updated long ago".into() };
    let c = auto(Precision::F32, &stale, &live(30.0, 62.0)).unwrap();
    assert_eq!(c.chosen, 16_384);
    assert_eq!(c.source, "broker-stale, conservative");
    assert!(c.reason.contains("stale"), "{}", c.reason);
    assert!(c.run_json().contains("\"source\": \"broker-stale, conservative\""));
}

#[test]
fn no_broker_uses_the_live_readings_alone() {
    let c = auto(Precision::F32, &BrokerState::Absent, &live(30.0, 62.0)).unwrap();
    assert_eq!(c.chosen, 83_232);
    assert_eq!(c.source, "live-only");
    assert_eq!(c.gpu_budget_gib, None);
    assert!(c.run_json().contains("\"gpu_budget_gib\": null"));
    // The live readings still bind.
    assert_eq!(auto(Precision::F32, &BrokerState::Absent, &live(9.0, 62.0)).unwrap().chosen, 32_768);
}

#[test]
fn f64_and_mix_return_16384_with_the_f32_only_reason() {
    for p in [Precision::F64, Precision::Mix, Precision::F32K] {
        let c = auto(p, &fresh(23.0, 61.0), &live(30.0, 62.0)).unwrap();
        assert_eq!(c.chosen, 16_384, "{p:?}");
        assert_eq!(c.reason, "width validated at F32 only");
    }
}

#[test]
fn auto_run_json_has_every_a2_3_field() {
    let c = auto(Precision::F32, &fresh(23.0, 61.0), &live(30.9, 62.0)).unwrap();
    let j = c.run_json();
    for key in [
        "gpu_free_gib", "gpu_budget_gib", "ram_available_gib", "source", "device_bytes_per_column", "ram_gib_per_column",
        "candidates", "chosen", "reason",
    ] {
        assert!(j.contains(&format!("\"{key}\"")), "{key} missing in {j}");
    }
    assert!(j.contains("\"chosen\": 83232") && j.contains("\"source\": \"broker\""), "{j}");
    // Every validated width is in the candidate list with its ratios.
    for w in ["83232", "32768", "16384"] {
        assert!(j.contains(&format!("\"width\": {w}")), "{w} missing in {j}");
    }
    assert!(j.contains("\"median_gpu_ratio\": 0.297238"), "{j}");
}

#[test]
fn fixed_width_dry_run_reports_the_device_estimate_and_the_validated_host_figure() {
    let (dev, ram) = reserve_for_fixed(32_768, &VALIDATED_WIDTHS);
    assert!((dev - 7.07).abs() < 0.01, "{dev}");
    assert!((ram.unwrap() - 14.68).abs() < 0.01, "{ram:?}");
    // A width Phase F did not measure has a device estimate and no host figure.
    let (_, ram) = reserve_for_fixed(20_000, &VALIDATED_WIDTHS);
    assert!(ram.is_none());
}

// ---- the PTX each kernel compiles to (NVRTC only, no device) ------------------------------------------------

/// The PTX SHA-256 of the three shipped kernels at F32 equals the one Phase F's U1 record `v4-U1-2c815797d427`
/// (binary 03953621, NVRTC 13.3, sm_120) holds for the same source file. Skipped when NVRTC 13.3 cannot be loaded, so a
/// machine without it still runs the rest of this file.
#[test]
fn shipped_kernels_compile_to_the_ptx_phase_f_measured() {
    use pi_snow_gpu::gpu::{compile_kernel, nvrtc_version_pair};
    let Some((major, minor)) = nvrtc_version_pair() else {
        eprintln!("NVRTC not loadable here; PTX pin skipped");
        return;
    };
    if (major, minor) != (13, 3) {
        eprintln!("NVRTC {major}.{minor}, the pins are for 13.3; skipped");
        return;
    }
    for (kernel, want) in [
        (Kernel::Pinned, "a048838ce0646f84ebc95abca9b0f5564150d241c4be259f79bf419fdf20d1f2"),
        (Kernel::K1h, "26dfad5606a57fb15b5f62eca00c4b2eda9c3dae3520c910ba770a6590ff7116"),
        (Kernel::K2, "32f881191239a830204f4c340f5b3d5ce6723c22434f22c8c9a368d72801c134"),
    ] {
        let (ptx, options) = compile_kernel(Precision::F32, kernel).unwrap();
        assert_eq!(options, ["--fmad=true", "--prec-div=true", "--prec-sqrt=true", "--ftz=false", "-DPI_F32"]);
        assert_eq!(pi_snow_gpu::sha256::sha256_hex(ptx.to_src().as_bytes()), want, "{kernel:?}");
    }
}
