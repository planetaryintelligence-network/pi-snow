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
    // The default (A2.4; ships.nonfinite_scan_default of the Phase F record is every-step) parses to every-step.
    assert_eq!(DEFAULT_NONFINITE_SCAN, "every-step");
    assert_eq!(NonfiniteScan::parse(DEFAULT_NONFINITE_SCAN, HOURLY_OFFERED).unwrap(), NonfiniteScan::EveryStep);
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
    let j = NonfiniteScan::EveryStep.run_json(Kernel::K1h, &Kernel::K1h.source_sha256(), true);
    for part in ["\"setting\": \"every-step\"", "\"kernel\": \"k1h\"", "\"kernel_source_sha256\": \"3cfaf22b", "\"equivalent_to_pinned\": true"] {
        assert!(j.contains(part), "{j}");
    }
    let j = NonfiniteScan::Hourly.run_json(Kernel::K2, &Kernel::K2.source_sha256(), false);
    assert!(j.contains("\"equivalent_to_pinned\": false") && j.contains("\"setting\": \"hourly\""), "{j}");
    // At F64 every-step runs the pinned kernel and says so.
    let j = NonfiniteScan::EveryStep.run_json(Kernel::Pinned, &Kernel::Pinned.source_sha256(), false);
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
    // The default (A2.4) is auto.
    assert_eq!(DEFAULT_BATCH, "auto");
    assert_eq!(BatchSetting::parse(DEFAULT_BATCH).unwrap(), BatchSetting::Auto);
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
    assert!(c.candidates.iter().all(|x| x.fits == Some(true)));
    assert_eq!(c.device_bytes_per_column, 182_424);
    assert!(c.reason.contains("smallest median GPU-line ratio"), "{}", c.reason);
    // The reserve to post: 83,232 columns at 182,424 B plus 1.5 GiB, and 26.7 GiB of RSS plus the 4 GiB margin.
    let (rg, rr) = (c.reserve_gpu_gib.unwrap(), c.reserve_ram_gib.unwrap());
    assert!((rg - 15.64).abs() < 0.01, "{rg}");
    assert!((rr - 30.70).abs() < 0.01, "{rr}");
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
    assert!(c.run_json(true).contains("\"source\": \"broker-stale, conservative\""));
}

#[test]
fn no_broker_uses_the_live_readings_alone() {
    let c = auto(Precision::F32, &BrokerState::Absent, &live(30.0, 62.0)).unwrap();
    assert_eq!(c.chosen, 83_232);
    assert_eq!(c.source, "live-only");
    assert_eq!(c.gpu_budget_gib, None);
    assert!(c.run_json(true).contains("\"gpu_budget_gib\": null"));
    // The live readings still bind.
    assert_eq!(auto(Precision::F32, &BrokerState::Absent, &live(9.0, 62.0)).unwrap().chosen, 32_768);
}

#[test]
fn f64_and_mix_return_16384_with_the_f32_only_reason_and_null_sizes() {
    for p in [Precision::F64, Precision::Mix, Precision::F32K] {
        let c = auto(p, &fresh(23.0, 61.0), &live(30.0, 62.0)).unwrap();
        assert_eq!(c.chosen, 16_384, "{p:?}");
        assert_eq!(c.reason, "width validated at F32 only");
        // The device and host sizes are F32 figures: off F32 they are null, and so is every fits flag.
        assert_eq!((c.reserve_gpu_gib, c.reserve_ram_gib), (None, None), "{p:?}");
        assert_eq!(c.note, Some("sizes measured at F32 only"));
        assert!(c.candidates.iter().all(|x| x.device_gib_needed.is_none() && x.ram_gib_needed.is_none() && x.fits.is_none()));
        let j = c.run_json(false);
        assert!(j.contains("\"fits\": null") && !j.contains("\"fits\": true") && !j.contains("\"fits\": false"), "{j}");
        assert!(j.contains("\"device_gib_needed\": null") && j.contains("\"note\": \"sizes measured at F32 only\""), "{j}");
    }
}

#[test]
fn auto_run_json_has_every_a2_3_field() {
    let c = auto(Precision::F32, &fresh(23.0, 61.0), &live(30.9, 62.0)).unwrap();
    let j = c.run_json(true);
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
    let (dev, ram) = reserve_for_fixed(Precision::F32, 32_768, &VALIDATED_WIDTHS);
    assert!((dev.unwrap() - 7.07).abs() < 0.01, "{dev:?}");
    assert!((ram.unwrap() - 14.68).abs() < 0.01, "{ram:?}");
    // A width Phase F did not measure has a device estimate and no host figure.
    let (dev, ram) = reserve_for_fixed(Precision::F32, 20_000, &VALIDATED_WIDTHS);
    assert!(dev.is_some() && ram.is_none());
    // Off F32 both are null: the sizes were measured at F32 only.
    for p in [Precision::F64, Precision::Mix, Precision::F32K] {
        assert_eq!(reserve_for_fixed(p, 16_384, &VALIDATED_WIDTHS), (None, None), "{p:?}");
    }
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

// ---- A2.6: the claim boundary in the JSON and the help ------------------------------------------------------

fn validated_on_fields_present(j: &str) {
    for part in [
        "\"validated_on\": {\"gpu\": \"RTX 5090\"", "\"arch\": \"sm_120\"", "\"driver\": 13040", "\"nvrtc\": \"13.3\"",
        "\"precision\": \"F32\"", "\"season\": \"2025-11-01 to 2026-04-30\"", "\"matches_validated_setup\"",
    ] {
        assert!(j.contains(part), "{part} missing in {j}");
    }
}

#[test]
fn both_run_json_blocks_carry_the_a2_6_boundary() {
    validated_on_fields_present(&NonfiniteScan::EveryStep.run_json(Kernel::K1h, &Kernel::K1h.source_sha256(), true));
    validated_on_fields_present(&auto(Precision::F32, &fresh(23.0, 61.0), &live(30.0, 62.0)).unwrap().run_json(true));
    let j = NonfiniteScan::EveryStep.run_json(Kernel::K1h, "x", false);
    assert!(j.contains("\"matches_validated_setup\": false"), "{j}");
}

#[test]
fn help_states_where_the_claims_hold() {
    for part in ["ai100's RTX 5090 (sm_120)", "driver 13040", "NVRTC 13.3", "F32", "2025-11-01 to 2026-04-30", "fit, not to be faster", "matches_validated_setup"] {
        assert!(GRID_HELP.contains(part), "{part} missing in the help");
    }
}

#[test]
fn help_says_to_run_with_the_printed_width_after_posting_the_reserve() {
    // Gates review of step P: a reserve posted for the width `auto` printed shrinks the budget a later `auto` reads.
    for part in ["--dry-run prints it", "run with `--batch N` as printed", "a posted reserve shrinks the GPU budget `auto` then reads", "narrower width than the dry run did"] {
        assert!(GRID_HELP.contains(part), "{part} missing in the help");
    }
}

#[test]
fn the_validated_ptx_table_covers_the_three_shipped_kernels() {
    let kernels: Vec<Kernel> = VALIDATED_PTX.iter().map(|(k, _)| *k).collect();
    assert_eq!(kernels, vec![Kernel::Pinned, Kernel::K1h, Kernel::K2]);
}

// ---- resuming an output directory (review change 1) ----------------------------------------------------------

fn batch_json(bi: usize, columns: usize, setting: &str, kernel: &str, precision: &str) -> String {
    format!(
        "{{\"batch\": {bi}, \"columns\": {columns}, \"nmax\": 2048, \"days\": 181, \"engine\": {{\"precision\": \"{precision}\", \"arch\": \"sm_120\"}}, \"nonfinite_scan\": {{\"setting\": \"{setting}\", \"kernel\": \"{kernel}\"}}, \"batch_auto\": null}}\n"
    )
}

fn scan_block(setting: &str, kernel: &str) -> String {
    format!("{{\"setting\": \"{setting}\", \"kernel\": \"{kernel}\"}}")
}

const ENGINE_F32: &str = "{\"precision\": \"F32\", \"arch\": \"sm_120\"}";

#[test]
fn resume_refuses_a_directory_written_at_another_width() {
    let dir = temp_dir("resume-width");
    // The first run chose 32,768 and finished batch 0 (columns 0 to 32,767), then stopped.
    std::fs::write(dir.join("batch-0000.json"), batch_json(0, 32_768, "every-step", "k1h", "F32")).unwrap();
    // The restart picks 83,232 over 83,232 pixels: batch 0 would hold all of them. Refused, with the width named.
    let e = pi_snow_gpu::grid::check_resume(&dir, 83_232, 83_232, &scan_block("every-step", "k1h"), ENGINE_F32).unwrap_err().to_string();
    assert!(e.contains("32768 columns") && e.contains("pass --batch 32768"), "{e}");
    // Restarting at the width that wrote it is fine, including a short last batch.
    pi_snow_gpu::grid::check_resume(&dir, 83_232, 32_768, &scan_block("every-step", "k1h"), ENGINE_F32).unwrap();
    std::fs::write(dir.join("batch-0002.json"), batch_json(2, 17_696, "every-step", "k1h", "F32")).unwrap();
    pi_snow_gpu::grid::check_resume(&dir, 83_232, 32_768, &scan_block("every-step", "k1h"), ENGINE_F32).unwrap();
    // A last batch of the wrong length is refused too.
    std::fs::write(dir.join("batch-0002.json"), batch_json(2, 16_384, "every-step", "k1h", "F32")).unwrap();
    assert!(pi_snow_gpu::grid::check_resume(&dir, 83_232, 32_768, &scan_block("every-step", "k1h"), ENGINE_F32).is_err());
}

#[test]
fn resume_refuses_a_different_kernel_setting_or_precision() {
    let dir = temp_dir("resume-setting");
    std::fs::write(dir.join("batch-0000.json"), batch_json(0, 16_384, "every-step", "k1h", "F32")).unwrap();
    let ok = |scan: &str, engine: &str| pi_snow_gpu::grid::check_resume(&dir, 83_232, 16_384, scan, engine);
    ok(&scan_block("every-step", "k1h"), ENGINE_F32).unwrap();
    assert!(ok(&scan_block("every-step", "pinned"), ENGINE_F32).is_err(), "another kernel");
    assert!(ok(&scan_block("hourly", "k2"), ENGINE_F32).is_err(), "another setting");
    assert!(ok(&scan_block("every-step", "k1h"), "{\"precision\": \"F64\"}").is_err(), "another precision");
}

#[test]
fn resume_treats_a_pre_settings_directory_as_the_pinned_kernel() {
    let dir = temp_dir("resume-old");
    // Written before the settings existed: no nonfinite_scan block, the pinned kernel.
    std::fs::write(dir.join("batch-0000.json"), "{\"batch\": 0, \"columns\": 16384, \"engine\": {\"precision\": \"F64\"}}\n").unwrap();
    let f64e = "{\"precision\": \"F64\"}";
    // Every-step at F64 runs the pinned kernel at F64: the same engine, so the resume is allowed.
    pi_snow_gpu::grid::check_resume(&dir, 83_232, 16_384, &scan_block("every-step", "pinned"), f64e).unwrap();
    // Another width, kernel or precision is still refused.
    assert!(pi_snow_gpu::grid::check_resume(&dir, 83_232, 32_768, &scan_block("every-step", "pinned"), f64e).is_err());
    assert!(pi_snow_gpu::grid::check_resume(&dir, 83_232, 16_384, &scan_block("every-step", "k1h"), "{\"precision\": \"F32\"}").is_err());
}

#[test]
fn resume_ignores_an_empty_or_missing_directory() {
    let none = temp_dir("resume-none").join("not-there");
    pi_snow_gpu::grid::check_resume(&none, 100, 50, &scan_block("every-step", "k1h"), ENGINE_F32).unwrap();
    let empty = temp_dir("resume-empty");
    pi_snow_gpu::grid::check_resume(&empty, 100, 50, &scan_block("every-step", "k1h"), ENGINE_F32).unwrap();
}
