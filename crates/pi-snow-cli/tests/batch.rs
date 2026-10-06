//! `pi-snow batch` writes what one `pi-snow -c ... -e ...` process per column
//! writes, and one failing unit does not stop the others.
//!
//! The columns are synthetic: a bare-ground start (the fixtures are ULAMB1's INI
//! and SNO) and six days of forcing that differ from unit to unit, with two
//! snowfalls, so that a column that picked up another's state would show.

use std::path::{Path, PathBuf};
use std::process::Command;

use pi_snow_cli::batch::{Job, run_batch};
use pi_snow_cli::unit::parse_end;
use pi_snow_core::PhysicsOptions;

const END: &str = "2026-01-07T00:00";
const PRO_AND_SMET: [&str; 2] = ["pro", "smet"];

fn scratch(name: &str) -> PathBuf {
    // Cargo's per-target scratch dir (rust/target/tmp), never /tmp.
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("pi-snow-batch-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Six days of 10-minute forcing; `k` shifts temperature and snowfall timing.
fn forcing(k: usize) -> String {
    let mut text = String::from(
        "SMET 1.1 ASCII\n[HEADER]\nstation_id = ULAMB1\nstation_name = Synthetic\n\
         latitude = 39.59478\nlongitude = -111.22507\naltitude = 2754\nnodata = -999\ntz = 0\n\
         fields = timestamp TA RH TSG VW DW ISWR ILWR PSUM\n[DATA]\n",
    );
    for step in 0..(6 * 144) {
        let minutes = step * 10;
        let hour = (minutes / 60) % 24;
        let day = minutes / 1440;
        let phase = (minutes % 1440) as f64 / 1440.0 * std::f64::consts::TAU;
        let ta = 267.0 + k as f64 * 0.7 + 4.0 * (phase - 2.0).sin();
        let iswr = (500.0 * (phase - 1.6).sin()).max(0.0);
        let snowing = (day == 0 || day == 3) && (8 + k % 3..14 + k % 3).contains(&hour);
        let psum = if snowing { 0.4 + 0.1 * k as f64 } else { 0.0 };
        text += &format!(
            "2026-01-{:02}T{:02}:{:02} {ta:.4} 0.8 273.15 {:.3} 200 {iswr:.3} 240 {psum}\n",
            1 + day,
            hour,
            minutes % 60,
            2.0 + k as f64
        );
    }
    text
}

fn stage(root: &Path, name: &str, k: usize) -> PathBuf {
    let unit = root.join(name);
    std::fs::create_dir_all(unit.join("input")).unwrap();
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    std::fs::copy(fixtures.join("snowpack.ini"), unit.join("snowpack.ini")).unwrap();
    std::fs::copy(fixtures.join("ULAMB1.sno"), unit.join("input/ULAMB1.sno")).unwrap();
    std::fs::write(unit.join("input/ULAMB1.smet"), forcing(k)).unwrap();
    unit
}

fn set_flag(unit: &Path, key: &str, value: &str) {
    let path = unit.join("snowpack.ini");
    let text = std::fs::read_to_string(&path).unwrap();
    let prefix = format!("{key} =");
    let mut found = false;
    let lines: Vec<_> = text
        .lines()
        .map(|line| {
            if line.trim_start().starts_with(&prefix) {
                found = true;
                format!("{key} = {value}")
            } else {
                line.to_string()
            }
        })
        .collect();
    assert!(found, "fixture lacks {key}");
    std::fs::write(path, lines.join("\n") + "\n").unwrap();
}

fn outputs(unit: &Path) -> Vec<Vec<u8>> {
    PRO_AND_SMET
        .iter()
        .map(|ext| {
            std::fs::read(unit.join(format!("output/ULAMB1_hrrr-terrain-v1.{ext}"))).unwrap()
        })
        .collect()
}

fn flags(physics: &[&str]) -> Vec<String> {
    physics
        .iter()
        .flat_map(|p| ["--physics".to_string(), p.to_string()])
        .collect()
}

fn one_process_each(units: &[PathBuf], physics: &[&str]) {
    for unit in units {
        let status = Command::new(env!("CARGO_BIN_EXE_pi-snow"))
            .args(["-c", unit.join("snowpack.ini").to_str().unwrap(), "-e", END])
            .args(flags(physics))
            .current_dir(unit)
            .status()
            .unwrap();
        assert!(status.success());
    }
}

#[test]
fn threads_equal_processes_byte_for_byte() {
    let root = scratch("equal");
    for physics in [&[][..], &["S5=LWC_INDEX_MITTERER2013"][..]] {
        let tag = physics.len();
        let per_process: Vec<PathBuf> = (0..5)
            .map(|k| stage(&root.join(format!("proc{tag}")), &format!("u{k}"), k))
            .collect();
        one_process_each(&per_process, physics);

        for (jobs, sub) in [(1, "j1"), (3, "j3"), (8, "j8")] {
            let batched: Vec<PathBuf> = (0..5)
                .map(|k| stage(&root.join(format!("{sub}-{tag}")), &format!("u{k}"), k))
                .collect();
            let mut cmd = Command::new(env!("CARGO_BIN_EXE_pi-snow"));
            cmd.args(["batch", "--jobs", &jobs.to_string(), "-e", END])
                .args(flags(physics))
                .args(&batched);
            assert!(cmd.status().unwrap().success());
            for (a, b) in per_process.iter().zip(&batched) {
                assert_eq!(outputs(a), outputs(b), "{sub} {physics:?} {}", b.display());
            }
        }
    }
    // The columns must differ from one another, or the comparison proves nothing.
    let first = outputs(&root.join("proc0/u0"));
    let second = outputs(&root.join("proc0/u3"));
    assert_ne!(first, second);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_failing_unit_is_named_and_the_others_still_write() {
    let root = scratch("fail");
    let good: Vec<PathBuf> = (0..3).map(|k| stage(&root, &format!("g{k}"), k)).collect();
    let reference: Vec<PathBuf> = (0..3)
        .map(|k| stage(&root.join("ref"), &format!("g{k}"), k))
        .collect();
    one_process_each(&reference, &[]);

    let broken = stage(&root, "broken", 9);
    std::fs::remove_file(broken.join("input/ULAMB1.smet")).unwrap();

    let end = parse_end(END).unwrap();
    let order = [&good[0], &broken, &good[1], &good[2]];
    let jobs: Vec<Job> = order
        .iter()
        .map(|u| Job {
            unit_dir: (*u).clone(),
            end,
        })
        .collect();
    let reports = run_batch(&jobs, &PhysicsOptions::default(), 2);

    assert_eq!(reports.len(), 4);
    assert!(reports[1].error.as_deref().unwrap().contains("ULAMB1.smet"));
    assert!(reports[0].error.is_none() && reports[2].error.is_none() && reports[3].error.is_none());
    for (g, r) in good.iter().zip(&reference) {
        assert_eq!(outputs(g), outputs(r));
    }

    // Through the binary: non-zero exit, and the report names the unit.
    let report = root.join("report.json");
    let status = Command::new(env!("CARGO_BIN_EXE_pi-snow"))
        .args([
            "batch",
            "--jobs",
            "2",
            "-e",
            END,
            "--report",
            report.to_str().unwrap(),
        ])
        .args(&good)
        .arg(&broken)
        .status()
        .unwrap();
    assert!(!status.success());
    let text = std::fs::read_to_string(&report).unwrap();
    assert!(text.contains("\"status\":\"error\""), "{text}");
    assert!(text.contains("broken"), "{text}");
    assert_eq!(text.matches("\"status\":\"ok\"").count(), 3, "{text}");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_list_file_gives_each_unit_its_own_end() {
    let root = scratch("list");
    let unit = stage(&root, "u", 1);
    let list = root.join("units.tsv");
    std::fs::write(&list, format!("{}\t{END}\n", unit.display())).unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_pi-snow"))
        .args(["batch", "--jobs", "1", "--list", list.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(status.success());
    let reference = stage(&root.join("ref"), "u", 1);
    one_process_each(std::slice::from_ref(&reference), &[]);
    assert_eq!(outputs(&unit), outputs(&reference));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn false_output_flags_gate_their_files_and_acdd_header() {
    let root = scratch("output-flags");
    let cases = [
        ("prof", "PROF_WRITE"),
        ("ts", "TS_WRITE"),
        ("acdd", "ACDD_WRITE"),
    ];
    for (name, key) in cases {
        let unit = stage(&root, name, 1);
        set_flag(&unit, key, "false");
        let status = Command::new(env!("CARGO_BIN_EXE_pi-snow"))
            .args(["-c", "snowpack.ini", "-e", END])
            .current_dir(&unit)
            .status()
            .unwrap();
        assert!(status.success(), "{key}=false: {status}");

        let stem = unit.join("output/ULAMB1_hrrr-terrain-v1");
        let pro = stem.with_extension("pro");
        let smet = stem.with_extension("smet");
        match key {
            "PROF_WRITE" => {
                assert!(!pro.exists());
                assert!(smet.is_file());
            }
            "TS_WRITE" => {
                assert!(pro.is_file());
                assert!(!smet.exists());
            }
            "ACDD_WRITE" => {
                assert!(pro.is_file());
                let text = std::fs::read_to_string(smet).unwrap();
                assert!(!text.lines().any(|line| line.starts_with("creator_name")));
            }
            _ => unreachable!(),
        }
    }
    let _ = std::fs::remove_dir_all(root);
}
