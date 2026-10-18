//! User-selectable settings of the `grid` command (gpu-alternatives-v4 plan, Amendment 2, step P).
//!
//! Two settings, each tied to a measured result in the Phase F record `v4-AF-2ed84da713b4`:
//!
//! * `--nonfinite-scan every-step|hourly`. `every-step` is the K1h kernel at F32 (byte-identical to the pinned
//!   kernel in every unit, shorter than it by the registered rule, `ships.every_step_form`), and the pinned kernel
//!   at every other precision, because K1h was validated at F32 only. `hourly` is K2, labelled non-equivalent, and is
//!   offered only when the record shows it shorter than the every-step form (A2.2); it is not today
//!   (`ships.hourly_offered` is false: K2 is 1.632 and 1.516 times K1h on HI and LO). Offering it needs a dated
//!   amendment (the record says no) and A2.5's K2 GPU gate; flipping the constant alone is not enough. There is no
//!   `off`: a switch that skipped the check would ship columns with non-finite state silently (engine-gpu-v1, G2).
//! * `--batch N|auto`. `auto` picks the validated launch width with the smallest median GPU-line ratio that fits the
//!   memory budget (A2.3). It reads the budget and never posts feedback; the caller posts the reserve.
//!
//! Everything here is CPU-only and pure apart from [`broker_state`] (reads a file) and [`read_meminfo_available_kib`].

use std::path::Path;

use anyhow::{Result, bail};
use chrono::{DateTime, Utc};

use crate::gpu::{Kernel, Precision};

/// The Phase F record this module's table and `HOURLY_OFFERED` come from.
pub const PHASE_F_RECORD: &str = "v4-AF-2ed84da713b4";

/// `ships.hourly_offered` of [`PHASE_F_RECORD`]. A2.2 offers `hourly` only if Phase F shows K2 shorter than the
/// chosen every-step form; it showed it longer, so this is false and `hourly` is refused with that reason. Setting it
/// true needs a dated amendment to the plan and A2.5's K2 GPU gate first.
pub const HOURLY_OFFERED: bool = false;

/// The defaults of the two settings (A2.4), pinned by tests. `ships.nonfinite_scan_default` of the record is
/// `every-step`; `--batch auto` is A2.4's shipped width default.
pub const DEFAULT_NONFINITE_SCAN: &str = "every-step";
pub const DEFAULT_BATCH: &str = "auto";

/// A2.6: where the identity and speed statements hold.
pub const VALIDATED_GPU: &str = "RTX 5090";
pub const VALIDATED_ARCH: &str = "sm_120";
pub const VALIDATED_DRIVER: i32 = 13040;
pub const VALIDATED_NVRTC: (i32, i32) = (13, 3);
pub const VALIDATED_PRECISION: &str = "F32";
pub const VALIDATED_SEASON: &str = "2025-11-01 to 2026-04-30";

/// PTX SHA-256 of each shipped kernel at F32, NVRTC 13.3, sm_120: the Phase F U1 record `v4-U1-2c815797d427`.
pub const VALIDATED_PTX: [(Kernel, &str); 3] = [
    (Kernel::Pinned, "a048838ce0646f84ebc95abca9b0f5564150d241c4be259f79bf419fdf20d1f2"),
    (Kernel::K1h, "26dfad5606a57fb15b5f62eca00c4b2eda9c3dae3520c910ba770a6590ff7116"),
    (Kernel::K2, "32f881191239a830204f4c340f5b3d5ce6723c22434f22c8c9a368d72801c134"),
];

/// The `validated_on` object written into both run-JSON blocks (A2.6).
pub fn validated_on_json() -> String {
    format!(
        "{{\"gpu\": \"{VALIDATED_GPU}\", \"arch\": \"{VALIDATED_ARCH}\", \"driver\": {VALIDATED_DRIVER}, \"nvrtc\": \"{}.{}\", \"precision\": \"{VALIDATED_PRECISION}\", \"season\": \"{VALIDATED_SEASON}\"}}",
        VALIDATED_NVRTC.0, VALIDATED_NVRTC.1
    )
}

/// Whether the running build is the one the identity and ratio claims were measured on: arch, driver, NVRTC, F32
/// and a PTX hash equal to the U1 record's for this kernel. A reader sees from the JSON whether the claim applies.
pub fn matches_validated_setup(build: &crate::gpu::Build, kernel: Kernel) -> bool {
    build.precision == Precision::F32
        && build.arch == VALIDATED_ARCH
        && build.driver_version == VALIDATED_DRIVER
        && build.nvrtc_version == VALIDATED_NVRTC
        && VALIDATED_PTX.iter().any(|(k, h)| *k == kernel && *h == build.ptx_sha256)
}

/// A2.3: device bytes per column at F32, measured; the context and working space are added on top.
pub const DEVICE_BYTES_PER_COLUMN: u64 = 182_424;
pub const DEVICE_CONTEXT_GIB: f64 = 1.5;
/// A2.3: host margin added to a width's measured peak RSS.
pub const RAM_MARGIN_GIB: f64 = 4.0;
/// A broker status file older than this is stale (the house rule: ten minutes).
pub const BROKER_STALE_SECS: i64 = 600;
/// The width `auto` falls back to; always valid, always last in the list.
pub const BASE_WIDTH: usize = 16_384;

const GIB: f64 = 1_073_741_824.0;
const KIB_PER_GIB: f64 = 1_048_576.0;

/// The `grid --help` text for these two options (a test pins the labels).
pub const GRID_HELP: &str = "\
pi-snow-gpu grid --forcing DIR --ini TEMPLATE --pixels FILE --out DIR
                 [--batch N|auto] [--nonfinite-scan every-step|hourly]
                 [--precision F64|MIX|F32|F32K] [--nmax N] [--threads T] [--dry-run]

  --batch N|auto          columns per launch. `auto` (the default) takes the validated width with the smallest
                          median GPU-line ratio that fits the broker budget and the live memory readings; it
                          returns 16384 when the broker file is stale, and at any precision other than F32. It
                          never posts a reservation: post one for the width it reports (--dry-run prints it).
  --nonfinite-scan S      every-step (default): the non-finite check after every step; the K1h kernel at F32, the
                          pinned kernel at other precisions; byte-identical to the pinned kernel.
                          hourly: K2, NON-EQUIVALENT: refusals can land up to five steps late and the refused set
                          can differ in both directions. Not offered unless the Phase F record shows it shorter
                          than every-step (it did not). There is no `off`: the check is never skipped.
  --dry-run               print the chosen settings and the reserve to post, then exit without running. Sizes are
                          measured at F32 only: at any other precision the reserve is null.

Where the claims hold (A2.6): the identity of every-step to the pinned kernel, and the speed ratios behind `auto`,
were measured on ai100's RTX 5090 (sm_120), driver 13040, NVRTC 13.3, F32, the 2025-11-01 to 2026-04-30 season, on
the 83,232-pixel grid. Elsewhere `auto` is claimed to fit, not to be faster, and every-step's identity is shown only
there; each run's JSON carries `validated_on` and `matches_validated_setup` so a reader can see which applies.
`auto` reads the broker's `gpu:` budget and `ram:` available figures as its limits (not the `Post feedback before
allocating more than N` threshold).
";

// ---- the non-finite scan -----------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonfiniteScan {
    EveryStep,
    Hourly,
}

impl NonfiniteScan {
    /// `hourly_offered` is [`HOURLY_OFFERED`] in production; a parameter so a test can pin both states.
    pub fn parse(s: &str, hourly_offered: bool) -> Result<Self> {
        match s {
            "every-step" => Ok(Self::EveryStep),
            "hourly" if hourly_offered => Ok(Self::Hourly),
            "hourly" => bail!(
                "--nonfinite-scan hourly is not offered: gpu-alternatives-v4 A2.2 offers it only when Phase F shows it \
                 shorter than every-step, and record {PHASE_F_RECORD} shows it longer (K2 over K1h 1.632 on HI, 1.516 on LO)"
            ),
            "off" => bail!(
                "--nonfinite-scan off does not exist: skipping the check would ship columns with non-finite state \
                 silently; use every-step"
            ),
            other => bail!("--nonfinite-scan {other:?} is not one of every-step, hourly"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::EveryStep => "every-step",
            Self::Hourly => "hourly",
        }
    }

    /// The kernel for this setting at a precision. `hourly` is validated at F32 only.
    pub fn kernel(self, precision: Precision) -> Result<Kernel> {
        match (self, precision) {
            (Self::EveryStep, Precision::F32) => Ok(Kernel::K1h),
            (Self::EveryStep, _) => Ok(Kernel::Pinned),
            (Self::Hourly, Precision::F32) => Ok(Kernel::K2),
            (Self::Hourly, p) => bail!("--nonfinite-scan hourly was built and measured at F32 only, not {}", p.label()),
        }
    }

    pub fn equivalent_to_pinned(self) -> bool {
        self == Self::EveryStep
    }

    /// The run-JSON block `nonfinite_scan` (A2.2), plus the kernel actually used and the A2.6 boundary.
    pub fn run_json(self, kernel: Kernel, kernel_source_sha256: &str, matches_validated: bool) -> String {
        format!(
            "{{\"setting\": \"{}\", \"kernel\": \"{}\", \"kernel_source_sha256\": \"{}\", \"equivalent_to_pinned\": {}, \"validated_on\": {}, \"matches_validated_setup\": {}}}",
            self.label(),
            kernel.label(),
            kernel_source_sha256,
            self.equivalent_to_pinned(),
            validated_on_json(),
            matches_validated
        )
    }
}

// ---- the launch width --------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchSetting {
    Fixed(usize),
    Auto,
}

impl BatchSetting {
    pub fn parse(s: &str) -> Result<Self> {
        if s == "auto" {
            return Ok(Self::Auto);
        }
        match s.parse::<usize>() {
            Ok(n) if n > 0 => Ok(Self::Fixed(n)),
            _ => bail!("--batch {s:?} is not a positive integer or auto"),
        }
    }
}

/// One validated launch width: every unit of it byte-identical to M2 chain B and shorter than 16,384 on both the GPU
/// line and the sum of batch walls by the registered rule, with the peak RSS measured at 4 host threads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValidatedWidth {
    pub width: usize,
    pub median_gpu_ratio: f64,
    pub median_wall_ratio: f64,
    /// Peak RSS in KiB; `None` means not recorded, and a width without it is not offered (A2.7).
    pub peak_rss_kib: Option<u64>,
}

/// `ships.batch_auto_widths` of [`PHASE_F_RECORD`]: every validated width, best first, then 16,384 (ratio 1.0).
/// The 83,232 and 32,768 peak RSS figures are Phase F's W-stage K0 units at 4 host threads (KW's K1h at 83,232
/// peaked 70 MiB higher, 28,065,136 KiB; the 4 GiB margin covers it). The 16,384 figure, 5,728,136 KiB, is the one A2.3
/// names (R8, I10): engine-gpu-v1's F32 grid at 16 threads; the record's ships block carries no RSS for 16,384.
/// Thread count moves the peak by under 2 percent (Phase F's 4-thread units at 16,384 peaked at 5.48 and 5.57 GiB).
pub const VALIDATED_WIDTHS: [ValidatedWidth; 3] = [
    ValidatedWidth { width: 83_232, median_gpu_ratio: 0.29723768016476354, median_wall_ratio: 0.3031243861353953, peak_rss_kib: Some(27_993_212) },
    ValidatedWidth { width: 32_768, median_gpu_ratio: 0.6369143894152312, median_wall_ratio: 0.6458919522134997, peak_rss_kib: Some(11_201_280) },
    ValidatedWidth { width: BASE_WIDTH, median_gpu_ratio: 1.0, median_wall_ratio: 1.0, peak_rss_kib: Some(5_728_136) },
];

/// What the broker's status file said.
#[derive(Debug, Clone, PartialEq)]
pub enum BrokerState {
    /// `~/.machine/` is absent: live readings alone.
    Absent,
    /// The file is missing, unparseable or older than ten minutes.
    Stale { reason: String },
    Fresh(BrokerReading),
}

#[derive(Debug, Clone, PartialEq)]
pub struct BrokerReading {
    pub updated: DateTime<Utc>,
    /// The `gpu:` directive's budget, GiB.
    pub gpu_budget_gib: f64,
    /// The `ram:` directive's available figure, GiB.
    pub ram_available_gib: f64,
}

/// Parse `updated:` and the `gpu:` and `ram:` directive lines of `MACHINE_STATUS.md`.
pub fn parse_status(text: &str) -> Result<BrokerReading> {
    let mut updated = None;
    let (mut gpu, mut ram) = (None, None);
    for line in text.lines() {
        let l = line.trim();
        if let Some(v) = l.strip_prefix("updated:") {
            updated = DateTime::parse_from_rfc3339(v.trim()).ok().map(|d| d.with_timezone(&Utc));
        } else if let Some(v) = l.strip_prefix("- gpu:") {
            gpu = number_after(v, "budget");
        } else if let Some(v) = l.strip_prefix("- ram:") {
            ram = number_before(v, "GiB available");
        }
    }
    match (updated, gpu, ram) {
        (Some(updated), Some(gpu_budget_gib), Some(ram_available_gib)) => Ok(BrokerReading { updated, gpu_budget_gib, ram_available_gib }),
        _ => bail!("status file lacks updated, a gpu budget or a ram available figure"),
    }
}

/// The number that follows `key ` in `s` ("budget 23 GiB of ..." gives 23).
fn number_after(s: &str, key: &str) -> Option<f64> {
    let rest = s.split(key).nth(1)?;
    rest.split_whitespace().next()?.parse().ok()
}

/// The number just before `key` in `s` ("61 GiB available of ..." gives 61).
fn number_before(s: &str, key: &str) -> Option<f64> {
    let head = s.split(key).next()?;
    head.split_whitespace().last()?.parse().ok()
}

/// Classify the broker at `now`. `machine_dir` is `~/.machine`; `status` is its `MACHINE_STATUS.md`.
pub fn broker_state(machine_dir: &Path, status: &Path, now: DateTime<Utc>) -> BrokerState {
    if !machine_dir.exists() {
        return BrokerState::Absent;
    }
    let text = match std::fs::read_to_string(status) {
        Ok(t) => t,
        Err(e) => return BrokerState::Stale { reason: format!("status file unreadable: {e}") },
    };
    match parse_status(&text) {
        Err(e) => BrokerState::Stale { reason: e.to_string() },
        Ok(r) if (now - r.updated).num_seconds() > BROKER_STALE_SECS => {
            BrokerState::Stale { reason: format!("status file updated {} is older than {} s", r.updated.to_rfc3339(), BROKER_STALE_SECS) }
        }
        Ok(r) => BrokerState::Fresh(r),
    }
}

/// `MemAvailable` from `/proc/meminfo` text, KiB.
pub fn parse_meminfo_available_kib(text: &str) -> Option<u64> {
    text.lines().find_map(|l| l.strip_prefix("MemAvailable:")?.split_whitespace().next()?.parse().ok())
}

pub fn read_meminfo_available_kib() -> Result<u64> {
    let text = std::fs::read_to_string("/proc/meminfo")?;
    parse_meminfo_available_kib(&text).ok_or_else(|| anyhow::anyhow!("/proc/meminfo has no MemAvailable"))
}

/// Live readings: free device bytes and available host KiB.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Live {
    pub gpu_free_bytes: u64,
    pub ram_available_kib: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub width: usize,
    pub median_gpu_ratio: f64,
    pub median_wall_ratio: f64,
    /// `None` is missing, never 0 (A2.7).
    pub peak_rss_gib: Option<f64>,
    /// `None` at any precision other than F32: the sizes were measured at F32 only.
    pub device_gib_needed: Option<f64>,
    pub ram_gib_needed: Option<f64>,
    pub fits: Option<bool>,
}

/// The note written wherever a size would be wrong off F32.
pub const F32_ONLY_SIZES: &str = "sizes measured at F32 only";

/// The result of `--batch auto`, and the run-JSON block `batch_auto` (A2.3).
#[derive(Debug, Clone, PartialEq)]
pub struct AutoChoice {
    pub gpu_free_gib: f64,
    /// `None` when no fresh broker was read.
    pub gpu_budget_gib: Option<f64>,
    pub ram_available_gib: f64,
    pub source: &'static str,
    pub device_bytes_per_column: u64,
    /// The chosen width's peak RSS per column, GiB.
    pub ram_gib_per_column: Option<f64>,
    pub candidates: Vec<Candidate>,
    pub chosen: usize,
    pub reason: String,
    /// What to reserve for the chosen width: device GiB and host GiB. `None` off F32 (see `note`).
    pub reserve_gpu_gib: Option<f64>,
    pub reserve_ram_gib: Option<f64>,
    pub note: Option<&'static str>,
}

pub fn device_gib_for(width: usize) -> f64 {
    (width as f64 * DEVICE_BYTES_PER_COLUMN as f64) / GIB + DEVICE_CONTEXT_GIB
}

fn ram_gib_for(w: &ValidatedWidth) -> Option<f64> {
    w.peak_rss_kib.map(|k| k as f64 / KIB_PER_GIB + RAM_MARGIN_GIB)
}

/// A2.3. Of the validated widths that fit, the one with the smallest median GPU-line ratio; the next one down the list
/// if that does not fit; 16,384 last; an error that names the shortfall if not even 16,384 fits. A stale broker, and any
/// precision other than F32, return 16,384 without a fit check; off F32 the sizes are `None` (measured at F32 only).
///
/// The live GPU reading the caller passes is taken after the CUDA context exists, so it is already net of the
/// context's own memory, and A2.3's 1.5 GiB for the context is counted again: conservative by about that much.
pub fn select_auto(precision: Precision, broker: &BrokerState, live: &Live, table: &[ValidatedWidth]) -> Result<AutoChoice> {
    let live_gpu_gib = live.gpu_free_bytes as f64 / GIB;
    let live_ram_gib = live.ram_available_kib as f64 / KIB_PER_GIB;
    let (source, budget, gpu_eff, ram_eff): (&'static str, Option<f64>, f64, f64) = match broker {
        BrokerState::Fresh(r) => ("broker", Some(r.gpu_budget_gib), live_gpu_gib.min(r.gpu_budget_gib), live_ram_gib.min(r.ram_available_gib)),
        BrokerState::Stale { .. } => ("broker-stale, conservative", None, live_gpu_gib, live_ram_gib),
        BrokerState::Absent => ("live-only", None, live_gpu_gib, live_ram_gib),
    };
    let sized = precision == Precision::F32;
    // Offered widths only: a width with no recorded peak RSS is not offered; the base width is always offered.
    let offered: Vec<&ValidatedWidth> = table.iter().filter(|w| w.peak_rss_kib.is_some() || w.width == BASE_WIDTH).collect();
    let candidates: Vec<Candidate> = offered
        .iter()
        .map(|w| {
            let dev = sized.then(|| device_gib_for(w.width));
            let ram = if sized { ram_gib_for(w) } else { None };
            Candidate {
                width: w.width,
                median_gpu_ratio: w.median_gpu_ratio,
                median_wall_ratio: w.median_wall_ratio,
                peak_rss_gib: w.peak_rss_kib.map(|k| k as f64 / KIB_PER_GIB),
                device_gib_needed: dev,
                ram_gib_needed: ram,
                fits: match (dev, ram) {
                    (Some(d), Some(r)) => Some(d <= gpu_eff && r <= ram_eff),
                    _ => None,
                },
            }
        })
        .collect();
    let base = candidates.iter().find(|c| c.width == BASE_WIDTH).expect("the base width is always listed");
    let finish = |chosen: &Candidate, reason: String| AutoChoice {
        gpu_free_gib: live_gpu_gib,
        gpu_budget_gib: budget,
        ram_available_gib: ram_eff,
        source,
        device_bytes_per_column: DEVICE_BYTES_PER_COLUMN,
        ram_gib_per_column: chosen.peak_rss_gib.map(|g| g / chosen.width as f64),
        candidates: candidates.clone(),
        chosen: chosen.width,
        reason,
        reserve_gpu_gib: chosen.device_gib_needed,
        reserve_ram_gib: chosen.ram_gib_needed,
        note: (!sized).then_some(F32_ONLY_SIZES),
    };
    if !sized {
        return Ok(finish(base, "width validated at F32 only".into()));
    }
    if let BrokerState::Stale { reason } = broker {
        return Ok(finish(base, format!("broker status is stale ({reason}); conservative width")));
    }
    // Best first by median GPU-line ratio (the table is already sorted; sort again so the rule does not depend on it).
    let mut order: Vec<&Candidate> = candidates.iter().collect();
    order.sort_by(|a, b| a.median_gpu_ratio.partial_cmp(&b.median_gpu_ratio).expect("finite ratios"));
    match order.iter().find(|c| c.fits == Some(true)) {
        Some(c) => {
            let why = if c.width == order[0].width {
                format!("smallest median GPU-line ratio ({:.3}) among the widths that fit", c.median_gpu_ratio)
            } else {
                let best = order[0];
                format!(
                    "{} does not fit (needs {:.1} GiB device and {:.1} GiB host, have {:.1} and {:.1}); next width down that fits, median GPU-line ratio {:.3}",
                    best.width,
                    best.device_gib_needed.unwrap_or(f64::NAN),
                    best.ram_gib_needed.unwrap_or(f64::NAN),
                    gpu_eff,
                    ram_eff,
                    c.median_gpu_ratio
                )
            };
            Ok(finish(c, why))
        }
        None => {
            let (dev, ram) = (base.device_gib_needed.unwrap_or(f64::NAN), base.ram_gib_needed.unwrap_or(f64::NAN));
            bail!(
                "--batch auto: not even {BASE_WIDTH} fits: it needs {dev:.1} GiB of device memory (have {gpu_eff:.1}, short {:.1}) and {ram:.1} GiB of host RAM (have {ram_eff:.1}, short {:.1}); free memory or pass --batch N",
                (dev - gpu_eff).max(0.0),
                (ram - ram_eff).max(0.0)
            )
        }
    }
}

fn opt(x: Option<f64>, digits: usize) -> String {
    x.map_or("null".to_string(), |v| format!("{v:.digits$}"))
}

impl AutoChoice {
    /// The run-JSON block `batch_auto` (A2.3), plus the A2.6 boundary.
    pub fn run_json(&self, matches_validated: bool) -> String {
        let cands: Vec<String> = self
            .candidates
            .iter()
            .map(|c| {
                format!(
                    "{{\"width\": {}, \"median_gpu_ratio\": {:.6}, \"median_wall_ratio\": {:.6}, \"peak_rss_gib\": {}, \"device_gib_needed\": {}, \"ram_gib_needed\": {}, \"fits\": {}}}",
                    c.width,
                    c.median_gpu_ratio,
                    c.median_wall_ratio,
                    opt(c.peak_rss_gib, 3),
                    opt(c.device_gib_needed, 3),
                    opt(c.ram_gib_needed, 3),
                    c.fits.map_or("null".to_string(), |f| f.to_string())
                )
            })
            .collect();
        format!(
            "{{\"gpu_free_gib\": {:.3}, \"gpu_budget_gib\": {}, \"ram_available_gib\": {:.3}, \"source\": \"{}\", \"device_bytes_per_column\": {}, \"ram_gib_per_column\": {}, \"candidates\": [{}], \"chosen\": {}, \"reason\": \"{}\", \"note\": {}, \"phase_f_record\": \"{PHASE_F_RECORD}\", \"validated_on\": {}, \"matches_validated_setup\": {}}}",
            self.gpu_free_gib,
            opt(self.gpu_budget_gib, 3),
            self.ram_available_gib,
            self.source,
            self.device_bytes_per_column,
            opt(self.ram_gib_per_column, 8),
            cands.join(", "),
            self.chosen,
            self.reason.replace('"', "'"),
            self.note.map_or("null".to_string(), |n| format!("\"{n}\"")),
            validated_on_json(),
            matches_validated
        )
    }
}

/// What a `--dry-run` prints for a fixed width: the device estimate and, if the width is validated, the host figure.
/// Both are `None` off F32 (sizes measured at F32 only).
pub fn reserve_for_fixed(precision: Precision, width: usize, table: &[ValidatedWidth]) -> (Option<f64>, Option<f64>) {
    if precision != Precision::F32 {
        return (None, None);
    }
    let ram = table.iter().find(|w| w.width == width).and_then(ram_gib_for);
    (Some(device_gib_for(width)), ram)
}
