//! Kernel against the CPU stepper on synthetic forcing, three days of lockstep. Skipped
//! (passes trivially, with a message) on a machine without a usable CUDA device.

use pi_snow_gpu::gpu::{Gpu, Precision};
use pi_snow_gpu::lockstep;
use pi_snow_gpu::runner::ColumnInput;
use pi_snow_gpu::synth;

#[test]
fn three_synthetic_days_agree_and_count_monin_obukhov_passes() {
    let Ok(gpu) = Gpu::new(Precision::F64) else {
        eprintln!("no CUDA device: skipped");
        return;
    };
    let cfg = synth::config();
    // Day 2 of the synthetic season starts with clear calm nights between storms.
    let s = synth::generate(4, 3);
    let inputs: Vec<ColumnInput> = s
        .sites
        .iter()
        .zip(s.rows)
        .map(|(site, rows)| ColumnInput { cfg: &cfg, site, start: synth::bare(site), rows, profile_times: None })
        .collect();
    let tally = lockstep::run(&gpu, &inputs, 2048, 0, 3 * 144 + 1).unwrap();
    assert_eq!(tally.agreeing, tally.comparisons, "{}", tally.to_json());
    assert_eq!(tally.count_mismatch_steps, 0);
    assert_eq!(tally.outcome_mismatch_steps, 0);
    assert!(tally.mo_multi_pass_steps_cpu > 0, "{}", tally.to_json());
    assert!(tally.mo_multi_pass_steps_gpu > 0, "{}", tally.to_json());
}
