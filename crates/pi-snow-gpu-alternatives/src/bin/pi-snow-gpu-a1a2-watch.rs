//! Registered W/G1d output interface for the compact FP32 candidate.
#[path = "../a1a2_gpu.rs"] mod a1a2_gpu;
#[path = "../a1a2_runner.rs"] mod a1a2_runner;
#[path = "../a1a2_watch_batch.rs"] mod a1a2_watch_batch;
pub use pi_snow_gpu_alternatives::{sha256, stepper, unit};
fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let preflight = args.first().map(String::as_str) == Some("--preflight");
    if preflight { args.remove(0); }
    if args.iter().any(|a| a == "--precision") { anyhow::bail!("candidate precision is fixed F32"); }
    args.extend(["--precision".into(), "F32".into()]);
    if preflight {
        let parsed = a1a2_watch_batch::parse_args(&args)?;
        let options = pi_snow_core::PhysicsOptions::default();
        let mut k = None;
        for job in &parsed.jobs {
            let (u, rows) = a1a2_watch_batch::load(job)?;
            let constants = stepper::Constants::new(&u.cfg, &options, &u.site);
            a1a2_gpu::check_supported(&u.cfg, &constants, &u.start)?;
            let current = a1a2_gpu::run_constants(&u.cfg, &constants)?.map(f64::to_bits);
            if k.is_some_and(|previous| previous != current) { anyhow::bail!("mixed run constants"); }
            k = Some(current);
            if rows.is_empty() { anyhow::bail!("empty forcing"); }
        }
        println!("{{\"state\":\"preflight-success\",\"units\":{}}}", parsed.jobs.len());
        Ok(())
    } else {
        a1a2_watch_batch::main(&args)
    }
}
