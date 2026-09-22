//! Feed the reference profile's own microstructure through our stability
//! formulations and compare with the indices SNOWPACK wrote for it. Same
//! inputs, so every difference is formulation, not state.
//!
//! Usage: stability_vs_reference <file.pro> [timestamp dd.mm.yyyy HH:MM:SS]
use pi_snow_io::pro::Profiles;
use pi_snow_physics::stability::{self, Layer, Penetration};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[1]).unwrap();
    let profiles = Profiles::parse(&text).unwrap();
    let step = profiles
        .timesteps
        .iter()
        .max_by_key(|t| t.element_count())
        .unwrap();
    let f = |c: &str| step.element_field(c).unwrap().to_vec();
    let (top, rho, gt, gs, hh, temp, age) = (
        f("0501"),
        f("0502"),
        f("0513"),
        f("0512"),
        f("0534"),
        f("0503"),
        f("0505"),
    );
    let hs = *top.last().unwrap();
    let layers: Vec<Layer> = (0..top.len())
        .map(|i| {
            let bottom = if i == 0 { 0.0 } else { top[i - 1] };
            Layer {
                depth_top: (hs - top[i]) / 100.0,
                thickness: (top[i] - bottom) / 100.0,
                density: rho[i],
                grain_type: gt[i] as u16,
                grain_size_mm: gs[i],
                hand_hardness: hh[i].abs(),
                temperature_c: temp[i],
                age_days: age[i],
            }
        })
        .collect();
    let (sn, sk, ssi, rc) = (f("0532"), f("0533"), f("0604"), f("0606"));
    println!(
        "timestamp {} HS {:.1} cm, {} elements",
        step.timestamp,
        hs,
        top.len()
    );
    for model in [
        Penetration::JamiesonJohnston1998,
        Penetration::ExtractorForm,
    ] {
        let ours = stability::indices_with(&layers, hs / 100.0, model);
        println!("\npenetration model {model:?}");
        let refs = [
            ("Sn38", &sn, 6.0),
            ("Sk38", &sk, 6.0),
            ("SSI", &ssi, 6.0),
            ("rc", &rc, 3.0),
        ];
        for (k, (name, reference, cap)) in refs.iter().enumerate() {
            let mut all = Vec::new();
            let mut evaluated = Vec::new();
            for (i, o) in ours.iter().enumerate() {
                let u = [o.sn38, o.sk38, o.ssi, o.critical_cut_length_m][k];
                let r = reference[i];
                if let Some(u) = u.filter(|_| r > -998.0) {
                    all.push((r, u));
                    if r < *cap - 1e-9 && u < *cap - 1e-9 {
                        evaluated.push((r, u));
                    }
                }
            }
            let stats = |p: &[(f64, f64)]| {
                let n = p.len().max(1) as f64;
                let bias = p.iter().map(|(r, u)| u - r).sum::<f64>() / n;
                let rmse = (p.iter().map(|(r, u)| (u - r).powi(2)).sum::<f64>() / n).sqrt();
                (p.len(), bias, rmse)
            };
            let (na, ba, ra) = stats(&all);
            let (ne, be, re) = stats(&evaluated);
            println!(
                "  {name:<5} all n={na:>2} bias {ba:+.3} rmse {ra:.3} | both below cap n={ne:>2} bias {be:+.3} rmse {re:.3}"
            );
        }
        let first_sk = ours
            .iter()
            .enumerate()
            .rev()
            .find(|(_, o)| o.sk38.is_some_and(|v| v < 6.0));
        let ref_sk = sk.iter().enumerate().rev().find(|(_, v)| **v < 6.0);
        if let (Some((i, _)), Some((j, _))) = (first_sk, ref_sk) {
            println!(
                "  shallowest evaluated Sk38: ours at {:.1} cm, reference at {:.1} cm",
                layers[i].depth_top * 100.0,
                layers[j].depth_top * 100.0
            );
        }
    }
}
