//! engine-soil-spinup-v2 plan test Z-T9 `zc_matches_linearise_golden`.
//!
//! `experiments/engine-soil-spinup-v2/artifacts/zn-golden.tsv` holds the bits
//! of `surface_layer_continuous::fluxes` from branch `p2/engine-surface-linearise`
//! at commit 92a1e6273e4978bcea4b97b983ed7deec92312b6 on 10,000 fixed inputs
//! (`zn_golden.sh` rebuilds it). This module's scheme must reproduce it bit
//! for bit. It certifies arithmetic equivalence only; the two engines that
//! reach that arithmetic are different (plan A1.1).

use pi_snow_physics::atmosphere::{specific_humidity, vapour_pressure_ice, vapour_pressure_water};
use pi_snow_physics::turbulence::{self, Scheme, SurfaceLayer};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn zc_matches_linearise_golden() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../experiments/engine-soil-spinup-v2/artifacts/zn-golden.tsv"
    );
    let golden = std::fs::read_to_string(path).expect("golden file");
    let mut lines = golden.lines();
    let mut r = Rng(0x2026_1012_5011_5eed);
    let mut checked = 0;
    for i in 0..10_000 {
        let wind = if i % 500 == 0 {
            0.0
        } else {
            0.05 + 15.0 * r.next()
        };
        let t_air = 250.0 + 50.0 * r.next();
        let dt = -30.0 + 60.0 * r.next();
        let near = if i % 7 == 0 {
            1e-6 * (r.next() - 0.5)
        } else {
            dt
        };
        let t_surface = t_air + near;
        let z0 = (2e-4f64.ln() + (1e-2f64.ln() - 2e-4f64.ln()) * r.next()).exp();
        let pressure = 60_000.0 + 40_000.0 * r.next();
        let rh = 0.05 + 0.9 * r.next();
        let es = if t_surface >= 273.15 {
            vapour_pressure_water(t_surface)
        } else {
            vapour_pressure_ice(t_surface)
        };
        let l = SurfaceLayer {
            wind,
            z_wind: 10.0,
            z_scalar: 2.0,
            z0,
            t_air,
            t_surface,
            q_air: specific_humidity(rh * vapour_pressure_water(t_air), pressure),
            q_surface: specific_humidity(es, pressure),
            pressure,
        };
        let f = turbulence::fluxes(&l, Scheme::MoHoltslagZetaContinuous);
        let expected = format!(
            "{i}\t{:016x}\t{:016x}\t{:016x}\t{:016x}",
            f.sensible.to_bits(),
            f.vapour_mass.to_bits(),
            f.u_star.to_bits(),
            f.obukhov_length.to_bits()
        );
        assert_eq!(lines.next().unwrap(), expected, "input {i}");
        checked += 1;
    }
    assert_eq!(checked, 10_000);
    assert!(lines.next().is_none());
}
