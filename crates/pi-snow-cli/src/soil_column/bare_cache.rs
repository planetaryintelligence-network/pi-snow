//! A5.3: exact per-forcing-row budgets, with arm-dependent thresholds kept local.
use super::bare_scan::Certification;
use pi_snow_physics::surface::{Atmosphere, SurfaceParameters};
use pi_snow_physics::turbulence::Scheme;
use std::{cell::RefCell, collections::HashMap, rc::Rc};

#[derive(Debug, Default, Clone)]
pub struct Samples {
    pub(super) maximum_slope: Option<f64>,
    pub(super) certification: Option<Certification>,
    pub(super) nodes: Option<[Vec<f64>; 2]>,
}

#[derive(Debug, Default)]
pub struct PredicateCache {
    rows: HashMap<(i64, [u64; 12]), Samples>,
}

/// Owned by one unit-season's runner and shared only by its soil arms.
pub type Shared = Rc<RefCell<PredicateCache>>;

impl PredicateCache {
    pub fn shared() -> Shared {
        Rc::new(RefCell::new(Self::default()))
    }

    pub(super) fn row(
        &mut self,
        stamp: i64,
        a: &Atmosphere,
        p: &SurfaceParameters,
    ) -> &mut Samples {
        let scheme = match p.scheme {
            Scheme::MoHoltslag => 0,
            Scheme::Louis => 1,
            Scheme::MoHoltslagScalarHeight => 2,
            Scheme::MoHoltslagZetaLimited => 3,
        };
        let bits = [
            a.t_air.to_bits(),
            a.relative_humidity.to_bits(),
            a.wind.to_bits(),
            a.absorbed_shortwave.to_bits(),
            a.ilwr.to_bits(),
            a.pressure.to_bits(),
            a.rain_rate.to_bits(),
            p.emissivity.to_bits(),
            p.z_wind.to_bits(),
            p.z_scalar.to_bits(),
            p.z0.to_bits(),
            scheme,
        ];
        self.rows.entry((stamp, bits)).or_default()
    }
}
