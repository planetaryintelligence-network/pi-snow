//! Aoki et al. (2003), Figure 6 and Table 2 temperature-rate snow albedo.
//!
//! VIS/NIR intercepts come from Table 1's subzero regressions. The rate is
//! floored at zero below its Table 2 zero crossing; Aoki says albedo is stable
//! below about -10 C but does not specify the fit's predictor range.

use std::collections::VecDeque;

use pi_snow_core::Column;

use crate::albedo::{self, Shortwave};

const DAY_SECONDS: f64 = 86_400.0;
const NEW_SNOW_RESET_M: f64 = 0.03;
const INITIAL_VIS: f64 = 0.927;
const INITIAL_NIR: f64 = 0.722;

/// Per-column Aoki surface state.
#[derive(Debug, Default)]
pub struct AokiAlbedo {
    active: bool,
    elapsed: f64,
    visible: f64,
    near_infrared: f64,
    snowfall: VecDeque<(f64, f64)>,
    temperatures: VecDeque<(f64, f64)>,
    temperature_seconds: f64,
    temperature_sum: f64,
}

impl AokiAlbedo {
    /// Initialize a column that already contains snow at the run's start.
    pub fn start_pack(&mut self) {
        self.active = true;
        self.visible = INITIAL_VIS;
        self.near_infrared = INITIAL_NIR;
        self.clear_temperatures();
        self.snowfall.clear();
    }

    /// Advance the surface state and return the three mapped band albedos.
    /// `new_pack` is true on the first deposited element after a bare interval.
    pub fn advance(
        &mut self,
        step_seconds: f64,
        temperature_c: f64,
        deposited_snow_m: f64,
        new_pack: bool,
    ) -> [f64; 3] {
        self.elapsed += step_seconds;
        let cutoff = self.elapsed - DAY_SECONDS;
        while self.snowfall.front().is_some_and(|(t, _)| *t < cutoff) {
            self.snowfall.pop_front();
        }
        let before = self.snowfall.iter().map(|(_, h)| h).sum::<f64>();
        if deposited_snow_m > 0.0 {
            self.snowfall.push_back((self.elapsed, deposited_snow_m));
        }
        let after = before + deposited_snow_m.max(0.0);

        let reset = new_pack || (before <= NEW_SNOW_RESET_M && after > NEW_SNOW_RESET_M);
        if reset {
            self.active = true;
            self.visible = INITIAL_VIS;
            self.near_infrared = INITIAL_NIR;
            self.clear_temperatures();
        } else if self.active {
            self.push_temperature(temperature_c, step_seconds);
            if self.temperature_seconds >= DAY_SECONDS {
                let t24 = self.temperature_sum / self.temperature_seconds;
                if t24.is_finite() {
                    let dt_days = step_seconds / DAY_SECONDS;
                    self.visible =
                        (self.visible - rate(9.61e-4, 0.0203, t24) * dt_days).clamp(0.0, 1.0);
                    self.near_infrared =
                        (self.near_infrared - rate(1.55e-3, 0.0358, t24) * dt_days).clamp(0.0, 1.0);
                }
            }
        }

        [self.visible, self.near_infrared, self.near_infrared]
    }

    /// Clear age and rolling windows after the column melts out.
    pub fn end_pack(&mut self) {
        self.active = false;
        self.visible = INITIAL_VIS;
        self.near_infrared = INITIAL_NIR;
        self.snowfall.clear();
        self.clear_temperatures();
    }

    fn push_temperature(&mut self, temperature_c: f64, duration: f64) {
        if !temperature_c.is_finite() || duration <= 0.0 {
            return;
        }
        let duration = duration.min(DAY_SECONDS);
        self.temperatures.push_back((temperature_c, duration));
        self.temperature_seconds += duration;
        self.temperature_sum += temperature_c * duration;
        let mut excess = (self.temperature_seconds - DAY_SECONDS).max(0.0);
        while excess > 0.0 {
            let Some((temperature, seconds)) = self.temperatures.pop_front() else {
                break;
            };
            let removed = seconds.min(excess);
            let retained = seconds - removed;
            self.temperature_seconds -= removed;
            self.temperature_sum -= temperature * removed;
            excess -= removed;
            if retained > 0.0 {
                self.temperatures.push_front((temperature, retained));
            }
        }
    }

    fn clear_temperatures(&mut self) {
        self.temperatures.clear();
        self.temperature_seconds = 0.0;
        self.temperature_sum = 0.0;
    }
}

/// Partition shortwave using Aoki surface bands and Vionnet pack extinction.
pub fn shortwave(column: &Column, incoming: f64, band_albedo: [f64; 3]) -> Shortwave {
    albedo::shortwave_with_band_albedo(column, incoming, band_albedo)
}

fn rate(a: f64, b: f64, temperature_c: f64) -> f64 {
    (a * temperature_c + b).max(0.0)
}
