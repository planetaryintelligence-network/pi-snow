//! Post-failure bisection for the bare-soil outer flux solve (engine-soil-spinup
//! v1 amendment A9, brought into the tree by engine-soil-spinup-v2 plan
//! section 6 and amendment A1.5).
//!
//! It runs only after the 100-iteration safeguarded Newton loop has finished
//! without accepting, so a row that converges never reaches it and the Newton
//! path is unchanged. It reads no environment variable and writes no file.

use super::bare::Diagnostics;

/// Cap on fallback iterations, counted apart from `newton_iterations`.
pub const MAX_ITERATIONS: usize = 100;
pub const COLLAPSED: &str = "bare-soil bisection fallback failed: bracket collapsed";
pub const CAPPED: &str = "bare-soil bisection fallback failed after 100 iterations";

/// Plain-midpoint bisection on the saved original bracket `[lo0, hi0]`.
///
/// `trial(q, d)` returns the residual `q - balance(phase0(q))` and a value to
/// keep if it is accepted. Acceptance is `|r| <= 1e-6`, unchanged. A collapsed
/// bracket (adjacent doubles, no midpoint strictly inside) and the cap both
/// fail with their fixed messages and record the final bracket bits.
pub(super) fn bisect<T>(
    lo0: f64,
    hi0: f64,
    d: &mut Diagnostics,
    mut trial: impl FnMut(f64, &mut Diagnostics) -> Result<(f64, T), String>,
) -> Result<(f64, f64, T), String> {
    d.fallback_entries += 1;
    let (mut lo, mut hi) = (lo0, hi0);
    let result = (|| {
        for _ in 0..MAX_ITERATIONS {
            let q = 0.5 * (lo + hi);
            if !(q > lo && q < hi) {
                return Err(COLLAPSED.to_string());
            }
            d.fallback_iterations += 1;
            let (r, value) = trial(q, d)?;
            if r.abs() <= 1e-6 {
                return Ok((q, r, value));
            }
            if r > 0.0 {
                hi = q;
            } else {
                lo = q;
            }
        }
        Err(CAPPED.to_string())
    })();
    if result.is_err() {
        d.fallback_failures += 1;
        d.failure_bracket_bits = Some([lo.to_bits(), hi.to_bits()]);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacent_jump_collapses_without_accepting() {
        let mut d = Diagnostics::default();
        let lo = 1.0_f64;
        let hi = f64::from_bits(lo.to_bits() + 1);
        let failure = bisect(lo, hi, &mut d, |_, _| Ok((-1.0, ()))).unwrap_err();
        assert_eq!(failure, COLLAPSED);
        assert_eq!(d.failure_bracket_bits, Some([lo.to_bits(), hi.to_bits()]));
        assert_eq!(
            (
                d.fallback_entries,
                d.fallback_iterations,
                d.fallback_failures
            ),
            (1, 0, 1)
        );
    }

    #[test]
    fn cap_and_trial_failures_are_retained() {
        let mut d = Diagnostics::default();
        let failure = bisect(0.0, 1e100, &mut d, |_, _| Ok((1.0, ()))).unwrap_err();
        assert_eq!(failure, CAPPED);
        assert_eq!((d.fallback_iterations, d.fallback_failures), (100, 1));
        let mut d = Diagnostics::default();
        assert_eq!(
            bisect::<()>(-1.0, 1.0, &mut d, |_, _| Err("inner error".into())).unwrap_err(),
            "inner error"
        );
        assert_eq!(d.fallback_failures, 1);
    }

    #[test]
    fn a_monotone_residual_is_accepted_at_the_root() {
        let mut d = Diagnostics::default();
        let (q, r, _) = bisect(-1000.0, 1000.0, &mut d, |q, _| Ok((q - 12.5, ()))).unwrap();
        assert!(r.abs() <= 1e-6 && (q - 12.5).abs() <= 1e-6);
        assert_eq!((d.fallback_entries, d.fallback_failures), (1, 0));
        assert!(d.fallback_iterations > 20 && d.fallback_iterations < 60);
    }
}
