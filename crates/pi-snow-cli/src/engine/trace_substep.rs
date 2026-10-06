use pi_snow_core::Column;
use pi_snow_physics::surface::Budget;

use crate::diagnostic::DiagnosticMode;

pub(super) fn capture_first_nonfinite(
    first: &mut Option<String>,
    mode: Option<DiagnosticMode>,
    stage: &str,
    column: &Column,
    budget: &Budget,
    deposition_buffer: f64,
) {
    if first.is_some() || !matches!(mode, Some(DiagnosticMode::Trace)) || column.elements.is_empty()
    {
        return;
    }
    if let Some(field) =
        crate::finite::column(column).or_else(|| crate::finite::budget(budget, deposition_buffer))
    {
        *first = Some(format!("{stage}: {field}"));
    }
}
