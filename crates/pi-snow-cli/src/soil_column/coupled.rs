//! One implicit snow/soil system, solved by eliminating snow into the soil row.
use super::{Bottom, SoilColumn, StepReport, Surface};
use pi_snow_core::Column;
use pi_snow_physics::surface::{self, Atmosphere, Budget, SurfaceParameters};
use pi_snow_physics::thermal::{self, Conductivity};

#[derive(Debug, Clone, Copy)]
pub struct CoupledReport {
    pub soil: StepReport,
    pub surface_flux_w_m2: f64,
    pub combined_ledger_residual_j_m2: f64,
}

/// Algebraic Schur elimination preserves the shared node's two storage terms.
/// Reduced rows use series fractions to avoid subtracting huge thin-snow terms.
pub fn linear_step(
    snow: &mut Column,
    soil: &mut SoilColumn,
    dt: f64,
    bottom: Bottom,
    a: f64,
    b: f64,
    sources: &[f64],
    ground_shortwave: f64,
    conductivity: Conductivity,
) -> Result<CoupledReport, String> {
    let n = snow.elements.len();
    if n == 0 || snow.soil_elements != 0 || !snow.is_well_formed() || sources.len() > n {
        return Err("coupled solve requires a separate soil and a nonempty snow column".into());
    }
    if !dt.is_finite()
        || dt <= 0.0
        || !a.is_finite()
        || !b.is_finite()
        || b > 0.0
        || !ground_shortwave.is_finite()
        || sources.iter().any(|x| !x.is_finite())
    {
        return Err("invalid coupled step or boundary".into());
    }
    if (snow.node_temperature[0] - soil.phase(0).temperature_k).abs() > 1e-8 {
        return Err("snow/soil shared node is not synchronized".into());
    }
    let mut capacity = vec![0.0; n + 1];
    let mut load = vec![0.0; n + 1];
    let mut links = vec![0.0; n];
    for (i, element) in snow.elements.iter().enumerate() {
        if !element.thickness.is_finite() || element.thickness <= 0.0 {
            return Err("invalid snow thickness in coupled solve".into());
        }
        links[i] = conductivity.of_density(element.density()) / element.thickness;
        let half = 0.5 * thermal::volumetric_heat_capacity(element) * element.thickness / dt;
        for j in [i, i + 1] {
            capacity[j] += half;
            load[j] +=
                half * snow.node_temperature[j] + 0.5 * sources.get(i).copied().unwrap_or(0.0);
        }
    }
    if links
        .iter()
        .chain(capacity.iter())
        .any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err("invalid snow coefficients in coupled solve".into());
    }
    let mut reduced = capacity.clone();
    reduced[n] -= b;
    load[n] += a;
    for i in (0..n).rev() {
        let fraction = links[i] / (links[i] + reduced[i + 1]);
        reduced[i] += fraction * reduced[i + 1];
        load[i] += fraction * load[i + 1];
    }
    let before = soil.energy_j_m2() + thermal::internal_energy(snow);
    let mut next_soil = soil.clone();
    let soil_report = next_soil.step(
        dt,
        Surface::Robin {
            transfer_w_m2_k: reduced[0],
            air_k: (load[0] + ground_shortwave) / reduced[0],
        },
        bottom,
    )?;
    let mut temperatures = snow.node_temperature.clone();
    temperatures[0] = next_soil.phase(0).temperature_k;
    for i in 1..=n {
        let previous = temperatures[i - 1];
        temperatures[i] =
            previous + (load[i] - reduced[i] * previous) / (links[i - 1] + reduced[i]);
    }
    if temperatures.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite coupled temperature".into());
    }
    let mut next_snow = snow.clone();
    next_snow.node_temperature = temperatures;
    let surface_flux = a + b * next_snow.node_temperature[n];
    let after = next_soil.energy_j_m2() + thermal::internal_energy(&next_snow);
    let residual = after
        - before
        - dt * (surface_flux + sources.iter().sum::<f64>() + ground_shortwave
            - soil_report.bottom_flux_w_m2);
    if !residual.is_finite() {
        return Err("nonfinite coupled ledger".into());
    }
    *soil = next_soil;
    *snow = next_snow;
    Ok(CoupledReport {
        soil: soil_report,
        surface_flux_w_m2: surface_flux,
        combined_ledger_residual_j_m2: residual,
    })
}

/// The native three Picard passes all start from the same full coupled state.
pub fn step(
    snow: &mut Column,
    soil: &mut SoilColumn,
    dt: f64,
    bottom: Bottom,
    atm: &Atmosphere,
    parameters: &SurfaceParameters,
    sources: &[f64],
    ground_shortwave: f64,
    conductivity: Conductivity,
) -> Result<(Budget, CoupledReport), String> {
    step_with_linearisation(
        snow,
        soil,
        dt,
        bottom,
        atm,
        parameters,
        sources,
        ground_shortwave,
        conductivity,
        surface::linearise,
    )
}

/// Preserve the run's existing snow surface-linearisation option.
pub fn step_with_linearisation(
    snow: &mut Column,
    soil: &mut SoilColumn,
    dt: f64,
    bottom: Bottom,
    atm: &Atmosphere,
    parameters: &SurfaceParameters,
    sources: &[f64],
    ground_shortwave: f64,
    conductivity: Conductivity,
    linearise: fn(&Atmosphere, &SurfaceParameters, f64) -> (f64, f64),
) -> Result<(Budget, CoupledReport), String> {
    let entry_snow = snow.clone();
    let entry_soil = soil.clone();
    let mut t_linear = *snow.node_temperature.last().ok_or("no surface node")?;
    let mut final_result = None;
    for _ in 0..3 {
        let (a, b) = linearise(atm, parameters, t_linear);
        let mut trial_snow = entry_snow.clone();
        let mut trial_soil = entry_soil.clone();
        let report = linear_step(
            &mut trial_snow,
            &mut trial_soil,
            dt,
            bottom,
            a,
            b,
            sources,
            ground_shortwave,
            conductivity,
        )?;
        t_linear = *trial_snow.node_temperature.last().unwrap();
        final_result = Some((trial_snow, trial_soil, report));
    }
    let (next_snow, next_soil, report) = final_result.unwrap();
    let mut budget = surface::budget(atm, parameters, t_linear);
    budget.total = report.surface_flux_w_m2;
    *snow = next_snow;
    *soil = next_soil;
    Ok((budget, report))
}
