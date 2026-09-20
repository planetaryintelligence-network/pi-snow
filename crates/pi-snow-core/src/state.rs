//! The snow column state.
//!
//! The discretisation is a one-dimensional finite-element mesh: `n+1` **nodes**
//! carry temperature and height, `n` **elements** between them carry mass and
//! microstructure. This split is not an implementation detail, it is visible in
//! the output format: PRO field `0503` is an *element* temperature even though
//! the thermal solve is nodal, and heights in `0501` are element interfaces.
//!
//! Element index 0 is the bottom of the column. Heights are measured vertically
//! from the ground, in metres, and are vertical rather than slope-normal because
//! the target configuration sets `PERP_TO_SLOPE = false`.

use crate::constants::{RHO_ICE, RHO_WATER};

/// Volumetric composition of one element. The four fractions sum to one.
///
/// Soil is carried because the state format has a slot for it, but the target
/// configuration sets `SNP_SOIL = false`, so `soil` is zero in every run this
/// engine is validated on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Composition {
    /// Volumetric ice fraction (1).
    pub ice: f64,
    /// Volumetric liquid water fraction (1).
    pub water: f64,
    /// Volumetric air (void) fraction (1).
    pub air: f64,
    /// Volumetric soil fraction (1).
    pub soil: f64,
}

impl Composition {
    /// Dry snow of a given bulk density, with no liquid water and no soil.
    pub fn dry_snow(density: f64) -> Self {
        let ice = density / RHO_ICE;
        Self { ice, water: 0.0, air: 1.0 - ice, soil: 0.0 }
    }

    /// Bulk density of the mixture, kg m^-3. Air is neglected, as it is
    /// everywhere in snow physics: at 1.1 kg m^-3 in a 0.9 void fraction it
    /// contributes about 1 kg m^-3, below the precision anything here is
    /// reported to.
    pub fn density(&self, soil_density: f64) -> f64 {
        self.ice * RHO_ICE + self.water * RHO_WATER + self.soil * soil_density
    }

    /// Pore volume available to liquid water (1): everything that is not ice or
    /// soil. This is the ceiling on `water`, and a state with `water` above it
    /// is unphysical rather than merely unusual.
    pub fn pore_volume(&self) -> f64 {
        1.0 - self.ice - self.soil
    }

    /// Whether the fractions sum to one within `tol` and none is negative.
    pub fn is_consistent(&self, tol: f64) -> bool {
        let sum = self.ice + self.water + self.air + self.soil;
        (sum - 1.0).abs() <= tol
            && self.ice >= 0.0
            && self.water >= 0.0
            && self.air >= -tol
            && self.soil >= 0.0
    }
}

/// Grain microstructure of one element.
///
/// `dendricity` and `sphericity` are the two shape coordinates the
/// SNOWPACK-lineage metamorphism evolves. Fresh snow starts fully dendritic
/// (`dendricity = 1`); once `dendricity` reaches zero the grain is described by
/// `sphericity` alone and grain growth takes over.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Microstructure {
    /// Dendricity, 1 for fresh dendritic crystals down to 0 for fully
    /// decomposed grains (1).
    pub dendricity: f64,
    /// Sphericity, 0 for fully faceted to 1 for fully rounded (1).
    pub sphericity: f64,
    /// Grain radius, m. Reported in PRO `0512` as a **diameter in mm**.
    pub grain_radius: f64,
    /// Bond radius, m. Reported in PRO `0511` as a **diameter in mm**.
    pub bond_radius: f64,
    /// Coordination number, the mean number of bonds per grain (1). PRO `0510`.
    pub coordination_number: f64,
    /// Marker carrying history that the shape coordinates cannot: melt-freeze,
    /// surface hoar, rain crust, and so on. The SNO format calls it `mk`.
    pub marker: u16,
}

impl Microstructure {
    /// Bond-to-grain radius ratio, the quantity the strength and viscosity
    /// parameterisations depend on. Returns 0 for a zero-radius grain rather
    /// than a NaN, because a fresh element is legitimately constructed before
    /// its radius is set.
    pub fn bond_ratio(&self) -> f64 {
        if self.grain_radius > 0.0 { self.bond_radius / self.grain_radius } else { 0.0 }
    }
}

/// One element: the mass and microstructure between node `i` and node `i+1`.
#[derive(Debug, Clone, PartialEq)]
pub struct Element {
    /// Vertical thickness, m.
    pub thickness: f64,
    /// Volumetric composition.
    pub composition: Composition,
    /// Grain microstructure.
    pub micro: Microstructure,
    /// Age since deposition, days. PRO `0505`.
    pub age_days: f64,
    /// Slope-normal overburden stress at the element's mid-height, Pa.
    /// Negative in compression. PRO `0517` reports it in kPa.
    pub stress: f64,
    /// Rate of change of stress, Pa s^-1. The SNO format calls it `CDot`.
    pub stress_rate: f64,
    /// Accumulated metamorphism indicator the SNO format calls `metamo`.
    pub metamorphism: f64,
    /// Density of the soil grains, kg m^-3. Meaningless when `soil` is zero.
    pub soil_density: f64,
    /// Thermal conductivity of the soil grains, W m^-1 K^-1.
    pub soil_conductivity: f64,
    /// Heat capacity of the soil grains, J kg^-1 K^-1.
    pub soil_heat_capacity: f64,
}

impl Element {
    /// Bulk density, kg m^-3. PRO `0502`.
    pub fn density(&self) -> f64 {
        self.composition.density(self.soil_density)
    }

    /// Mass per unit area, kg m^-2.
    pub fn mass(&self) -> f64 {
        self.density() * self.thickness
    }

    /// Ice mass per unit area, kg m^-2.
    pub fn ice_mass(&self) -> f64 {
        self.composition.ice * RHO_ICE * self.thickness
    }

    /// Liquid water mass per unit area, kg m^-2.
    pub fn water_mass(&self) -> f64 {
        self.composition.water * RHO_WATER * self.thickness
    }
}

/// The snow column at one instant.
///
/// Invariant: `node_temperature.len() == elements.len() + 1` whenever there is
/// at least one element, and `node_temperature.len() == 1` for a bare-ground
/// column. A bare-ground column is the normal starting state for these runs,
/// since they begin from UAC's deployed empty profiles.
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    /// Elements, index 0 at the bottom.
    pub elements: Vec<Element>,
    /// Nodal temperatures, K. Node 0 is the ground surface.
    pub node_temperature: Vec<f64>,
    /// Slope angle, degrees from horizontal.
    pub slope_angle_deg: f64,
    /// Slope azimuth, degrees clockwise from north.
    pub slope_azimuth_deg: f64,
    /// Mass per unit area of surface hoar standing on the surface, kg m^-2.
    /// Surface hoar is tracked separately until it is buried, because before
    /// burial it is not an element.
    pub surface_hoar_mass: f64,
    /// Number of soil elements at the bottom of `elements`. Zero in every
    /// configuration this engine is validated against.
    pub soil_elements: usize,
}

impl Column {
    /// An empty column on bare ground at a uniform temperature.
    pub fn bare_ground(ground_temperature: f64, slope_angle_deg: f64, slope_azimuth_deg: f64) -> Self {
        Self {
            elements: Vec::new(),
            node_temperature: vec![ground_temperature],
            slope_angle_deg,
            slope_azimuth_deg,
            surface_hoar_mass: 0.0,
            soil_elements: 0,
        }
    }

    /// Number of snow elements, excluding any soil elements.
    pub fn snow_element_count(&self) -> usize {
        self.elements.len() - self.soil_elements
    }

    /// Snow height above the ground, m. PRO `0501` and SMET `HS_mod`, which
    /// report it in centimetres.
    pub fn snow_height(&self) -> f64 {
        self.elements[self.soil_elements..].iter().map(|e| e.thickness).sum()
    }

    /// Snow water equivalent, kg m^-2. SMET `SWE`.
    pub fn swe(&self) -> f64 {
        self.elements[self.soil_elements..].iter().map(|e| e.mass()).sum()
    }

    /// Total liquid water in the snow, kg m^-2. SMET `MS_Water`.
    pub fn liquid_water(&self) -> f64 {
        self.elements[self.soil_elements..].iter().map(|e| e.water_mass()).sum()
    }

    /// Height of each element's **upper** interface above the ground, m, bottom
    /// element first. This is what PRO `0501` reports, in centimetres.
    pub fn element_top_heights(&self) -> Vec<f64> {
        let mut h = 0.0;
        self.elements
            .iter()
            .map(|e| {
                h += e.thickness;
                h
            })
            .collect()
    }

    /// Mean of the two bounding nodal temperatures for each element, K. This is
    /// what PRO `0503` reports, in degrees Celsius.
    pub fn element_temperatures(&self) -> Vec<f64> {
        self.node_temperature
            .windows(2)
            .map(|w| 0.5 * (w[0] + w[1]))
            .collect()
    }

    /// Whether the nodal count matches the element count.
    pub fn is_well_formed(&self) -> bool {
        if self.elements.is_empty() {
            self.node_temperature.len() == 1
        } else {
            self.node_temperature.len() == self.elements.len() + 1
        }
    }
}
