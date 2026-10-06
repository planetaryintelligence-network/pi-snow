//! Grid forcing: the flattened forcing B (`engine_gpu.forcing_extract`) to 10-minute rows.
//!
//! Two rules of the station lane, ported (plan.md section 3.6):
//!
//! * `open_data.hrrr_snowpack.hourly_forcing`: wind as U and V; an hour missing any of
//!   TA, RH, ISWR, ILWR, VW, DW is filled by linear interpolation of TA, RH, ISWR, ILWR, U
//!   and V between the bounding known hours; a missing PSUM is zero.
//! * `open_data.hrrr_snowpack.expand_hourly`: a boundary row at the first hour with PSUM
//!   0, then six 10-minute rows per hour, linearly interpolated, VW and DW from the
//!   interpolated U and V, and each hour's PSUM divided by six.
//!
//! Python's `math.radians`, `math.degrees` and `%` are reproduced exactly; `math.hypot`
//! is CPython's own and may differ from libm's in the last bit, which G1c (a) allows.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Duration, TimeZone, Utc};

use crate::stepper::{ForcingRow, Site};

pub const FIELDS: usize = 7; // TA RH VW DW ISWR ILWR PSUM
pub const HOURS: usize = 4344;
pub const GROUND_TEMPERATURE_K: f64 = 273.15;

pub fn season_start() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2025, 11, 1, 0, 0, 0).unwrap()
}

/// One row of `pixels.tsv`, keeping the strings the SMET and SNO files carry.
#[derive(Debug, Clone)]
pub struct Pixel {
    pub index: usize,
    pub key: String,
    pub row: u32,
    pub column: u32,
    pub latitude: String,
    pub longitude: String,
    pub altitude: String,
    pub slope: String,
    pub azimuth: String,
    pub elevation_m: f64,
}

impl Pixel {
    /// The site as `pi-snow` would read it from this pixel's SMET and SNO.
    pub fn site(&self) -> Result<Site> {
        Ok(Site {
            station_id: self.key.clone(),
            station_name: format!("grid pixel {},{}", self.row, self.column),
            latitude: self.latitude.parse()?,
            longitude: self.longitude.parse()?,
            altitude: self.altitude.parse()?,
            slope_angle: self.slope.parse()?,
            slope_azimuth: self.azimuth.parse()?,
            soil_albedo: 0.09,
        })
    }
}

pub fn read_pixels(path: &Path) -> Result<Vec<Pixel>> {
    let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 10 {
            bail!("{}:{}: expected 10 fields", path.display(), n + 1);
        }
        out.push(Pixel {
            index: f[0].parse()?,
            key: f[1].to_string(),
            row: f[2].parse()?,
            column: f[3].parse()?,
            latitude: f[4].to_string(),
            longitude: f[5].to_string(),
            altitude: f[6].to_string(),
            slope: f[7].to_string(),
            azimuth: f[8].to_string(),
            elevation_m: f[9].parse()?,
        });
        if out.last().unwrap().index != out.len() - 1 {
            bail!("{}:{}: indices must run 0..n in order", path.display(), n + 1);
        }
    }
    Ok(out)
}

/// One pixel's hourly season, `[hour][field]` as float32 from the extract.
pub fn read_hourly(file: &mut File, index: usize) -> Result<Vec<f32>> {
    let bytes = HOURS * FIELDS * 4;
    let mut buf = vec![0u8; bytes];
    file.seek(SeekFrom::Start((index * bytes) as u64))?;
    file.read_exact(&mut buf)?;
    Ok(buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// `hrrr_snowpack.wind_to_vector`.
pub fn wind_to_vector(speed: f64, direction_from: f64) -> (f64, f64) {
    let r = direction_from.to_radians();
    (-speed * r.sin(), -speed * r.cos())
}

/// Python's `x % 360.0` (sign of the divisor, +0.0 for an exact multiple).
fn py_mod_360(x: f64) -> f64 {
    let r = x % 360.0;
    if r == 0.0 {
        0.0
    } else if r < 0.0 {
        r + 360.0
    } else {
        r
    }
}

/// `aorc_snowpack.vector_to_wind`.
pub fn vector_to_wind(u: f64, v: f64) -> (f64, f64) {
    if u == 0.0 && v == 0.0 {
        return (0.0, 0.0);
    }
    (u.hypot(v), py_mod_360(270.0 - v.atan2(u).to_degrees()))
}

/// The filled hourly state: TA, RH, ISWR, ILWR, U, V and PSUM per hour.
#[derive(Debug, Clone)]
pub struct Hourly {
    pub ta: Vec<f64>,
    pub rh: Vec<f64>,
    pub iswr: Vec<f64>,
    pub ilwr: Vec<f64>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub psum: Vec<f64>,
    pub filled_hours: usize,
    pub missing_psum_hours: usize,
}

/// `hrrr_snowpack.hourly_forcing` over the season (float32 values widened to f64).
pub fn hourly_forcing(raw: &[f32]) -> Result<Hourly> {
    let at = |h: usize, f: usize| raw[h * FIELDS + f] as f64;
    let known: Vec<bool> = (0..HOURS)
        .map(|h| [0, 1, 4, 5, 2, 3].iter().all(|&f| at(h, f).is_finite()))
        .collect();
    if !known[0] || !known[HOURS - 1] {
        bail!("forcing is missing at the first or last hour");
    }
    let state = |h: usize| {
        let (u, v) = wind_to_vector(at(h, 2), at(h, 3));
        [at(h, 0), at(h, 1), at(h, 4), at(h, 5), u, v]
    };
    let mut cols: [Vec<f64>; 6] = Default::default();
    let mut psum = Vec::with_capacity(HOURS);
    let (mut filled, mut missing_psum) = (0, 0);
    let mut previous = 0usize;
    for h in 0..HOURS {
        let s = if known[h] {
            previous = h;
            state(h)
        } else {
            let following = (h + 1..HOURS).find(|&k| known[k]).unwrap();
            let weight = (h - previous) as f64 / (following - previous) as f64;
            let (a, b) = (state(previous), state(following));
            filled += 1;
            std::array::from_fn(|i| a[i] + (b[i] - a[i]) * weight)
        };
        for (c, x) in cols.iter_mut().zip(s) {
            c.push(x);
        }
        let p = at(h, 6);
        if !p.is_finite() {
            if h != 0 {
                missing_psum += 1;
            }
            psum.push(0.0);
        } else if p < 0.0 {
            bail!("negative PSUM at hour {h}");
        } else {
            psum.push(p);
        }
    }
    let [ta, rh, iswr, ilwr, u, v] = cols;
    Ok(Hourly { ta, rh, iswr, ilwr, u, v, psum, filled_hours: filled, missing_psum_hours: missing_psum })
}

/// Rows of `expand_hourly` for hourly intervals ending at hours `first_hour..=last_hour`
/// (1-based interval index into the season), plus the boundary row when `first_hour`
/// is 1. Row `k` of hour `i` is `start + i hours - (6 - k) * 10 minutes`.
pub fn expand(h: &Hourly, first_hour: usize, last_hour: usize) -> Vec<ForcingRow> {
    let start = season_start();
    let mut rows = Vec::with_capacity((last_hour + 1 - first_hour) * 6 + 1);
    let row = |t: DateTime<Utc>, ta: f64, rh: f64, u: f64, v: f64, iswr: f64, ilwr: f64, psum: f64| {
        let (vw, dw) = vector_to_wind(u, v);
        ForcingRow {
            t,
            ta: Some(ta),
            rh: Some(rh),
            vw: Some(vw),
            dw: Some(dw),
            iswr: Some(iswr),
            ilwr: Some(ilwr),
            psum: Some(psum),
            tsg: Some(GROUND_TEMPERATURE_K),
        }
    };
    if first_hour == 1 {
        rows.push(row(start, h.ta[0], h.rh[0], h.u[0], h.v[0], h.iswr[0], h.ilwr[0], 0.0));
    }
    for i in first_hour..=last_hour {
        let (l, r) = (i - 1, i);
        let interval_end = start + Duration::hours(i as i64);
        for step in 1..=6 {
            let w = step as f64 / 6.0;
            let lerp = |a: &[f64]| a[l] + (a[r] - a[l]) * w;
            rows.push(row(
                interval_end - Duration::minutes(10 * (6 - step) as i64),
                lerp(&h.ta),
                lerp(&h.rh),
                lerp(&h.u),
                lerp(&h.v),
                lerp(&h.iswr),
                lerp(&h.ilwr),
                h.psum[r] / 6.0,
            ));
        }
    }
    rows
}

/// The pixel's 10-minute forcing as a SMET `pi-snow` reads, shortest round-trip
/// formatting so its parse gives back these exact values.
pub fn smet_text(p: &Pixel, rows: &[ForcingRow]) -> String {
    let mut s = format!(
        "SMET 1.1 ASCII\n[HEADER]\nstation_id = {}\nstation_name = grid pixel {},{}\nlatitude = {}\nlongitude = {}\naltitude = {}\nnodata = -999\ntz = 0\nsource = engine-gpu-v1 forcing B, station-lane 10-minute expansion\nfields = timestamp TA RH TSG VW DW ISWR ILWR PSUM\n[DATA]\n",
        p.key, p.row, p.column, p.latitude, p.longitude, p.altitude
    );
    for r in rows {
        let v = |x: Option<f64>| x.unwrap();
        s.push_str(&format!(
            "{} {} {} {} {} {} {} {} {}\n",
            r.t.format("%Y-%m-%dT%H:%M"),
            v(r.ta),
            v(r.rh),
            v(r.tsg),
            v(r.vw),
            v(r.dw),
            v(r.iswr),
            v(r.ilwr),
            v(r.psum)
        ));
    }
    s
}

/// `grid.snowpack_grid.generate_grid_sno` for this pixel (bare ground).
pub fn sno_text(p: &Pixel) -> String {
    format!(
        "SMET 1.1 ASCII\n[HEADER]\nstation_id = {}\nstation_name = grid pixel {},{}\nlongitude = {}\nlatitude = {}\naltitude = {}\nnodata = -999\ntz = 0\nsource = pi-pod grid empty autumn initialization\nProfileDate = 2025-11-01T00:00\nHS_Last = 0\nSlopeAngle = {}\nSlopeAzi = {}\nnSoilLayerData = 0\nnSnowLayerData = 0\nSoilAlbedo = 0.09\nBareSoil_z0 = 0.200\nCanopyHeight = 0.00\nCanopyLeafAreaIndex = 0.00\nCanopyDirectThroughfall = 1.00\nWindScalingFactor = 1.00\nErosionLevel = 0\nTimeCountDeltaHS = 0.000000\nfields = timestamp Layer_Thick T Vol_Frac_I Vol_Frac_W Vol_Frac_V Vol_Frac_S Rho_S Conduc_S HeatCapac_S rg rb dd sp mk mass_hoar ne CDot metamo\n[DATA]\n",
        p.key, p.row, p.column, p.longitude, p.latitude, p.altitude, p.slope, p.azimuth
    )
}

/// The v3b INI with this pixel's station id and the grid experiment name, so the
/// outputs are `<key>_grid.smet` and `<key>_grid.pro` as Phase 2's scorers expect.
pub fn ini_text(template: &str, key: &str) -> Result<String> {
    let mut out = String::new();
    let mut seen = (0, 0, 0);
    for line in template.lines() {
        let k = line.split('=').next().unwrap_or("").trim().to_ascii_uppercase();
        let new = match k.as_str() {
            "STATION1" => {
                seen.0 += 1;
                format!("STATION1 = {key}")
            }
            "SNOWFILE1" => {
                seen.1 += 1;
                format!("SNOWFILE1 = {key}")
            }
            "EXPERIMENT" => {
                seen.2 += 1;
                "EXPERIMENT = grid".to_string()
            }
            _ => line.to_string(),
        };
        out.push_str(&new);
        out.push('\n');
    }
    if seen != (1, 1, 1) {
        bail!("the INI template must carry STATION1, SNOWFILE1 and EXPERIMENT once each");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_modulo_and_wind_round_trip() {
        assert_eq!(py_mod_360(-90.0), 270.0);
        assert_eq!(py_mod_360(720.0).to_bits(), 0.0f64.to_bits());
        let (u, v) = wind_to_vector(5.0, 225.0);
        let (s, d) = vector_to_wind(u, v);
        assert!((s - 5.0).abs() < 1e-12 && (d - 225.0).abs() < 1e-9);
        assert_eq!(vector_to_wind(0.0, 0.0), (0.0, 0.0));
    }

    #[test]
    fn gap_hours_are_filled_and_psum_divided_by_six() {
        let mut raw = vec![0f32; HOURS * FIELDS];
        for h in 0..HOURS {
            let row = [260.0 + (h % 10) as f32, 0.8, 3.0, 90.0, 100.0, 250.0, 1.2];
            raw[h * FIELDS..(h + 1) * FIELDS].copy_from_slice(&row);
        }
        for f in 0..FIELDS {
            raw[5 * FIELDS + f] = f32::NAN;
        }
        let h = hourly_forcing(&raw).unwrap();
        assert_eq!(h.filled_hours, 1);
        assert_eq!(h.missing_psum_hours, 1);
        assert_eq!(h.psum[5], 0.0);
        assert!((h.ta[5] - 0.5 * (264.0 + 266.0)).abs() < 1e-12);
        let rows = expand(&h, 1, 2);
        assert_eq!(rows.len(), 13);
        assert_eq!(rows[0].psum, Some(0.0));
        assert_eq!(rows[1].t, season_start() + Duration::minutes(10));
        assert_eq!(rows[6].t, season_start() + Duration::hours(1));
        assert!((rows[1].psum.unwrap() - 1.2f32 as f64 / 6.0).abs() == 0.0);
        let later = expand(&h, 3, 3);
        assert_eq!(later[0].t, season_start() + Duration::hours(2) + Duration::minutes(10));
    }
}
