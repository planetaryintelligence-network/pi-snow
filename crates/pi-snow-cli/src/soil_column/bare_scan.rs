//! Registered A4 path scan. It diagnoses roots and never chooses one.
use super::{Bottom, FREEZING_K, LATENT_J_KG, Phase, SoilColumn, Surface};

pub const TAU: f64 = 1e-6;
const EPS: f64 = 1e-10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Segment {
    Frozen,
    Plateau,
    Liquid,
}
#[derive(Debug, Clone)]
pub struct Cell {
    pub segment: Segment,
    pub lo: f64,
    pub hi: f64,
}
#[derive(Debug, Clone)]
pub struct Root {
    pub cells: Vec<Cell>,
}
#[derive(Debug, Clone)]
pub struct Node {
    pub coordinate: f64,
    pub budget: f64,
    pub flux: Option<f64>,
    pub sign: Option<i8>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Agreement {
    Agree,
    OutsideRootCells,
    FailedWithRoots,
    ConvergedWithoutRoots,
}
#[derive(Debug, Clone)]
pub struct RootScan {
    pub nodes: [Vec<Node>; 3],
    pub roots: Vec<Root>,
    pub jump_crossings: [bool; 2],
    pub resolve_solves: usize,
    pub agreement: Option<Agreement>,
}
#[derive(Debug, Clone)]
pub struct Certification {
    pub failed: [bool; 3],
    pub jumps: [f64; 2],
    pub plateau_budget: Vec<f64>,
    pub corner_budget: [f64; 2],
}

pub fn sign(residual: f64) -> i8 {
    if residual > TAU {
        1
    } else if residual < -TAU {
        -1
    } else {
        0
    }
}

pub fn certify(
    balance: &impl Fn(Phase) -> f64,
    maximum_slope: f64,
    lower: f64,
    width: f64,
) -> Result<Certification, String> {
    let plateau_budget: Vec<f64> = (0..=100)
        .map(|j| {
            balance(Phase {
                temperature_k: FREEZING_K,
                liquid_fraction: j as f64 / 100.0,
                dtemperature_denthalpy: 0.0,
            })
        })
        .collect();
    let corner_budget = [
        balance(super::bare::pure_phase(FREEZING_K - EPS)),
        balance(super::bare::pure_phase(FREEZING_K + EPS)),
    ];
    if plateau_budget
        .iter()
        .chain(corner_budget.iter())
        .any(|x| !x.is_finite())
    {
        return Err("nonfinite bare-soil certification budget".into());
    }
    let jumps = [
        plateau_budget[0] - corner_budget[0],
        corner_budget[1] - plateau_budget[100],
    ];
    let max_plateau_slope = plateau_budget
        .windows(2)
        .map(|p| (p[1] - p[0]) / 0.01)
        .fold(f64::NEG_INFINITY, f64::max);
    Ok(Certification {
        failed: [
            maximum_slope >= lower,
            max_plateau_slope >= width,
            jumps.iter().any(|j| j.abs() > TAU),
        ],
        jumps,
        plateau_budget,
        corner_budget,
    })
}

pub fn dirichlet(entry: &SoilColumn, dt: f64, bottom: Bottom, t: f64) -> Result<f64, String> {
    let mut column = entry.clone();
    Ok(column
        .step(dt, Surface::Temperature(t), bottom)?
        .top_flux_w_m2)
}

fn set_flux(node: &mut Node, flux: f64) {
    node.flux = Some(flux);
    node.sign = Some(sign(flux - node.budget));
}

fn resolve(
    nodes: &mut [Node],
    a: usize,
    b: usize,
    lower: f64,
    upper: f64,
    solve: &mut impl FnMut(f64) -> Result<f64, String>,
    spent: &mut usize,
) -> Result<(), String> {
    if b <= a + 1 {
        return Ok(());
    }
    let (ta, tb) = (nodes[a].coordinate, nodes[b].coordinate);
    let (qa, qb) = (nodes[a].flux.unwrap(), nodes[b].flux.unwrap());
    let mut unknown = false;
    for node in &mut nodes[a + 1..b] {
        if node.sign.is_some() {
            continue;
        }
        let (da, db) = (node.coordinate - ta, tb - node.coordinate);
        let lo = (qa + lower * da).max(qb - upper * db);
        let hi = (qa + upper * da).min(qb - lower * db);
        if lo - node.budget > TAU {
            node.sign = Some(1);
        } else if hi - node.budget < -TAU {
            node.sign = Some(-1);
        } else {
            unknown = true;
        }
    }
    if unknown {
        let m = (a + b) / 2;
        let flux = solve(nodes[m].coordinate)?;
        *spent += 1;
        set_flux(&mut nodes[m], flux);
        resolve(nodes, a, m, lower, upper, solve, spent)?;
        resolve(nodes, m, b, lower, upper, solve, spent)?;
    }
    Ok(())
}

pub fn scan(
    entry: &SoilColumn,
    dt: f64,
    bottom: Bottom,
    balance: &impl Fn(Phase) -> f64,
    lower: f64,
    bracket: [f64; 2],
    certificate: &Certification,
) -> Result<RootScan, String> {
    let qb = dirichlet(entry, dt, bottom, FREEZING_K)?;
    let width = entry.material.water_kg_m3() * LATENT_J_KG * SoilColumn::node_volume(0) / dt;
    let qa = qb - width;
    let mut frozen: Vec<Node> = (0..=631)
        .map(|k| {
            let t = (2100 + k) as f64 / 10.0;
            Node {
                coordinate: t,
                budget: balance(super::bare::pure_phase(t)),
                flux: None,
                sign: None,
            }
        })
        .collect();
    frozen.push(Node {
        coordinate: FREEZING_K,
        budget: certificate.corner_budget[0],
        flux: None,
        sign: None,
    });
    let plateau: Vec<Node> = (0..=100)
        .map(|j| {
            let f = j as f64 / 100.0;
            let mut node = Node {
                coordinate: f,
                budget: certificate.plateau_budget[j],
                flux: None,
                sign: None,
            };
            set_flux(&mut node, qb - (1.0 - f) * width);
            node
        })
        .collect();
    let mut liquid = vec![Node {
        coordinate: FREEZING_K,
        budget: certificate.corner_budget[1],
        flux: None,
        sign: None,
    }];
    liquid.extend((632..=1300).map(|k| {
        let t = (2100 + k) as f64 / 10.0;
        Node {
            coordinate: t,
            budget: balance(super::bare::pure_phase(t)),
            flux: None,
            sign: None,
        }
    }));
    if frozen
        .iter()
        .chain(liquid.iter())
        .any(|n| !n.budget.is_finite())
    {
        return Err("nonfinite bare-soil path scan budget".into());
    }
    set_flux(&mut frozen[0], bracket[0]);
    set_flux(frozen.last_mut().unwrap(), qa);
    set_flux(&mut liquid[0], qb);
    set_flux(liquid.last_mut().unwrap(), bracket[1]);
    let g0 = super::solver::conductances(entry, &entry.enthalpy).0[0];
    let mut spent = 0;
    for (nodes, fraction) in [(&mut frozen, 0.0), (&mut liquid, 1.0)] {
        let upper = entry.material.capacity(fraction) * SoilColumn::node_volume(0) / dt + g0;
        let mut solve = |t| dirichlet(entry, dt, bottom, t);
        let last = nodes.len() - 1;
        resolve(nodes, 0, last, lower, upper, &mut solve, &mut spent)?;
    }
    let nodes = [frozen, plateau, liquid];
    let crossing = |a: &Node, b: &Node| a.sign.unwrap() * b.sign.unwrap() < 0;
    let jump_crossings = [
        crossing(nodes[0].last().unwrap(), &nodes[1][0]),
        crossing(nodes[1].last().unwrap(), &nodes[2][0]),
    ];
    let mut roots: Vec<Root> = Vec::new();
    let mut previous: Option<(Segment, &Node)> = None;
    let mut in_zero_run = false;
    for (segment, entries) in [Segment::Frozen, Segment::Plateau, Segment::Liquid]
        .into_iter()
        .zip(&nodes)
    {
        for node in entries {
            let s = node.sign.unwrap();
            if s == 0 {
                let cell = Cell {
                    segment,
                    lo: node.coordinate,
                    hi: node.coordinate,
                };
                if in_zero_run {
                    roots.last_mut().unwrap().cells.push(cell);
                } else {
                    roots.push(Root { cells: vec![cell] });
                }
            } else if let Some((old_segment, old)) = previous {
                if segment == old_segment && old.sign.unwrap() * s < 0 {
                    roots.push(Root {
                        cells: vec![Cell {
                            segment,
                            lo: old.coordinate,
                            hi: node.coordinate,
                        }],
                    });
                }
            }
            in_zero_run = s == 0;
            previous = Some((segment, node));
        }
    }
    Ok(RootScan {
        nodes,
        roots,
        jump_crossings,
        resolve_solves: spent,
        agreement: None,
    })
}

impl RootScan {
    pub fn assess(&mut self, accepted: Option<Phase>) {
        self.agreement = Some(match accepted {
            None if self.roots.is_empty() => Agreement::Agree,
            None => Agreement::FailedWithRoots,
            Some(_) if self.roots.is_empty() => Agreement::ConvergedWithoutRoots,
            Some(phase) => {
                let contains = self.roots.iter().flat_map(|r| &r.cells).any(|cell| {
                    let coordinate = match cell.segment {
                        Segment::Frozen if phase.liquid_fraction == 0.0 => {
                            Some(phase.temperature_k)
                        }
                        Segment::Liquid if phase.liquid_fraction == 1.0 => {
                            Some(phase.temperature_k)
                        }
                        Segment::Plateau if phase.temperature_k == FREEZING_K => {
                            Some(phase.liquid_fraction)
                        }
                        _ => None,
                    };
                    coordinate.is_some_and(|x| x >= cell.lo && x <= cell.hi)
                });
                if contains {
                    Agreement::Agree
                } else {
                    Agreement::OutsideRootCells
                }
            }
        });
    }
}
