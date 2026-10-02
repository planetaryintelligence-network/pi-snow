// pi-snow's Phase 2 default physics as a CUDA kernel: one thread advances one column.
//
// A line-by-line port of rust/crates/pi-snow-cli/src/engine.rs (run with no --physics
// flag) and the pi-snow-physics functions it calls. Every expression keeps the Rust
// operation order, because the build turns FMA contraction off (--fmad=false) and the
// lockstep gate compares the two step by step: a reordered sum is a real difference.
// Each function names its Rust original. Constants that the Rust code derives
// (gas constants, per-column trigonometry of the slope, surface pressure) are computed
// on the host by the Rust crates themselves and passed in, so they are bit-identical.
//
// State is struct-of-arrays, element i of column c at [i * B + c]. Soil, stress_rate,
// metamorphism and the Lafaysse history are not carried (plan.md section 3.3).
//
// PI_MIX (the MIX arm) evaluates the transcendental functions of the
// parameterisations in single precision; state and bookkeeping stay double.

#define T_MELT 273.15
#define RHO_ICE 917.0
#define RHO_WATER 1000.0
#define RHO_AIR 1.1
#define C_ICE 2000.0
#define C_WATER 4190.0
#define C_AIR 1005.0
#define LH_FUSION 333500.0
#define LH_SUBLIMATION 2838000.0
#define LH_VAPORISATION (LH_SUBLIMATION - LH_FUSION)
#define STEFAN_BOLTZMANN 5.670374419e-8
#define VON_KARMAN 0.4
#define GRAVITY 9.80665
#define NODATA (-999.0)
#define PI_D 3.14159265358979323846
#define DEG2RAD 0.017453292519943295
#define RAD2DEG 57.29577951308232
#define MIN_MEASUREMENT_HEIGHT 0.5
#define MIN_COS_ZENITH 0.08715574274765817
#define SOLAR_CONSTANT 1360.8
#define MELTED_OUT_ICE 1e-6
#define NEW_SNOW_BOND_RADIUS 0.04e-3
#define MARK_REFROZEN 1u

#ifdef PI_MIX
#define XEXP(x) ((double)expf((float)(x)))
#define XLOG(x) ((double)logf((float)(x)))
#define XPOW(x, y) ((double)powf((float)(x), (float)(y)))
#define XSQRT(x) ((double)sqrtf((float)(x)))
#define XTANH(x) ((double)tanhf((float)(x)))
#define XATAN(x) ((double)atanf((float)(x)))
#define XATAN2(y, x) ((double)atan2f((float)(y), (float)(x)))
#define XACOS(x) ((double)acosf((float)(x)))
#define XSIN(x) ((double)sinf((float)(x)))
#define XCOS(x) ((double)cosf((float)(x)))
#define XTAN(x) ((double)tanf((float)(x)))
#else
#define XEXP(x) exp(x)
#define XLOG(x) log(x)
#define XPOW(x, y) pow(x, y)
#define XSQRT(x) sqrt(x)
#define XTANH(x) tanh(x)
#define XATAN(x) atan(x)
#define XATAN2(y, x) atan2(y, x)
#define XACOS(x) acos(x)
#define XSIN(x) sin(x)
#define XCOS(x) cos(x)
#define XTAN(x) tan(x)
#endif

// Failure codes.
#define FAIL_CRAZY 1
#define FAIL_OVERFLOW 2

// Indices into the run-constant array `k` (host: gpu.rs, RunConst).
enum {
    K_THRESH_RAIN_C, K_HEIGHT_NEW_ELEM, K_GRAIN_RADIUS, K_HEIGHT_WIND, K_HEIGHT_METEO,
    K_Z0, K_MIN_ELEMENT, K_T_CRAZY_MIN, K_T_CRAZY_MAX, K_SCHEME, K_DENSITY_SCHEME,
    K_R_DRY, K_CP_AIR, K_EPSILON, K_EMISSIVITY, K_COUNT
};
// Indices into the per-column parameter array (host: ColumnParams).
enum {
    P_LAT, P_LON, P_SLOPE, P_AZI, P_SOIL_ALBEDO, P_PRESSURE, P_COS_SLOPE, P_SIN_SLOPE,
    P_SIN_PHI, P_COS_PHI, P_STRESS_FACTOR, P_COUNT
};
// Forcing fields, [(r * F_COUNT + f) * B + c]; NaN stands for a missing value.
enum { F_TA, F_RH, F_VW, F_DW, F_ISWR, F_ILWR, F_PSUM, F_TSG, F_COUNT };
// Accumulators, as stepper::Accum::to_array.
enum {
    A_N, A_SECONDS, A_QS, A_QL, A_QR, A_QG0, A_OLWR, A_ILWR, A_LWR_NET, A_OSWR, A_ISWR,
    A_ISWR_H, A_ISWR_DIR, A_ISWR_DIFF, A_QW, A_ALBEDO, A_SNOW, A_RAIN, A_RUNOFF,
    A_SUBLIMATION, A_COUNT
};
enum { L_TA, L_RH, L_VW, L_DW, L_TSG, L_COUNT };
// Daily fields written in grid mode at 12:00 UTC, [(d * D_COUNT + k) * B + c].
enum { D_HS, D_SWE, D_TSS, D_WATER, D_N, D_FAIL, D_COUNT };

#define ROW_ABSENT (-9223372036854775807LL - 1LL)

struct Arrays {
    // elements
    double *thick, *ice, *water, *air, *dend, *sph, *gr, *br, *cn, *age, *stress;
    unsigned int *marker;
    // nodes
    double *T;
    // scratch, node-length
    double *s_start, *s_src, *s_energy, *s_cond, *s_hc, *s_c, *s_d, *s_wbefore;
    // per column
    int *n, *fail, *first_out, *has_prev, *out_ready, *cursor, *maxn;
    long long *prev_t, *steps, *fail_t;
    double *buffer, *acc, *last, *par;
    // forcing rows
    long long *ft;
    double *fv;
    // daily output
    double *daily;
    // Monin-Obukhov passes of the step's final surface budget (G1a listings)
    int *mo_passes;
};

struct Ctx {
    const Arrays *a;
    const double *k;
    int B, c, nmax;
};

#define EL(arr, i) (a->arr[(size_t)(i) * B + c])

// ---- small helpers ---------------------------------------------------------------

__device__ __forceinline__ double clampd(double x, double lo, double hi) {
    // f64::clamp: NaN passes through.
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

__device__ __forceinline__ long long floor_div(long long a, long long b) {
    long long q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) q -= 1;
    return q;
}

__device__ long long days_from_civil(long long y, int m, int d) {
    y -= m <= 2;
    long long era = floor_div(y, 400);
    long long yoe = y - era * 400;
    long long doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    long long doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

// chrono's ordinal(), hour(), minute(), second() of a UTC instant in milliseconds.
__device__ void decompose(long long ms, double *ordinal, double *hour, double *minute,
                          double *second) {
    long long secs = floor_div(ms, 1000);
    long long days = floor_div(secs, 86400);
    long long sod = secs - days * 86400;
    *hour = (double)(sod / 3600);
    *minute = (double)((sod % 3600) / 60);
    *second = (double)(sod % 60);
    long long z = days + 719468;
    long long era = floor_div(z, 146097);
    long long doe = z - era * 146097;
    long long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    long long y = yoe + era * 400;
    long long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    long long mp = (5 * doy + 2) / 153;
    long long m = mp < 10 ? mp + 3 : mp - 9;
    y += (m <= 2);
    *ordinal = (double)(days - days_from_civil(y, 1, 1) + 1);
}

// ---- atmosphere.rs ------------------------------------------------------------------

__device__ double vapour_pressure_ice(double t) {
    return XEXP(9.550426 - 5723.265 / t + 3.53068 * XLOG(t) - 0.00728332 * t);
}

__device__ double vapour_pressure_water(double t) {
    return XEXP(54.842763 - 6763.22 / t - 4.210 * XLOG(t) + 0.000367 * t +
                XTANH(0.0415 * (t - 218.8)) *
                    (53.878 - 1331.22 / t - 9.44523 * XLOG(t) + 0.014025 * t));
}

// ---- solar.rs -----------------------------------------------------------------------

struct Sun { double zenith_deg, azimuth_deg; };

__device__ void spencer(double fdoy, double *decl, double *eot) {
    double g = 2.0 * PI_D * (fdoy - 1.0) / 365.0;
    *decl = 0.006918 - 0.399912 * XCOS(g) + 0.070257 * XSIN(g) - 0.006758 * XCOS(2.0 * g) +
            0.000907 * XSIN(2.0 * g) - 0.002697 * XCOS(3.0 * g) + 0.00148 * XSIN(3.0 * g);
    *eot = (1440.0 / (2.0 * PI_D)) *
           (0.0000075 + 0.001868 * XCOS(g) - 0.032077 * XSIN(g) - 0.014615 * XCOS(2.0 * g) -
            0.040849 * XSIN(2.0 * g));
}

__device__ Sun sun_position(double ordinal, double h, double mi, double s, double lon,
                            double sin_phi, double cos_phi) {
    double utc_hours = h + mi / 60.0 + s / 3600.0;
    double decl, eot;
    spencer(ordinal + utc_hours / 24.0, &decl, &eot);
    double solar_hours = utc_hours + lon / 15.0 + eot / 60.0;
    double hour_angle = (15.0 * (solar_hours - 12.0)) * DEG2RAD;
    double cos_zenith = sin_phi * XSIN(decl) + cos_phi * XCOS(decl) * XCOS(hour_angle);
    double zenith = XACOS(clampd(cos_zenith, -1.0, 1.0));
    double wos = XATAN2(XSIN(hour_angle), XCOS(hour_angle) * sin_phi - XTAN(decl) * cos_phi);
    double az = fmod(wos * RAD2DEG + 180.0, 360.0);
    if (az < 0.0) az = az + 360.0;
    Sun out;
    out.zenith_deg = zenith * RAD2DEG;
    out.azimuth_deg = az;
    return out;
}

__device__ double cos_incidence(Sun sun, double cos_b, double sin_b, double aspect_deg) {
    double z = sun.zenith_deg * DEG2RAD;
    double relative = (sun.azimuth_deg - aspect_deg) * DEG2RAD;
    return XCOS(z) * cos_b + sin_b * XSIN(z) * XCOS(relative);
}

__device__ double erbs_diffuse_fraction(double kt) {
    double k = fmax(kt, 0.0);
    if (k <= 0.22) return 1.0 - 0.09 * k;
    if (k <= 0.8) {
        double k2 = k * k;
        return 0.9511 - 0.1604 * k + 4.388 * k2 - 16.638 * (k * k2) + 12.336 * (k2 * k2);
    }
    return 0.165;
}

struct SlopeSW { double direct_on_slope, sky_diffuse_on_slope, global_on_slope; };

__device__ SlopeSW on_slope(double ghi, double dhi, Sun sun, double cos_b, double sin_b,
                            double aspect_deg, double ground_albedo) {
    dhi = clampd(dhi, 0.0, fmax(ghi, 0.0));
    double direct_horizontal = fmax(ghi - dhi, 0.0);
    double cos_zenith = XCOS(sun.zenith_deg * DEG2RAD);
    double direct = 0.0;
    if (cos_zenith > 0.0 && direct_horizontal > 0.0) {
        direct = direct_horizontal * fmax(cos_incidence(sun, cos_b, sin_b, aspect_deg), 0.0) /
                 fmax(cos_zenith, MIN_COS_ZENITH);
    }
    double c = cos_b;
    double sky = dhi * (1.0 + c) / 2.0;
    double ground = fmax(ghi, 0.0) * ground_albedo * (1.0 - c) / 2.0;
    SlopeSW out;
    out.direct_on_slope = direct;
    out.sky_diffuse_on_slope = sky;
    out.global_on_slope = direct + sky + ground;
    return out;
}

__device__ double eccentricity_factor(double ordinal, double h, double mi, double s) {
    double hours = h + mi / 60.0 + s / 3600.0;
    double b = 2.0 * PI_D * (ordinal - 1.0 + hours / 24.0) / 365.0;
    return 1.00011 + 0.034221 * XCOS(b) + 0.00128 * XSIN(b) + 0.000719 * XCOS(2.0 * b) +
           7.7e-5 * XSIN(2.0 * b);
}

// Returns the diffuse part (the direct part is not used by the engine).
__device__ double split_global_diffuse(double ghi, Sun sun, double ordinal, double h, double mi,
                                       double s) {
    ghi = fmax(ghi, 0.0);
    double cos_z = XCOS(sun.zenith_deg * DEG2RAD);
    if (cos_z <= 0.0 || ghi == 0.0) return ghi;
    double extraterrestrial = SOLAR_CONSTANT * eccentricity_factor(ordinal, h, mi, s) * cos_z;
    double kt = fmin(ghi / extraterrestrial, 1.0);
    return erbs_diffuse_fraction(kt) * ghi;
}

// ---- element helpers (pi-snow-core state.rs) ----------------------------------------

#define DENSITY(i) (EL(ice, i) * RHO_ICE + EL(water, i) * RHO_WATER)
#define MASS(i) (DENSITY(i) * EL(thick, i))
#define ICE_MASS(i) (EL(ice, i) * RHO_ICE * EL(thick, i))
#define WATER_MASS(i) (EL(water, i) * RHO_WATER * EL(thick, i))
// thermal::volumetric_heat_capacity (soil term is an exact zero)
#define VHC(i) (EL(ice, i) * RHO_ICE * C_ICE + EL(water, i) * RHO_WATER * C_WATER + \
                EL(air, i) * RHO_AIR * C_AIR)

// albedo::optical_diameter
__device__ __forceinline__ double optical_diameter(double d, double s, double gr) {
    if (d > 0.0) return 1e-4 * (d + (1.0 - d) * (4.0 - s));
    double gs = 2.0 * gr;
    return gs * s + (1.0 - s) * fmax(4e-4, gs / 2.0);
}

// ---- albedo.rs ----------------------------------------------------------------------

// albedo::shortwave. Returns the broadband albedo; with `per` non-null also writes the
// absorbed shortwave per element into it and the surface part into *surface.
__device__ double shortwave(const Ctx &x, int n, double incoming, double pressure,
                            double *surface, bool fill) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    // surface_properties
    double depth = 0.0, d_sum = 0.0, age_sum = 0.0;
    for (int i = n - 1; i >= 0; --i) {
        double take = fmin(EL(thick, i), 0.03 - depth);
        if (take <= 0.0) break;
        d_sum += optical_diameter(EL(dend, i), EL(sph, i), EL(gr, i)) * take;
        age_sum += EL(age, i) * take;
        depth += take;
    }
    double d_opt = d_sum / depth, age = age_sum / depth;
    double root = XSQRT(d_opt);
    double aging = clampd(pressure / 87000.0, 0.5, 1.0) * 0.2 * age / 60.0;
    double a1 = fmax(fmin(0.92, 0.96 - 1.58 * root) - aging, 0.6);
    double a2 = fmax(0.9 - 15.4 * root, 0.3);
    double dp = fmin(d_opt, 0.0023);
    double a3 = 346.3 * dp - 32.31 * XSQRT(dp) + 0.88;
    double albedo = -0.0;
    albedo = albedo + 0.71 * a1;
    albedo = albedo + 0.21 * a2;
    albedo = albedo + 0.08 * a3;
    if (!fill) return albedo;
    incoming = fmax(incoming, 0.0);
    for (int i = 0; i < n; ++i) EL(s_src, i) = 0.0;
    *surface = 0.08 * (1.0 - a3) * incoming;
    double band_albedo[2] = {a1, a2};
    double fraction[2] = {0.71, 0.21};
    for (int band = 0; band < 2; ++band) {
        double flux = fraction[band] * (1.0 - band_albedo[band]) * incoming;
        for (int i = n - 1; i >= 0; --i) {
            double root_e = XSQRT(optical_diameter(EL(dend, i), EL(sph, i), EL(gr, i)));
            double rho = DENSITY(i);
            double beta = band == 0 ? fmax(0.00192 * rho / root_e, 40.0)
                                    : fmax(0.01098 * rho / root_e, 100.0);
            double transmitted = flux * XEXP(-beta * EL(thick, i));
            EL(s_src, i) += flux - transmitted;
            flux = transmitted;
        }
    }
    return albedo;
}

// ---- turbulence.rs ------------------------------------------------------------------

__device__ double holtslag_de_bruin(double z) {
    const double A = 0.7, Bc = 0.75, C = 5.0, D = 0.35;
    return -(A * z + Bc * (z - C / D) * XEXP(-D * z) + Bc * C / D);
}

__device__ double psi_m(double zeta) {
    if (zeta >= 0.0) return holtslag_de_bruin(zeta);
    double x = XPOW(1.0 - 16.0 * zeta, 0.25);
    return 2.0 * XLOG((1.0 + x) / 2.0) + XLOG((1.0 + x * x) / 2.0) - 2.0 * XATAN(x) +
           PI_D / 2.0;
}

__device__ double psi_h(double zeta) {
    if (zeta >= 0.0) return holtslag_de_bruin(zeta);
    double x = XPOW(1.0 - 16.0 * zeta, 0.25);
    return 2.0 * XLOG((1.0 + x * x) / 2.0);
}

struct Fluxes { double sensible, vapour_mass; int passes; };

__device__ Fluxes fluxes(const double *k, double wind, double z_wind, double z_scalar,
                         double z0, double t_air, double t_surface, double q_air,
                         double q_surface, double pressure) {
    double mean_t = 0.5 * (t_air + t_surface);
    double rho = pressure / (k[K_R_DRY] * mean_t);
    Fluxes f;
    f.sensible = 0.0;
    f.vapour_mass = 0.0;
    f.passes = 0;
    if (wind <= 0.0) return f;
    double cp = k[K_CP_AIR];
    if (k[K_SCHEME] == 0.0) {
        // monin_obukhov
        double dt = t_air - t_surface, dq = q_air - q_surface;
        double inverse_l = 0.0, u_star = 0.0, theta_star = 0.0, q_star = 0.0;
        for (int it = 0; it < 200; ++it) {
            f.passes += 1;
            double dm = XLOG(z_wind / z0) - psi_m(z_wind * inverse_l) + psi_m(z0 * inverse_l);
            double dh = XLOG(z_scalar / z0) - psi_h(z_scalar * inverse_l) + psi_h(z0 * inverse_l);
            u_star = VON_KARMAN * wind / dm;
            theta_star = VON_KARMAN * dt / dh;
            q_star = VON_KARMAN * dq / dh;
            double target = VON_KARMAN * GRAVITY * theta_star / (u_star * u_star * mean_t);
            double next = 0.5 * (inverse_l + target);
            if (fabs(next - inverse_l) <= 1e-10 * fmax(fabs(next), 1e-6)) {
                inverse_l = next;
                break;
            }
            inverse_l = next;
        }
        f.sensible = rho * cp * u_star * theta_star;
        f.vapour_mass = rho * u_star * q_star;
    } else {
        // louis
        const double KAPPA_M = 0.35, KAPPA_H = 0.41, Bc = 4.7, C_M = 7.4, C_H = 5.3;
        double z = z_scalar;
        double lg = XLOG((z + z0) / z0);
        double a_m = KAPPA_M / lg, a_h = KAPPA_H / lg;
        double ri = GRAVITY * z * (t_air - t_surface) / (mean_t * wind * wind);
        double ch;
        if (ri >= 0.0) {
            double q = 1.0 + Bc * ri;
            double fct = 1.0 / (q * q);
            ch = a_h * a_h * fct;
        } else {
            double root = XSQRT(fabs(ri) * (z + z0) / z0);
            ch = a_h * a_h * (1.0 - 2.0 * Bc * ri / (1.0 + 2.0 * a_h * a_h * Bc * C_H * root));
        }
        (void)a_m; (void)C_M;
        f.sensible = rho * cp * ch * wind * (t_air - t_surface);
        f.vapour_mass = rho * ch * wind * (q_air - q_surface);
    }
    return f;
}

// ---- surface.rs ---------------------------------------------------------------------

struct Atm { double t_air, rh, wind, absorbed_sw, ilwr, pressure, rain_rate; };
struct SP { double z_wind, z_scalar, z0; };
struct Budget { double olwr, lwr_net, sensible, latent, rain, vapour_mass, total; int mo_passes; };

__device__ double specific_humidity(const double *k, double e, double p) {
    double eps = k[K_EPSILON];
    return eps * e / (p - (1.0 - eps) * e);
}

__device__ Budget budget(const double *k, const Atm &atm, const SP &p, double t_surface) {
    double em = k[K_EMISSIVITY];
    double ts = fmin(t_surface, T_MELT);
    double ts2 = ts * ts;
    double olwr = em * STEFAN_BOLTZMANN * (ts2 * ts2) + (1.0 - em) * atm.ilwr;
    double lwr_net = atm.ilwr - olwr;
    bool melting = ts >= T_MELT;
    double e_surface = melting ? vapour_pressure_water(ts) : vapour_pressure_ice(ts);
    double e_air = atm.rh * vapour_pressure_water(atm.t_air);
    Fluxes f = fluxes(k, atm.wind, p.z_wind, p.z_scalar, p.z0, atm.t_air, ts,
                      specific_humidity(k, e_air, atm.pressure),
                      specific_humidity(k, e_surface, atm.pressure), atm.pressure);
    double latent_heat = melting ? LH_VAPORISATION : LH_SUBLIMATION;
    Budget b;
    b.olwr = olwr;
    b.lwr_net = lwr_net;
    b.sensible = f.sensible;
    b.latent = latent_heat * f.vapour_mass;
    b.rain = atm.rain_rate * C_WATER * (atm.t_air - ts);
    b.vapour_mass = f.vapour_mass;
    b.mo_passes = f.passes;
    b.total = atm.absorbed_sw + lwr_net + f.sensible + b.latent + b.rain;
    return b;
}

__device__ void linearise(const double *k, const Atm &atm, const SP &p, double t0, double *ra,
                          double *rb) {
    const double DELTA = 0.01;
    double t = fmin(t0, T_MELT - DELTA);
    double f0 = budget(k, atm, p, t).total;
    double f1 = budget(k, atm, p, t + DELTA).total;
    double b = fmin((f1 - f0) / DELTA, 0.0);
    *ra = f0 - b * t;
    *rb = b;
}

// thermal::internal_energy
__device__ double internal_energy(const Ctx &x, int n) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    double sum = -0.0;
    for (int i = 0; i < n; ++i) {
        double t = 0.5 * (EL(T, i) + EL(T, i + 1));
        sum = sum + VHC(i) * EL(thick, i) * t;
    }
    return sum;
}

// surface::step_general with thermal::step_general (Dirichlet ground, linear surface flux,
// element sources in s_src). The assembly and the forward elimination below the top row
// do not depend on the linearisation, so they are done once; every value is the one the
// Rust code recomputes on each of its three passes.
__device__ Budget step_energy(const Ctx &x, int n, const Atm &atm, const SP &p, double tsg,
                              double dt) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    const double *k = x.k;
    int nodes = n + 1, top = n;
    for (int j = 0; j < nodes; ++j) EL(s_start, j) = EL(T, j);
    // Per-element conductance and lumped half capacity.
    for (int i = 0; i < n; ++i) {
        double h = EL(thick, i);
        double rho = DENSITY(i);
        EL(s_cond, i) = 2.22362 * XPOW(rho / 1000.0, 1.885) / h;
        EL(s_hc, i) = 0.5 * VHC(i) * h / dt;
    }
    // Forward elimination for rows 0..top-1. Row 0 is Dirichlet: diag 1, sup 0, rhs tsg.
    // s_c holds c[j], s_d holds d[j].
    EL(s_c, 0) = 0.0 / 1.0;
    EL(s_d, 0) = tsg / 1.0;
    for (int j = 1; j < top; ++j) {
        double diag = -0.0;
        diag = 0.0 + (EL(s_cond, j - 1) + EL(s_hc, j - 1));
        diag += EL(s_cond, j) + EL(s_hc, j);
        double tj = EL(s_start, j);
        double rhs = 0.0 + EL(s_hc, j - 1) * tj;
        rhs += EL(s_hc, j) * tj;
        rhs += 0.5 * EL(s_src, j - 1);
        rhs += 0.5 * EL(s_src, j);
        double sub = -EL(s_cond, j - 1);
        double sup = -EL(s_cond, j);
        double den = diag - sub * EL(s_c, j - 1);
        EL(s_c, j) = sup / den;
        EL(s_d, j) = (rhs - sub * EL(s_d, j - 1)) / den;
    }
    // Top row before its boundary term.
    double diag_top0 = 0.0 + (EL(s_cond, n - 1) + EL(s_hc, n - 1));
    double rhs_top0 = 0.0 + EL(s_hc, n - 1) * EL(s_start, top);
    rhs_top0 += 0.5 * EL(s_src, n - 1);
    double sub_top = -EL(s_cond, n - 1);

    double t_lin = EL(s_start, top);
    double ua = 0.0, ub = 0.0;
    for (int pass = 0; pass < 3; ++pass) {
        linearise(k, atm, p, t_lin, &ua, &ub);
        double rhs = rhs_top0 + ua;
        double diag = diag_top0 - ub;
        double den = diag - sub_top * EL(s_c, top - 1);
        double d_top = (rhs - sub_top * EL(s_d, top - 1)) / den;
        // Back substitution into T.
        EL(T, top) = d_top;
        for (int j = top - 1; j >= 0; --j) EL(T, j) = EL(s_d, j) - EL(s_c, j) * EL(T, j + 1);
        t_lin = EL(T, top);
    }
    Budget report = budget(k, atm, p, t_lin);
    report.total = ua + ub * t_lin;
    return report;
}

// ---- water.rs -----------------------------------------------------------------------

// node_capacities into s_c (reused as scratch), node_energies into s_energy.
__device__ void node_energies(const Ctx &x, int n) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    for (int j = 0; j <= n; ++j) EL(s_c, j) = 0.0;
    for (int i = 0; i < n; ++i) {
        double half = 0.5 * VHC(i) * EL(thick, i);
        EL(s_c, i) += half;
        EL(s_c, i + 1) += half;
    }
    for (int j = 0; j <= n; ++j) EL(s_energy, j) = EL(s_c, j) * (EL(T, j) - T_MELT);
}

__device__ void set_temperatures_from_energies(const Ctx &x, int n) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    int nodes = n == 0 ? 1 : n + 1;
    for (int j = 0; j < nodes; ++j) EL(s_c, j) = 0.0;
    for (int i = 0; i < n; ++i) {
        double half = 0.5 * VHC(i) * EL(thick, i);
        EL(s_c, i) += half;
        EL(s_c, i + 1) += half;
    }
    for (int j = 0; j < nodes; ++j) {
        double kk = EL(s_c, j);
        if (kk > 0.0) EL(T, j) = T_MELT + EL(s_energy, j) / kk;
    }
}

__device__ double holding_capacity(double ti) {
    double theta;
    if (ti <= 0.23) theta = 0.0264 + 0.0099 * (1.0 - ti) / ti;
    else if (ti <= 0.812) theta = 0.08 - 0.1023 * (ti - 0.03);
    else theta = 0.0;
    return clampd(theta, 0.0, fmax(1.0 - ti, 0.0));
}

// Remove element i (shifting the ones above down) and node i + 1.
__device__ void remove_element(const Ctx &x, int &n, int i, bool shift_energy) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    for (int j = i; j < n - 1; ++j) {
        EL(thick, j) = EL(thick, j + 1); EL(ice, j) = EL(ice, j + 1);
        EL(water, j) = EL(water, j + 1); EL(air, j) = EL(air, j + 1);
        EL(dend, j) = EL(dend, j + 1); EL(sph, j) = EL(sph, j + 1);
        EL(gr, j) = EL(gr, j + 1); EL(br, j) = EL(br, j + 1); EL(cn, j) = EL(cn, j + 1);
        EL(age, j) = EL(age, j + 1); EL(stress, j) = EL(stress, j + 1);
        EL(marker, j) = EL(marker, j + 1);
    }
    for (int j = i + 1; j < n; ++j) {
        EL(T, j) = EL(T, j + 1);
        if (shift_energy) EL(s_energy, j) = EL(s_energy, j + 1);
    }
    n -= 1;
}

// water::phase_change; returns the drained mass.
__device__ double phase_change(const Ctx &x, int &n) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    if (n == 0) return 0.0;
    node_energies(x, n);
    // element temperatures before any change
    for (int i = 0; i < n; ++i) EL(s_cond, i) = 0.5 * (EL(T, i) + EL(T, i + 1));
    for (int i = 0; i < n; ++i) {
        double sensible = VHC(i) * EL(thick, i) * (EL(s_cond, i) - T_MELT);
        double ice = ICE_MASS(i), liquid = WATER_MASS(i);
        double melt;
        if (sensible > 0.0 && ice > 0.0) melt = fmin(sensible / LH_FUSION, ice);
        else if (sensible < 0.0 && liquid > 0.0) melt = -(fmin(-sensible / LH_FUSION, liquid));
        else melt = 0.0;
        if (melt == 0.0) continue;
        double latent = LH_FUSION * melt;
        EL(s_energy, i) -= 0.5 * latent;
        EL(s_energy, i + 1) -= 0.5 * latent;
        double new_ice = ice - melt, new_liquid = liquid + melt;
        double thickness = (melt > 0.0 && ice > 0.0) ? EL(thick, i) * new_ice / ice : EL(thick, i);
        double solid_volume = new_ice / RHO_ICE + new_liquid / RHO_WATER;
        thickness = fmax(thickness, solid_volume);
        EL(thick, i) = thickness;
        if (thickness > 0.0) {
            EL(ice, i) = new_ice / (RHO_ICE * thickness);
            EL(water, i) = new_liquid / (RHO_WATER * thickness);
            EL(air, i) = fmax(1.0 - EL(ice, i) - EL(water, i), 0.0);
        }
    }
    // remove_melted_out
    double drained = 0.0;
    int index = n;
    while (index > 0) {
        index -= 1;
        if (ICE_MASS(index) >= MELTED_OUT_ICE) continue;
        double ice = ICE_MASS(index), liquid = WATER_MASS(index);
        double node_energy = EL(s_energy, index + 1);
        remove_element(x, n, index, true);
        EL(s_energy, index) += node_energy;
        if (index == 0) {
            drained += liquid + ice;
            continue;
        }
        int b = index - 1;
        double thickness = EL(thick, b);
        EL(water, b) += liquid / (RHO_WATER * thickness);
        EL(ice, b) += ice / (RHO_ICE * thickness);
        EL(air, b) = fmax(1.0 - EL(ice, b) - EL(water, b), 0.0);
    }
    set_temperatures_from_energies(x, n);
    return drained;
}

// water::cascade_with(Wever2014Printed); returns runoff.
__device__ double cascade(const Ctx &x, int n) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    if (n == 0) return 0.0;
    node_energies(x, n);
    double carry = 0.0;
    for (int i = n - 1; i >= 0; --i) {
        double thickness = EL(thick, i);
        EL(water, i) += carry / (RHO_WATER * thickness);
        double capacity = holding_capacity(EL(ice, i));
        if (EL(water, i) > capacity) {
            double excess = (EL(water, i) - capacity) * RHO_WATER * thickness;
            EL(water, i) = capacity;
            carry = excess;
        } else {
            carry = 0.0;
        }
        EL(air, i) = fmax(1.0 - EL(ice, i) - EL(water, i), 0.0);
    }
    set_temperatures_from_energies(x, n);
    return carry;
}

// ---- settlement.rs ------------------------------------------------------------------

__device__ void settlement(const Ctx &x, int n, double dt, double stress_factor) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    double above = 0.0;
    for (int i = n - 1; i >= 0; --i) {
        double own = MASS(i);
        EL(stress, i) = -stress_factor * (above + 0.5 * own);
        above += own;
    }
    for (int i = 0; i < n; ++i) {
        double temperature = 0.5 * (EL(T, i) + EL(T, i + 1));
        // Br92::viscosity
        double rho = DENSITY(i);
        double f1 = 1.0 / (1.0 + 60.0 * EL(water, i));
        double gs = 2.0 * EL(gr, i);
        double f2 = fmin(XEXP(fmin(0.4e-3, gs - 0.2e-3) / 0.1e-3), 4.0);
        double eta = f1 * f2 * 7.62237e6 * (rho / 250.0) *
                     XEXP(0.1 * (T_MELT - temperature) + 0.023 * rho);
        // settle
        double ratio = exp(EL(stress, i) * dt / eta);
        double solid = EL(ice, i) + EL(water, i);
        ratio = fmax(ratio, solid);
        EL(thick, i) *= ratio;
        EL(ice, i) /= ratio;
        EL(water, i) /= ratio;
        EL(air, i) = fmax(1.0 - EL(ice, i) - EL(water, i), 0.0);
    }
}

// ---- metamorphism.rs ----------------------------------------------------------------

__device__ double marbouty_f(double t) {
    double c = t - T_MELT;
    if (c < -40.0) return 0.0;
    if (c < -22.0) return 0.011 * (c + 40.0);
    if (c < -6.0) return 0.2 + 0.05 * (c + 22.0);
    return 1.0 - 0.05 * c;
}
__device__ double marbouty_h(double rho) {
    if (rho < 150.0) return 1.0;
    if (rho < 400.0) return 1.0 - 0.004 * (rho - 150.0);
    return 0.0;
}
__device__ double marbouty_g(double g) {
    if (g < 15.0) return 0.0;
    if (g < 25.0) return 0.01 * (g - 15.0);
    if (g < 40.0) return 0.1 + 0.037 * (g - 25.0);
    if (g < 50.0) return 0.65 + 0.02 * (g - 40.0);
    if (g < 70.0) return 0.85 + 0.0075 * (g - 50.0);
    return 1.0;
}

__device__ void metamorphism(const Ctx &x, int n, double dt) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    double days = dt / 86400.0;
    for (int i = 0; i < n; ++i) {
        double t = 0.5 * (EL(T, i) + EL(T, i + 1));
        double gradient = fabs(EL(T, i + 1) - EL(T, i)) / fmax(EL(thick, i), 1e-6);
        double mass = MASS(i);
        double lwc = mass > 0.0 ? 100.0 * WATER_MASS(i) / mass : 0.0;
        double density = DENSITY(i);
        double d = EL(dend, i), s = EL(sph, i);
        double arrhenius = XEXP(-6000.0 / t);
        double g = fabs(gradient);
        if (lwc > 0.0) {
            double rate = lwc * (lwc * lwc) / 16.0 * days;
            if (d > 0.0) { d -= rate; s += rate; }
            else if (s < 1.0) { s += rate; }
        } else if (d > 0.0) {
            if (g <= 5.0) {
                d -= 2e8 * arrhenius * days;
                s += 1e9 * arrhenius * days;
            } else {
                double rate = 2e8 * arrhenius * XPOW(g, 0.4) * days;
                d -= rate;
                s -= rate;
            }
        } else if (g <= 5.0) {
            s += 1e9 * arrhenius * days;
        } else if (s > 0.0 || g <= 15.0) {
            s -= 2e8 * arrhenius * XPOW(g, 0.4) * days;
        } else {
            double growth = marbouty_f(t) * marbouty_h(density) * marbouty_g(g) * 1.0417e-9 * dt;
            EL(gr, i) += 0.5 * growth;
        }
        EL(dend, i) = clampd(d, 0.0, 1.0);
        EL(sph, i) = clampd(s, 0.0, 1.0);
        EL(age, i) += dt / 86400.0;
    }
}

// ---- new_snow.rs --------------------------------------------------------------------

__device__ double new_snow_density(const double *k, double t_air, double rh, double wind) {
    double ta = t_air - T_MELT;
    if (k[K_DENSITY_SCHEME] == 0.0) {
        return fmax(109.0 + 6.0 * ta + 26.0 * sqrt(fmax(wind, 0.0)), 50.0);
    }
    double rh_percent = 100.0 * rh;
    double base = 3.27820894 + 0.03132918 * ta -
                  0.74783337 * asin(sqrt(clampd(rh_percent / 100.0, 0.0, 1.0))) +
                  0.30943402 * log10(fmax(wind, 2.0));
    return pow(10.0, ta < -14.0 ? base : base - 0.35976206);
}

// Deposition::add; returns the deposited mass, or -1 on overflow.
__device__ double deposit(const Ctx &x, int &n, double solid, double t_air, double rh,
                          double wind) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    const double *k = x.k;
    double buffer = a->buffer[c] + fmax(solid, 0.0);
    a->buffer[c] = buffer;
    double density = new_snow_density(k, t_air, rh, wind);
    double element_mass = k[K_HEIGHT_NEW_ELEM] * density;
    double count_f = floor((buffer + 1e-12) / element_mass);
    if (!(count_f >= 1.0)) return 0.0;
    long long count = (long long)count_f;
    if (n + count > x.nmax) return -1.0;
    double node_t = fmin(t_air, T_MELT);
    double ice = density / RHO_ICE;
    for (long long q = 0; q < count; ++q) {
        int i = n;
        EL(thick, i) = k[K_HEIGHT_NEW_ELEM];
        EL(ice, i) = ice;
        EL(water, i) = 0.0;
        EL(air, i) = 1.0 - ice;
        EL(dend, i) = 1.0;
        EL(sph, i) = 0.5;
        EL(gr, i) = k[K_GRAIN_RADIUS];
        EL(br, i) = NEW_SNOW_BOND_RADIUS;
        EL(cn, i) = 1.0;
        EL(marker, i) = 0u;
        EL(age, i) = 0.0;
        EL(stress, i) = 0.0;
        EL(T, i + 1) = node_t;
        n += 1;
    }
    double deposited = (double)count * element_mass;
    a->buffer[c] = fmax(buffer - deposited, 0.0);
    return deposited;
}

// new_snow::merge_thin_elements with combine.
__device__ void merge_thin(const Ctx &x, int &n, double min_length) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    if (n < 2) return;
    node_energies(x, n);
    int i = 0;
    while (i < n && n >= 2) {
        if (EL(thick, i) >= min_length) { i += 1; continue; }
        int lower = i > 0 ? i - 1 : 0;
        int upper = i > 0 ? i : 1;
        int shared = lower + 1;
        double ha = EL(thick, lower), hb = EL(thick, upper);
        double h = ha + hb;
        double ma = MASS(lower), mb = MASS(upper);
        double msum = ma + mb;
        bool has_mass = msum > 0.0;
#define VW(f) ((EL(f, lower) * ha + EL(f, upper) * hb) / h)
#define MW(f) (has_mass ? (EL(f, lower) * ma + EL(f, upper) * mb) / msum \
                        : 0.5 * (EL(f, lower) + EL(f, upper)))
        double n_ice = VW(ice), n_water = VW(water), n_air = VW(air);
        double n_dend = MW(dend), n_sph = MW(sph), n_gr = MW(gr), n_br = MW(br), n_cn = MW(cn);
        double n_age = MW(age);
        unsigned int n_marker = max(EL(marker, lower), EL(marker, upper));
#undef VW
#undef MW
        EL(thick, lower) = h;
        EL(ice, lower) = n_ice; EL(water, lower) = n_water; EL(air, lower) = n_air;
        EL(dend, lower) = n_dend; EL(sph, lower) = n_sph; EL(gr, lower) = n_gr;
        EL(br, lower) = n_br; EL(cn, lower) = n_cn; EL(age, lower) = n_age;
        EL(marker, lower) = n_marker;
        // stress stays the lower element's
        double node_energy = EL(s_energy, shared);
        // remove element `upper` and node `shared` (== upper): shift everything above.
        for (int j = upper; j < n - 1; ++j) {
            EL(thick, j) = EL(thick, j + 1); EL(ice, j) = EL(ice, j + 1);
            EL(water, j) = EL(water, j + 1); EL(air, j) = EL(air, j + 1);
            EL(dend, j) = EL(dend, j + 1); EL(sph, j) = EL(sph, j + 1);
            EL(gr, j) = EL(gr, j + 1); EL(br, j) = EL(br, j + 1); EL(cn, j) = EL(cn, j + 1);
            EL(age, j) = EL(age, j + 1); EL(stress, j) = EL(stress, j + 1);
            EL(marker, j) = EL(marker, j + 1);
        }
        for (int j = shared; j < n; ++j) {
            EL(T, j) = EL(T, j + 1);
            EL(s_energy, j) = EL(s_energy, j + 1);
        }
        n -= 1;
        EL(s_energy, lower) += 0.5 * node_energy;
        EL(s_energy, lower + 1) += 0.5 * node_energy;
        i = lower;
    }
    set_temperatures_from_energies(x, n);
}

// ---- the step (engine.rs loop body) -------------------------------------------------

// Returns: 0 no step, 1 skipped, 2 advanced, 3 advanced with output, 4 failed.
__device__ int step_row(const Ctx &x, int &n, long long t, const double *fv, int r) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    const double *k = x.k;
#define FV(f) fv[((size_t)r * F_COUNT + (f)) * B + c]
#define PAR(p) a->par[(size_t)(p) * B + c]
#define ACC(q) a->acc[(size_t)(q) * B + c]
    bool had = a->has_prev[c] != 0;
    long long t_prev = a->prev_t[c];
    a->prev_t[c] = t;
    a->has_prev[c] = 1;
    if (!had) return 0;
    double dt = (double)(t - t_prev);
    if (dt <= 0.0) return 0;
    a->steps[c] += 1;
    double ta = FV(F_TA), rh = FV(F_RH), vw = FV(F_VW), iswr_h = FV(F_ISWR), ilwr = FV(F_ILWR);
    if (isnan(ta) || isnan(rh) || isnan(vw) || isnan(iswr_h) || isnan(ilwr)) return 1;
    double dw = FV(F_DW);
    if (isnan(dw)) dw = NODATA;
    double psum = FV(F_PSUM);
    psum = fmax(isnan(psum) ? 0.0 : psum, 0.0);
    double tsg = FV(F_TSG);
    if (isnan(tsg)) tsg = T_MELT;

    double pressure = PAR(P_PRESSURE);
    double soil_albedo = PAR(P_SOIL_ALBEDO);

    // 1. Radiation on the slope.
    long long mid_ms = t * 1000LL - (long long)(dt * 500.0);
    double ordinal, hh, mi, ss;
    decompose(mid_ms, &ordinal, &hh, &mi, &ss);
    Sun sun = sun_position(ordinal, hh, mi, ss, PAR(P_LON), PAR(P_SIN_PHI), PAR(P_COS_PHI));
    double diffuse_h = split_global_diffuse(iswr_h, sun, ordinal, hh, mi, ss);
    double ground_albedo = n == 0 ? soil_albedo : shortwave(x, n, 1.0, pressure, 0, false);
    SlopeSW sw = on_slope(iswr_h, diffuse_h, sun, PAR(P_COS_SLOPE), PAR(P_SIN_SLOPE),
                          PAR(P_AZI), ground_albedo);
    double iswr = sw.global_on_slope;

    // 2. Precipitation.
    double solid, liquid;
    if (ta < T_MELT + k[K_THRESH_RAIN_C]) { solid = psum; liquid = 0.0; }
    else { solid = 0.0; liquid = psum; }
    double deposited = deposit(x, n, solid, ta, rh, vw);
    if (deposited < 0.0) {
        a->fail[c] = FAIL_OVERFLOW;
        a->fail_t[c] = t;
        return 4;
    }
    ACC(A_SNOW) += deposited;
    ACC(A_RAIN) += liquid;
    double rain_rate = 0.0;
    if (liquid > 0.0) {
        if (n > 0) {
            int tp = n - 1;
            EL(water, tp) += liquid / (RHO_WATER * EL(thick, tp));
            EL(air, tp) = fmax(1.0 - EL(ice, tp) - EL(water, tp), 0.0);
            rain_rate = liquid / dt;
        } else {
            ACC(A_RUNOFF) += liquid;
        }
    }

    double hs = -0.0;
    for (int i = 0; i < n; ++i) hs = hs + EL(thick, i);
    SP sp;
    sp.z_wind = fmax(k[K_HEIGHT_WIND] - hs, MIN_MEASUREMENT_HEIGHT);
    sp.z_scalar = fmax(k[K_HEIGHT_METEO] - hs, MIN_MEASUREMENT_HEIGHT);
    sp.z0 = k[K_Z0];

    Budget bud;
    double albedo_now, qg0;
    if (n == 0) {
        Atm atm = {ta, rh, vw, 0.0, ilwr, pressure, 0.0};
        EL(T, 0) = tsg;
        bud = budget(k, atm, sp, tsg);
        albedo_now = soil_albedo;
        qg0 = 0.0;
    } else {
        // 3. Shortwave through the pack (into s_src).
        double surface_sw = 0.0;
        double alb = shortwave(x, n, iswr, pressure, &surface_sw, true);
        Atm atm = {ta, rh, vw, surface_sw, ilwr, pressure, rain_rate};
        // 4. Energy balance and conduction.
        double before = internal_energy(x, n);
        bud = step_energy(x, n, atm, sp, tsg, dt);
        double sources = -0.0;
        for (int i = 0; i < n; ++i) sources = sources + EL(s_src, i);
        qg0 = (internal_energy(x, n) - before) / dt - bud.total - sources;

        // 5. Phase change and water.
        for (int i = 0; i < n; ++i) EL(s_wbefore, i) = WATER_MASS(i);
        int n_before = n;
        double drained = phase_change(x, n);
        if (n == n_before) {
            for (int i = 0; i < n; ++i) {
                if (WATER_MASS(i) < EL(s_wbefore, i) - 1e-9) EL(marker, i) |= MARK_REFROZEN;
            }
        }
        double runoff = cascade(x, n);
        ACC(A_RUNOFF) += drained + runoff;

        // 6. Sublimation or deposition on the top element.
        double vapour = bud.vapour_mass * dt;
        if (n > 0) {
            int tp = n - 1;
            double ice = fmax(ICE_MASS(tp) + vapour, 0.0);
            double im = ICE_MASS(tp);
            double scale = im > 0.0 ? ice / im : 1.0;
            EL(thick, tp) *= scale;
            if (EL(thick, tp) > 0.0) {
                EL(water, tp) /= scale;
                EL(air, tp) = fmax(1.0 - EL(ice, tp) - EL(water, tp), 0.0);
            }
        }
        ACC(A_SUBLIMATION) += fmax(-vapour, 0.0);

        // 7. Settlement.
        settlement(x, n, dt, PAR(P_STRESS_FACTOR));
        // 8. Metamorphism and ageing.
        metamorphism(x, n, dt);
        // 9. Merge thin elements.
        merge_thin(x, n, k[K_MIN_ELEMENT]);
        albedo_now = alb;
    }

    a->mo_passes[c] = bud.mo_passes;
    int nodes = n + 1;
    for (int j = 0; j < nodes; ++j) {
        double tn = EL(T, j);
        if (tn < k[K_T_CRAZY_MIN] || tn > k[K_T_CRAZY_MAX]) {
            a->fail[c] = FAIL_CRAZY;
            a->fail_t[c] = t;
            return 4;
        }
    }

    // 10. Accumulate.
    ACC(A_N) += 1.0;
    ACC(A_SECONDS) += dt;
    ACC(A_QS) += bud.sensible;
    ACC(A_QL) += bud.latent;
    ACC(A_QR) += bud.rain;
    ACC(A_QG0) += qg0;
    ACC(A_OLWR) += bud.olwr;
    ACC(A_ILWR) += ilwr;
    ACC(A_LWR_NET) += bud.lwr_net;
    ACC(A_OSWR) += albedo_now * iswr;
    ACC(A_ISWR) += iswr;
    ACC(A_ISWR_H) += iswr_h;
    ACC(A_ISWR_DIR) += sw.direct_on_slope;
    ACC(A_ISWR_DIFF) += sw.sky_diffuse_on_slope;
    ACC(A_QW) += (1.0 - albedo_now) * iswr;
    ACC(A_ALBEDO) += albedo_now;
    a->last[(size_t)L_TA * B + c] = ta;
    a->last[(size_t)L_RH * B + c] = rh;
    a->last[(size_t)L_VW * B + c] = vw;
    a->last[(size_t)L_DW * B + c] = dw;
    a->last[(size_t)L_TSG * B + c] = tsg;

    bool on_the_hour = (t % 3600LL) == 0;
    bool output = a->first_out[c] != 0 || on_the_hour;
    if (output) a->first_out[c] = 0;
    return output ? 3 : 2;
#undef FV
#undef PAR
#undef ACC
}

// Advance every column from its cursor through at most `max_rows` rows of the
// resident forcing (`rows` rows in all).
//   mode 0 (hourly output): stop a column right after a step that writes output, with
//          out_ready set; the host formats it and the next launch resets the
//          accumulators before going on.
//   mode 1 (grid): never stop; reset the accumulators at each output step, and at
//          12:00 UTC write the daily fields to slot (t - day0) / 86400.
extern "C" __global__ void pi_snow_advance(const Arrays *arrays, const double *k, int B,
                                           int nmax, int rows, int max_rows, int mode,
                                           long long day0) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= B) return;
    const Arrays *a = arrays;
    Ctx x;
    x.a = a; x.k = k; x.B = B; x.c = c; x.nmax = nmax;
    if (a->out_ready[c]) {
        for (int q = 0; q < A_COUNT; ++q) a->acc[(size_t)q * B + c] = 0.0;
        a->out_ready[c] = 0;
    }
    if (a->fail[c]) return;
    int n = a->n[c];
    int r = a->cursor[c];
    int done = 0;
    while (r < rows && done < max_rows) {
        long long t = a->ft[(size_t)r * B + c];
        if (t == ROW_ABSENT) { r = rows; break; }
        int outcome = step_row(x, n, t, a->fv, r);
        r += 1;
        done += 1;
        if (outcome == 4) break;
        if (n > a->maxn[c]) a->maxn[c] = n;
        if (outcome == 3) {
            if (mode == 0) {
                a->out_ready[c] = 1;
                break;
            }
            for (int q = 0; q < A_COUNT; ++q) a->acc[(size_t)q * B + c] = 0.0;
        }
        if (mode == 1 && outcome >= 2 && (t % 86400LL) == 43200LL) {
            long long day = floor_div(t - day0, 86400LL);
            double hs = -0.0, swe = -0.0, water = -0.0;
            for (int i = 0; i < n; ++i) {
                hs = hs + EL(thick, i);
                swe = swe + MASS(i);
                water = water + WATER_MASS(i);
            }
            double *d = a->daily + (size_t)day * D_COUNT * B;
            d[(size_t)D_HS * B + c] = hs;
            d[(size_t)D_SWE * B + c] = swe;
            d[(size_t)D_TSS * B + c] = EL(T, n);
            d[(size_t)D_WATER * B + c] = water;
            d[(size_t)D_N * B + c] = (double)n;
            d[(size_t)D_FAIL * B + c] = 0.0;
        }
    }
    a->n[c] = n;
    a->cursor[c] = r;
}
