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
// parameterisations in single precision; state and bookkeeping stay Real.

#define C(x) ((Real)(x))
#if defined(PI_F32) || defined(PI_F32K)
using Real = float;
#define T_MELT C(0.0)
#define ABS_TEMP(t) ((t) + C(273.15))
#ifdef PI_F32K
using AccumReal = double;
#else
using AccumReal = Real;
#endif
#define MELTED_OUT_ICE C(1e-4)
#define DEPOSITION_EPSILON C(1e-7)
#define MO_BREAK C(1e-5)
#define MARK_REFROZEN_FRACTION C(1e-3)
#else
using Real = double;
using AccumReal = Real;
#define T_MELT C(273.15)
#define ABS_TEMP(t) (t)
#define MELTED_OUT_ICE C(1e-6)
#define DEPOSITION_EPSILON C(1e-12)
#define MO_BREAK C(1e-10)
#define MARK_REFROZEN_FRACTION C(0.0)
#endif

#define RHO_ICE C(917.0)
#define RHO_WATER C(1000.0)
#define RHO_AIR C(1.1)
#define C_ICE C(2000.0)
#define C_WATER C(4190.0)
#define C_AIR C(1005.0)
#define LH_FUSION C(333500.0)
#define LH_SUBLIMATION C(2838000.0)
#define LH_VAPORISATION (LH_SUBLIMATION - LH_FUSION)
#define STEFAN_BOLTZMANN C(5.670374419e-8)
#define VON_KARMAN C(0.4)
#define GRAVITY C(9.80665)
#define NODATA (-C(999.0))
#define PI_D C(3.14159265358979323846)
#define DEG2RAD C(0.017453292519943295)
#define RAD2DEG C(57.29577951308232)
#define MIN_MEASUREMENT_HEIGHT C(0.5)
#define MIN_COS_ZENITH C(0.08715574274765817)
#define SOLAR_CONSTANT C(1360.8)
#define NEW_SNOW_BOND_RADIUS C(0.04e-3)
#define MARK_REFROZEN 1u
#define MARK_SURFACE_HOAR 2u

#if defined(PI_F32) || defined(PI_F32K)
#define XEXP(x) expf((float)(x))
#define XLOG(x) logf((float)(x))
#define XPOW(x, y) powf((float)(x), (float)(y))
#define XSQRT(x) sqrtf((float)(x))
#define XTANH(x) tanhf((float)(x))
#define XATAN(x) atanf((float)(x))
#define XATAN2(y, x) atan2f((float)(y), (float)(x))
#define XACOS(x) acosf((float)(x))
#define XSIN(x) sinf((float)(x))
#define XCOS(x) cosf((float)(x))
#define XTAN(x) tanf((float)(x))
#define XASIN(x) asinf((float)(x))
#define XLOG10(x) log10f((float)(x))
#define XEXP10(x) powf(10.0f, (float)(x))
#elif defined(PI_MIX)
#define XEXP(x) ((Real)expf((float)(x)))
#define XLOG(x) ((Real)logf((float)(x)))
#define XPOW(x, y) ((Real)powf((float)(x), (float)(y)))
#define XSQRT(x) ((Real)sqrtf((float)(x)))
#define XTANH(x) ((Real)tanhf((float)(x)))
#define XATAN(x) ((Real)atanf((float)(x)))
#define XATAN2(y, x) ((Real)atan2f((float)(y), (float)(x)))
#define XACOS(x) ((Real)acosf((float)(x)))
#define XSIN(x) ((Real)sinf((float)(x)))
#define XCOS(x) ((Real)cosf((float)(x)))
#define XTAN(x) ((Real)tanf((float)(x)))
#define XASIN(x) ((Real)asinf((float)(x)))
#define XLOG10(x) ((Real)log10f((float)(x)))
#define XEXP10(x) ((Real)powf(10.0f, (float)(x)))
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
#define XASIN(x) asin(x)
#define XLOG10(x) log10(x)
#define XEXP10(x) pow(10.0, (x))
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
    Real *thick, *ice, *water, *air, *dend, *sph, *gr, *br, *cn, *age, *stress;
    unsigned int *marker;
    // nodes
    Real *T;
    // scratch, node-length
    Real *s_start, *s_src, *s_energy, *s_cond, *s_hc, *s_c, *s_d, *s_wbefore;
    // per column
    int *n, *fail, *first_out, *has_prev, *out_ready, *cursor, *maxn;
    long long *prev_t, *steps, *fail_t;
    AccumReal *buffer, *acc;
    Real *last, *par;
    // forcing rows
    long long *ft;
    Real *fv;
    // daily output
    Real *daily;
    // Monin-Obukhov passes of the step's final surface budget (G1a listings)
    int *mo_passes;
    unsigned int *tolerance_flags;
    unsigned int *count_change_cause;
};

struct Ctx {
    const Arrays *a;
    const Real *k;
    int B, c, nmax;
};

#define EL(arr, i) (a->arr[(size_t)(i) * B + c])

// ---- small helpers ---------------------------------------------------------------

__device__ __forceinline__ Real clampd(Real x, Real lo, Real hi) {
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
__device__ void decompose(long long ms, Real *ordinal, Real *hour, Real *minute,
                          Real *second) {
    long long secs = floor_div(ms, 1000);
    long long days = floor_div(secs, 86400);
    long long sod = secs - days * 86400;
    *hour = (Real)(sod / 3600);
    *minute = (Real)((sod % 3600) / 60);
    *second = (Real)(sod % 60);
    long long z = days + 719468;
    long long era = floor_div(z, 146097);
    long long doe = z - era * 146097;
    long long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    long long y = yoe + era * 400;
    long long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    long long mp = (5 * doy + 2) / 153;
    long long m = mp < 10 ? mp + 3 : mp - 9;
    y += (m <= 2);
    *ordinal = (Real)(days - days_from_civil(y, 1, 1) + 1);
}

// ---- atmosphere.rs ------------------------------------------------------------------

__device__ Real vapour_pressure_ice(Real t) {
    Real tk = ABS_TEMP(t);
    return XEXP(C(9.550426) - C(5723.265) / tk + C(3.53068) * XLOG(tk) - C(0.00728332) * tk);
}

__device__ Real vapour_pressure_water(Real t) {
    Real tk = ABS_TEMP(t);
    return XEXP(C(54.842763) - C(6763.22) / tk - C(4.210) * XLOG(tk) + C(0.000367) * tk +
                XTANH(C(0.0415) * (tk - C(218.8))) *
                    (C(53.878) - C(1331.22) / tk - C(9.44523) * XLOG(tk) + C(0.014025) * tk));
}

// ---- solar.rs -----------------------------------------------------------------------

struct Sun { Real zenith_deg, azimuth_deg; };

__device__ void spencer(Real fdoy, Real *decl, Real *eot) {
    Real g = C(2.0) * PI_D * (fdoy - C(1.0)) / C(365.0);
    *decl = C(0.006918) - C(0.399912) * XCOS(g) + C(0.070257) * XSIN(g) - C(0.006758) * XCOS(C(2.0) * g) +
            C(0.000907) * XSIN(C(2.0) * g) - C(0.002697) * XCOS(C(3.0) * g) + C(0.00148) * XSIN(C(3.0) * g);
    *eot = (C(1440.0) / (C(2.0) * PI_D)) *
           (C(0.0000075) + C(0.001868) * XCOS(g) - C(0.032077) * XSIN(g) - C(0.014615) * XCOS(C(2.0) * g) -
            C(0.040849) * XSIN(C(2.0) * g));
}

__device__ Sun sun_position(Real ordinal, Real h, Real mi, Real s, Real lon,
                            Real sin_phi, Real cos_phi) {
    Real utc_hours = h + mi / C(60.0) + s / C(3600.0);
    Real decl, eot;
    spencer(ordinal + utc_hours / C(24.0), &decl, &eot);
    Real solar_hours = utc_hours + lon / C(15.0) + eot / C(60.0);
    Real hour_angle = (C(15.0) * (solar_hours - C(12.0))) * DEG2RAD;
    Real cos_zenith = sin_phi * XSIN(decl) + cos_phi * XCOS(decl) * XCOS(hour_angle);
    Real zenith = XACOS(clampd(cos_zenith, -C(1.0), C(1.0)));
    Real wos = XATAN2(XSIN(hour_angle), XCOS(hour_angle) * sin_phi - XTAN(decl) * cos_phi);
    Real az = fmod(wos * RAD2DEG + C(180.0), C(360.0));
    if (az < C(0.0)) az = az + C(360.0);
    Sun out;
    out.zenith_deg = zenith * RAD2DEG;
    out.azimuth_deg = az;
    return out;
}

__device__ Real cos_incidence(Sun sun, Real cos_b, Real sin_b, Real aspect_deg) {
    Real z = sun.zenith_deg * DEG2RAD;
    Real relative = (sun.azimuth_deg - aspect_deg) * DEG2RAD;
    return XCOS(z) * cos_b + sin_b * XSIN(z) * XCOS(relative);
}

__device__ Real erbs_diffuse_fraction(Real kt) {
    Real k = fmax(kt, C(0.0));
    if (k <= C(0.22)) return C(1.0) - C(0.09) * k;
    if (k <= C(0.8)) {
        Real k2 = k * k;
        return C(0.9511) - C(0.1604) * k + C(4.388) * k2 - C(16.638) * (k * k2) + C(12.336) * (k2 * k2);
    }
    return C(0.165);
}

struct SlopeSW { Real direct_on_slope, sky_diffuse_on_slope, global_on_slope; };

__device__ SlopeSW on_slope(Real ghi, Real dhi, Sun sun, Real cos_b, Real sin_b,
                            Real aspect_deg, Real ground_albedo) {
    dhi = clampd(dhi, C(0.0), fmax(ghi, C(0.0)));
    Real direct_horizontal = fmax(ghi - dhi, C(0.0));
    Real cos_zenith = XCOS(sun.zenith_deg * DEG2RAD);
    Real direct = C(0.0);
    if (cos_zenith > C(0.0) && direct_horizontal > C(0.0)) {
        direct = direct_horizontal * fmax(cos_incidence(sun, cos_b, sin_b, aspect_deg), C(0.0)) /
                 fmax(cos_zenith, MIN_COS_ZENITH);
    }
    Real c = cos_b;
    Real sky = dhi * (C(1.0) + c) / C(2.0);
    Real ground = fmax(ghi, C(0.0)) * ground_albedo * (C(1.0) - c) / C(2.0);
    SlopeSW out;
    out.direct_on_slope = direct;
    out.sky_diffuse_on_slope = sky;
    out.global_on_slope = direct + sky + ground;
    return out;
}

__device__ Real eccentricity_factor(Real ordinal, Real h, Real mi, Real s) {
    Real hours = h + mi / C(60.0) + s / C(3600.0);
    Real b = C(2.0) * PI_D * (ordinal - C(1.0) + hours / C(24.0)) / C(365.0);
    return C(1.00011) + C(0.034221) * XCOS(b) + C(0.00128) * XSIN(b) + C(0.000719) * XCOS(C(2.0) * b) +
           C(7.7e-5) * XSIN(C(2.0) * b);
}

// Returns the diffuse part (the direct part is not used by the engine).
__device__ Real split_global_diffuse(Real ghi, Sun sun, Real ordinal, Real h, Real mi,
                                       Real s) {
    ghi = fmax(ghi, C(0.0));
    Real cos_z = XCOS(sun.zenith_deg * DEG2RAD);
    if (cos_z <= C(0.0) || ghi == C(0.0)) return ghi;
    Real extraterrestrial = SOLAR_CONSTANT * eccentricity_factor(ordinal, h, mi, s) * cos_z;
    Real kt = fmin(ghi / extraterrestrial, C(1.0));
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
__device__ __forceinline__ Real optical_diameter(Real d, Real s, Real gr) {
    if (d > C(0.0)) return C(1e-4) * (d + (C(1.0) - d) * (C(4.0) - s));
    Real gs = C(2.0) * gr;
    return gs * s + (C(1.0) - s) * fmax(C(4e-4), gs / C(2.0));
}

// ---- albedo.rs ----------------------------------------------------------------------

// albedo::shortwave. Returns the broadband albedo; with `per` non-null also writes the
// absorbed shortwave per element into it and the surface part into *surface.
__device__ Real shortwave(const Ctx &x, int n, Real incoming, Real pressure,
                            Real *surface, bool fill) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    // surface_properties
    Real depth = C(0.0), d_sum = C(0.0), age_sum = C(0.0);
    for (int i = n - 1; i >= 0; --i) {
        Real take = fmin(EL(thick, i), C(0.03) - depth);
        if (take <= C(0.0)) break;
        d_sum += optical_diameter(EL(dend, i), EL(sph, i), EL(gr, i)) * take;
        age_sum += EL(age, i) * take;
        depth += take;
    }
    Real d_opt = d_sum / depth, age = age_sum / depth;
    Real root = XSQRT(d_opt);
    Real aging = clampd(pressure / C(87000.0), C(0.5), C(1.0)) * C(0.2) * age / C(60.0);
    Real a1 = fmax(fmin(C(0.92), C(0.96) - C(1.58) * root) - aging, C(0.6));
    Real a2 = fmax(C(0.9) - C(15.4) * root, C(0.3));
    Real dp = fmin(d_opt, C(0.0023));
    Real a3 = C(346.3) * dp - C(32.31) * XSQRT(dp) + C(0.88);
    Real albedo = -C(0.0);
    albedo = albedo + C(0.71) * a1;
    albedo = albedo + C(0.21) * a2;
    albedo = albedo + C(0.08) * a3;
    if (!fill) return albedo;
    incoming = fmax(incoming, C(0.0));
    for (int i = 0; i < n; ++i) EL(s_src, i) = C(0.0);
    *surface = C(0.08) * (C(1.0) - a3) * incoming;
    Real band_albedo[2] = {a1, a2};
    Real fraction[2] = {C(0.71), C(0.21)};
    for (int band = 0; band < 2; ++band) {
        Real flux = fraction[band] * (C(1.0) - band_albedo[band]) * incoming;
        for (int i = n - 1; i >= 0; --i) {
            Real root_e = XSQRT(optical_diameter(EL(dend, i), EL(sph, i), EL(gr, i)));
            Real rho = DENSITY(i);
            Real beta = band == 0 ? fmax(C(0.00192) * rho / root_e, C(40.0))
                                    : fmax(C(0.01098) * rho / root_e, C(100.0));
            Real transmitted = flux * XEXP(-beta * EL(thick, i));
            EL(s_src, i) += flux - transmitted;
            flux = transmitted;
        }
    }
    return albedo;
}

// ---- turbulence.rs ------------------------------------------------------------------

__device__ Real holtslag_de_bruin(Real z) {
    const Real A = C(0.7), Bc = C(0.75), C = C(5.0), D = C(0.35);
    return -(A * z + Bc * (z - C / D) * XEXP(-D * z) + Bc * C / D);
}

__device__ Real psi_m(Real zeta) {
    if (zeta >= C(0.0)) return holtslag_de_bruin(zeta);
    Real x = XPOW(C(1.0) - C(16.0) * zeta, C(0.25));
    return C(2.0) * XLOG((C(1.0) + x) / C(2.0)) + XLOG((C(1.0) + x * x) / C(2.0)) - C(2.0) * XATAN(x) +
           PI_D / C(2.0);
}

__device__ Real psi_h(Real zeta) {
    if (zeta >= C(0.0)) return holtslag_de_bruin(zeta);
    Real x = XPOW(C(1.0) - C(16.0) * zeta, C(0.25));
    return C(2.0) * XLOG((C(1.0) + x * x) / C(2.0));
}

struct Fluxes { Real sensible, vapour_mass; int passes; };

__device__ Fluxes fluxes(const Real *k, Real wind, Real z_wind, Real z_scalar,
                         Real z0, Real t_air, Real t_surface, Real q_air,
                         Real q_surface, Real pressure) {
    Real mean_t = ABS_TEMP(C(0.5) * (t_air + t_surface));
    Real rho = pressure / (k[K_R_DRY] * mean_t);
    Fluxes f;
    f.sensible = C(0.0);
    f.vapour_mass = C(0.0);
    f.passes = 0;
    if (wind <= C(0.0)) return f;
    Real cp = k[K_CP_AIR];
    if (k[K_SCHEME] == C(0.0)) {
        // monin_obukhov
        Real dt = t_air - t_surface, dq = q_air - q_surface;
        Real inverse_l = C(0.0), u_star = C(0.0), theta_star = C(0.0), q_star = C(0.0);
        for (int it = 0; it < 200; ++it) {
            f.passes += 1;
            Real dm = XLOG(z_wind / z0) - psi_m(z_wind * inverse_l) + psi_m(z0 * inverse_l);
            Real dh = XLOG(z_scalar / z0) - psi_h(z_scalar * inverse_l) + psi_h(z0 * inverse_l);
            u_star = VON_KARMAN * wind / dm;
            theta_star = VON_KARMAN * dt / dh;
            q_star = VON_KARMAN * dq / dh;
            Real target = VON_KARMAN * GRAVITY * theta_star / (u_star * u_star * mean_t);
            Real next = C(0.5) * (inverse_l + target);
            if (fabs(next - inverse_l) <= MO_BREAK * fmax(fabs(next), C(1e-6))) {
                inverse_l = next;
                break;
            }
            inverse_l = next;
        }
        f.sensible = rho * cp * u_star * theta_star;
        f.vapour_mass = rho * u_star * q_star;
    } else {
        // louis
        const Real KAPPA_M = C(0.35), KAPPA_H = C(0.41), Bc = C(4.7), C_M = C(7.4), C_H = C(5.3);
        Real z = z_scalar;
        Real lg = XLOG((z + z0) / z0);
        Real a_m = KAPPA_M / lg, a_h = KAPPA_H / lg;
        Real ri = GRAVITY * z * (t_air - t_surface) / (mean_t * wind * wind);
        Real ch;
        if (ri >= C(0.0)) {
            Real q = C(1.0) + Bc * ri;
            Real fct = C(1.0) / (q * q);
            ch = a_h * a_h * fct;
        } else {
            Real root = XSQRT(fabs(ri) * (z + z0) / z0);
            ch = a_h * a_h * (C(1.0) - C(2.0) * Bc * ri / (C(1.0) + C(2.0) * a_h * a_h * Bc * C_H * root));
        }
        (void)a_m; (void)C_M;
        f.sensible = rho * cp * ch * wind * (t_air - t_surface);
        f.vapour_mass = rho * ch * wind * (q_air - q_surface);
    }
    return f;
}

// ---- surface.rs ---------------------------------------------------------------------

struct Atm { Real t_air, rh, wind, absorbed_sw, ilwr, pressure, rain_rate; };
struct SP { Real z_wind, z_scalar, z0; };
struct Budget { Real olwr, lwr_net, sensible, latent, rain, vapour_mass, total; int mo_passes; };

__device__ Real specific_humidity(const Real *k, Real e, Real p) {
    Real eps = k[K_EPSILON];
    return eps * e / (p - (C(1.0) - eps) * e);
}

__device__ Budget budget(const Real *k, const Atm &atm, const SP &p, Real t_surface) {
    Real em = k[K_EMISSIVITY];
    Real ts = fmin(t_surface, T_MELT);
    Real ts_abs = ABS_TEMP(ts);
    Real ts2 = ts_abs * ts_abs;
    Real olwr = em * STEFAN_BOLTZMANN * (ts2 * ts2) + (C(1.0) - em) * atm.ilwr;
    Real lwr_net = atm.ilwr - olwr;
    bool melting = ts >= T_MELT;
    Real e_surface = melting ? vapour_pressure_water(ts) : vapour_pressure_ice(ts);
    Real e_air = atm.rh * vapour_pressure_water(atm.t_air);
    Fluxes f = fluxes(k, atm.wind, p.z_wind, p.z_scalar, p.z0, atm.t_air, ts,
                      specific_humidity(k, e_air, atm.pressure),
                      specific_humidity(k, e_surface, atm.pressure), atm.pressure);
    Real latent_heat = melting ? LH_VAPORISATION : LH_SUBLIMATION;
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

__device__ void linearise(const Real *k, const Atm &atm, const SP &p, Real t0, Real *ra,
                          Real *rb) {
    const Real DELTA = C(0.01);
    Real t = fmin(t0, T_MELT - DELTA);
    Real f0 = budget(k, atm, p, t).total;
    Real f1 = budget(k, atm, p, t + DELTA).total;
    Real b = fmin((f1 - f0) / DELTA, C(0.0));
    *ra = f0 - b * t;
    *rb = b;
}

// thermal::internal_energy
__device__ Real internal_energy(const Ctx &x, int n) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    Real sum = -C(0.0);
    for (int i = 0; i < n; ++i) {
        Real t = C(0.5) * (EL(T, i) + EL(T, i + 1));
        sum = sum + VHC(i) * EL(thick, i) * t;
    }
    return sum;
}

// surface::step_general with thermal::step_general (Dirichlet ground, linear surface flux,
// element sources in s_src). The assembly and the forward elimination below the top row
// do not depend on the linearisation, so they are done once; every value is the one the
// Rust code recomputes on each of its three passes.
__device__ Budget step_energy(const Ctx &x, int n, const Atm &atm, const SP &p, Real tsg,
                              Real dt) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    a->tolerance_flags[c] |= 1u << 5;
    const Real *k = x.k;
    int nodes = n + 1, top = n;
    for (int j = 0; j < nodes; ++j) EL(s_start, j) = EL(T, j);
    // Per-element conductance and lumped half capacity.
    for (int i = 0; i < n; ++i) {
        Real h = EL(thick, i);
        Real rho = DENSITY(i);
        EL(s_cond, i) = C(2.22362) * XPOW(rho / C(1000.0), C(1.885)) / h;
        EL(s_hc, i) = C(0.5) * VHC(i) * h / dt;
    }
    // Forward elimination for rows 0 through top-1. Row 0 is Dirichlet with rhs tsg.
    // s_c holds c[j], s_d holds d[j].
    EL(s_c, 0) = C(0.0) / C(1.0);
#if defined(PI_F32) || defined(PI_F32K)
    EL(s_d, 0) = tsg - EL(s_start, 0);
#else
    EL(s_d, 0) = tsg / C(1.0);
#endif
    for (int j = 1; j < top; ++j) {
        Real diag = -C(0.0);
        diag = C(0.0) + (EL(s_cond, j - 1) + EL(s_hc, j - 1));
        diag += EL(s_cond, j) + EL(s_hc, j);
        Real tj = EL(s_start, j);
#if defined(PI_F32) || defined(PI_F32K)
        Real sub = -EL(s_cond, j - 1);
        Real sup = -EL(s_cond, j);
        Real rhs = C(0.0) + EL(s_hc, j - 1) * tj;
        rhs += EL(s_hc, j) * tj;
        rhs += C(0.5) * EL(s_src, j - 1);
        rhs += C(0.5) * EL(s_src, j);
        rhs -= diag * tj + sub * EL(s_start, j - 1) + sup * EL(s_start, j + 1);
#else
        Real rhs = C(0.0) + EL(s_hc, j - 1) * tj;
        rhs += EL(s_hc, j) * tj;
        rhs += C(0.5) * EL(s_src, j - 1);
        rhs += C(0.5) * EL(s_src, j);
        Real sub = -EL(s_cond, j - 1);
        Real sup = -EL(s_cond, j);
#endif
        Real den = diag - sub * EL(s_c, j - 1);
        EL(s_c, j) = sup / den;
        EL(s_d, j) = (rhs - sub * EL(s_d, j - 1)) / den;
    }
    // Top row before its boundary term.
    Real diag_top0 = C(0.0) + (EL(s_cond, n - 1) + EL(s_hc, n - 1));
#if defined(PI_F32) || defined(PI_F32K)
    Real rhs_top0 = C(0.0) + EL(s_hc, n - 1) * EL(s_start, top);
    rhs_top0 += C(0.5) * EL(s_src, n - 1);
    Real sub_top = -EL(s_cond, n - 1);
    rhs_top0 -= diag_top0 * EL(s_start, top) + sub_top * EL(s_start, top - 1);
#else
    Real rhs_top0 = C(0.0) + EL(s_hc, n - 1) * EL(s_start, top);
    rhs_top0 += C(0.5) * EL(s_src, n - 1);
    Real sub_top = -EL(s_cond, n - 1);
#endif

    Real t_lin = EL(s_start, top);
    Real ua = C(0.0), ub = C(0.0);
    for (int pass = 0; pass < 3; ++pass) {
        linearise(k, atm, p, t_lin, &ua, &ub);
        Real rhs = rhs_top0 + ua;
        Real diag = diag_top0 - ub;
#if defined(PI_F32) || defined(PI_F32K)
        rhs += ub * EL(s_start, top);
#endif
        Real den = diag - sub_top * EL(s_c, top - 1);
        Real d_top = (rhs - sub_top * EL(s_d, top - 1)) / den;
#if defined(PI_F32) || defined(PI_F32K)
        // Solve for dT, then add it to the Celsius state.
        EL(s_energy, top) = d_top;
        for (int j = top - 1; j >= 0; --j)
            EL(s_energy, j) = EL(s_d, j) - EL(s_c, j) * EL(s_energy, j + 1);
        for (int j = 0; j <= top; ++j) EL(T, j) = EL(s_start, j) + EL(s_energy, j);
#else
        // Back substitution into T.
        EL(T, top) = d_top;
        for (int j = top - 1; j >= 0; --j) EL(T, j) = EL(s_d, j) - EL(s_c, j) * EL(T, j + 1);
#endif
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
    for (int j = 0; j <= n; ++j) EL(s_c, j) = C(0.0);
    for (int i = 0; i < n; ++i) {
        Real half = C(0.5) * VHC(i) * EL(thick, i);
        EL(s_c, i) += half;
        EL(s_c, i + 1) += half;
    }
    for (int j = 0; j <= n; ++j) EL(s_energy, j) = EL(s_c, j) * (EL(T, j) - T_MELT);
}

__device__ void set_temperatures_from_energies(const Ctx &x, int n) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    int nodes = n == 0 ? 1 : n + 1;
    for (int j = 0; j < nodes; ++j) EL(s_c, j) = C(0.0);
    for (int i = 0; i < n; ++i) {
        Real half = C(0.5) * VHC(i) * EL(thick, i);
        EL(s_c, i) += half;
        EL(s_c, i + 1) += half;
    }
    for (int j = 0; j < nodes; ++j) {
        Real kk = EL(s_c, j);
        if (kk > C(0.0)) EL(T, j) = T_MELT + EL(s_energy, j) / kk;
    }
}

__device__ Real holding_capacity(Real ti) {
    Real theta;
    if (ti <= C(0.23)) theta = C(0.0264) + C(0.0099) * (C(1.0) - ti) / ti;
    else if (ti <= C(0.812)) theta = C(0.08) - C(0.1023) * (ti - C(0.03));
    else theta = C(0.0);
    return clampd(theta, C(0.0), fmax(C(1.0) - ti, C(0.0)));
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
__device__ Real phase_change(const Ctx &x, int &n) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    if (n == 0) return C(0.0);
    node_energies(x, n);
    // element temperatures before any change
    for (int i = 0; i < n; ++i) EL(s_cond, i) = C(0.5) * (EL(T, i) + EL(T, i + 1));
    for (int i = 0; i < n; ++i) {
        Real sensible = VHC(i) * EL(thick, i) * (EL(s_cond, i) - T_MELT);
        Real ice = ICE_MASS(i), liquid = WATER_MASS(i);
        Real melt;
        if (sensible > C(0.0) && ice > C(0.0)) melt = fmin(sensible / LH_FUSION, ice);
        else if (sensible < C(0.0) && liquid > C(0.0)) melt = -(fmin(-sensible / LH_FUSION, liquid));
        else melt = C(0.0);
        if (melt == C(0.0)) continue;
        Real latent = LH_FUSION * melt;
        EL(s_energy, i) -= C(0.5) * latent;
        EL(s_energy, i + 1) -= C(0.5) * latent;
        Real new_ice = ice - melt, new_liquid = liquid + melt;
        Real thickness = (melt > C(0.0) && ice > C(0.0)) ? EL(thick, i) * new_ice / ice : EL(thick, i);
        Real solid_volume = new_ice / RHO_ICE + new_liquid / RHO_WATER;
        thickness = fmax(thickness, solid_volume);
        EL(thick, i) = thickness;
        if (thickness > C(0.0)) {
            EL(ice, i) = new_ice / (RHO_ICE * thickness);
            EL(water, i) = new_liquid / (RHO_WATER * thickness);
            EL(air, i) = fmax(C(1.0) - EL(ice, i) - EL(water, i), C(0.0));
        }
    }
    // remove_melted_out
    Real drained = C(0.0);
    int index = n;
    while (index > 0) {
        index -= 1;
        if (ICE_MASS(index) >= MELTED_OUT_ICE) continue;
        a->tolerance_flags[c] |= 1u << 1;
        if (a->count_change_cause[c] == 0) a->count_change_cause[c] = 3;
        Real ice = ICE_MASS(index), liquid = WATER_MASS(index);
        Real node_energy = EL(s_energy, index + 1);
        remove_element(x, n, index, true);
        EL(s_energy, index) += node_energy;
        if (index == 0) {
            drained += liquid + ice;
            continue;
        }
        int b = index - 1;
        Real thickness = EL(thick, b);
        EL(water, b) += liquid / (RHO_WATER * thickness);
        EL(ice, b) += ice / (RHO_ICE * thickness);
        EL(air, b) = fmax(C(1.0) - EL(ice, b) - EL(water, b), C(0.0));
    }
    set_temperatures_from_energies(x, n);
    return drained;
}

// water::cascade_with(Wever2014Printed); returns runoff.
__device__ Real cascade(const Ctx &x, int n) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    if (n == 0) return C(0.0);
    node_energies(x, n);
    Real carry = C(0.0);
    for (int i = n - 1; i >= 0; --i) {
        Real thickness = EL(thick, i);
        EL(water, i) += carry / (RHO_WATER * thickness);
        Real capacity = holding_capacity(EL(ice, i));
        if (EL(water, i) > capacity) {
            Real excess = (EL(water, i) - capacity) * RHO_WATER * thickness;
            EL(water, i) = capacity;
            carry = excess;
        } else {
            carry = C(0.0);
        }
        EL(air, i) = fmax(C(1.0) - EL(ice, i) - EL(water, i), C(0.0));
    }
    set_temperatures_from_energies(x, n);
    return carry;
}

// ---- settlement.rs ------------------------------------------------------------------

__device__ void settlement(const Ctx &x, int n, Real dt, Real stress_factor) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    Real above = C(0.0);
    for (int i = n - 1; i >= 0; --i) {
        Real own = MASS(i);
        EL(stress, i) = -stress_factor * (above + C(0.5) * own);
        above += own;
    }
    for (int i = 0; i < n; ++i) {
        Real temperature = C(0.5) * (EL(T, i) + EL(T, i + 1));
        // Br92::viscosity
        Real rho = DENSITY(i);
        Real f1 = C(1.0) / (C(1.0) + C(60.0) * EL(water, i));
        Real gs = C(2.0) * EL(gr, i);
        Real f2 = fmin(XEXP(fmin(C(0.4e-3), gs - C(0.2e-3)) / C(0.1e-3)), C(4.0));
        Real eta = f1 * f2 * C(7.62237e6) * (rho / C(250.0)) *
                     XEXP(C(0.1) * (T_MELT - temperature) + C(0.023) * rho);
        // settle
    Real ratio = XEXP(EL(stress, i) * dt / eta);
        Real solid = EL(ice, i) + EL(water, i);
        ratio = fmax(ratio, solid);
        EL(thick, i) *= ratio;
        EL(ice, i) /= ratio;
        EL(water, i) /= ratio;
        EL(air, i) = fmax(C(1.0) - EL(ice, i) - EL(water, i), C(0.0));
    }
}

// ---- metamorphism.rs ----------------------------------------------------------------

__device__ Real marbouty_f(Real t) {
    Real c = t - T_MELT;
    if (c < -C(40.0)) return C(0.0);
    if (c < -C(22.0)) return C(0.011) * (c + C(40.0));
    if (c < -C(6.0)) return C(0.2) + C(0.05) * (c + C(22.0));
    return C(1.0) - C(0.05) * c;
}
__device__ Real marbouty_h(Real rho) {
    if (rho < C(150.0)) return C(1.0);
    if (rho < C(400.0)) return C(1.0) - C(0.004) * (rho - C(150.0));
    return C(0.0);
}
__device__ Real marbouty_g(Real g) {
    if (g < C(15.0)) return C(0.0);
    if (g < C(25.0)) return C(0.01) * (g - C(15.0));
    if (g < C(40.0)) return C(0.1) + C(0.037) * (g - C(25.0));
    if (g < C(50.0)) return C(0.65) + C(0.02) * (g - C(40.0));
    if (g < C(70.0)) return C(0.85) + C(0.0075) * (g - C(50.0));
    return C(1.0);
}

__device__ void metamorphism(const Ctx &x, int n, Real dt) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    Real days = dt / C(86400.0);
    for (int i = 0; i < n; ++i) {
        Real t = C(0.5) * (EL(T, i) + EL(T, i + 1));
        if (EL(thick, i) < C(1e-6)) a->tolerance_flags[c] |= 1u << 4;
        Real gradient = fabs(EL(T, i + 1) - EL(T, i)) / fmax(EL(thick, i), C(1e-6));
        Real mass = MASS(i);
        Real lwc = mass > C(0.0) ? C(100.0) * WATER_MASS(i) / mass : C(0.0);
        Real density = DENSITY(i);
        Real d = EL(dend, i), s = EL(sph, i);
        Real arrhenius = XEXP(-C(6000.0) / ABS_TEMP(t));
        Real g = fabs(gradient);
        if (lwc > C(0.0)) {
            Real rate = lwc * (lwc * lwc) / C(16.0) * days;
            if (d > C(0.0)) { d -= rate; s += rate; }
            else if (s < C(1.0)) { s += rate; }
        } else if (d > C(0.0)) {
            if (g <= C(5.0)) {
                d -= C(2e8) * arrhenius * days;
                s += C(1e9) * arrhenius * days;
            } else {
                Real rate = C(2e8) * arrhenius * XPOW(g, C(0.4)) * days;
                d -= rate;
                s -= rate;
            }
        } else if (g <= C(5.0)) {
            s += C(1e9) * arrhenius * days;
        } else if (s > C(0.0) || g <= C(15.0)) {
            s -= C(2e8) * arrhenius * XPOW(g, C(0.4)) * days;
        } else {
            Real growth = marbouty_f(t) * marbouty_h(density) * marbouty_g(g) * C(1.0417e-9) * dt;
            EL(gr, i) += C(0.5) * growth;
        }
        EL(dend, i) = clampd(d, C(0.0), C(1.0));
        EL(sph, i) = clampd(s, C(0.0), C(1.0));
        EL(age, i) += dt / C(86400.0);
    }
}

// ---- new_snow.rs --------------------------------------------------------------------

__device__ Real new_snow_density(const Real *k, Real t_air, Real rh, Real wind) {
    Real ta = t_air - T_MELT;
    if (k[K_DENSITY_SCHEME] == C(0.0)) {
        return fmax(C(109.0) + C(6.0) * ta + C(26.0) * XSQRT(fmax(wind, C(0.0))), C(50.0));
    }
    Real rh_percent = C(100.0) * rh;
    Real base = C(3.27820894) + C(0.03132918) * ta -
                  C(0.74783337) * XASIN(XSQRT(clampd(rh_percent / C(100.0), C(0.0), C(1.0)))) +
                  C(0.30943402) * XLOG10(fmax(wind, C(2.0)));
    return XEXP10(ta < -C(14.0) ? base : base - C(0.35976206));
}

// Deposition::add; returns the deposited mass, or -1 on overflow.
__device__ Real deposit(const Ctx &x, int &n, Real solid, Real t_air, Real rh,
                          Real wind) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    const Real *k = x.k;
    AccumReal buffer = a->buffer[c] + fmax(solid, C(0.0));
    a->buffer[c] = buffer;
    Real density = new_snow_density(k, t_air, rh, wind);
    Real element_mass = k[K_HEIGHT_NEW_ELEM] * density;
    Real count_f = floor((buffer + DEPOSITION_EPSILON) / element_mass);
    if (count_f > floor(buffer / element_mass)) a->tolerance_flags[c] |= 1u << 0;
    if (!(count_f >= C(1.0))) return C(0.0);
    long long count = (long long)count_f;
    if (n + count > x.nmax) return -C(1.0);
    if (a->count_change_cause[c] == 0) a->count_change_cause[c] = 1;
    Real node_t = fmin(t_air, T_MELT);
    Real ice = density / RHO_ICE;
    for (long long q = 0; q < count; ++q) {
        int i = n;
        EL(thick, i) = k[K_HEIGHT_NEW_ELEM];
        EL(ice, i) = ice;
        EL(water, i) = C(0.0);
        EL(air, i) = C(1.0) - ice;
        EL(dend, i) = C(1.0);
        EL(sph, i) = C(0.5);
        EL(gr, i) = k[K_GRAIN_RADIUS];
        EL(br, i) = NEW_SNOW_BOND_RADIUS;
        EL(cn, i) = C(1.0);
        EL(marker, i) = 0u;
        EL(age, i) = C(0.0);
        EL(stress, i) = C(0.0);
        EL(T, i + 1) = node_t;
        n += 1;
    }
    Real deposited = (Real)count * element_mass;
    AccumReal remainder = buffer - deposited;
    a->buffer[c] = remainder > C(0.0) ? remainder : C(0.0);
    return deposited;
}

// new_snow::merge_thin_elements with combine.
__device__ void merge_thin(const Ctx &x, int &n, Real min_length) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    if (n < 2) return;
    node_energies(x, n);
    int i = 0;
    while (i < n && n >= 2) {
        if (EL(thick, i) >= min_length) { i += 1; continue; }
        int lower = i > 0 ? i - 1 : 0;
        int upper = i > 0 ? i : 1;
        int shared = lower + 1;
        Real ha = EL(thick, lower), hb = EL(thick, upper);
        Real h = ha + hb;
        Real ma = MASS(lower), mb = MASS(upper);
        Real msum = ma + mb;
        bool has_mass = msum > C(0.0);
#define VW(f) ((EL(f, lower) * ha + EL(f, upper) * hb) / h)
#define MW(f) (has_mass ? (EL(f, lower) * ma + EL(f, upper) * mb) / msum \
                        : C(0.5) * (EL(f, lower) + EL(f, upper)))
        Real n_ice = VW(ice), n_water = VW(water), n_air = VW(air);
        Real n_dend = MW(dend), n_sph = MW(sph), n_gr = MW(gr), n_br = MW(br), n_cn = MW(cn);
        Real n_age = MW(age);
        unsigned int n_marker = max(EL(marker, lower), EL(marker, upper));
#undef VW
#undef MW
        EL(thick, lower) = h;
        EL(ice, lower) = n_ice; EL(water, lower) = n_water; EL(air, lower) = n_air;
        EL(dend, lower) = n_dend; EL(sph, lower) = n_sph; EL(gr, lower) = n_gr;
        EL(br, lower) = n_br; EL(cn, lower) = n_cn; EL(age, lower) = n_age;
        EL(marker, lower) = n_marker;
        // stress stays the lower element's
        Real node_energy = EL(s_energy, shared);
        if (a->count_change_cause[c] == 0) a->count_change_cause[c] = 2;
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
        EL(s_energy, lower) += C(0.5) * node_energy;
        EL(s_energy, lower + 1) += C(0.5) * node_energy;
        i = lower;
    }
    set_temperatures_from_energies(x, n);
}

// A6 deep cap: preserve the active top metre, merge only the deeper slice.
__device__ unsigned short a6_class(const Ctx &x, int i) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    if (EL(marker, i) & MARK_REFROZEN) return 772;
    if (EL(water, i) > C(0.0)) return 770;
    if (EL(marker, i) & MARK_SURFACE_HOAR) return 660;
    Real d = EL(dend, i), s = EL(sph, i);
    if (d >= C(0.75)) return 110;
    if (d >= C(0.5)) return 120;
    if (d > C(0.0)) return s >= C(0.5) ? 223 : 224;
    if (s >= C(0.5)) return s >= C(0.75) ? 330 : 334;
    if (C(2000.0) * EL(gr, i) >= C(1.5) && s < C(0.2)) return 550;
    return s >= C(0.3) ? 443 : 440;
}

__device__ void a6_cap_deep(const Ctx &x, int &n) {
    if (n < 2) return;
    const Arrays *a = x.a; int B = x.B, c = x.c;
    int deep = 0;
    Real height = C(0.0);
    Real deep_limit = C(0.0);
    for (int i = 0; i < n; ++i) deep_limit += EL(thick, i);
    deep_limit -= C(1.0);
    height = C(0.0);
    while (deep < n) {
        height += EL(thick, deep);
        if (height <= deep_limit) deep += 1;
        else break;
    }
    while (deep > 50) {
        int best = -1;
        bool have_same = false;
        Real best_thickness = C(0.0);
        for (int i = 0; i < deep - 1; ++i) {
            bool same = a6_class(x, i) == a6_class(x, i + 1);
            if (have_same && !same) continue;
            Real combined = EL(thick, i) + EL(thick, i + 1);
            if (best < 0 || (same && !have_same) ||
                (same == have_same && (combined < best_thickness ||
                 (combined == best_thickness && i > best)))) {
                best = i;
                best_thickness = combined;
                if (same) have_same = true;
            }
        }
        if (best < 0) break;
        int lower = best, upper = best + 1, shared = upper;
        const Arrays *a = x.a;
        int B = x.B, c = x.c;
        Real ha = EL(thick, lower), hb = EL(thick, upper), h = ha + hb;
        Real ma = MASS(lower), mb = MASS(upper), msum = ma + mb;
        bool has_mass = msum > C(0.0);
#define A6VW(f) ((EL(f, lower) * ha + EL(f, upper) * hb) / h)
#define A6MW(f) (has_mass ? (EL(f, lower) * ma + EL(f, upper) * mb) / msum \
                         : C(0.5) * (EL(f, lower) + EL(f, upper)))
        Real ni = A6VW(ice), nw = A6VW(water), na = A6VW(air);
        Real nd = A6MW(dend), ns = A6MW(sph), ng = A6MW(gr), nb = A6MW(br);
        Real nc = A6MW(cn), nage = A6MW(age);
        unsigned int nm = max(EL(marker, lower), EL(marker, upper));
#undef A6VW
#undef A6MW
        Real internal_energy = EL(s_energy, shared);
        EL(thick, lower) = h; EL(ice, lower) = ni; EL(water, lower) = nw;
        EL(air, lower) = na; EL(dend, lower) = nd; EL(sph, lower) = ns;
        EL(gr, lower) = ng; EL(br, lower) = nb; EL(cn, lower) = nc;
        EL(age, lower) = nage; EL(marker, lower) = nm;
        for (int j = upper; j < n - 1; ++j) {
            EL(thick, j) = EL(thick, j + 1); EL(ice, j) = EL(ice, j + 1);
            EL(water, j) = EL(water, j + 1); EL(air, j) = EL(air, j + 1);
            EL(dend, j) = EL(dend, j + 1); EL(sph, j) = EL(sph, j + 1);
            EL(gr, j) = EL(gr, j + 1); EL(br, j) = EL(br, j + 1);
            EL(cn, j) = EL(cn, j + 1); EL(age, j) = EL(age, j + 1);
            EL(stress, j) = EL(stress, j + 1); EL(marker, j) = EL(marker, j + 1);
        }
        for (int j = shared; j < n; ++j) {
            EL(T, j) = EL(T, j + 1);
            EL(s_energy, j) = EL(s_energy, j + 1);
        }
        n -= 1;
        EL(s_energy, lower) += C(0.5) * internal_energy;
        EL(s_energy, lower + 1) += C(0.5) * internal_energy;
        deep -= 1;
        set_temperatures_from_energies(x, n);
    }
}

// ---- the step (engine.rs loop body) -------------------------------------------------

// Returns: 0 no step, 1 skipped, 2 advanced, 3 advanced with output, 4 failed.
__device__ int step_row(const Ctx &x, int &n, long long t, const Real *fv, int r) {
    const Arrays *a = x.a; int B = x.B, c = x.c;
    const Real *k = x.k;
#define FV(f) fv[((size_t)r * F_COUNT + (f)) * B + c]
#define PAR(p) a->par[(size_t)(p) * B + c]
#define ACC(q) a->acc[(size_t)(q) * B + c]
    a->mo_passes[c] = 0;
    a->tolerance_flags[c] = 0;
    a->count_change_cause[c] = 0;
    bool had = a->has_prev[c] != 0;
    long long t_prev = a->prev_t[c];
    a->prev_t[c] = t;
    a->has_prev[c] = 1;
    if (!had) return 0;
    Real dt = (Real)(t - t_prev);
    if (dt <= C(0.0)) return 0;
    a->steps[c] += 1;
    Real ta = FV(F_TA), rh = FV(F_RH), vw = FV(F_VW), iswr_h = FV(F_ISWR), ilwr = FV(F_ILWR);
    if (isnan(ta) || isnan(rh) || isnan(vw) || isnan(iswr_h) || isnan(ilwr)) return 1;
    Real dw = FV(F_DW);
    if (isnan(dw)) dw = NODATA;
    Real psum = FV(F_PSUM);
    psum = fmax(isnan(psum) ? C(0.0) : psum, C(0.0));
    Real tsg = FV(F_TSG);
    if (isnan(tsg)) tsg = T_MELT;

    Real pressure = PAR(P_PRESSURE);
    Real soil_albedo = PAR(P_SOIL_ALBEDO);

    // 1. Radiation on the slope.
    long long mid_ms = t * 1000LL - (long long)(dt * C(500.0));
    Real ordinal, hh, mi, ss;
    decompose(mid_ms, &ordinal, &hh, &mi, &ss);
    Sun sun = sun_position(ordinal, hh, mi, ss, PAR(P_LON), PAR(P_SIN_PHI), PAR(P_COS_PHI));
    Real diffuse_h = split_global_diffuse(iswr_h, sun, ordinal, hh, mi, ss);
    Real ground_albedo = n == 0 ? soil_albedo : shortwave(x, n, C(1.0), pressure, 0, false);
    SlopeSW sw = on_slope(iswr_h, diffuse_h, sun, PAR(P_COS_SLOPE), PAR(P_SIN_SLOPE),
                          PAR(P_AZI), ground_albedo);
    Real iswr = sw.global_on_slope;

    // 2. Precipitation.
    Real solid, liquid;
    if (ta < T_MELT + k[K_THRESH_RAIN_C]) { solid = psum; liquid = C(0.0); }
    else { solid = C(0.0); liquid = psum; }
    Real deposited = deposit(x, n, solid, ta, rh, vw);
    if (deposited < C(0.0)) {
        a->fail[c] = FAIL_OVERFLOW;
        a->fail_t[c] = t;
        return 4;
    }
    ACC(A_SNOW) += deposited;
    ACC(A_RAIN) += liquid;
    Real rain_rate = C(0.0);
    if (liquid > C(0.0)) {
        if (n > 0) {
            int tp = n - 1;
            EL(water, tp) += liquid / (RHO_WATER * EL(thick, tp));
            EL(air, tp) = fmax(C(1.0) - EL(ice, tp) - EL(water, tp), C(0.0));
            rain_rate = liquid / dt;
        } else {
            ACC(A_RUNOFF) += liquid;
        }
    }

    Real hs = -C(0.0);
    for (int i = 0; i < n; ++i) hs = hs + EL(thick, i);
    SP sp;
    sp.z_wind = fmax(k[K_HEIGHT_WIND] - hs, MIN_MEASUREMENT_HEIGHT);
    sp.z_scalar = fmax(k[K_HEIGHT_METEO] - hs, MIN_MEASUREMENT_HEIGHT);
    sp.z0 = k[K_Z0];

    Budget bud;
    Real albedo_now, qg0;
    if (n == 0) {
        Atm atm = {ta, rh, vw, C(0.0), ilwr, pressure, C(0.0)};
        EL(T, 0) = tsg;
        bud = budget(k, atm, sp, tsg);
        albedo_now = soil_albedo;
        qg0 = C(0.0);
    } else {
        // 3. Shortwave through the pack (into s_src).
        Real surface_sw = C(0.0);
        Real alb = shortwave(x, n, iswr, pressure, &surface_sw, true);
        Atm atm = {ta, rh, vw, surface_sw, ilwr, pressure, rain_rate};
        // 4. Energy balance and conduction.
        Real before = internal_energy(x, n);
        bud = step_energy(x, n, atm, sp, tsg, dt);
        Real sources = -C(0.0);
        for (int i = 0; i < n; ++i) sources = sources + EL(s_src, i);
        qg0 = (internal_energy(x, n) - before) / dt - bud.total - sources;

        // 5. Phase change and water.
        for (int i = 0; i < n; ++i) {
            EL(s_wbefore, i) = WATER_MASS(i);
            EL(s_hc, i) = MASS(i);
        }
        int n_before = n;
        Real drained = phase_change(x, n);
        if (n == n_before) {
            for (int i = 0; i < n; ++i) {
#if defined(PI_F32) || defined(PI_F32K)
                if (EL(s_wbefore, i) - WATER_MASS(i) > MARK_REFROZEN_FRACTION * EL(s_hc, i)) {
                    if ((EL(marker, i) & MARK_REFROZEN) == 0) a->tolerance_flags[c] |= 1u << 2;
                    EL(marker, i) |= MARK_REFROZEN;
                }
#else
                if (WATER_MASS(i) < EL(s_wbefore, i) - C(1e-9)) EL(marker, i) |= MARK_REFROZEN;
#endif
            }
        }
        Real runoff = cascade(x, n);
        ACC(A_RUNOFF) += drained + runoff;

        // 6. Sublimation or deposition on the top element.
        Real vapour = bud.vapour_mass * dt;
        if (n > 0) {
            int tp = n - 1;
            Real ice = fmax(ICE_MASS(tp) + vapour, C(0.0));
            Real im = ICE_MASS(tp);
            Real scale = im > C(0.0) ? ice / im : C(1.0);
            EL(thick, tp) *= scale;
            if (EL(thick, tp) > C(0.0)) {
                EL(water, tp) /= scale;
                EL(air, tp) = fmax(C(1.0) - EL(ice, tp) - EL(water, tp), C(0.0));
            }
        }
        ACC(A_SUBLIMATION) += fmax(-vapour, C(0.0));

        // 7. Settlement.
        settlement(x, n, dt, PAR(P_STRESS_FACTOR));
        // 8. Metamorphism and ageing.
        metamorphism(x, n, dt);
        // 9. Merge thin elements.
        merge_thin(x, n, k[K_MIN_ELEMENT]);
        a6_cap_deep(x, n);
        albedo_now = alb;
    }

    a->mo_passes[c] = bud.mo_passes;
    if (a->mo_passes[c] > 0 && a->mo_passes[c] < 200) a->tolerance_flags[c] |= 1u << 3;
    int nodes = n + 1;
    for (int j = 0; j < nodes; ++j) {
        Real tn = EL(T, j);
        if (tn < k[K_T_CRAZY_MIN] || tn > k[K_T_CRAZY_MAX]) {
            a->fail[c] = FAIL_CRAZY;
            a->fail_t[c] = t;
            return 4;
        }
    }

    // 10. Accumulate.
    ACC(A_N) += C(1.0);
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
    ACC(A_QW) += (C(1.0) - albedo_now) * iswr;
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
extern "C" __global__ void pi_snow_advance(const Arrays *arrays, const Real *k, int B,
                                           int nmax, int rows, int max_rows, int mode,
                                           long long day0) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= B) return;
    const Arrays *a = arrays;
    Ctx x;
    x.a = a; x.k = k; x.B = B; x.c = c; x.nmax = nmax;
    if (a->out_ready[c]) {
        for (int q = 0; q < A_COUNT; ++q) a->acc[(size_t)q * B + c] = C(0.0);
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
            for (int q = 0; q < A_COUNT; ++q) a->acc[(size_t)q * B + c] = C(0.0);
        }
        if (mode == 1 && outcome >= 2 && (t % 86400LL) == 43200LL) {
            long long day = floor_div(t - day0, 86400LL);
            Real hs = -C(0.0), swe = -C(0.0), water = -C(0.0);
            for (int i = 0; i < n; ++i) {
                hs = hs + EL(thick, i);
                swe = swe + MASS(i);
                water = water + WATER_MASS(i);
            }
            Real *d = a->daily + (size_t)day * D_COUNT * B;
            d[(size_t)D_HS * B + c] = hs;
            d[(size_t)D_SWE * B + c] = swe;
            d[(size_t)D_TSS * B + c] = ABS_TEMP(EL(T, n));
            d[(size_t)D_WATER * B + c] = water;
            d[(size_t)D_N * B + c] = (Real)n;
            d[(size_t)D_FAIL * B + c] = C(0.0);
        }
    }
    a->n[c] = n;
    a->cursor[c] = r;
}
