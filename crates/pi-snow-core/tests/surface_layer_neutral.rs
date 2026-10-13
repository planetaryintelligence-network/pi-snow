//! engine-soil-spinup-v2 plan Z-T8 (parser half): `SURFACE_LAYER_NEUTRAL`.

use pi_snow_core::options::{PhysicsOptions, SnowAlbedo, SurfaceLayer, SurfaceLayerNeutral};

#[test]
fn key_absent_is_native_and_default() {
    assert_eq!(
        PhysicsOptions::default().surface_layer_neutral,
        SurfaceLayerNeutral::Native
    );
    let o = PhysicsOptions::from_settings(["SURFACE_LAYER=ZETA_LIMIT_CLM5"]).unwrap();
    assert_eq!(o.surface_layer_neutral, SurfaceLayerNeutral::Native);
}

#[test]
fn continuous_parses_in_either_order_and_sets_no_surface_layer_value() {
    let a = PhysicsOptions::from_settings([
        "SURFACE_LAYER=ZETA_LIMIT_CLM5",
        "SURFACE_LAYER_NEUTRAL=CONTINUOUS",
    ])
    .unwrap();
    let b = PhysicsOptions::from_settings([
        "SURFACE_LAYER_NEUTRAL=CONTINUOUS",
        "SURFACE_LAYER=ZETA_LIMIT_CLM5",
    ])
    .unwrap();
    assert_eq!(a, b);
    assert_eq!(a.surface_layer, SurfaceLayer::ZetaLimitClm5);
    assert_eq!(a.surface_layer_neutral, SurfaceLayerNeutral::Continuous);
}

#[test]
fn unknown_values_and_misspelt_values_refuse() {
    for v in [
        "NATIVE",
        "PHASE2",
        "",
        "continuous",
        "ZETA_LIMIT_CONTINUOUS",
    ] {
        let setting = format!("SURFACE_LAYER_NEUTRAL={v}");
        assert!(
            PhysicsOptions::from_settings([setting.as_str()]).is_err(),
            "{v:?}"
        );
    }
    // No value is added to SURFACE_LAYER itself (plan A1.1).
    assert!(PhysicsOptions::from_settings(["SURFACE_LAYER=ZETA_LIMIT_CONTINUOUS"]).is_err());
}

#[test]
fn composes_with_the_albedo_schemes_that_require_clm5() {
    for albedo in ["AOKI2003_TEMP_RATE", "CHAN2022_MERRA2"] {
        let setting = format!("SNOW_ALBEDO={albedo}");
        let o = PhysicsOptions::from_settings([
            "SURFACE_LAYER=ZETA_LIMIT_CLM5",
            "SURFACE_LAYER_NEUTRAL=CONTINUOUS",
            setting.as_str(),
        ])
        .unwrap();
        assert_ne!(o.snow_albedo, SnowAlbedo::Vionnet2012);
        assert_eq!(o.surface_layer, SurfaceLayer::ZetaLimitClm5);
    }
}
