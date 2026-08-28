POI_COLUMNS = [
    "poi_restaurant", "poi_school", "poi_university", "poi_fuel",
    "poi_hospital", "poi_clinic", "poi_shop", "poi_tourism",
    "poi_industrial_landuse", "poi_commercial_landuse",
    "poi_residential_landuse", "poi_industrial_building",
    "poi_commercial_building", "poi_retail_building", "poi_warehouse",
    "poi_parking", "poi_bus_station",
]

MODIS_COLUMNS = [
    "modis_ndvi_mean", "modis_ndvi_mean_pixel_count", "modis_evi_mean",
    "modis_evi_mean_pixel_count", "modis_red_reflectance_mean",
    "modis_red_reflectance_mean_pixel_count", "modis_nir_reflectance_mean",
    "modis_nir_reflectance_mean_pixel_count",
]

VIIRS_COLUMNS = [
    "ntl_radiance_mean", "ntl_radiance_max", "ntl_radiance_sum",
    "ntl_valid_pixel_count", "ntl_is_missing", "ntl_is_imputed",
    "ntl_log_radiance_mean",
]

WEATHER_COLUMNS = [
    "temperature_2m_mean_c", "relative_humidity_2m_mean_pct",
    "precipitation_sum_mm", "wind_speed_10m_max_kmh",
    "shortwave_radiation_sum_mj_m2",
]

MONTH_COLUMNS = [f"month_{month:02d}" for month in range(1, 13)]
MISSING_COLUMNS = [f"{name}_is_missing" for name in MODIS_COLUMNS + WEATHER_COLUMNS]
TABULAR_FEATURES = POI_COLUMNS + MODIS_COLUMNS + VIIRS_COLUMNS + WEATHER_COLUMNS + MONTH_COLUMNS + MISSING_COLUMNS
REMOTE_FEATURES = MODIS_COLUMNS + VIIRS_COLUMNS
ENVIRONMENT_FEATURES = WEATHER_COLUMNS

MODEL_NAMES = [
    "lightgbm", "bpnn", "carbongcn", "opencarbon_core",
    "opencarbon_monthly", "opencarbon_monthly_noviirs",
    "opencarbon_monthly_precomputed", "opencarbon_monthly_vrex",
]
