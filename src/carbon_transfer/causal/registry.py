from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from carbon_transfer.constants import MODIS_COLUMNS, POI_COLUMNS, VIIRS_COLUMNS, WEATHER_COLUMNS


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    role: str
    estimable: bool
    intervention: str
    default_lags: Sequence[int]
    caveat: str

    def to_dict(self) -> Dict:
        value = asdict(self)
        value["default_lags"] = list(self.default_lags)
        return value


def _modis_role(name: str) -> tuple[str, bool, str, str]:
    if name.endswith("pixel_count"):
        return (
            "measurement_quality", False, "not_applicable",
            "Valid-pixel count describes measurement quality, not a physical intervention.",
        )
    if "ndvi" in name or "evi" in name:
        return (
            "proxy_exposure", True, "one within-panel standard-deviation increase",
            "Vegetation indices are proxies; weather and land-development confounding remain.",
        )
    return (
        "proxy_exposure", True, "one within-panel standard-deviation increase",
        "Raw reflectance mixes land cover, material, snow, water, and atmospheric effects.",
    )


def build_feature_registry(
    columns: Iterable[str],
    *,
    lags: Sequence[int] = (0, 1, 3, 6, -1),
    overrides: Optional[Mapping[str, Mapping]] = None,
) -> List[FeatureSpec]:
    """Classify every known feature before any effect is estimated."""
    available = set(columns)
    specs: List[FeatureSpec] = []
    for name in POI_COLUMNS:
        specs.append(FeatureSpec(
            name, "poi", "proxy_exposure", name in available,
            "one within-panel standard-deviation increase in POI count",
            tuple(lags),
            "POI counts proxy land use; causal interpretation requires temporal openings/closures.",
        ))
    for name in MODIS_COLUMNS:
        role, estimable, intervention, caveat = _modis_role(name)
        specs.append(FeatureSpec(
            name, "modis", role, estimable and name in available, intervention,
            tuple(lags) if estimable else (), caveat,
        ))
    for name in VIIRS_COLUMNS:
        quality = name in {"ntl_valid_pixel_count", "ntl_is_missing", "ntl_is_imputed"}
        specs.append(FeatureSpec(
            name, "viirs", "measurement_quality" if quality else "outcome_proxy",
            (not quality) and name in available,
            "not_applicable" if quality else "one within-panel standard-deviation increase",
            () if quality else tuple(lags),
            "VIIRS may share causes with emissions or enter label construction; contemporaneous values are not automatically causal.",
        ))
    for name in WEATHER_COLUMNS:
        specs.append(FeatureSpec(
            name, "weather", "external_exposure", name in available,
            "one within-panel standard-deviation increase",
            tuple(lags),
            "Weather is comparatively exogenous, but seasonality and label measurement processes can confound estimates.",
        ))

    known = {spec.name for spec in specs}
    for name in sorted(available):
        if name not in known and (name.startswith("month_") or name.endswith("_is_missing")):
            specs.append(FeatureSpec(
                name, "calendar" if name.startswith("month_") else "missingness",
                "control_or_measurement", False, "not_applicable", (),
                "Calendar and missingness indicators are controls or measurement variables, not interventions.",
            ))
    override_map = dict(overrides or {})
    output = []
    for spec in specs:
        values = spec.to_dict()
        values.update(override_map.get(spec.name, {}))
        values["default_lags"] = tuple(int(value) for value in values["default_lags"])
        output.append(FeatureSpec(**values))
    return output
