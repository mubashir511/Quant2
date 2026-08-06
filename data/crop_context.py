from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# One representative growing-region coordinate per country — a simplified
# single-point proxy for a whole country's harvest, not precise agronomic
# modeling. Keyed by the same crop display names data/underlying.py already
# produces, so detection can just intersect with resolved Market Watch names.
CROP_TRADE_PROFILES: dict[str, dict[str, list[tuple[str, float, float]]]] = {
    "Wheat": {
        "exporters": [
            ("Russia", 45.0, 39.0),  # Krasnodar
            ("USA", 38.5, -98.0),  # Kansas
            ("Canada", 52.0, -106.0),  # Saskatchewan
        ],
        "importers": [
            ("Egypt", 30.0, 31.2),
            ("Indonesia", -6.2, 106.8),
        ],
    },
    "Corn": {
        "exporters": [
            ("USA", 42.0, -93.5),  # Iowa
            ("Brazil", -12.6, -56.1),  # Mato Grosso
            ("Argentina", -31.4, -64.2),  # Cordoba/Pampas
        ],
        "importers": [
            ("Mexico", 23.6, -102.5),
            ("China", 47.0, 128.0),  # Heilongjiang corn belt
        ],
    },
    "Rice": {
        "exporters": [
            ("India", 31.1, 75.3),  # Punjab
            ("Thailand", 15.9, 100.9),
            ("Vietnam", 10.0, 105.8),  # Mekong Delta
        ],
        "importers": [
            ("Philippines", 12.9, 121.8),
            ("Nigeria", 9.1, 8.7),
        ],
    },
    "Sugar": {
        "exporters": [
            ("Brazil", -22.0, -48.0),  # Sao Paulo cane belt
            ("India", 19.8, 75.3),  # Maharashtra
            ("Thailand", 15.9, 100.9),
        ],
        "importers": [
            ("Indonesia", -6.2, 106.8),
            ("China", 23.8, 108.8),  # Guangxi
        ],
    },
}


@dataclass
class WeatherSnapshot:
    past_30d_precip_mm: float | None
    forecast_7d_precip_mm: float | None
    temp_max_avg_c: float | None
    temp_min_avg_c: float | None


def fetch_weather_snapshot(lat: float, lon: float) -> WeatherSnapshot:
    """Recent rainfall (history proxy) + near-term forecast for one point.
    Any failure degrades to an all-None snapshot rather than raising."""
    import requests

    try:
        response = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
                "past_days": 30,
                "forecast_days": 7,
                "timezone": "auto",
            },
            timeout=10,
        )
        response.raise_for_status()
        daily = response.json()["daily"]
        precip = daily["precipitation_sum"]
        tmax = daily["temperature_2m_max"]
        tmin = daily["temperature_2m_min"]
    except Exception:
        return WeatherSnapshot(None, None, None, None)

    if len(precip) <= 7:
        return WeatherSnapshot(None, None, None, None)

    past_precip = [v for v in precip[:-7] if v is not None]
    forecast_precip = [v for v in precip[-7:] if v is not None]
    past_tmax = [v for v in tmax[:-7] if v is not None]
    past_tmin = [v for v in tmin[:-7] if v is not None]

    return WeatherSnapshot(
        past_30d_precip_mm=round(sum(past_precip), 1) if past_precip else None,
        forecast_7d_precip_mm=round(sum(forecast_precip), 1) if forecast_precip else None,
        temp_max_avg_c=round(sum(past_tmax) / len(past_tmax), 1) if past_tmax else None,
        temp_min_avg_c=round(sum(past_tmin) / len(past_tmin), 1) if past_tmin else None,
    )


def fetch_crop_supply_demand_context(crop_names: set[str]) -> str:
    """Per-country weather context for the given crops' major exporters/
    importers. Names not in CROP_TRADE_PROFILES are silently skipped. Every
    region is an independent HTTP call, fetched in parallel — several crops
    at once would otherwise mean 10-15+ sequential round-trips."""
    crops = sorted(crop_names & CROP_TRADE_PROFILES.keys())
    if not crops:
        return ""

    jobs = []  # (crop, role, country_name, lat, lon), in display order
    for crop in crops:
        profile = CROP_TRADE_PROFILES[crop]
        for role, countries in (
            ("Exporter", profile["exporters"]),
            ("Importer", profile["importers"]),
        ):
            for country_name, lat, lon in countries:
                jobs.append((crop, role, country_name, lat, lon))

    with ThreadPoolExecutor(max_workers=min(len(jobs), 16) or 1) as pool:
        snapshots = list(pool.map(lambda j: fetch_weather_snapshot(j[3], j[4]), jobs))

    lines = []
    current_crop = None
    for (crop, role, country_name, _lat, _lon), snapshot in zip(jobs, snapshots):
        if crop != current_crop:
            lines.append(f"{crop} supply/demand context:")
            current_crop = crop

        bits = []
        if snapshot.past_30d_precip_mm is not None:
            bits.append(f"past 30d rainfall {snapshot.past_30d_precip_mm}mm")
        if snapshot.forecast_7d_precip_mm is not None:
            bits.append(f"next 7d forecast {snapshot.forecast_7d_precip_mm}mm")
        if snapshot.temp_max_avg_c is not None and snapshot.temp_min_avg_c is not None:
            bits.append(f"recent temp range {snapshot.temp_min_avg_c}-{snapshot.temp_max_avg_c}°C")
        detail = ", ".join(bits) if bits else "weather data unavailable"
        lines.append(f"- {role} {country_name} (growing region): {detail}")

    return "\n".join(lines)
