# SPARK weather-unit resolution

## Decision

The canonical source values are preserved exactly. The three disputed columns
should be interpreted as follows when a GOLD table is eventually designed:

| Published column | Published label | Delivered range (2018–2024) | Actual likely unit | Decision |
|---|---:|---:|---|---|
| `humidity_%` | percent | 0.15–1.00 | decimal fraction (0–1) | resolved; retain raw values and rename semantically before modelling, or multiply by 100 only with an explicit derived-column name |
| `cloud_cover_total_oktas_(0–8)` | oktas | -0.125–1.00 | decimal fraction of eighths | resolved; retain raw values; `-0.125` corresponds to DWD's special `-1` eighth value after division by 8 and must be handled as a flag/invalid value, not physical negative cloud |
| `sunshine_duration_minutes_per_hour` | minutes/hour | 0–3600 | seconds within the hour | resolved; retain raw values and rename semantically, or derive minutes by division by 60 with provenance |

Temperature and radiation are independently consistent with their published
units: temperature is degrees Celsius (-15.4 to 38.7 °C), and global radiation
is hourly energy in J/cm² (0 to 360 J/cm²).

## Evidence chain

1. The official SPARK `00exogenous_data_details.json` labels humidity as `%`,
   cloud cover as oktas, and sunshine duration as minutes/hour.
2. The SPARK preprocessing notebook `scripts/05_External_Features.ipynb` fetched
   DWD data through Wetterdienst 0.113.0, did no value transformation itself,
   and then hard-coded those labels into column names. Repository commit:
   `c0e4ac1c3789b88e2bc2bb918464ae2e4dae255a`.
3. In Wetterdienst 0.113.0, the DWD mappings identify humidity as source unit
   `percent`, cloud cover as `one_eighth`, and sunshine duration as `minute`.
   Its default target units are respectively `decimal`, `decimal`, and
   `second`; its explicit conversion functions divide percent by 100, divide
   eighths by 8, and multiply minutes by 60.
4. DWD's source descriptions independently specify relative humidity in `%`,
   total cloud cover in eighths, sunshine duration in minutes, and global
   radiation in J/cm². The delivered ranges exactly match the Wetterdienst
   conversions, not the labels later attached by SPARK.

Primary DWD documentation:

- <https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/moisture/DESCRIPTION_obsgermany_climate_hourly_moisture_en.pdf>
- <https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/cloudiness/DESCRIPTION_obsgermany_climate_hourly_cloudiness_en.pdf>
- <https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/solar/DESCRIPTION_obsgermany_climate_hourly_solar_en.pdf>

## Modelling implication

These variables no longer need exclusion solely because of unit ambiguity, but
their misleading source column names must not be propagated unchanged into
GOLD. Weather is observed historical weather, not historical forecast/NWP, so
it is unsuitable as a point-in-time-safe day-ahead covariate unless a forecast
source is added.
