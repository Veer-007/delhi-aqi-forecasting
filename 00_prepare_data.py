"""Build one clean daily Delhi AQI series from the CPCB bulletin files published in
https://github.com/urbanemissions-info/AQI_bulletins  (data/Processed/*.csv).

Usage: python 00_prepare_data.py /path/to/AQI_bulletins/data/Processed
"""
import sys, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "aqi", "data", "Processed")
files = ["AllIndiaBulletins_Master_2024.csv", "AllIndiaBulletins_Master2025.csv",
         "AllIndiaBulletins_2025.csv", "AllIndiaBulletins_2026.csv"]
parts = []
for f in files:
    x = pd.read_csv(os.path.join(src, f), low_memory=False)
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x[x["City"].astype(str).str.strip().str.lower() == "delhi"]
    parts.append(x[["date", "No. Stations", "Air Quality", "Index Value", "Prominent Pollutant"]])
d = pd.concat(parts)
d["Index Value"] = pd.to_numeric(d["Index Value"], errors="coerce")
d = d.dropna(subset=["Index Value"])
# The files overlap; duplicate dates carry identical values (checked: max difference = 0),
# so keep one row per date.
d = d.sort_values("date").groupby("date").agg(
    aqi=("Index Value", "mean"), stations=("No. Stations", "last"),
    category=("Air Quality", "last"), pollutant=("Prominent Pollutant", "last"))
full = pd.date_range(d.index.min(), d.index.max(), freq="D")
d = d.reindex(full)
d.index.name = "date"
d["interpolated"] = d["aqi"].isna().astype(int)
d["aqi"] = d["aqi"].interpolate("linear")           # only isolated gaps (see report)
d["aqi"] = d["aqi"].round(1)
d.to_csv(os.path.join(HERE, "..", "data", "delhi_aqi_daily.csv"))
print(d.shape, d.index.min().date(), d.index.max().date(), "interpolated days:", int(d["interpolated"].sum()))
print(d["aqi"].describe())
