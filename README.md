corals_TSA
Time-series analysis pipeline for coral stable isotope records from the Western Solomon Islands.
This repository supports the PhD dissertation of Mehmet Ege Karaesmen (Jackson School of Geosciences, UT Austin) on using skeletal δ¹³C in Porites microatolls as a proxy for vertical tectonic displacement.

Repo Structure

corals_TSA/
├── 20260401_read_data.py          # Main data pipeline (current version)
├── 20260323_read_data.py          # Previous pipeline version (archived)
├── agemodel_pathsorted.xlsx       # Primary age model — 574 samples, paths H–O, 1937–2007
├── fred_age_model_v2.xlsx         # Preliminary dataset — 118 samples, 1994–2012
├── climate_df.csv                 # ORAS5 reanalysis: monthly SST, SSS, synthetic δ¹⁸O
├── ENSO_Time_Series.csv           # Oceanic Niño Index (ONI) monthly time series
├── memo_edited.csv                # Honiara tide gauge sea level anomalies
├── data_depth.csv                 # Fred's 2024 depth calibration samples (paths A–G, 68 samples)
├── 2024depthABCDE_memoedited_Agemodeled.xlsx  # Memo's 2024 depth calibration (paths A–E, 141 samples)
└── suesseffect.xlsx               # Suess effect correction time series (after Linsley et al. 2019)
