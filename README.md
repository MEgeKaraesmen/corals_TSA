# corals_TSA

Time-series analysis pipeline for coral stable isotope records from the Western Solomon Islands.  
This repository supports the PhD dissertation of Mehmet Ege Karaesmen (Jackson School of Geosciences, UT Austin) on using skeletal δ¹³C in *Porites* microatolls as a proxy for vertical tectonic displacement.

---

## Repository structure

```
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
```

---

## Data files

### `agemodel_pathsorted.xlsx`
Main isotope time series for coral **12-SPL-A** (Vonovono Island, Western Solomon Islands).  
- **574 samples** from 8 horizontal drilling paths (H, I, K, L, M, N, O and path subdivisions)  
- Key columns: `path_name`, `index`, `sample`, `d13c`, `d18o`, `age2` (X-ray anchor), `age_end`, `age_final`  
- `age_final` is the calibrated decimal year from the peak-matching age model  
- `age2` / `age_end` mark X-ray dark band start/end points used as stratigraphic anchors

### `fred_age_model_v2.xlsx`
Preliminary drilling campaign results (2021/2023, path H* / F*).  
- **118 time-series samples** covering the 2007 M8.1 event, plus 68 depth calibration samples  
- Use the `d18o` column directly (the `d18o_pdb` column contains a known conversion error)  
- Ages span 1994–2012 in decimal years

### `climate_df.csv`
Monthly ORAS5 ocean reanalysis extracted at the study site (8.22°S, 157.00°E).  
- Columns: `datetime`, `SST`, `SSS`, `age_absolute`, `synthetic_d18o`, `synthetic_d18o_2`  
- `synthetic_d18o` is computed from the Thompson et al. (2011) forward model:  
  δ¹⁸O = −0.22 · SST + 0.80 · SSS  
- Spans January 1958 – present at monthly resolution

### `ENSO_Time_Series.csv`
Oceanic Niño Index (ONI) monthly values.  
- Columns: `Date`, `ENSO`  
- Used for ENSO event annotation in rolling correlation and spectral figures

### `memo_edited.csv`
Honiara tide gauge sea level anomaly (JASL / NOAA NCEI).  
- Columns: `Date` (decimal year), `Sea Level` (m anomaly)  
- Record starts ~1968; used as sea level climate driver in rolling correlations

### `data_depth.csv`
Fred Taylor's 2024 depth calibration campaign on 12-SPL-A.  
- **68 samples** from 7 vertical paths (A–G), SMOW scale  
- Columns: `id`, `d13c`, `d18o`, `path`, `idx`, `depth`  
- Depths in cm below sea surface at time of sampling

### `2024depthABCDE_memoedited_Agemodeled.xlsx`
Memo's 2024 depth calibration campaign on 12-SPL-A.  
- **141 samples** from 5 vertical paths (A–E), SMOW scale  
- Includes age-modelled dates assigned by matching to the horizontal time series

### `suesseffect.xlsx`
Annual Suess effect correction values for atmospheric δ¹³C (after Linsley et al. 2019).  
- Reference epoch: 1997.75 (midpoint of the 1995–2000 depth calibration window)  
- Applied to δ¹³C before depth calibration to account for anthropogenic ¹³C depletion  
- Not applied to the time-series analysis (secular trend absorbed into SG filter)

---

## Main pipeline: `20260401_read_data.py`

Run at the top of any analysis notebook:

```python
%run 20260401_read_data.py
```

### What it produces

| Variable | Description |
|---|---|
| `df` | Raw isotope data sorted by path, with per-path 3-sample moving average (`path_ma`) |
| `df_sorted` | All samples sorted by `age_final` (irregular time series, 574 rows) |
| `df_resampled` | 3-month regular grid (resample → interpolate → rolling statistics) |
| `dark_bands` | DataFrame of paired X-ray dark band start/end positions per path |
| `coefs_centered` | Rolling OLS slope of δ¹⁸O and δ¹³C vs. time (window = 36 quarters) |

### Key computed columns on `df_sorted`

| Column | Description |
|---|---|
| `d18o_pdb` | δ¹⁸O converted to VPDB: `0.97001 × d18o − 29.98` |
| `d13c_suess` | δ¹³C with Suess effect correction applied |
| `d13c_sg_trend` / `d18o_sg_trend` | Savitzky–Golay long-term (tectonic) trend, 73-month window, poly order 3 |
| `d13c_no_tec_sg` / `d18o_no_tec_sg` | Detrended residual (climate + noise band) |
| `d13c_ma`, `d13c_ma2`, `d13c_ma3` | 3-, 36-, 72-sample centred moving averages |
| `d13c_no_tectonic` | δ¹³C minus 36-sample MA (alternative detrend) |

### Savitzky–Golay filter parameters
- `window_length = 73` months on a monthly grid  
- `polyorder = 3`, `mode = 'interp'`  
- Applied on a temporary monthly grid then interpolated back to irregular raw times — zero NaNs, no edge loss

---

## Isotope conventions

| Proxy | Standard | Conversion |
|---|---|---|
| δ¹³C | VPDB | — (measured directly on PDB scale) |
| δ¹⁸O (raw) | VSMOW | — |
| δ¹⁸O (converted) | VPDB | `d18o_pdb = 0.97001 × d18o_smow − 29.98` |

---

## Age model summary

- **Method**: pseudo-coral forward model (Thompson et al. 2011) matched to ORAS5 synthetic δ¹⁸O  
- **Calibrated range**: 1958–2007 (r = 0.729 vs. climate reference)  
- **Pre-1958 extrapolation**: long-term annual climatological average used as tie-point reference  
- **Anchor**: 2007 M8.1 earthquake coseismic uplift (73 cm) and corresponding δ¹³C excursion

---

## Dependencies

```
pandas
numpy
scipy
matplotlib
statsmodels
```

Install with:

```bash
pip install pandas numpy scipy matplotlib statsmodels
```

---

## Reference

Karaesmen, M. E. (in preparation). *Coral δ¹³C as a proxy for vertical tectonic displacement: decadal interseismic signatures in the Western Solomon Islands.* PhD Dissertation, Jackson School of Geosciences, The University of Texas at Austin.
