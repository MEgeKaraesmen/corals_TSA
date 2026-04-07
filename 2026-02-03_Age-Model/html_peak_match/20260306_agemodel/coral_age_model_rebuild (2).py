"""
coral_age_model_rebuild.py
==========================
Rebuilds the coral age model from coral_age_final_20260220.xlsx
(the previously working peak-based age model) and runs the full
notebook-equivalent analysis against the climate reference.

Outputs:
  - coral_age_model_rebuilt.csv      (574 samples, calendar_year + d18o_pdb)
  - coral_age_final_rebuilt.xlsx     (drop-in replacement for read_data.py)
  - rebuilt_age_model_analysis.png   (4-panel diagnostic figure)

Usage:
  python coral_age_model_rebuild.py

Requirements:
  pandas, numpy, matplotlib, openpyxl
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
OLD_AGE_MODEL  = 'coral_age_final_20260220.xlsx'
CLIMATE_CSV    = 'climate_df.csv'
OUT_CSV        = 'coral_age_model_rebuilt.csv'    # written to cwd
OUT_XLSX       = 'coral_age_final_rebuilt.xlsx'   # written to cwd
OUT_FIG        = 'rebuilt_age_model_analysis.png' # written to cwd


# ── Helper functions (from 20260223_read_data.py) ─────────────────────────────

def decimal_year_to_datetime_arr(decimal_year):
    """Vectorised decimal-year → pandas Timestamp."""
    decimal_year = np.asarray(decimal_year, dtype=float)
    year = decimal_year.astype(int)
    decimal_part = decimal_year - year
    start_of_year = pd.to_datetime([f'{y}-01-01' for y in year])
    seconds_in_year = (
        pd.to_datetime([f'{y+1}-01-01' for y in year]) - start_of_year
    ).total_seconds()
    seconds = decimal_part * seconds_in_year
    return start_of_year + pd.to_timedelta(seconds, unit='s')


def decimal_year_to_datetime_scalar(x):
    """Scalar decimal-year → pandas Timestamp."""
    year = int(x); frac = x - year
    start = pd.Timestamp(f'{year}-01-01')
    end   = pd.Timestamp(f'{year+1}-01-01')
    return start + pd.Timedelta(seconds=frac * (end - start).total_seconds())


def datetime_to_decimal_year(dt):
    """pandas Timestamp / datetime → decimal year."""
    year = dt.year
    days = (dt.date() - datetime.date(year, 1, 1)).days
    return year + days / 365.25


def to_month(x):
    """Decimal year → calendar month (1-12)."""
    return decimal_year_to_datetime_scalar(x).month


def interp_climate(yr, cx, cy):
    """
    Linear interpolation of climate synthetic d18O at decimal year `yr`.
    cx, cy = age_absolute and synthetic_d18o arrays from climate_df.csv.
    Returns NaN if yr is outside the climate record.
    """
    idx = int(np.searchsorted(cx, yr))
    if idx == 0 or idx >= len(cx):
        return np.nan
    t = (yr - cx[idx - 1]) / (cx[idx] - cx[idx - 1])
    return float(cy[idx - 1] + t * (cy[idx] - cy[idx - 1]))


# ── 1. Load old age model ──────────────────────────────────────────────────────

print("Loading old age model …")
df_raw = pd.read_excel(OLD_AGE_MODEL)

# Sort by age_final (calendar year from old peak-based model)
df = df_raw.sort_values('age_final').reset_index(drop=True)

# VSMOW → PDB conversion (standard formula)
df['d18o_pdb'] = 0.97001 * df['d18o'] - 29.99

# Rename age_final → calendar_year for downstream consistency
df.rename(columns={'age_final': 'calendar_year'}, inplace=True)

# Age source label
df['age_source'] = 'climate_matched'
df.loc[df['calendar_year'] < 1958, 'age_source'] = 'pre1958_peak_match'

print(f"  Samples : {len(df)}")
print(f"  Cal year: {df['calendar_year'].min():.4f} → {df['calendar_year'].max():.4f}")
print(f"  Pre-1958: {(df['calendar_year'] < 1958).sum()} samples")
print(f"  Post-1958: {(df['calendar_year'] >= 1958).sum()} samples")


# ── 2. Load climate reference ──────────────────────────────────────────────────

print("\nLoading climate reference …")
climate = pd.read_csv(CLIMATE_CSV)
climate['datetime'] = pd.to_datetime(climate['datetime'])
climate['month']    = climate['datetime'].dt.month
cx = climate['age_absolute'].values
cy = climate['synthetic_d18o'].values


# ── 3. Per-sample climate interpolation & correlation ─────────────────────────

df['clim_interp'] = df['calendar_year'].apply(lambda y: interp_climate(y, cx, cy))
df['month']       = df['calendar_year'].apply(to_month)

mask = ~np.isnan(df['clim_interp'])
r_all = np.corrcoef(df.loc[mask, 'd18o_pdb'], df.loc[mask, 'clim_interp'])[0, 1]
print(f"\n  r (d18o_pdb vs synthetic_d18o, all): {r_all:.4f}  n={mask.sum()}")

mask_post = mask & (df['calendar_year'] >= 1958)
r_post = np.corrcoef(df.loc[mask_post, 'd18o_pdb'],
                      df.loc[mask_post, 'clim_interp'])[0, 1]
print(f"  r (post-1958 only):                  {r_post:.4f}  n={mask_post.sum()}")


# ── 4. Monthly resampled time series (mirrors read_data.py) ───────────────────

window_size  = 3
window_size2 = 36

df['age_dt'] = decimal_year_to_datetime_arr(df['calendar_year'].values)

df_r_ = df[['calendar_year', 'age_dt', 'd13c', 'd18o', 'd18o_pdb']].set_index('age_dt')
df_r  = df_r_.resample('ME').mean()
df_r.interpolate(inplace=True)
df_r['age_res'] = [datetime_to_decimal_year(df_r.index[i]) for i in range(len(df_r))]

df_r[['d13c_ma',  'd18o_ma',  'd18o_pdb_ma' ]] = (
    df_r[['d13c', 'd18o', 'd18o_pdb']].rolling(window_size,  center=True).mean()
)
df_r[['d13c_ma2', 'd18o_ma2', 'd18o_pdb_ma2']] = (
    df_r[['d13c', 'd18o', 'd18o_pdb']].rolling(window_size2, center=True).mean()
)

df_r['d18o_pdb_no_tectonic'] = df_r['d18o_pdb'] - df_r['d18o_pdb_ma2']

df_r['clim_interp'] = df_r['age_res'].apply(lambda y: interp_climate(y, cx, cy))

r_res = np.corrcoef(
    df_r.dropna(subset=['d18o_pdb', 'clim_interp'])['d18o_pdb'],
    df_r.dropna(subset=['d18o_pdb', 'clim_interp'])['clim_interp']
)[0, 1]
print(f"  r (monthly resampled):               {r_res:.4f}  n={df_r['clim_interp'].notna().sum()}")

rc      = df_r['d18o_pdb'].rolling(30, center=True).corr(df_r['clim_interp'])
rc_mean = rc.dropna().mean()
print(f"  Rolling-corr mean (30 mo window):    {rc_mean:.3f}")


# ── 5. Seasonal cycle ─────────────────────────────────────────────────────────

coral_monthly = df.groupby('month')['d18o_pdb'].mean()
clim_monthly  = climate.groupby('month')['synthetic_d18o'].mean()
amp = coral_monthly.max() - coral_monthly.min()
print(f"\n  Coral seasonal amplitude: {amp:.4f} ‰ PDB")
print(f"  Coral peak month: {coral_monthly.idxmax()}  |  Climate peak month: {clim_monthly.idxmax()}")


# ── 6. Save outputs ───────────────────────────────────────────────────────────

print("\nSaving outputs …")

out_csv = df[['index', 'Name', 'sample', 'path_name', 'path_num',
               'd13c', 'd18o', 'd18o_pdb', 'calendar_year', 'age_source']].copy()
out_csv.to_csv(OUT_CSV, index=False)
print(f"  {OUT_CSV}")

df.to_excel(OUT_XLSX, index=False)
print(f"  {OUT_XLSX}")


# ── 7. Four-panel diagnostic figure ──────────────────────────────────────────

print("\nPlotting …")
fig = plt.figure(figsize=(20, 18))
gs  = plt.GridSpec(4, 1, hspace=0.38, figure=fig)

# ── Panel 1: raw d18O vs climate (dual y-axis) ────────────────────────────────
ax1  = fig.add_subplot(gs[0])
ax1r = ax1.twinx()

df_s = df.sort_values('calendar_year')
ma3  = df_s['d18o'].rolling(3, center=True).mean()

ax1.scatter(df_s['calendar_year'], df_s['d18o'],
            c='#3492eb', s=9, alpha=0.55, zorder=2, label='raw d18O (VSMOW)')
ax1.plot(df_s['calendar_year'], ma3,
         color='#134b80', lw=1.2, label='3pt MA', zorder=3)
ax1r.plot(climate['age_absolute'], climate['synthetic_d18o'],
          color='#F54927', lw=1.3, label='synthetic d18O (climate)', alpha=0.9)

ax1.set_ylabel('δ¹⁸O VSMOW (‰)', color='#134b80', fontsize=10)
ax1r.set_ylabel('Climate δ¹⁸O anomaly (‰)', color='#F54927', fontsize=10)
ax1.set_title(f'd18O Observed vs Climate Synthetic   r = {r_all:.3f}',
              fontsize=12, fontweight='bold')
l1, b1 = ax1.get_legend_handles_labels()
l2, b2 = ax1r.get_legend_handles_labels()
ax1.legend(l1 + l2, b1 + b2, fontsize=8, loc='upper left')
ax1.grid(alpha=0.25)
ax1.set_xlim(1936, 2008)

# ── Panel 2: detrended d18O_PDB vs climate ───────────────────────────────────
ax2  = fig.add_subplot(gs[1], sharex=ax1)
ax2r = ax2.twinx()

ax2.plot(df_r['age_res'], df_r['d18o_pdb_no_tectonic'],
         color='#3492eb', lw=0.9, label='d18O_PDB detrended (−36mo MA)')
ax2r.plot(climate['age_absolute'], climate['synthetic_d18o'],
          color='#F54927', lw=1.2, label='climate synthetic')

ax2.set_ylabel('d18O_PDB detrended (‰)', color='#3492eb', fontsize=10)
ax2r.set_ylabel('Climate anomaly (‰)',   color='#F54927', fontsize=10)
ax2.set_title('d18O PDB Detrended (−36mo MA) vs Climate — Monthly Resampled', fontsize=11)
l1, b1 = ax2.get_legend_handles_labels()
l2, b2 = ax2r.get_legend_handles_labels()
ax2.legend(l1 + l2, b1 + b2, fontsize=8)
ax2.grid(alpha=0.25)

# ── Panel 3: rolling correlation ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2], sharex=ax1)

xrc = df_r['age_res'].values
yrc = rc.values
cmap = plt.cm.RdBu_r
norm = mpl.colors.Normalize(vmin=-1, vmax=1)

ax3.plot(xrc, yrc, color='black', lw=0.8, zorder=3)
for i in range(len(xrc) - 1):
    if not (np.isnan(yrc[i]) or np.isnan(yrc[i + 1])):
        ax3.axvspan(xrc[i], xrc[i + 1],
                    color=cmap(norm(yrc[i])), alpha=0.6, linewidth=0)
ax3.axhline(0, color='k', lw=0.5, ls='--')
ax3.set_ylim(-1, 1)
ax3.set_ylabel('Rolling r (30mo)', fontsize=10)
ax3.set_title(
    f'Rolling Correlation d18O_PDB vs Climate   [mean r = {rc_mean:.3f}]',
    fontsize=11)
ax3.grid(alpha=0.25, zorder=0)

# ── Panel 4: seasonal cycles ──────────────────────────────────────────────────
ax4  = fig.add_subplot(gs[3])
ax4r = ax4.twinx()

mo_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

ax4.plot(range(1, 13), clim_monthly.values,  'b-o',  lw=2, label='Climate synthetic d18O')
ax4r.plot(range(1, 13), coral_monthly.values, 'r--o', lw=2, label='Coral d18O PDB')

ax4.set_xticks(range(1, 13))
ax4.set_xticklabels(mo_labels)
ax4.set_ylabel('Climate d18O anomaly (‰)', color='blue', fontsize=10)
ax4r.set_ylabel('Coral d18O PDB (‰)',      color='red',  fontsize=10)
ax4.set_title(
    f'Seasonal Cycle  |  coral amp = {amp:.3f} ‰  |  '
    f'peaks: coral = Sep, climate = Aug  |  '
    f'r = {np.corrcoef(coral_monthly.values, clim_monthly.values)[0,1]:.3f}',
    fontsize=11)
l1, b1 = ax4.get_legend_handles_labels()
l2, b2 = ax4r.get_legend_handles_labels()
ax4.legend(l1 + l2, b1 + b2, fontsize=9)
ax4.grid(alpha=0.3)

plt.suptitle(
    'Rebuilt Age Model (coral_age_final_20260220.xlsx)\n'
    f'Range: {df["calendar_year"].min():.2f}–{df["calendar_year"].max():.2f}  |  '
    f'n = {len(df)}  |  r(d18O_PDB vs climate) = {r_all:.3f}',
    fontsize=14, fontweight='bold', y=1.005)

plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
plt.close()
print(f"  {OUT_FIG}")

print("\nDone.")
