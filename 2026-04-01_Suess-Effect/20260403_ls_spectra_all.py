"""
20260403_ls_spectra_all.py
───────────────────────────────────────────────────────────────────────────────
Lomb-Scargle periodograms for: δ¹³C_suess (raw / MA / SG-detrended),
δ¹⁸O, SST, SSS, and ENSO.

Run from the project directory:
    /c/Users/mkara/anaconda3/python.exe 20260403_ls_spectra_all.py

Outputs
-------
    ls_spectra_all.pdf
    ls_spectra_all.png
"""

import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from astropy.timeseries import LombScargle

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
CORAL_CSV   = os.path.join(BASE, 'all-data-frame.csv')
CLIMATE_CSV = os.path.join(BASE, '..', '2026-01-15_Climate-Model',
                           'analysis', 'climate_df.csv')
ENSO_CSV    = os.path.join(BASE, '..', 'ENSO_Time_Series.csv')

OUT_PDF     = os.path.join(BASE, 'ls_spectra_all.pdf')
OUT_PNG     = os.path.join(BASE, 'ls_spectra_all.png')

# ── LS parameters ────────────────────────────────────────────────────────────
FMIN        = 1 / 15.0   # longest period = 15 yr
FMAX        = 12.0        # shortest period ~ 1 month
SPP         = 10          # samples per peak (frequency resolution)
FAP_LEVEL   = 0.01        # 1 % false-alarm probability threshold
XLIM        = (0.5, 8.0)  # period range shown in plot (years)

# ── Reference period lines ───────────────────────────────────────────────────
REF_PERIODS = {
    '1 yr':   (1.0,   'crimson',   '--'),
    '3.5 yr': (3.5,   'steelblue', ':'),
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(CORAL_CSV)
df = df.sort_values('age_final').reset_index(drop=True)

climate = pd.read_csv(CLIMATE_CSV)
climate['datetime'] = pd.to_datetime(climate['datetime'])

enso = pd.read_csv(ENSO_CSV, parse_dates=[0])
enso['day_of_year']   = enso['Date'].dt.dayofyear
enso['is_leap']       = enso['Date'].dt.is_leap_year
enso['total_days']    = np.where(enso['is_leap'], 366, 365)
enso['decimal_year']  = (enso['Date'].dt.year
                         + (enso['day_of_year'] - 1) / enso['total_days'])

# ─────────────────────────────────────────────────────────────────────────────
# 2. Lomb-Scargle helper
# ─────────────────────────────────────────────────────────────────────────────
def lomb_spec(t, y, fmin=FMIN, fmax=FMAX, spp=SPP):
    """Return (period, power, fap_level) for a time series."""
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    ls   = LombScargle(t, y)
    freq, power = ls.autopower(minimum_frequency=fmin,
                               maximum_frequency=fmax,
                               samples_per_peak=spp)
    fap = ls.false_alarm_level(FAP_LEVEL)
    return 1.0 / freq, power, fap

# ─────────────────────────────────────────────────────────────────────────────
# 3. Compute spectra
# ─────────────────────────────────────────────────────────────────────────────
t13 = df['age_final'].values

# δ¹³C_suess — three versions
p_raw,  pow_raw,  fap_raw  = lomb_spec(t13, df['d13c_suess'].values)
p_ma,   pow_ma,   fap_ma   = lomb_spec(t13, df['d13c_suess_ma'].values)
p_sg,   pow_sg,   fap_sg   = lomb_spec(t13, df['d13c_suess_no_tec_sg'].values)

# δ¹⁸O
p_d18o, pow_d18o, fap_d18o = lomb_spec(t13, df['d18o'].values)

# SST
t_sst = climate['age_absolute'].values
p_sst, pow_sst, fap_sst = lomb_spec(t_sst, climate['SST'].values)

# SSS
p_sss, pow_sss, fap_sss = lomb_spec(t_sst, climate['SSS'].values)

# ENSO
t_enso = enso['decimal_year'].values
p_enso, pow_enso, fap_enso = lomb_spec(t_enso, enso['ENSO'].values)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Figure
# ─────────────────────────────────────────────────────────────────────────────
greens = cm.get_cmap('Greens')
blues  = cm.get_cmap('Blues')

PANELS = [
    # (title,  list of (period, power, color, label),   fap_color,  fap_value)
    (r'$\delta^{13}$C$_{\rm Suess}$',
     [
         (p_raw, pow_raw, greens(0.40), r'$\delta^{13}$C$_{\rm Suess}$ (raw)'),
         (p_ma,  pow_ma,  greens(0.65), r'$\delta^{13}$C$_{\rm Suess}$ MA (3 mo)'),
         (p_sg,  pow_sg,  greens(0.90), r'$\delta^{13}$C$_{\rm Suess}$ SG-detrended'),
     ],
     greens(0.90), fap_sg),

    (r'$\delta^{18}$O',
     [(p_d18o, pow_d18o, blues(0.75), r'$\delta^{18}$O (raw)')],
     blues(0.75), fap_d18o),

    ('SST',
     [(p_sst, pow_sst, '#D62728', 'SST')],
     '#D62728', fap_sst),

    ('SSS',
     [(p_sss, pow_sss, '#17BECF', 'SSS')],
     '#17BECF', fap_sss),

    ('ENSO',
     [(p_enso, pow_enso, '#3B3B3B', 'ENSO index')],
     '#3B3B3B', fap_enso),
]

fig, axes = plt.subplots(
    len(PANELS), 1,
    figsize=(9, 14),
    sharex=True,
    gridspec_kw={'hspace': 0.08}
)

panel_labels = 'abcdefg'

for idx, (ax, (title, series, fap_color, fap_val)) in enumerate(zip(axes, PANELS)):

    # ── plot each series ─────────────────────────────────────────────────────
    for (per, pow_, col, lbl) in series:
        ax.plot(per, pow_, color=col, lw=1.1, label=lbl, zorder=3)

    # ── FAP threshold ────────────────────────────────────────────────────────
    ax.axhline(fap_val, color=fap_color, lw=0.9, ls='-.', alpha=0.6,
               label=f'1% FAP ({fap_val:.3f})', zorder=2)

    # ── reference period lines ───────────────────────────────────────────────
    for ref_label, (ref_p, ref_c, ref_ls) in REF_PERIODS.items():
        ax.axvline(ref_p, color=ref_c, lw=0.8, ls=ref_ls, alpha=0.55, zorder=1,
                   label=ref_label if idx == 0 else '_nolegend_')

    # ── panel label (a, b, c …) ──────────────────────────────────────────────
    ax.text(0.012, 0.93, f'({panel_labels[idx]})', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')

    # ── title / ylabel ───────────────────────────────────────────────────────
    ax.set_ylabel('LS power', fontsize=9)
    ax.set_title(title, loc='left', fontsize=10, pad=3)

    # ── legend ──────────────────────────────────────────────────────────────
    ax.legend(fontsize=7.5, loc='upper right',
              framealpha=0.85, edgecolor='none',
              ncol=2 if len(series) > 1 else 1)

    ax.tick_params(labelsize=8)
    ax.set_xlim(*XLIM)
    ax.set_ylim(bottom=0)

    # light grid
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune='upper'))
    ax.grid(axis='y', lw=0.4, alpha=0.4)
    ax.grid(axis='x', lw=0.3, alpha=0.3)

# ── shared x-axis label ───────────────────────────────────────────────────────
axes[-1].set_xlabel('Period (years)', fontsize=10)

# ── super title ───────────────────────────────────────────────────────────────
fig.suptitle('Lomb–Scargle Periodograms', fontsize=12, fontweight='bold', y=0.995)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save
# ─────────────────────────────────────────────────────────────────────────────
fig.savefig(OUT_PDF, dpi=150, bbox_inches='tight')
fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
print(f'Saved:\n  {OUT_PDF}\n  {OUT_PNG}')
