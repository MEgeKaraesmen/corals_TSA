import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import datetime
import scipy.stats
import matplotlib.dates as mdates
import matplotlib
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS


def decimal_year_to_datetime(decimal_year):
    year = decimal_year.astype(int)
    decimal_part = decimal_year - year
    start_of_year = pd.to_datetime([f'{y}-01-01' for y in year])
    seconds_in_year = (pd.to_datetime([f'{y+1}-01-01' for y in year]) - start_of_year).total_seconds()
    seconds = decimal_part * seconds_in_year
    return start_of_year + pd.to_timedelta(seconds, unit='s')

def datetime_to_decimal_year(datetime_):
    year = datetime_.year
    days = (datetime_.date() - datetime.date(year, 1,1)).days
    res = year + days/365.25
    return res


def FFT_res(x, si):
    n = len(x)
    x_FFT_shifted = np.fft.fftshift(np.fft.fft(x))
    x_FFT_freq_shifted = np.fft.fftshift(np.fft.fftfreq(n, d=si))
    return x_FFT_freq_shifted, x_FFT_shifted


def fft_noramlize(fft_res):
    n = len(fft_res)
    norm_fft = np.abs(fft_res)/(np.max(np.abs(fft_res)))
    return norm_fft

def line_fit(X,Y):
    P = np.polyfit(X,Y,1)
    dx = np.max(X) - np.min(X)
    X_new = np.array([np.min(X)-dx*0.05, np.max(X)+dx*0.05])
    Y_new = np.polyval(P, X_new)
    return X_new, Y_new, P

##------------------------------------------------------------------------

window_size = 3
window_size2 = 36
window_size3 = 72


df = pd.read_excel('C:\\Users\\mkara\\Desktop\\Academics\\Coral\\2026-02-03_Age-Model\\html_peak_match\\20260317_agemodel\\agemodel_pathsorted.xlsx')
paths = list(df['path_name'].value_counts().index.values)
paths = np.flip(np.sort(paths))
for path in paths:
    df.loc[df['path_name']==path, "path_ma"] = df.loc[df['path_name']==path, 'd13c'].rolling(window_size, center=True).mean()
#df[['d13c_ma','d18o_ma']] = df[['d13c','d18o']].rolling(window_size, center=True).mean()
df_sorted = df.sort_values("age_final")
df_sorted['d18o_pdb'] = 0.97001*df_sorted['d18o']-29.98

df_sorted['age_ws'] = df_sorted['age_final'].rolling(window_size, center=True).mean()
df_sorted['age_ws2'] = df_sorted['age_final'].rolling(window_size2, center=True).mean()

df_sorted[['d13c_ma','d18o_ma','d18o_pdb_ma']] = df_sorted[['d13c','d18o','d18o_pdb']].rolling(window_size, center=True).mean()
df_sorted[['d13c_ma2','d18o_ma2','d18o_pdb_ma2']] = df_sorted[['d13c','d18o','d18o_pdb']].rolling(window_size2, center=True).mean()
df_sorted[['d13c_ma3','d18o_ma3','d18o_pdb_ma3']] = df_sorted[['d13c','d18o','d18o_pdb']].rolling(window_size3, center=True).mean()
df_sorted[['d13c_std','d18o_std', 'd18o_pdb_std']] = df_sorted[['d13c','d18o','d18o_pdb']].rolling(window_size2, center=True).std()
#df_sorted['d13c_no_tectonic'] = df_sorted['d13c'] - df_sorted['d13c_ma2'].fillna(0)
#df_sorted['d18o_no_tectonic'] = df_sorted['d18o'] - df_sorted['d18o_ma2'].fillna(0)
#df_sorted['d18o_pdb_no_tectonic'] = df_sorted['d18o_pdb'] - df_sorted['d18o_pdb_ma2'].fillna(0)
df_sorted['d13c_no_tectonic'] = df_sorted['d13c'] - df_sorted['d13c_ma2']
df_sorted['d18o_no_tectonic'] = df_sorted['d18o'] - df_sorted['d18o_ma2']
df_sorted['d18o_pdb_no_tectonic'] = df_sorted['d18o_pdb'] - df_sorted['d18o_pdb_ma2']
df_sorted[['d13c_notec_ma','d18o_notec_ma','d18o_pdb_notec_ma']] = df_sorted[['d13c_no_tectonic','d18o_no_tectonic','d18o_pdb_no_tectonic']].rolling(window_size, center=True).mean()

df_sorted['d13c_no_tectonic2'] = df_sorted['d13c'] - df_sorted['d13c_ma3']
df_sorted['d18o_no_tectonic2'] = df_sorted['d18o'] - df_sorted['d18o_ma3']
df_sorted['d18o_pdb_no_tectonic2'] = df_sorted['d18o_pdb'] - df_sorted['d18o_pdb_ma3']

#-----------------------RESAMPLE MEAN & STD-----------------------------------------------------

df_sorted['age_dt'] = decimal_year_to_datetime(df_sorted['age_final'])
df_resampled_ = df_sorted[['age_final', 'age_dt', 'd13c','d18o', 'd18o_pdb']].set_index('age_dt')
df_resampled = df_resampled_.resample('3ME').mean()

df_resampled.interpolate(inplace=True)
df_resampled['age_resampled'] = [datetime_to_decimal_year(df_resampled.index[i]) for i in range(len(df_resampled))]
df_resampled[['d13c_ma','d18o_ma','d18o_pdb_ma']] = df_resampled[['d13c','d18o','d18o_pdb']].rolling(window_size, center=True).mean()
df_resampled[['d13c_ma2','d18o_ma2', 'd18o_pdb_ma2']] = df_resampled[['d13c','d18o','d18o_pdb']].rolling(window_size2, center=True).mean()
df_resampled[['d13c_ma3','d18o_ma3','d18o_pdb_ma3']] = df_resampled[['d13c','d18o','d18o_pdb']].rolling(window_size3, center=True).mean()
df_resampled[['d13c_std','d18o_std', 'd18o_pdb_std']] = df_resampled[['d13c','d18o','d18o_pdb']].rolling(window_size2, center=True).std()

df_resampled['d13c_no_tectonic'] = df_resampled['d13c'] - df_resampled['d13c_ma2']
df_resampled['d18o_no_tectonic'] = df_resampled['d18o'] - df_resampled['d18o_ma2']
df_resampled['d18o_pdb_no_tectonic'] = df_resampled['d18o_pdb'] - df_resampled['d18o_pdb_ma2']
df_resampled[['d13c_notec_ma','d18o_notec_ma','d18o_pdb_notec_ma']] = df_resampled[['d13c_no_tectonic','d18o_no_tectonic','d18o_pdb_no_tectonic']].rolling(window_size, center=True).mean()
#-----------------------------------------------------------------------------------------------
#-----------------------Savitzky-Golay tectonic trend (zero NaNs, no edge loss)-----------------

# SG requires a uniform grid → build monthly grid temporarily, apply SG,
# then interpolate back to both df_sorted (raw) and df_resampled (3-month).
#
# SG_LONG (~6yr): recommended for tectonic removal — keeps ENSO band intact.
# SG_SHORT (~3yr): shown for reference only — removes ENSO, do not use for LS.
#
# New columns on df_sorted:   d13c_sg_trend, d13c_no_tec_sg
#                              d18o_sg_trend, d18o_no_tec_sg
# New columns on df_resampled: d13c_sg_trend, d13c_no_tec_sg
#                              d18o_sg_trend, d18o_no_tec_sg

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

SG_WINDOW   = 73   # months (~6yr) on the monthly grid
SG_POLYORDER = 3

# Temporary monthly grid (used only to apply SG)
_df_monthly = (df_sorted[['age_final', 'age_dt', 'd13c', 'd18o', 'd18o_pdb']]
               .set_index('age_dt')
               .resample('ME').mean()
               .interpolate())
_t_monthly = _df_monthly['age_final'].values

for col in ['d13c', 'd18o', 'd18o_pdb']:
    # SG trend on monthly grid — no NaNs, edges handled by polynomial extrapolation
    _sg = savgol_filter(_df_monthly[col].values,
                        window_length=SG_WINDOW,
                        polyorder=SG_POLYORDER,
                        mode='interp')

    # Interpolator: monthly grid → anywhere
    _fn = interp1d(_t_monthly, _sg, bounds_error=False, fill_value='extrapolate')

    # Apply to raw irregular times (df_sorted)
    df_sorted[f'{col}_sg_trend']  = _fn(df_sorted['age_final'].values)
    df_sorted[f'{col}_no_tec_sg'] = df_sorted[col] - df_sorted[f'{col}_sg_trend']

    # Apply to 3-month resampled times (df_resampled)
    df_resampled[f'{col}_sg_trend']  = _fn(df_resampled['age_resampled'].values)
    df_resampled[f'{col}_no_tec_sg'] = df_resampled[col] - df_resampled[f'{col}_sg_trend']


#-----------------------CENTRAL DIFF METHOD-----------------------------------------------------
window = 12
col = 'd18o'

last = df_resampled.rolling(window, center=True).agg(lambda x: x.iloc[-1])
first = df_resampled.rolling(window, center=True).agg(lambda x: x.iloc[0])
df_resampled[col+'_cdm'] = (last[col] - first[col]) / (last['age_resampled'] - first['age_resampled'])

window = 12
col = 'd13c'

last = df_resampled.rolling(window, center=True).agg(lambda x: x.iloc[-1])
first = df_resampled.rolling(window, center=True).agg(lambda x: x.iloc[0])
df_resampled[col+'_cdm'] = (last[col] - first[col]) / (last['age_resampled'] - first['age_resampled'])
#-----------------------------------------------------------------------------------------------


#-----------------------------Moving OSL---------------------------------------
coefs_centered = pd.DataFrame()

# Define multiple windows and target columns
windows = [36, 36]
cols = ['d18o', 'd13c']

for window, col in zip(windows, cols):
    
    # Prepare data (assumes df_resampled contains all variables)
    data = df_resampled[[col, 'age_resampled']].dropna()

    if len(data) < window:
        print(f"Not enough data for column '{col}' with window {window}. Skipping.")
        continue
    
    # Setup endogenous and exogenous variables
    endog = data[col]
    exog = sm.add_constant(data['age_resampled'])

    # Fit rolling OLS
    rols = RollingOLS(endog, exog, window=window)
    rres = rols.fit()

    # Shift result to center and rename columns
    coefs = rres.params.shift(-(window // 2)).dropna()
    coefs = coefs.rename(columns={'const': f'{col}_intercept', 'age_resampled': f'{col}_slope'})

    # Merge into result
    coefs_centered = coefs_centered.join(coefs, how='outer')

# Add shared age column (assumes time index is same)
coefs_centered['age_resampled'] = df_resampled['age_resampled']

# Drop rows with NaNs in all slope columns (optional)
slope_cols = [f"{col}_slope" for col in cols]
coefs_centered.dropna(subset=slope_cols, how='all', inplace=True)
 
#-----------------------------------------------------------------------------------------------


#---------------------------------------FFT-----------------------------------------------------
dt = df_sorted['age_dt'].diff().mean().total_seconds()/((365*24+6)*60*60)
d13c_fft_freq, d13c_fft = FFT_res(df_resampled['d13c'], dt)
d18o_fft_freq, d18o_fft = FFT_res(df_resampled['d18o'], dt)

df_resampled[['d13c_std','d18o_std']] = df_resampled[['d13c','d18o']].rolling(window=window_size2, center=True).std()
d13c_stds = df_resampled.dropna()['d13c_std']
d18o_stds = df_resampled.dropna()['d18o_std']
d13c_std_fft_freq, d13c_std_fft = FFT_res(d13c_stds, dt)
d18o_std_fft_freq, d18o_std_fft = FFT_res(d18o_stds, dt)
#-----------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------
#Dark Bands 

pairs = []

for path, g in df.groupby("path_name", sort=False):

    ends = g.loc[g["age_end"].notna(),
                 ["index", "sample", "path_num", "age_end", "age2", "age_final", "d13c", "d18o"]]

    starts = g.loc[g["age2"].notna(),
                   ["index", "sample", "path_num", "age_end", "age2", "age_final", "d13c", "d18o"]]

    m = ends.merge(
        starts,
        left_on="age_end",
        right_on="age2",
        how="inner",
        suffixes=("_end", "_start")
    )

    if not m.empty:
        m.insert(0, "path_name", path)
        pairs.append(m)

dark_bands = pd.concat(pairs, ignore_index=True)
dark_bands = dark_bands[dark_bands['index_end'] > dark_bands['index_start']].reset_index(drop=True)
#-----------------------------------------------------------------------------------------------


df_resampled