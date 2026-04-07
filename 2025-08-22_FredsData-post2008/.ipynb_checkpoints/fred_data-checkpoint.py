import pandas as pd
import numpy as np

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


fred = pd.read_excel('C:\\Users\\mkara\\Desktop\\Academics\\Coral\\2025-08-22_FredsData-post2008\\FredData_editedMemo.xlsx').dropna()
fred['d18o_pdb'] = -(fred['d18o']-20.91)/1.03091
fred[['d13c_ma','d18o_ma']] = fred[['d13c','d18o_pdb']].rolling(window_size, center=True).mean()
fred[['d13c_std','d18o_std']] = fred[['d13c','d18o_pdb']].rolling(10, center=True).std()
fred_resampled = fred.drop(fred.columns[3:], axis=1)
fred_resampled['age_dt'] = decimal_year_to_datetime(fred_resampled['decimal_year'])
fred_resampled.set_index('age_dt', inplace=True)
fred_resampled = fred_resampled.resample('30D').mean()
fred_resampled.interpolate(inplace=True)
fred_resampled