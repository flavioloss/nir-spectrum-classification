import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import fft, fftfreq
import dsatools
from dsatools import utilits as ut
from dsatools._base._imf_decomposition import emd
import matplotlib.pyplot as plt

def create_emd(df, method, degree):
    """
    """
    data_array = np.array(df)
    data_emd = []
    for signal in data_array:
        imfs = emd(signal, order=16, method=method, max_itter=100)
        data_emd.append(imfs[degree])
    return data_emd


def SNV(df):
    """
    Standard Normal Variation (SNV): substract the row mean from each row and scales to unit variance
    """
    data_array = np.array(df)
    data_snv = []
    for i in data_array:
        data_snv.append(i - np.mean(i))
    return data_snv


def create_features(X, n_subsets, mode):
    if mode == 'all':
        return create_features_all(X, n_subsets)
    elif mode == 'stats':
        return create_features_stats(X, n_subsets)
    elif mode == 'freq':
        return create_features_freq(X, n_subsets)


def create_features_all(X, n_subsets):
    """
    Create a Dataframe of features of signal, given a number of subdivisions of the amplitudes
    """
    all_cols = np.array(X.columns)
    splits = np.array_split(all_cols, n_subsets)
    df_features = pd.DataFrame()
    for subset, vals in zip(range(1, n_subsets+1), splits):
        means = []; stds = []; medians = []; kurts = []; skews = []; maxs = []; mins = []
        peaks = []; rms = []; p2p = []; var = []; crest_factor = []
        max_f = []; sum_f = []; mean_f = []; var_f = []; peak_f = []; skew_f = []; kurt_f = []
        df_sub = X[vals]

        for idx, row in df_sub.iterrows():
            means.append(np.mean(row.values))
            stds.append(np.std(row.values))
            medians.append(np.median(row.values))
            kurts.append(stats.kurtosis(row.values))
            skews.append(stats.skew(row.values))
            maxs.append(np.max(row.values))
            mins.append(np.min(row.values))
            peaks.append(np.max(np.abs(row.values)))
            rms.append(np.sqrt(np.mean(row.values**2)))
            p2p.append(np.ptp(row.values))
            var.append(np.var(row.values))
            crest_factor.append(np.max(np.abs(row.values))/np.sqrt(np.mean(row.values**2)))

            # frequency domain
            ft = fft(row.values)
            S = np.abs(ft**2) / len(df_sub)
            max_f.append(np.max(S))
            mean_f.append(np.mean(S))
            sum_f.append(np.sum(S))
            var_f.append(np.var(S))
            peak_f.append(np.max(np.abs(S)))
            skew_f.append(stats.skew(S))
            kurt_f.append(stats.kurtosis(S))

            # peaks, _ = signal.find_peaks(row.values)
            # prominences = signal.peak_prominences(row.values, peaks)[0]
            # peaks.append(prominences[0])

        df_temp = pd.DataFrame({    
                                f"mean_subset{subset}": means, f"std_subset{subset}": stds,
                                f"median_subset{subset}": medians, f"kurtosis_subset{subset}": kurts,
                                f"skew_subset{subset}": skews, f"max_subset{subset}": maxs, f"min_subset{subset}": mins,
                                f"peaks_subset{subset}": peaks, f"rms_subset{subset}": rms, f"p2p_subset{subset}": p2p,
                                f"var_subset{subset}": var, f"crest_factor_subset{subset}": crest_factor,
                                f"max_f_subset{subset}": max_f, f"mean_f_subset{subset}": mean_f, f"sum_f_subset{subset}": sum_f,
                                f"var_f_subset{subset}": var_f, f"peak_f_subset{subset}": peak_f, 
                                f"skew_f_subset{subset}": skew_f, f"kurt_f_subset{subset}": kurt_f
                                })
        df_features = pd.concat([df_features, df_temp], axis=1)
    return df_features


def create_features_stats(X, n_subsets):
    """
    Create a Dataframe of features of signal, given a number of subdivisions of the amplitudes
    """
    all_cols = np.array(X.columns)
    splits = np.array_split(all_cols, n_subsets)
    df_features = pd.DataFrame()
    for subset, vals in zip(range(1, n_subsets+1), splits):
        means = []; stds = []; medians = []; kurts = []; skews = []; maxs = []; mins = []
        peaks = []; rms = []; p2p = []; var = []; crest_factor = []
        df_sub = X[vals]

        for idx, row in df_sub.iterrows():
            means.append(np.mean(row.values))
            stds.append(np.std(row.values))
            medians.append(np.median(row.values))
            kurts.append(stats.kurtosis(row.values))
            skews.append(stats.skew(row.values))
            maxs.append(np.max(row.values))
            mins.append(np.min(row.values))
            peaks.append(np.max(np.abs(row.values)))
            rms.append(np.sqrt(np.mean(row.values**2)))
            p2p.append(np.ptp(row.values))
            var.append(np.var(row.values))
            crest_factor.append(np.max(np.abs(row.values))/np.sqrt(np.mean(row.values**2)))

            # peaks, _ = signal.find_peaks(row.values)
            # prominences = signal.peak_prominences(row.values, peaks)[0]
            # peaks.append(prominences[0])

        df_temp = pd.DataFrame({    
                                f"mean_subset{subset}": means, f"std_subset{subset}": stds,
                                f"median_subset{subset}": medians, f"kurtosis_subset{subset}": kurts,
                                f"skew_subset{subset}": skews, f"max_subset{subset}": maxs, f"min_subset{subset}": mins,
                                f"peaks_subset{subset}": peaks, f"rms_subset{subset}": rms, f"p2p_subset{subset}": p2p,
                                f"var_subset{subset}": var, f"crest_factor_subset{subset}": crest_factor
                                })
        df_features = pd.concat([df_features, df_temp], axis=1)
    return df_features


def create_features_freq(X, n_subsets):
    """
    Create a Dataframe of features of signal, given a number of subdivisions of the amplitudes
    """
    all_cols = np.array(X.columns)
    splits = np.array_split(all_cols, n_subsets)
    df_features = pd.DataFrame()
    for subset, vals in zip(range(1, n_subsets+1), splits):
        max_f = []; sum_f = []; mean_f = []; var_f = []; peak_f = []; skew_f = []; kurt_f = []
        df_sub = X[vals]

        for idx, row in df_sub.iterrows():
            # frequency domain
            ft = fft(row.values)
            S = np.abs(ft**2) / len(df_sub)
            max_f.append(np.max(S))
            mean_f.append(np.mean(S))
            sum_f.append(np.sum(S))
            var_f.append(np.var(S))
            peak_f.append(np.max(np.abs(S)))
            skew_f.append(stats.skew(S))
            kurt_f.append(stats.kurtosis(S))

            # peaks, _ = signal.find_peaks(row.values)
            # prominences = signal.peak_prominences(row.values, peaks)[0]
            # peaks.append(prominences[0])

        df_temp = pd.DataFrame({
                                f"max_f_subset{subset}": max_f, f"mean_f_subset{subset}": mean_f, f"sum_f_subset{subset}": sum_f,
                                f"var_f_subset{subset}": var_f, f"peak_f_subset{subset}": peak_f, 
                                f"skew_f_subset{subset}": skew_f, f"kurt_f_subset{subset}": kurt_f
                                })
        df_features = pd.concat([df_features, df_temp], axis=1)
    return df_features



def create_features_means(X, n_subsets):
    """
    Create a Dataframe of features of signal, given a number of subdivisions of the amplitudes
    """
    all_cols = np.array(X.columns)
    splits = np.array_split(all_cols, n_subsets)
    df_features = pd.DataFrame()
    for subset, vals in zip(range(1, n_subsets+1), splits):
        means = []
        df_sub = X[vals]

        for idx, row in df_sub.iterrows():
            means.append(np.mean(row.values))

        df_temp = pd.DataFrame({    
                                f"mean_subset{subset}": means
                                })
        df_features = pd.concat([df_features, df_temp], axis=1)
    return df_features
