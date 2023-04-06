from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.simplefilter('ignore')
sns.set_style('darkgrid')
sns.set_palette("icefire")


def plot_mean_labels(df, column, figsize=(9, 6), title="Subsection ", n_subsets=1):
    """
    Funcao para criar medias para cada doenca
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    df_means = df.groupby(column).mean()
    labels = df_means.index.to_list()
    range_spectrum = df_means.columns.to_numpy(dtype=np.float64)
    if n_subsets > 1:
        ymin = np.min(df_means.min().values)
        ymax = np.max(df_means.max().values)
        lines_location = [i[0] for i in np.array_split(np.array(range_spectrum), n_subsets)]
        ax.vlines(lines_location, ymin=ymin, ymax=ymax, color='r', alpha=0.4)
    for i in range(len(df_means)):
        sns.lineplot(range_spectrum, np.array(df_means)[i], label=labels[i])


def plot_labels(df, column, figsize=(9, 6), title="Grafico da Media de Cada classe"):
    fig = plt.figure(figsize=figsize)
    for col in df[column].value_counts().index:
        df_sub = df.loc[df[column] == col]
        tidy = df_sub.iloc[:, :-2].stack().reset_index().rename(columns={"level_1": "comp_onda", 0: "absortion"})
        ax = sns.lineplot(data=tidy, x='comp_onda', y='absortion', label=col)



def plot_transpose(df, figsize=(9, 6), title="Grafico de linha dos espectros"):
    df = df[df.columns[1:-2]].transpose().reset_index()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    for spec in df.columns[1:]:
        sns.lineplot(data=df, x="index", y=spec, ax=ax)


def plot_mean_dist(df, column):
    """
    Plota a distribuicao de classes
    """
    classes = df[column].value_counts().index
    df_hist = pd.DataFrame(columns=['mean', 'label'])
    for classe in classes:
        class_mean = []
        df_subset = df.loc[df[column] == classe]
        df_transpose = df_subset[df_subset.columns[1:-2]].transpose()
        for col in df_transpose.columns:
            class_mean.append(df_transpose[col].mean())
        df_hist = pd.concat([df_hist, pd.DataFrame({'mean': class_mean, 'label': classe})])
    df_hist.reset_index(drop=True, inplace=True)
    sns.displot(data=df_hist, x='mean', col='label')


def plot_spectre(X, y, figsize=(14, 8), title="Spectres line plot"):
    plt.figure(figsize=figsize)
    for (idx, row), y_label in zip(X.iterrows(), y):
        if y_label == 0:
            plt.plot(X.columns.to_numpy(dtype=np.float64), row, label="Benign", color='navy')
        if y_label == 1:
            plt.plot(X.columns.to_numpy(dtype=np.float64), row, label="Malignant", color='firebrick')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel('Wavelength')
    plt.ylabel('Absortion level')
    plt.title(title)


def SNV(data):
    """
    Standard Normal Variation (SNV): substract the row mean from each row and scales to unit variance
    """
    data_array = np.array(data)
    data_snv = []
    for i in data_array:
        line_snv = (i - np.mean(i)) / np.std(i)
        data_snv.append(line_snv)

    return np.array(data_snv)