import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sample_hist(sample, title, xlab, ylab, xlim_min, xlim_max, bin_no):
    par_df = pd.DataFrame(np.concatenate(sample))
    fig = plt.figure()
    plt.style.use('seaborn-colorblind')
    plt.xlim(xlim_min, xlim_max)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.hist(par_df[~np.isnan(par_df)], bins = bin_no, histtype='barstacked', stacked=True, normed=True, rwidth=0.85,
             edgecolor='black', linewidth=1)
    plt.title(title)
    plt.tight_layout()
    return fig






