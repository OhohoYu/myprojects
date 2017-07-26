import matplotlib.pyplot as plt

# TRACE PLOTS
def posterior_dist(param, fit):
    fig = fit.plot(param)
    plt.style.use('seaborn-colorblind')
    plt.tight_layout()
    return fig
