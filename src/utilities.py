import pylab


def set_fonts(extra_params={}):
    params = {
        "font.family": "Serif",
        # "font.sans-serif": ["Tahoma", "DejaVu Sans", "Lucida Grande", "Verdana"],
        "mathtext.fontset": "cm",
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 12,
    }
    for key, value in extra_params.items():
        params[key] = value
    pylab.rcParams.update(params)
