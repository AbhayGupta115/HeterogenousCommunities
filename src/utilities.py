def set_fonts(extra_params={}):
    params = {
        "font.family": "Serif",
        # "font.sans-serif": ["Tahoma", "DejaVu Sans", "Lucida Grande", "Verdana"],
        "mathtext.fontset": "cm",
        "legend.fontsize": 16,
        "legend.title_fontsize": 14,
        "axes.labelsize": 28,
        "axes.titlesize": 28,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "figure.titlesize": 20,
    }
    for key, value in extra_params.items():
        params[key] = value
    pylab.rcParams.update(params)
