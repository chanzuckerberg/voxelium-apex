import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def get_default_cmap():
    cm_tropical = [
        (1.000, 1.000, 1.000), (0.267, 0.987, 0.988), (0.154, 0.934, 0.722), (0.429, 0.843, 0.431),
        (0.647, 0.719, 0.203), (0.772, 0.580, 0.031), (0.837, 0.429, 0.067), (0.850, 0.273, 0.195),
        (0.808, 0.111, 0.354), (0.699, 0.022, 0.528), (0.565, 0.054, 0.646)
    ]

    cm_blush = [
       "#FFFFFF", "#7ef5ff", "#7757de", "#f83dff", "#C71078"#, "#d35fdd", "#D061B0", "#ce3266", "#9f223b", "#7a0018"
    ]

    for i in range(len(cm_blush)):
        h = cm_blush[i].lstrip('#')
        cm_blush[i] = tuple(int(h[i:i+2], 16) / 256 for i in (0, 2, 4))

    n_bins = 20  # Use 100 bins for smooth transitions
    return LinearSegmentedColormap.from_list('positron', np.array(cm_blush), N=n_bins)
