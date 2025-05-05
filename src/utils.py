### src/utils.py
from scipy.interpolate import interp1d

def smooth_curve(x_vals, y_vals, x_dense, kind='cubic'):
    interpolator = interp1d(x_vals, y_vals, kind=kind)
    return interpolator(x_dense)
