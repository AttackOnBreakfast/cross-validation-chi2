# -----------------------------
# src/utils.py
# -----------------------------
import numpy as np
from scipy.interpolate import interp1d

def rescale(x):
    return 2 * (x - x.min()) / (x.max() - x.min()) - 1

def smooth_curve(x, y, x_dense, kind='cubic'):
    interpolator = interp1d(x, y, kind=kind)
    return interpolator(x_dense)