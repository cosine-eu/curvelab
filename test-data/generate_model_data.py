"""Generate one test CSV per model in the CurveLab registry.

Run from the test-data directory:
    python generate_model_data.py

Each file has columns: x, y, y_noisy, yerr
so you can test fits with and without error bars.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
NOISE_FRAC = 0.03  # 3% of signal range as noise sigma


def save(name, x, y_clean, noise_sigma=None):
    if noise_sigma is None:
        noise_sigma = max(NOISE_FRAC * (y_clean.max() - y_clean.min()), 1e-6)
    noise = rng.normal(0, noise_sigma, size=len(x))
    y_noisy = y_clean + noise
    yerr = np.full_like(x, noise_sigma)
    df = pd.DataFrame({"x": x, "y": y_noisy, "y_clean": y_clean, "yerr": yerr})
    path = f"model_{name}.csv"
    df.to_csv(path, index=False)
    print(f"  {path} ({len(x)} points)")


# ---- Peak / Line-shape models ----

x = np.linspace(-10, 10, 200)

# Gaussian: amplitude=5, center=0, sigma=1.5
y = 5 * np.exp(-x**2 / (2 * 1.5**2))
save("gaussian", x, y)

# Lorentzian: amplitude=5, center=1, sigma=0.8
y = (5 / np.pi) * (0.8 / ((x - 1)**2 + 0.8**2))
save("lorentzian", x, y)

# Voigt: approximate as weighted sum of Gaussian and Lorentzian
from scipy.special import voigt_profile
y = 4 * voigt_profile(x - 0.5, 0.8, 1.0)
save("voigt", x, y)

# PseudoVoigt: fraction * Lorentzian + (1-fraction) * Gaussian
frac = 0.4
g = np.exp(-(x - 0.5)**2 / (2 * 1.2**2))
l = (1 / np.pi) * (1.2 / ((x - 0.5)**2 + 1.2**2))
y = 5 * (frac * l + (1 - frac) * g)
save("pseudovoigt", x, y)

# Moffat: amplitude * (1 + ((x-center)/sigma)**2)**(-beta)
y = 6 * (1 + ((x - 0.3) / 1.5)**2)**(-2.5)
save("moffat", x, y)

# Pearson7: like Moffat with different parameterization
y = 5 * (1 + ((x - 0) / 1.0)**2 * (2**(1/3) - 1))**(-3)
save("pearson7", x, y)

# StudentT: center=0, sigma=1.5 (heavy tails)
from scipy.stats import t as t_dist
y = 4 * t_dist.pdf(x, df=3, loc=0, scale=1.5)
save("studentt", x, y)

# BreitWigner: amplitude * sigma / ((x-center)**2 + sigma**2)  (resonance)
y = 8 * 1.0 / ((x - 0.5)**2 + 1.0**2)
save("breitwigner", x, y)

# SplitLorentzian: different widths on each side
sigma_l, sigma_r = 0.6, 1.8
y_left = 5 * (sigma_l / ((x - 1)**2 + sigma_l**2))
y_right = 5 * (sigma_r / ((x - 1)**2 + sigma_r**2))
y = np.where(x < 1, y_left, y_right)
save("splitlorentzian", x, y)

# SkewedGaussian: Gaussian * (1 + erf(gamma * (x-center)))
from scipy.special import erf
y = 4 * np.exp(-(x - 1)**2 / (2 * 1.5**2)) * (1 + erf(1.5 * (x - 1) / 1.5))
save("skewedgaussian", x, y)

# SkewedVoigt: approximate with skewed Voigt profile
y = 3 * voigt_profile(x - 0.5, 0.6, 1.0) * (1 + erf(0.8 * (x - 0.5)))
save("skewedvoigt", x, y)

# ExponentialGaussian: convolution of exponential decay and Gaussian
from scipy.signal import fftconvolve
x_eg = np.linspace(-5, 20, 300)
gauss_kernel = np.exp(-np.linspace(-5, 5, 200)**2 / (2 * 1.0**2))
gauss_kernel /= gauss_kernel.sum()
exp_decay = np.where(x_eg >= 0, 3 * np.exp(-x_eg / 3.0), 0)
y_conv = fftconvolve(exp_decay, gauss_kernel, mode="same")
save("exponentialgaussian", x_eg, y_conv)

# Pearson4: asymmetric peak (use lmfit to generate)
from lmfit.models import Pearson4Model
m = Pearson4Model()
p = m.make_params(amplitude=5, center=0, sigma=1.5, expon=3)
y = m.eval(p, x=x)
save("pearson4", x, y)

# Doniach: asymmetric line shape (XPS)
from lmfit.models import DoniachModel
m = DoniachModel()
p = m.make_params(amplitude=5, center=0, sigma=0.8, height=1)
p["height"].vary = False
y = m.eval(p, x=x)
if np.all(np.isfinite(y)):
    save("doniach", x, y)
else:
    # Fallback: narrower range
    x_d = np.linspace(-5, 5, 200)
    y = m.eval(p, x=x_d)
    save("doniach", x_d, y)


# ---- Oscillatory / Decay models ----

x_osc = np.linspace(0, 10, 300)

# DampedOscillator: amplitude * exp(-x/tau) * cos(omega*x)
y = 5 * np.exp(-x_osc / 3.0) * np.cos(2 * np.pi * 1.5 * x_osc)
save("dampedoscillator", x_osc, y)

# DampedHarmonicOscillator (resonance curve)
# |H(w)|^2 = A / ((w^2 - w0^2)^2 + (gamma*w)^2)
x_freq = np.linspace(0.1, 10, 300)
w0, gamma, A = 5.0, 0.5, 100
y = A / np.sqrt((x_freq**2 - w0**2)**2 + (gamma * x_freq)**2)
save("dampedharmonicoscillator", x_freq, y)

# Sine: amplitude * sin(2*pi*frequency*x + shift) + offset
y = 3 * np.sin(2 * np.pi * 0.8 * x_osc + 0.5) + 1.0
save("sine", x_osc, y)

# Exponential: amplitude * exp(-x / decay)
x_exp = np.linspace(0, 10, 200)
y = 8 * np.exp(-x_exp / 2.5)
save("exponential", x_exp, y)

# PowerLaw: amplitude * x**exponent
x_pow = np.linspace(0.1, 10, 200)
y = 3.0 * x_pow**(-1.5)
save("powerlaw", x_pow, y)

# Lognormal
from scipy.stats import lognorm
x_ln = np.linspace(0.01, 10, 200)
y = 5 * lognorm.pdf(x_ln, s=0.8, scale=np.exp(1.0))
save("lognormal", x_ln, y)

# ThermalDistribution: Bose-Einstein-like
x_th = np.linspace(0.1, 5, 200)
kt = 0.5
y = 4 * x_th**2 / (np.exp(x_th / kt) - 1 + 1e-10)
save("thermaldistribution", x_th, y)


# ---- Polynomial / Background models ----

x_poly = np.linspace(-5, 5, 150)

# Constant
y = np.full_like(x_poly, 3.7)
save("constant", x_poly, y)

# Linear: intercept + slope * x
y = 2.0 + 1.5 * x_poly
save("linear", x_poly, y)

# Quadratic
y = 1.0 - 0.5 * x_poly + 0.3 * x_poly**2
save("quadratic", x_poly, y)

# Polynomial3
y = 0.5 + 0.2 * x_poly - 0.1 * x_poly**2 + 0.05 * x_poly**3
save("polynomial3", x_poly, y)

# Polynomial5 (skip 4, 6, 7 — similar)
y = (1.0 + 0.3 * x_poly - 0.05 * x_poly**2 + 0.01 * x_poly**3
     - 0.002 * x_poly**4 + 0.0003 * x_poly**5)
save("polynomial5", x_poly, y)


# ---- Step / Rectangle models ----

x_step = np.linspace(-5, 5, 200)

# Step: sigmoid step
y = 2.0 + 3.0 / (1 + np.exp(-(x_step - 0.5) / 0.3))
save("step", x_step, y)

# Rectangle: two opposing steps
y = 4.0 / (1 + np.exp(-(x_step + 2) / 0.3)) - 4.0 / (1 + np.exp(-(x_step - 2) / 0.3))
save("rectangle", x_step, y)


# ---- Composite models (for testing multi-component fits) ----

x_comp = np.linspace(-10, 10, 300)

# Two Gaussians + Linear baseline
g1 = 4 * np.exp(-(x_comp - (-3))**2 / (2 * 0.8**2))
g2 = 6 * np.exp(-(x_comp - 3)**2 / (2 * 1.2**2))
baseline = 0.5 + 0.1 * x_comp
y = g1 + g2 + baseline
save("composite_two_gaussians_linear", x_comp, y)

# Gaussian * Exponential decay (e.g., modulated signal)
x_mod = np.linspace(0, 15, 300)
y = 5 * np.exp(-(x_mod - 7)**2 / (2 * 2.0**2)) * np.exp(-x_mod / 10)
save("composite_gaussian_times_exponential", x_mod, y)

# Three peaks with different shapes (Gaussian + Lorentzian + Voigt)
g = 3 * np.exp(-(x_comp - (-4))**2 / (2 * 0.6**2))
l = (4 / np.pi) * (0.5 / ((x_comp - 0)**2 + 0.5**2))
v = 3 * voigt_profile(x_comp - 4, 0.5, 0.8)
y = g + l + v + 0.5
save("composite_mixed_peaks", x_comp, y)


print("\nDone! Generated test data for all models.")
