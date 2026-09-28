"""
TA Analyzer - LPSVD (Linear Prediction Singular Value Decomposition) analysis.

Ported from the Jupyter notebook
``LPSVD_Lorentzian_Data Export_ver3.ipynb``. Two complementary models are
exposed:

* **Lorentzian (LPSVD)** - exponentially-damped sinusoids extracted from the
  Hankel SVD of the time-domain trace.
* **Gaussian** - sum of Gaussian-damped oscillators fitted by non-linear
  least squares with initial frequencies seeded from the FFT peaks.

All routines operate on a 1D time series; the Coherence dialog calls them
with the residual averaged or sliced over wavelength.

Internal frequency convention is Hz; helper conversions map to THz / cm^-1.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import svd, lstsq
from scipy.signal import detrend as _scipy_detrend, find_peaks
from scipy.optimize import curve_fit
from scipy.special import i0


# =====================================================================
# Unit helpers
# =====================================================================
_TU2S = {'fs': 1e-15, 'ps': 1e-12, 'ns': 1e-9, 'us': 1e-6}
C_CM = 2.99792458e10            # speed of light, cm/s


def tu_to_s(time_unit: str) -> float:
    """Return seconds-per-(time_unit). Defaults to picoseconds."""
    return _TU2S.get(str(time_unit).lower(), 1e-12)


def hz_to(freq_hz: np.ndarray | float, unit: str):
    """Convert Hz → ``unit`` ('Hz' | 'THz' | 'cm-1')."""
    u = (unit or 'cm-1').lower()
    if u == 'cm-1':
        return np.asarray(freq_hz) / C_CM
    if u == 'thz':
        return np.asarray(freq_hz) / 1e12
    return np.asarray(freq_hz)


# =====================================================================
# Preprocessing
# =====================================================================
def preprocess_signal(t, y, *, t_start=None, t_end=None,
                      do_polyfit=True, polyfit_deg=6,
                      do_detrend=True, do_center=True,
                      append_count=0):
    """Crop ``[t_start, t_end]``, optionally polynomial-baseline-remove,
    linearly detrend, mean-center, and optionally tile the trace.

    Returns dict with keys
        t_late      : (N,) time axis (same unit as input)
        y_raw       : (N,) raw signal after crop (and tile)
        y_centered  : (N,) preprocessed (zero-mean) signal
        bg          : (N,) total baseline removed so y_raw = y_centered + bg
        bg_poly     : (N,) polynomial baseline (zeros if disabled)
        bg_detrend  : (N,) linear detrend line removed after the polynomial
                          step (zeros if disabled)
        bg_center   : scalar mean offset removed (0.0 if disabled)
        poly_coefs  : np.ndarray | None polynomial coefficients in highest-
                          power-first order, exactly as returned by
                          ``np.polyfit``.  ``None`` if the polynomial step
                          was disabled or skipped.
        poly_degree : int the polynomial degree used (0 if disabled).
        dt          : scalar median Δt (same unit as input)
    """
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if t.size != y.size:
        raise ValueError('t and y must have the same length.')
    if t_start is None:
        t_start = float(t.min())
    if t_end is None:
        t_end = float(t.max())
    if t_end < t_start:
        t_start, t_end = t_end, t_start
    mask = (t >= t_start) & (t <= t_end)
    t_late = t[mask]
    y_raw = y[mask]
    if t_late.size < 8:
        raise ValueError(f'Time window too short (N={t_late.size}; need ≥ 8).')

    fin = np.isfinite(y_raw)
    if not fin.all():
        y_raw = np.where(fin, y_raw, 0.0)

    y_proc = y_raw.copy()
    bg_poly = np.zeros_like(y_proc)
    poly_coefs = None
    poly_deg_used = 0
    if do_polyfit and y_proc.size > int(polyfit_deg) + 1:
        try:
            poly_coefs = np.polyfit(t_late, y_proc, int(polyfit_deg))
            bg_poly = np.polyval(poly_coefs, t_late)
            y_proc = y_proc - bg_poly
            poly_deg_used = int(polyfit_deg)
        except Exception:
            poly_coefs = None
            bg_poly = np.zeros_like(y_proc)
            poly_deg_used = 0

    bg_detrend = np.zeros_like(y_proc)
    if do_detrend and y_proc.size >= 2:
        y_after = _scipy_detrend(y_proc, type='linear')
        bg_detrend = y_proc - y_after
        y_proc = y_after

    bg_center = 0.0
    if do_center:
        bg_center = float(np.mean(y_proc))
        y_proc = y_proc - bg_center

    bg = y_raw - y_proc

    dts = np.diff(t_late)
    dt_med = float(np.median(dts)) if dts.size else 0.0
    if dt_med <= 0:
        raise ValueError('Non-positive median Δt in preprocessed window.')

    n_app = int(max(0, append_count))
    if n_app > 0:
        y_proc = np.tile(y_proc, n_app + 1)
        y_raw = np.tile(y_raw, n_app + 1)
        bg = np.tile(bg, n_app + 1)
        bg_poly = np.tile(bg_poly, n_app + 1)
        bg_detrend = np.tile(bg_detrend, n_app + 1)
        N_new = y_proc.size
        t_late = t_start + np.arange(N_new) * dt_med

    return {'t_late': t_late, 'y_raw': y_raw, 'y_centered': y_proc,
            'bg': bg, 'bg_poly': bg_poly, 'bg_detrend': bg_detrend,
            'bg_center': bg_center, 'poly_coefs': poly_coefs,
            'poly_degree': poly_deg_used, 'dt': dt_med}


# =====================================================================
# Windowing
# =====================================================================
def apply_window(t_shifted, *, window_type='kaiser', beta=8.0, sigma=2.5):
    """Return a windowing array with the same length as ``t_shifted``.

    ``t_shifted`` must start at 0 and be expressed in the same unit as
    ``sigma`` (e.g. ps).  Unknown window names fall back to a rectangular
    (all-ones) window.
    """
    t = np.asarray(t_shifted, dtype=float)
    wt = (window_type or 'none').lower()
    if wt == 'kaiser':
        tmax = float(np.max(t)) if t.size else 0.0
        if tmax <= 0:
            return np.ones_like(t)
        arg = np.clip(1.0 - (t / tmax) ** 2, 0.0, None)
        return i0(float(beta) * np.sqrt(arg)) / i0(float(beta))
    if wt == 'gaussian':
        s = float(sigma)
        if s <= 0:
            return np.ones_like(t)
        return np.exp(-0.5 * (t / s) ** 2)
    return np.ones_like(t)


def window_label(window_type: str, beta: float, sigma: float) -> str:
    wt = (window_type or 'none').lower()
    if wt == 'kaiser':
        return rf'Kaiser ($\beta$={beta:g})'
    if wt == 'gaussian':
        return rf'Gaussian ($\sigma$={sigma:g})'
    return 'No window'


# =====================================================================
# Lorentzian (LPSVD) core
# =====================================================================
def lpsvd_core(x, dt_s, F):
    """Run LPSVD on signal ``x`` with sampling period ``dt_s`` (seconds).

    Returns ``(amps, freqs_Hz, dampings_Hz, phases_rad)``; the frequency
    array may contain negative entries (mirror modes).
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    if N < 4:
        raise ValueError('LPSVD needs at least 4 samples.')
    M = N // 2
    L = N - M
    F = int(min(max(1, F), M))
    H = np.zeros((L, M), dtype=complex)
    for i in range(L):
        H[i, :] = x[i:i + M]
    h = x[M:M + L].astype(complex)
    U, S, Vh = svd(H, full_matrices=False)
    F_eff = int(min(F, S.size))
    U_F = U[:, :F_eff]
    S_F = np.diag(S[:F_eff])
    V_F = Vh[:F_eff, :].conj().T
    H_inv = V_F @ np.linalg.inv(S_F) @ U_F.conj().T
    q = H_inv @ h
    # Forward-LP characteristic polynomial: z^M = Σ q[j] z^j (j = 0..M-1),
    # i.e. z^M - q[M-1] z^{M-1} - … - q[0] = 0.  numpy.roots expects
    # descending-power coefficients, so q must be reversed before negation.
    # (The reference notebook uses the un-reversed order, which biases pole
    # frequencies — see TA Analyzer test_lpsvd.test_lpsvd_recovers_lorentzian.)
    roots = np.roots(np.concatenate(([1.0], -q[::-1])))
    if roots.size == 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]))
    # Sort by |z| descending so the slowest-damping modes come first
    z_F = roots[np.argsort(np.abs(roots))[::-1]][:F_eff]
    freqs_hz = np.angle(z_F) / (2.0 * np.pi * dt_s)
    damp_hz = -np.log(np.abs(z_F)) / dt_s
    # Recover amplitudes/phases from the over-determined system Z·c = x
    n_idx = np.arange(N)
    Z = z_F[None, :] ** n_idx[:, None]
    c, *_ = lstsq(Z, x.astype(complex))
    amps = np.abs(c)
    phases = np.angle(c)
    return amps, freqs_hz, damp_hz, phases


def lpsvd_modes(y, dt_s, num_modes, *, F_target=None):
    """Extract the top ``num_modes`` positive-frequency Lorentzian modes.

    Each mode is a dict ``{'freq_Hz', 'damp_Hz', 'amp', 'phase'}`` and the
    list is returned sorted by amplitude descending. Amplitudes are doubled
    so the real-signal reconstruction ``A·exp(-Γt)·cos(2πft+φ)`` matches.
    """
    if F_target is None:
        F_target = max(20, int(num_modes) * 3)
    amps, freqs, damps, phases = lpsvd_core(y, dt_s, F_target)
    if freqs.size == 0:
        return []
    pos = freqs > 0
    p_freq = freqs[pos]
    p_amp = amps[pos] * 2.0
    p_damp = damps[pos]
    p_phase = phases[pos]
    if p_freq.size == 0:
        return []
    n_take = int(min(num_modes, p_freq.size))
    order = np.argsort(p_amp)[::-1][:n_take]
    return [{'freq_Hz': float(p_freq[i]),
             'damp_Hz': float(p_damp[i]),
             'amp': float(p_amp[i]),
             'phase': float(p_phase[i])} for i in order]


def reconstruct_lorentzian(t_shifted_s, mode):
    """Real time-domain trace of a single Lorentzian-damped oscillator."""
    a = mode['amp']
    f = mode['freq_Hz']
    d = mode['damp_Hz']
    p = mode['phase']
    t = np.asarray(t_shifted_s, dtype=float)
    val = ((a / 2.0) * np.exp(1j * p) * np.exp((-d + 1j * 2.0 * np.pi * f) * t)
           + (a / 2.0) * np.exp(-1j * p) * np.exp((-d - 1j * 2.0 * np.pi * f) * t))
    return np.real(val)


# =====================================================================
# Gaussian damped oscillator (curve_fit)
# =====================================================================
def _gaussian_osc_ps(t_ps, A, f_THz, gamma_per_ps2, phi):
    return A * np.exp(-gamma_per_ps2 * t_ps ** 2) * np.cos(
        2.0 * np.pi * f_THz * t_ps + phi)


def _multi_gaussian_ps(t_ps, *params):
    out = np.zeros_like(t_ps, dtype=float)
    for i in range(len(params) // 4):
        out = out + _gaussian_osc_ps(t_ps, *params[i * 4: i * 4 + 4])
    return out


def gaussian_fit_modes(y, t_shifted_s, dt_s, num_modes, *,
                       freq_window_THz=(0.1, 8.0)):
    """Fit ``y`` with a sum of ``num_modes`` Gaussian-damped oscillators.

    Parameters
    ----------
    y : (N,) preprocessed zero-mean signal
    t_shifted_s : (N,) time axis (starts at 0) in seconds
    dt_s : sampling period in seconds
    num_modes : number of oscillators to fit
    freq_window_THz : (lo, hi) frequency band scanned for initial peaks.

    Returns
    -------
    list of dicts (sorted by ascending frequency):
        'amp', 'freq_Hz', 'damp_per_s2' (Γ in 1/s²), 'phase'.

    Raises ``RuntimeError`` if the non-linear fit fails to converge.
    """
    y = np.asarray(y, dtype=float).ravel()
    t_s = np.asarray(t_shifted_s, dtype=float).ravel()
    if y.size != t_s.size:
        raise ValueError('y and t_shifted_s must have the same length.')
    N = y.size
    if N < 8:
        raise ValueError('Need at least 8 samples for Gaussian fit.')
    if num_modes < 1:
        raise ValueError('num_modes must be ≥ 1.')

    # ---- FFT-based initial guesses (in THz) ----
    freq_axis_hz = np.fft.fftfreq(N, d=dt_s)
    mag = np.abs(np.fft.fft(y)) * dt_s
    f_lo = float(freq_window_THz[0]) * 1e12
    f_hi = float(freq_window_THz[1]) * 1e12
    pos = (freq_axis_hz > f_lo) & (freq_axis_hz < f_hi)
    if not pos.any():
        pos = freq_axis_hz > 0
    f_pos_THz = freq_axis_hz[pos] * 1e-12
    mag_pos = mag[pos]

    initial_THz = []
    if mag_pos.size:
        peaks, _ = find_peaks(mag_pos, distance=2)
        ordered = sorted(peaks, key=lambda i: mag_pos[i], reverse=True)
        for k in range(num_modes):
            if k < len(ordered):
                initial_THz.append(float(f_pos_THz[ordered[k]]))
            else:
                # Fallback fillers spaced inside the window
                lo = float(freq_window_THz[0])
                initial_THz.append(lo + (k + 1) * 0.5)
    else:
        lo = float(freq_window_THz[0])
        initial_THz = [lo + (k + 1) * 0.5 for k in range(num_modes)]

    # ---- Build bounds & initial parameters (work in ps / THz) ----
    t_ps = t_s * 1e12
    y_scale = float(np.max(np.abs(y))) if np.max(np.abs(y)) > 0 else 1.0
    p0, lo_b, hi_b = [], [], []
    for f_THz in initial_THz:
        p0.extend([0.2 * y_scale, f_THz, 0.1, 0.0])
        lo_b.extend([0.0, max(0.01, f_THz - 0.5), 1e-3, -np.pi])
        hi_b.extend([10.0 * y_scale + 1e-12, f_THz + 0.5, 15.0, np.pi])

    try:
        popt, _ = curve_fit(_multi_gaussian_ps, t_ps, y, p0=p0,
                            bounds=(lo_b, hi_b), maxfev=150000)
    except Exception as e:
        raise RuntimeError(f'Gaussian curve_fit failed: {e}') from e

    modes = []
    for i in range(num_modes):
        A, f_THz, gamma_ps2, phi = popt[i * 4: i * 4 + 4]
        modes.append({
            'amp': float(A),
            'freq_Hz': float(f_THz * 1e12),
            'damp_per_s2': float(gamma_ps2 * 1e24),   # 1/ps² → 1/s²
            'phase': float(phi),
        })
    modes.sort(key=lambda m: m['freq_Hz'])
    return modes


def reconstruct_gaussian(t_shifted_s, mode):
    """Real time-domain trace of a single Gaussian-damped oscillator."""
    a = mode['amp']
    f = mode['freq_Hz']
    g = mode['damp_per_s2']
    p = mode['phase']
    t = np.asarray(t_shifted_s, dtype=float)
    return a * np.exp(-g * t * t) * np.cos(2.0 * np.pi * f * t + p)


# =====================================================================
# Reconstruction dispatcher + reliability evaluation
# =====================================================================
def reconstruct_modes(t_shifted_s, modes, model):
    """Per-mode reconstructions + their sum."""
    if model == 'gaussian':
        recons = [reconstruct_gaussian(t_shifted_s, m) for m in modes]
    else:
        recons = [reconstruct_lorentzian(t_shifted_s, m) for m in modes]
    total = (np.sum(np.stack(recons, axis=0), axis=0)
             if recons else np.zeros_like(t_shifted_s, dtype=float))
    return recons, total


def evaluate_reliability(modes, std_residual):
    """Tag each mode with ``reliable`` = amp > σ(residual)."""
    out = []
    for m in modes:
        d = dict(m)
        d['reliable'] = bool(d['amp'] > std_residual)
        out.append(d)
    return out


# =====================================================================
# End-to-end driver used by the GUI
# =====================================================================
def run_lpsvd_analysis(t, y, *, time_unit='ps', t_start=None, t_end=None,
                       do_polyfit=True, polyfit_deg=6,
                       do_detrend=True, do_center=True,
                       append_count=0,
                       model='lorentzian', num_modes=6,
                       window_type='kaiser', beta=8.0,
                       gaussian_sigma=2.5, apply_window_to_fft=True,
                       fft_pad_factor=4, fft_pad_min=8192):
    """One-call wrapper: preprocess → fit (LPSVD / Gaussian) → FFT.

    Returns a result dictionary with all arrays needed to draw the four
    LPSVD result panels and export tables.
    """
    pre = preprocess_signal(t, y, t_start=t_start, t_end=t_end,
                            do_polyfit=do_polyfit, polyfit_deg=polyfit_deg,
                            do_detrend=do_detrend, do_center=do_center,
                            append_count=append_count)
    t_late = pre['t_late']
    y_raw = pre['y_raw']
    y_cen = pre['y_centered']
    bg = pre['bg']
    bg_poly = pre['bg_poly']
    bg_detrend = pre['bg_detrend']
    bg_center = pre['bg_center']
    poly_coefs = pre['poly_coefs']
    poly_deg_used = pre['poly_degree']
    dt = pre['dt']                            # input units (e.g. ps)
    tu2s = tu_to_s(time_unit)
    dt_s = dt * tu2s
    t_shifted = t_late - t_late[0]
    t_shifted_s = t_shifted * tu2s

    # --- Fit ---
    if str(model).lower() == 'gaussian':
        modes = gaussian_fit_modes(y_cen, t_shifted_s, dt_s, int(num_modes))
    else:
        modes = lpsvd_modes(y_cen, dt_s, int(num_modes))
    recons, total_cen = reconstruct_modes(t_shifted_s, modes, model)
    total_with_bg = total_cen + bg
    residual = y_raw - total_with_bg
    std_res = float(np.std(residual)) if residual.size else 0.0
    modes_tagged = evaluate_reliability(modes, std_res)

    # --- FFT (optionally with window) ---
    if apply_window_to_fft:
        win = apply_window(t_shifted, window_type=window_type,
                           beta=beta, sigma=gaussian_sigma)
    else:
        win = np.ones_like(t_shifted)

    N_pad = max(int(fft_pad_min), int(fft_pad_factor) * t_late.size)
    freq_hz = np.fft.fftshift(np.fft.fftfreq(N_pad, d=dt_s))
    fft_orig = np.fft.fftshift(np.fft.fft(y_cen * win, n=N_pad)) * dt_s
    fft_fit = np.fft.fftshift(np.fft.fft(total_cen * win, n=N_pad)) * dt_s
    fft_modes = [np.fft.fftshift(np.fft.fft(r * win, n=N_pad)) * dt_s
                 for r in recons]

    # Record what was used so the GUI and CSV bundle can document the
    # full processing condition (without the GUI having to peek at the
    # internals).
    t_win_used = (float(t_late[0]), float(t_late[-1])) if t_late.size else (
        float('nan'), float('nan'))
    params = {
        'time_unit': str(time_unit).lower(),
        't_start_req': (None if t_start is None else float(t_start)),
        't_end_req': (None if t_end is None else float(t_end)),
        't_start_used': t_win_used[0],
        't_end_used': t_win_used[1],
        'N_window': int(t_late.size),
        'do_polyfit': bool(do_polyfit),
        'polyfit_deg_requested': int(polyfit_deg),
        'polyfit_deg_used': int(poly_deg_used),
        'poly_coefs': (None if poly_coefs is None
                       else np.asarray(poly_coefs, dtype=float).tolist()),
        'do_detrend': bool(do_detrend),
        'do_center': bool(do_center),
        'bg_center': float(bg_center),
        'append_count': int(append_count),
        'model': str(model).lower(),
        'num_modes': int(num_modes),
        'window_type': str(window_type),
        'beta': float(beta),
        'gaussian_sigma': float(gaussian_sigma),
        'apply_window_to_fft': bool(apply_window_to_fft),
        'fft_pad_factor': int(fft_pad_factor),
        'fft_pad_min': int(fft_pad_min),
        'fft_N_pad': int(N_pad),
    }

    # Envelope shown in the time-domain plot (matches notebook plot 0,0)
    y_amp = float(np.max(np.abs(y_cen))) if y_cen.size else 1.0
    bg_mean = float(np.mean(bg)) if bg.size else 0.0
    win_envelope = win * y_amp + bg_mean

    return {
        'modes': modes_tagged,
        'model': str(model).lower(),
        'window_type': window_type,
        'window': win,
        'window_envelope': win_envelope,
        'window_label': window_label(window_type, beta, gaussian_sigma),
        't_late': t_late,
        't_shifted': t_shifted,
        'dt': dt,
        'dt_s': dt_s,
        'y_raw': y_raw,
        'y_centered': y_cen,
        'bg': bg,
        'bg_poly': bg_poly,
        'bg_detrend': bg_detrend,
        'bg_center': bg_center,
        'poly_coefs': poly_coefs,
        'poly_degree': poly_deg_used,
        'params': params,
        'recons': recons,
        'total_centered': total_cen,
        'total_with_bg': total_with_bg,
        'residual': residual,
        'std_residual': std_res,
        'freq_hz': freq_hz,
        'fft_orig': fft_orig,
        'fft_fit': fft_fit,
        'fft_modes': fft_modes,
        'time_unit': str(time_unit).lower(),
    }
