"""
TA Analyzer - core numerical routines and file I/O.

Ported from the MATLAB class TAAnalyzer (handle class).
Author: Python port generated from MATLAB original.

This module contains the pure-numerical functions that the MATLAB version
defined as methods of the class but which have no GUI dependency:
    * parseDataFile / writeDataFile
    * expIRFConv, buildGaBasis
    * fitGlobalAnalysis (VARPRO)
    * computeEADSFromDADS
    * chirpModel, fitChirpParams
    * computeRidge helper
    * getColormap / divergingColormap
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc, erfcx
from scipy.optimize import minimize, least_squares
from scipy.ndimage import correlate1d
from matplotlib import cm as _mpl_cm


# =====================================================================
# File I/O
# =====================================================================
def _detect_delimiter(sample: str) -> str | None:
    """Pick the delimiter (tab, comma, or whitespace) by count in the sample.

    Returns ``None`` to mean "any run of whitespace" (for loadtxt-style files).
    """
    n_tab = sample.count('\t')
    n_comma = sample.count(',')
    if n_tab >= 5 and n_tab >= n_comma:
        return '\t'
    if n_comma >= 5:
        return ','
    return None


def _to_float(s) -> float:
    """Lenient string→float that returns NaN on anything non-numeric."""
    if s is None:
        return np.nan
    s = str(s).strip()
    if not s:
        return np.nan
    if s.lower() in ('nan', 'na', 'n/a', '-', 'inf', '+inf', '-inf'):
        if s.lower() == 'inf' or s.lower() == '+inf':
            return np.inf
        if s.lower() == '-inf':
            return -np.inf
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _is_numeric(s) -> bool:
    """Return True iff ``s`` parses as a finite or ±inf float (but not NaN-like)."""
    if s is None:
        return False
    s = str(s).strip()
    if not s:
        return False
    if s.lower() in ('nan', 'na', 'n/a', '-'):
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_data_file(fullpath: str):
    """Parse a TA data file (CSV/DAT/TXT, comma/tab/space delimited).

    Two file layouts are auto-detected:

    **Format A (MATLAB style, 1 header row + 1 header column):**
        row[0]    = [corner, t1, t2, ..., tN]    (delay times)
        col[0]    = [corner, l1, l2, ..., lM]    (wavelengths)
        row[1:][1:]                              = ΔA matrix (M x N)

    **Format B (2-column metadata prefix):**
        row[0][0]    = metadata string (ignored)
        row[0][1]    = ignored (often 0)
        row[0][2:]   = [t1, t2, ..., tN]         (delay times)
        row[i][0]    = metadata / empty (ignored)
        row[i][1]    = wavelength i              (for i >= 1)
        row[i][2:]   = ΔA row i                  (for i >= 1)

    NaN / empty / non-numeric cells inside the matrix become NaN. Rows or
    columns whose axis value (wavelength or delay) itself fails to parse
    are dropped — this is how stray blank cells, trailing tabs, and
    metadata rows are handled without the user having to clean them first.

    Implementation notes
    --------------------
    Speed matters: a 2136 × 192 file (~4 MB) used to take ~900 ms with a
    per-cell try/except loop in pure Python.  We now hand the bulk of
    the work to ``numpy.genfromtxt``, which is C-implemented and runs
    in a few tens of milliseconds.  Only format detection and a peek at
    the first row stay in Python — both O(1) in the file size.

    Returns
    -------
    wl : ndarray (M,)  sorted ascending
    t  : ndarray (N,)  sorted ascending
    A  : ndarray (M, N)
    """
    # ---- Sniff the first few lines: enough to detect delimiter + format ----
    with open(fullpath, 'r', encoding='utf-8', errors='replace') as f:
        sniff = f.read(8192)
    if not sniff.strip():
        raise ValueError('File is empty.')

    delim = _detect_delimiter(sniff)

    # Find first non-empty line for format detection
    first_line = ''
    for line in sniff.splitlines():
        if line.strip():
            first_line = line
            break
    if not first_line:
        raise ValueError('File is empty.')

    if delim is None:
        first_fields = first_line.split()
    else:
        first_fields = first_line.split(delim)
    if len(first_fields) < 2:
        raise ValueError('First row has fewer than 2 columns.')

    # Format B has a non-numeric first cell ("Integration time: ...");
    # Format A has either a number or an empty corner cell.
    is_format_b = not _is_numeric(first_fields[0])
    # Edge case: Format A with blank corner AND non-numeric second cell
    # would be misclassified, so flip back if row 1 col 0 is numeric.
    if is_format_b and len(first_fields) >= 2 \
            and not _is_numeric(first_fields[1]):
        # Try second line: if its col 0 is numeric → Format A
        lines = sniff.splitlines()
        if len(lines) >= 2:
            second_fields = (lines[1].split() if delim is None
                             else lines[1].split(delim))
            if second_fields and _is_numeric(second_fields[0]):
                is_format_b = False

    # genfromtxt expects an explicit string for the delimiter; passing
    # ``None`` makes it treat any run of whitespace as the separator,
    # which is what _detect_delimiter wanted to signal.
    np_delim = delim  # None ⇒ whitespace

    # The custom "missing values" let NaN, blanks, dashes, and Inf
    # collapse into NaN at parse time without per-cell Python work.
    missing = ('NaN', 'nan', 'NAN', 'NA', 'n/a', 'N/A', '-', '',
               'inf', 'Inf', 'INF', '+inf', '-inf', '-Inf', '-INF')

    if is_format_b:
        # Skip the metadata column (column 0). Determine how many
        # columns the first row has so we can give explicit usecols.
        n_cols_row0 = len(first_fields)
        # Guard against ragged rows by passing invalid_raise=False.
        try:
            mat = np.genfromtxt(
                fullpath, delimiter=np_delim,
                usecols=range(1, n_cols_row0),
                missing_values=missing, filling_values=np.nan,
                invalid_raise=False, dtype=float, comments=None)
        except Exception:
            # Fallback: parse without usecols and slice
            mat = np.genfromtxt(
                fullpath, delimiter=np_delim,
                missing_values=missing, filling_values=np.nan,
                invalid_raise=False, dtype=float, comments=None)
            mat = mat[:, 1:] if mat.ndim == 2 else mat
        if mat.ndim != 2 or mat.shape[0] < 2 or mat.shape[1] < 2:
            raise ValueError(
                'Could not parse file: too few rows or columns after '
                'dropping the metadata column.')
        # mat[0, :] = [<corner>, t1, t2, ...]  →  delays at index 1:
        # mat[i, 0] = wavelength i             →  wl from col 0
        # mat[i, 1:] = ΔA row i                →  matrix
        t = mat[0, 1:].astype(float, copy=False)
        wl = mat[1:, 0].astype(float, copy=False)
        A = mat[1:, 1:].astype(float, copy=False)
    else:
        # Format A: feed the file straight into genfromtxt.
        try:
            mat = np.genfromtxt(
                fullpath, delimiter=np_delim,
                missing_values=missing, filling_values=np.nan,
                invalid_raise=False, dtype=float, comments=None)
        except Exception as e:
            raise ValueError(f'Failed to parse file: {e}')
        if mat.ndim != 2 or mat.shape[0] < 2 or mat.shape[1] < 2:
            raise ValueError('Data matrix is too small.')
        t = mat[0, 1:].astype(float, copy=False)
        wl = mat[1:, 0].astype(float, copy=False)
        A = mat[1:, 1:].astype(float, copy=False)

    M, N = len(wl), len(t)
    if M == 0 or N == 0:
        raise ValueError('Parsed file produced empty wavelength or '
                         'delay axis.')
    if A.shape != (M, N):
        # Padding / truncation in case of ragged genfromtxt result
        AA = np.full((M, N), np.nan)
        rr = min(M, A.shape[0])
        cc = min(N, A.shape[1])
        AA[:rr, :cc] = A[:rr, :cc]
        A = AA

    # ---- Drop rows/cols whose axis value failed to parse ----
    # Skip the relatively expensive np.ix_ + fancy indexing when nothing
    # actually needs filtering — the common case in real data files.
    valid_wl = np.isfinite(wl)
    valid_t = np.isfinite(t)
    if not valid_wl.any() or not valid_t.any():
        raise ValueError('Parsed file produced no finite wavelengths '
                         'or delay times.')
    if not valid_wl.all() or not valid_t.all():
        wl = wl[valid_wl]
        t = t[valid_t]
        A = A[np.ix_(valid_wl, valid_t)]

    # Convert stray ±inf values in the data matrix to NaN — they always
    # come from measurement artefacts (division by zero, saturated
    # detector) and would otherwise poison plots and the global fit.
    # ``np.isnan(A) | np.isinf(A)`` is faster than ``~np.isfinite(A)``
    # in NumPy because the latter does both checks under the hood.
    bad = np.isinf(A)
    if bad.any():
        A[bad] = np.nan

    # ---- Sort ascending on both axes (required by the rest of the app) ----
    # Skip np.argsort + fancy indexing on the M ~ 2k matrix when it is
    # already sorted (typical for fresh measurement files).
    if not np.all(wl[1:] >= wl[:-1]):
        iw = np.argsort(wl)
        wl = wl[iw]
        A = A[iw, :]
    if not np.all(t[1:] >= t[:-1]):
        it = np.argsort(t)
        t = t[it]
        A = A[:, it]

    return wl, t, A


def write_data_file(fullpath: str, wl: np.ndarray, t: np.ndarray,
                    A: np.ndarray, delim: str = ','):
    """Write wl/t/A in the same layout accepted by parse_data_file.

    First row = [0, t1..tN]; first col = [0, l1..lM]; interior = A.
    """
    wl = np.asarray(wl).ravel()
    t = np.asarray(t).ravel()
    M, N = len(wl), len(t)
    if A.shape != (M, N):
        raise ValueError(
            f'Data size mismatch: A is {A.shape}, expected ({M}, {N})')
    out = np.zeros((M + 1, N + 1))
    out[0, 0] = 0.0
    out[0, 1:] = t
    out[1:, 0] = wl
    out[1:, 1:] = A
    np.savetxt(fullpath, out, delimiter=delim, fmt='%.10g')


def write_data_excel(fullpath: str, wl: np.ndarray, t: np.ndarray,
                     A: np.ndarray, sheet_name: str = 'TA Data'):
    """Write wl/t/A to an Excel (.xlsx) file in Format A layout.

    Top row holds delays, left column holds wavelengths, interior is ΔA.
    Re-readable with parse_data_file.

    Uses openpyxl in write-only mode for memory efficiency on large
    datasets (typical TA matrices are ~2k × 200 cells).

    NaN values in ``A`` are written as empty cells, which round-trip
    cleanly through parse_data_file (empty cell → NaN).
    """
    try:
        from openpyxl import Workbook
    except ImportError as e:
        raise RuntimeError(
            'openpyxl is required to export Excel files. '
            'Install with: pip install openpyxl'
        ) from e

    wl = np.asarray(wl, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    M, N = len(wl), len(t)
    if A.shape != (M, N):
        raise ValueError(
            f'Data size mismatch: A is {A.shape}, expected ({M}, {N})')

    wb = Workbook(write_only=True)
    ws = wb.create_sheet(sheet_name)

    # Header row: empty corner + delays
    header = [None] + [float(x) for x in t]
    ws.append(header)

    # Each wavelength row
    for i in range(M):
        row = [float(wl[i])]
        ai = A[i, :]
        for j in range(N):
            v = ai[j]
            row.append(float(v) if np.isfinite(v) else None)
        ws.append(row)

    wb.save(fullpath)


# =====================================================================
# Colormaps
# =====================================================================
def diverging_colormap(N: int, c1, c_mid, c2) -> np.ndarray:
    half = N // 2
    t1 = np.linspace(0, 1, half).reshape(-1, 1)
    t2 = np.linspace(0, 1, N - half).reshape(-1, 1)
    c1 = np.asarray(c1, float).reshape(1, 3)
    c_mid = np.asarray(c_mid, float).reshape(1, 3)
    c2 = np.asarray(c2, float).reshape(1, 3)
    part1 = (1 - t1) * c1 + t1 * c_mid
    part2 = (1 - t2) * c_mid + t2 * c2
    return np.vstack([part1, part2])


def get_colormap_array(name: str, N: int = 256) -> np.ndarray:
    """Return a N x 3 RGB array for the named colormap.

    Supported: turbo, parula (approx), jet, hot, cool, gray, RdBu, BWR,
    seismic.
    """
    name = name.lower()
    if name in ('turbo',):
        cmap = _mpl_cm.get_cmap('turbo', N)
    elif name in ('parula',):
        # parula is not shipped with matplotlib; use viridis as close analogue
        cmap = _mpl_cm.get_cmap('viridis', N)
    elif name in ('jet',):
        cmap = _mpl_cm.get_cmap('jet', N)
    elif name in ('hot',):
        cmap = _mpl_cm.get_cmap('hot', N)
    elif name in ('cool',):
        cmap = _mpl_cm.get_cmap('cool', N)
    elif name in ('gray', 'grey'):
        cmap = _mpl_cm.get_cmap('gray', N)
    elif name == 'rdbu':
        return diverging_colormap(N,
            [0.02, 0.19, 0.38], [1, 1, 1], [0.40, 0, 0.12])
    elif name == 'bwr':
        return diverging_colormap(N, [0, 0, 1], [1, 1, 1], [1, 0, 0])
    elif name == 'seismic':
        return diverging_colormap(N,
            [0, 0, 0.30], [1, 1, 1], [0.55, 0, 0])
    else:
        cmap = _mpl_cm.get_cmap('turbo', N)
    return cmap(np.linspace(0, 1, N))[:, :3]


# =====================================================================
# Chirp model (Sellmeier-like)
# =====================================================================
def chirp_model(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """t0(w) = a * sqrt((b*w^2 - 1) / (c*w^2 - 1)) + d.

    p = [a, b, c, d]; w may be a vector; out-of-domain entries are NaN.
    """
    w = np.asarray(w, dtype=float).ravel()
    a, b, c, d = p
    num = b * w ** 2 - 1.0
    den = c * w ** 2 - 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = num / den
    bad = (ratio < 0) | ~np.isfinite(ratio)
    ratio = np.where(bad, np.nan, ratio)
    with np.errstate(invalid='ignore'):
        return a * np.sqrt(ratio) + d


def align_solvent_to_sample(wl_solv: np.ndarray, t_solv: np.ndarray,
                            A_solv: np.ndarray,
                            wl_sample: np.ndarray,
                            t_sample: np.ndarray) -> np.ndarray:
    """Resample a solvent ΔA matrix onto the sample's (λ, t) grid.

    The user typically measures the pure-solvent reference with the
    same spectrograph and pump-probe stage as the sample, so the two
    grids are usually similar but rarely identical (a few wavelengths
    may differ, the delay step pattern may have been changed between
    runs, etc.).  We resample by 2-D linear interpolation:

        * along wavelength (axis 0): linear interp per delay column
        * along time       (axis 1): linear interp per wavelength row

    Points that fall outside the solvent grid are set to NaN; the
    caller should treat those positions as "no solvent data here"
    and avoid subtracting at them.

    Parameters
    ----------
    wl_solv, t_solv : 1-D arrays.  Solvent grid (sorted ascending).
    A_solv : (M_solv, N_solv) solvent ΔA matrix.
    wl_sample, t_sample : 1-D arrays.  Sample grid we resample onto.

    Returns
    -------
    A_solv_on_sample : (M_sample, N_sample) ndarray.  NaN where the
        sample grid steps outside the solvent's grid.
    """
    wl_solv = np.asarray(wl_solv, float).ravel()
    t_solv = np.asarray(t_solv, float).ravel()
    A_solv = np.asarray(A_solv, float)
    wl_sample = np.asarray(wl_sample, float).ravel()
    t_sample = np.asarray(t_sample, float).ravel()

    M_solv, N_solv = A_solv.shape
    if len(wl_solv) != M_solv or len(t_solv) != N_solv:
        raise ValueError(
            f'Solvent shape mismatch: A {A_solv.shape}, '
            f'wl {wl_solv.shape}, t {t_solv.shape}')

    # ---- Step 1: interpolate each wavelength row of A_solv along t -> sample t
    # We don't use scipy here so this stays a pure-numpy hot path.
    # Out-of-range t values become NaN.
    t_lo, t_hi = float(t_solv[0]), float(t_solv[-1])
    n_M, n_N = len(wl_sample), len(t_sample)
    inter_t = np.empty((M_solv, n_N), dtype=float)
    for i in range(M_solv):
        row = np.interp(t_sample, t_solv, A_solv[i, :])
        bad = (t_sample < t_lo) | (t_sample > t_hi)
        if bad.any():
            row[bad] = np.nan
        inter_t[i, :] = row

    # ---- Step 2: interpolate along wavelength per delay column
    wl_lo, wl_hi = float(wl_solv[0]), float(wl_solv[-1])
    out = np.empty((n_M, n_N), dtype=float)
    for j in range(n_N):
        col = np.interp(wl_sample, wl_solv, inter_t[:, j])
        bad = (wl_sample < wl_lo) | (wl_sample > wl_hi)
        if bad.any():
            col[bad] = np.nan
        out[:, j] = col
    return out


def apply_solvent_subtraction(sample: np.ndarray,
                              solvent_aligned: np.ndarray,
                              scale: float = 1.0) -> np.ndarray:
    """Subtract a scaled solvent ΔA from the sample ΔA.

    ``solvent_aligned`` must already be on the sample's (λ, t) grid
    (typically produced by :func:`align_solvent_to_sample`).  Cells
    where solvent_aligned is NaN (i.e. sample grid stepped outside
    the solvent's measured range) are left **unchanged**, NOT NaN'd,
    so partial-overlap experiments still work.

    Returns
    -------
    out : ndarray, same shape as ``sample``.
        ``sample - scale * solvent_aligned``  where finite, else
        ``sample`` itself.
    """
    sample = np.asarray(sample, float)
    solvent_aligned = np.asarray(solvent_aligned, float)
    if sample.shape != solvent_aligned.shape:
        raise ValueError(
            f'Shape mismatch: sample {sample.shape} vs '
            f'aligned solvent {solvent_aligned.shape}')
    out = sample.copy()
    fin = np.isfinite(solvent_aligned)
    if fin.any():
        out[fin] = sample[fin] - float(scale) * solvent_aligned[fin]
    return out


def apply_chirp_shift(data: np.ndarray, t: np.ndarray, wl: np.ndarray,
                      chirp_params: np.ndarray) -> np.ndarray:
    """Shift each row of ``data`` along the time axis by t0(λ).

    For wavelength index ``i``, the corrected value at ``t[j]`` is
    obtained by sampling row ``i`` at ``t[j] + t0[i]``.  Out-of-range
    samples become NaN.

    Implementation note: we benchmarked three approaches on a real
    2136 × 192 dataset:

        * per-row ``np.interp`` loop                   ≈ 23 ms
        * ``scipy.interpolate.RegularGridInterpolator`` ≈ 42 ms
        * ``scipy.ndimage.map_coordinates``             ≈ 31 ms

    The simple loop wins because ``np.interp`` does a tight C-level
    binary search over the (sorted) time axis, and the M = 2136
    Python iterations are cheap relative to the 410 K interpolated
    samples.  The "fancier" SciPy paths cannot beat that without
    re-doing essentially the same lookup work.  We therefore keep the
    loop, lifted out of ``recompute`` so it can be reused (e.g. tests).
    """
    data = np.asarray(data, dtype=float)
    t = np.asarray(t, dtype=float)
    wl = np.asarray(wl, dtype=float)
    M, N = data.shape
    if len(wl) != M or len(t) != N:
        raise ValueError(
            f'Shape mismatch: data {data.shape}, wl {wl.shape}, t {t.shape}')

    t0 = chirp_model(chirp_params, wl)        # (M,)
    t_lo, t_hi = float(t[0]), float(t[-1])
    out = np.empty_like(data)

    for i in range(M):
        ti = t0[i]
        if not np.isfinite(ti):
            out[i, :] = np.nan
            continue
        new_t = t + ti
        # np.interp clamps to edge values for out-of-range queries; we
        # need NaN there.  Compute, then NaN-out the offending entries.
        row = np.interp(new_t, t, data[i, :])
        bad = (new_t < t_lo) | (new_t > t_hi)
        if bad.any():
            row[bad] = np.nan
        out[i, :] = row
    return out


INTERP_METHODS = ('linear', 'cubic', 'pchip', 'akima')
INTERP_METHODS_2D = INTERP_METHODS + ('bilinear',)


def interpolate_missing_columns(A: np.ndarray, t: np.ndarray,
                                drop_idx, method: str = 'linear') -> np.ndarray:
    """Replace ``A[:, drop_idx]`` by interpolating row-wise along ``t``.

    Each wavelength row is interpolated independently from the columns
    that are *kept*, evaluating at the dropped t-values.  Useful for
    eliminating shutter/glitch spectra at a few specific delays without
    losing those delay-grid points.

    Parameters
    ----------
    A : (M, N) ndarray
        Data matrix (wavelength × delay).
    t : (N,) ndarray
        Delay axis matching the columns of ``A`` (must be strictly
        increasing).
    drop_idx : sequence of int
        Column indices in ``A`` whose values will be replaced.
    method : {'linear', 'cubic', 'pchip', 'akima'}
        Interpolator used per row.  ``cubic`` = natural cubic spline
        (S-spline); ``pchip`` and ``akima`` are monotone / non-overshoot
        cubic variants that are robust against sharp transients.

    Returns
    -------
    out : (M, N) ndarray
        Copy of ``A`` with the dropped columns overwritten.
    """
    A = np.asarray(A, dtype=float)
    t = np.asarray(t, dtype=float).ravel()
    if A.ndim != 2 or A.shape[1] != t.size:
        raise ValueError(
            f'Shape mismatch: A {A.shape}, t {t.shape}')
    if method not in INTERP_METHODS:
        raise ValueError(
            f'Unknown method {method!r}; expected one of {INTERP_METHODS}')

    drop = np.unique(np.asarray(list(drop_idx), dtype=int))
    if drop.size == 0:
        return A.copy()
    if drop.min() < 0 or drop.max() >= t.size:
        raise ValueError(
            f'drop_idx out of range for t of length {t.size}')

    keep_mask = np.ones(t.size, dtype=bool)
    keep_mask[drop] = False
    if keep_mask.sum() < 2:
        raise ValueError(
            'Need at least 2 kept delays to interpolate; '
            f'got {int(keep_mask.sum())}.')

    t_keep = t[keep_mask]
    t_drop = t[drop]

    # Cubic-family interpolators need ≥ 4 anchor points to behave well;
    # silently fall back to linear when there are too few.
    eff_method = method
    n_keep = t_keep.size
    if method in ('cubic', 'pchip', 'akima') and n_keep < 4:
        eff_method = 'linear'

    out = A.copy()
    from scipy.interpolate import (
        interp1d, CubicSpline, PchipInterpolator, Akima1DInterpolator,
    )

    A_keep = A[:, keep_mask]   # (M, n_keep)

    if eff_method == 'linear':
        # Vectorise: np.interp handles the per-row binary search in C.
        for i in range(A_keep.shape[0]):
            row = A_keep[i, :]
            finite = np.isfinite(row)
            if finite.sum() < 2:
                out[i, drop] = np.nan
                continue
            out[i, drop] = np.interp(t_drop, t_keep[finite], row[finite])
        return out

    # All cubic-family variants share the per-row pattern: build a 1-D
    # interpolator from the finite samples in this row, evaluate at the
    # dropped delays, fall back to linear (or NaN) if the row has too
    # few finite points.
    for i in range(A_keep.shape[0]):
        row = A_keep[i, :]
        finite = np.isfinite(row)
        if finite.sum() < 2:
            out[i, drop] = np.nan
            continue
        tk = t_keep[finite]
        yk = row[finite]
        if tk.size < 4:
            out[i, drop] = np.interp(t_drop, tk, yk)
            continue
        if eff_method == 'cubic':
            interp = CubicSpline(tk, yk, bc_type='natural',
                                 extrapolate=False)
        elif eff_method == 'pchip':
            interp = PchipInterpolator(tk, yk, extrapolate=False)
        else:  # 'akima'
            interp = Akima1DInterpolator(tk, yk)
        vals = interp(t_drop)
        # Extrapolated targets (outside [tk[0], tk[-1]]) come back as NaN;
        # repair them with edge-clamped linear so we never leak NaNs into
        # otherwise-finite columns.
        bad = ~np.isfinite(vals)
        if bad.any():
            vals[bad] = np.interp(t_drop[bad], tk, yk)
        out[i, drop] = vals
    return out


def interpolate_missing_rows(A: np.ndarray, w: np.ndarray,
                             drop_idx, method: str = 'linear') -> np.ndarray:
    """Replace ``A[drop_idx, :]`` by interpolating column-wise along ``w``.

    Mirror of :func:`interpolate_missing_columns` but operating on rows:
    each delay column is interpolated independently from the wavelength
    rows that are *kept*, evaluated at the dropped wavelength values.
    Useful for removing glitched detector channels at a few specific
    wavelengths without losing those wavelength-grid points.

    Parameters
    ----------
    A : (M, N) ndarray
        Data matrix (wavelength x delay).
    w : (M,) ndarray
        Wavelength axis matching the rows of ``A`` (strictly increasing).
    drop_idx : sequence of int
        Row indices in ``A`` whose values will be replaced.
    method : {'linear', 'cubic', 'pchip', 'akima'}
        Interpolator used per column.
    """
    if method == 'bilinear':
        raise ValueError("Use interpolate_missing_2d for 'bilinear'.")
    A = np.asarray(A, dtype=float)
    w = np.asarray(w, dtype=float).ravel()
    if A.ndim != 2 or A.shape[0] != w.size:
        raise ValueError(
            f'Shape mismatch: A {A.shape}, w {w.shape}')
    return interpolate_missing_columns(A.T, w, drop_idx, method=method).T


def interpolate_missing_2d(A: np.ndarray, w: np.ndarray, t: np.ndarray,
                           drop_w_idx, drop_t_idx) -> np.ndarray:
    """Bilinear (2D-surface) fill of cells flagged as missing.

    Treats the union ``(drop_w_idx rows) U (drop_t_idx columns)`` as
    missing and fills them by linear interpolation over the 2D surface
    defined by the remaining (kept) cells.  Uses
    :func:`scipy.interpolate.griddata` with ``method='linear'`` (a
    Delaunay triangulation), which honours both wavelength and delay
    spacing — the value at an intersection cell is the bilinear-like
    blend of its 2D neighbours, not a sequential axis-by-axis fill.

    Missing cells that fall outside the convex hull of the kept cells
    (e.g. an entire wavelength row at the edge of the grid) are filled
    by nearest-neighbour as a fallback so the result has no NaNs unless
    the input did.

    Parameters
    ----------
    A : (M, N) ndarray
        Data matrix (wavelength x delay).
    w : (M,) ndarray
        Wavelength axis.
    t : (N,) ndarray
        Delay axis.
    drop_w_idx, drop_t_idx : sequence of int
        Row / column indices to mark as missing.

    Returns
    -------
    out : (M, N) ndarray
        Copy of ``A`` with the missing cells overwritten.
    """
    A = np.asarray(A, dtype=float)
    w = np.asarray(w, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    if A.ndim != 2 or A.shape != (w.size, t.size):
        raise ValueError(
            f'Shape mismatch: A {A.shape}, w {w.shape}, t {t.shape}')

    drop_w = np.unique(np.asarray(list(drop_w_idx), dtype=int)) \
        if len(list(drop_w_idx)) else np.array([], dtype=int)
    drop_t = np.unique(np.asarray(list(drop_t_idx), dtype=int)) \
        if len(list(drop_t_idx)) else np.array([], dtype=int)
    if drop_w.size and (drop_w.min() < 0 or drop_w.max() >= w.size):
        raise ValueError(
            f'drop_w_idx out of range for w of length {w.size}')
    if drop_t.size and (drop_t.min() < 0 or drop_t.max() >= t.size):
        raise ValueError(
            f'drop_t_idx out of range for t of length {t.size}')

    mask_drop = np.zeros(A.shape, dtype=bool)
    if drop_w.size:
        mask_drop[drop_w, :] = True
    if drop_t.size:
        mask_drop[:, drop_t] = True
    if not mask_drop.any():
        return A.copy()

    mask_keep = (~mask_drop) & np.isfinite(A)
    if mask_keep.sum() < 4:
        raise ValueError(
            'Need at least 4 finite kept cells for bilinear surface '
            f'interpolation; got {int(mask_keep.sum())}.')

    # Scale axes so neither dimension dominates the Delaunay metric
    # (wavelength values can be 100x larger than delay in absolute
    # magnitude — without scaling the triangulation becomes 1D-like).
    w_span = float(w.max() - w.min()) or 1.0
    t_span = float(t.max() - t.min()) or 1.0
    W, T = np.meshgrid(w / w_span, t / t_span, indexing='ij')

    from scipy.interpolate import griddata
    pts_keep = np.column_stack([W[mask_keep], T[mask_keep]])
    vals_keep = A[mask_keep]
    pts_drop = np.column_stack([W[mask_drop], T[mask_drop]])

    vals = griddata(pts_keep, vals_keep, pts_drop, method='linear')
    # Fall back to nearest-neighbour at cells outside the convex hull
    # (e.g. drops at the very edge of the grid).
    bad = ~np.isfinite(vals)
    if bad.any():
        vals[bad] = griddata(pts_keep, vals_keep, pts_drop[bad],
                             method='nearest')

    out = A.copy()
    out[mask_drop] = vals
    return out


RESAMPLE_MODES = ('average', 'decimate')


def resample_wavelength(wl: np.ndarray, A: np.ndarray, dx: float,
                        mode: str = 'average', origin: float | None = None,
                        force: bool = False):
    """Reduce the wavelength axis (axis 0) onto uniform bins of width ``dx``.

    Bins are ``[origin + k*dx, origin + (k+1)*dx)``; a point lying exactly
    on a boundary belongs to the upper bin.  ``origin`` defaults to
    ``wl[0]``.  Bins that contain no original point are simply absent from
    the output, so the result keeps only measured spectral regions.

    ``mode='average'``  : λ and ΔA are averaged over each bin (ΔA with
                          nanmean; an all-NaN bin stays NaN).
    ``mode='decimate'`` : each bin keeps the original row closest to the
                          bin centre (ties → lower index); λ and ΔA are
                          taken unchanged from that row.

    When ``dx`` is not larger than the mean original spacing
    ``(wl[-1] - wl[0]) / (n - 1)`` (or fewer than 2 points are given) the
    call is a no-op unless ``force`` is set: copies of the inputs are
    returned with ``info['applied'] = False``.

    Returns
    -------
    wl_new : (M',) ndarray
    A_new  : (M', ...) ndarray  (same trailing shape as ``A``)
    info   : dict with ``applied, mode, dx, origin, mean_spacing, n_in,
             n_out`` plus ``starts`` (average) or ``idx`` (decimate) — the
             grouping, reusable on another matrix via
             :func:`apply_resample_info`.
    """
    wl = np.asarray(wl, dtype=float).ravel()
    A = np.asarray(A, dtype=float)
    n = wl.size
    if A.shape[0] != n:
        raise ValueError(f'Shape mismatch: A {A.shape}, wl {wl.shape}')
    if n > 1 and np.any(np.diff(wl) <= 0):
        raise ValueError('wavelength axis must be strictly ascending')
    mode = str(mode).lower()
    if mode == 'moving':
        raise ValueError('moving average is not a bin mode; use '
                         'resample_by_spec / moving_average_wavelength')
    if mode not in RESAMPLE_MODES:
        raise ValueError(f'mode must be one of {RESAMPLE_MODES}, got {mode!r}')
    dx = float(dx)
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError(f'dx must be positive, got {dx}')

    mean_spacing = float((wl[-1] - wl[0]) / (n - 1)) if n > 1 else float('nan')
    origin = float(wl[0]) if origin is None else float(origin)
    info = {'applied': False, 'mode': mode, 'dx': dx, 'origin': origin,
            'mean_spacing': mean_spacing, 'n_in': n, 'n_out': n}
    if not force and (n < 2 or dx <= mean_spacing):
        return wl.copy(), A.copy(), info

    k = np.floor((wl - origin) / dx + 1e-9).astype(np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(k)) + 1]
    info.update(applied=True, n_out=int(starts.size))
    if mode == 'average':
        info['starts'] = starts
        wl_new = np.add.reduceat(wl, starts) / np.diff(np.r_[starts, n])
    else:
        centres = origin + (k[starts] + 0.5) * dx
        ends = np.r_[starts[1:], n]
        idx = np.array([s + int(np.argmin(np.abs(wl[s:e] - c)))
                        for s, e, c in zip(starts, ends, centres)],
                       dtype=np.int64)
        info['idx'] = idx
        wl_new = wl[idx]
    return wl_new, apply_resample_info(A, info), info


def apply_resample_info(A: np.ndarray, info: dict) -> np.ndarray:
    """Apply the operation described by a resampling ``info`` to another
    matrix ``A`` defined on the same (pre-resample) wavelength grid —
    e.g. a solvent reference aligned to the sample grid.

    Average groups use nanmean (all-NaN group → NaN, no RuntimeWarning);
    decimate groups pick the same rows; moving averages reuse the same
    kernel (a row count other than ``info['n_in']`` raises).  A no-op
    ``info`` returns a copy.
    """
    A = np.asarray(A, dtype=float)
    if not info.get('applied'):
        return A.copy()
    if info['mode'] == MOVING_MODE:
        if A.shape[0] != info['n_in']:
            raise ValueError(f"moving average of {info['n_in']} rows "
                             f'applied to {A.shape[0]} rows')
        return _moving_apply(A, moving_kernel_weights(info['k'],
                                                      info['kernel']))
    if info['mode'] == 'decimate':
        return A[info['idx']].copy()
    starts = info['starts']
    finite = np.isfinite(A)
    sums = np.add.reduceat(np.where(finite, A, 0.0), starts, axis=0)
    counts = np.add.reduceat(finite.astype(np.int64), starts, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = sums / counts
    out[counts == 0] = np.nan
    return out


apply_bin_groups = apply_resample_info   # v1.1.0 name


MOVING_MODE = 'moving'
ALL_RESAMPLE_MODES = RESAMPLE_MODES + (MOVING_MODE,)
MOVING_KERNELS = ('box', 'triangular', 'gaussian')
MOVING_K_RANGE = (3, 101)


def moving_kernel_weights(k: int, kernel: str = 'box') -> np.ndarray:
    """Weights of an odd-length moving-average kernel.

    box        : all ones
    triangular : 1, 2, …, m, …, 2, 1   with m = (k + 1) / 2
    gaussian   : exp(-x² / 2σ²), x = -(k-1)/2 … (k-1)/2, σ = k / 6
                 (the window spans ±3σ)
    """
    k = int(k)
    if kernel == 'box':
        return np.ones(k)
    if kernel == 'triangular':
        m = (k + 1) // 2
        return np.r_[1:m + 1, m - 1:0:-1].astype(float)
    if kernel == 'gaussian':
        x = np.arange(k) - k // 2
        sigma = k / 6.0
        return np.exp(-x ** 2 / (2.0 * sigma ** 2))
    raise ValueError(f'kernel must be one of {MOVING_KERNELS}, got {kernel!r}')


def _moving_apply(A: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted moving average of ``A`` along axis 0 with weights ``w``.

    Positions outside the array and NaN samples are left out and the
    remaining weights renormalised, so the output keeps the input shape;
    a window without any finite sample gives NaN.
    """
    A = np.asarray(A, dtype=float)
    A2 = A.reshape(A.shape[0], -1)       # 1-D → column; 2-D unchanged
    finite = np.isfinite(A2)
    num = correlate1d(np.where(finite, A2, 0.0), w, axis=0,
                      mode='constant', cval=0.0)
    den = correlate1d(finite.astype(float), w, axis=0,
                      mode='constant', cval=0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    out[den <= 0] = np.nan
    return out.reshape(A.shape)


def _as_window(k):
    """``k`` as an int when it is an integer value (not bool), else None."""
    if isinstance(k, (bool, np.bool_)):
        return None
    if isinstance(k, (int, np.integer)):
        return int(k)
    if isinstance(k, (float, np.floating)) and float(k).is_integer():
        return int(k)
    return None


def moving_average_wavelength(wl: np.ndarray, A: np.ndarray, k,
                              kernel: str = 'box'):
    """Smooth ΔA along the wavelength axis with a pixel-window kernel.

    The wavelength axis and the number of points are unchanged: each
    point becomes the weighted mean of the ``k`` points centred on it
    (odd ``k``; edges and NaNs renormalised, see :func:`_moving_apply`).
    ``kernel`` is one of :data:`MOVING_KERNELS`.

    ``k`` must be an odd integer in :data:`MOVING_K_RANGE` and smaller
    than the number of points; otherwise — or for an unknown kernel —
    copies of the inputs are returned with ``info['applied'] = False``.

    Returns ``(wl, A_smoothed, info)`` with ``info`` holding ``applied,
    mode='moving', k, kernel, n_in, n_out`` — reusable on another matrix
    of the same length via :func:`apply_resample_info`.
    """
    wl = np.asarray(wl, dtype=float).ravel()
    A = np.asarray(A, dtype=float)
    n = wl.size
    if A.shape[0] != n:
        raise ValueError(f'Shape mismatch: A {A.shape}, wl {wl.shape}')
    kk = _as_window(k)
    info = {'applied': False, 'mode': MOVING_MODE, 'k': kk if kk is not None else k,
            'kernel': kernel, 'n_in': n, 'n_out': n}
    lo, hi = MOVING_K_RANGE
    if (kk is None or kk % 2 == 0 or not lo <= kk <= hi or kk >= n
            or kernel not in MOVING_KERNELS):
        return wl.copy(), A.copy(), info
    info['applied'] = True
    return wl.copy(), _moving_apply(A, moving_kernel_weights(kk, kernel)), info


def resample_by_spec(wl: np.ndarray, A: np.ndarray, spec: dict):
    """Dispatch a Crop-dialog spec ``{'mode', 'dx' | 'window', 'kernel'}``
    to :func:`resample_wavelength` (bin modes) or
    :func:`moving_average_wavelength`."""
    if spec['mode'] == MOVING_MODE:
        return moving_average_wavelength(wl, A, spec['window'],
                                         spec['kernel'])
    return resample_wavelength(wl, A, spec['dx'], spec['mode'])


def fit_chirp_params(pts: np.ndarray):
    """Multi-start Nelder-Mead fit of chirp_model to (w, t) points.

    Returns (p_vec (4,), rms_residual).
    """
    pts = np.asarray(pts, dtype=float)
    if pts.shape[0] < 4:
        raise ValueError('At least 4 points are required.')
    w = pts[:, 0]
    t_obs = pts[:, 1]

    w0 = float(np.median(w))
    ws = w / w0  # scaled wavelengths

    def obj(p):
        try:
            pred = chirp_model(p, ws)
        except Exception:
            return 1e20
        finite = np.isfinite(pred)
        if not finite.all():
            # Penalize partial-NaN predictions, like the MATLAB version
            return float(np.nansum((t_obs - pred) ** 2)) + 1e6
        return float(np.sum((t_obs - pred) ** 2))

    t_range = float(np.max(t_obs) - np.min(t_obs))
    t_mid = float((np.max(t_obs) + np.min(t_obs)) / 2.0)
    if t_range == 0:
        t_range = 1.0

    starts = [
        np.array([ t_range,    1.2, 1.5, t_mid - t_range]),
        np.array([-t_range,    1.2, 1.5, t_mid + t_range]),
        np.array([ t_range/2,  2.0, 3.0, t_mid]),
        np.array([ t_range,    1.1, 2.0, t_mid - t_range/2]),
        np.array([ 2*t_range,  0.8, 1.2, t_mid - t_range]),
        np.array([ t_range,    5.0, 10.0, t_mid - t_range]),
    ]

    best_p = None
    best_r = np.inf
    for x0 in starts:
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'xatol': 1e-9, 'fatol': 1e-11,
                                    'maxiter': 20000, 'maxfev': 20000,
                                    'disp': False})
            pk = res.x
            rk = float(res.fun)
            if np.all(np.isfinite(pk)) and rk < best_r:
                best_r = rk
                best_p = pk.copy()
        except Exception:
            continue
    if best_p is None:
        raise RuntimeError('Chirp fit did not converge. Try adjusting points.')

    # Unscale b, c so that the returned model works in original wl units
    best_p[1] = best_p[1] / w0 ** 2
    best_p[2] = best_p[2] / w0 ** 2
    rms = float(np.sqrt(best_r / len(w)))
    return best_p, rms


# =====================================================================
# Ridge helper
# =====================================================================
def compute_ridge(wavelength, delay, data,
                  method: str = 'max|dA/dt|',
                  t_min: float = -0.5, t_max: float = 1.5,
                  smooth_n: int = 5):
    """Locate the t-position of maximum |dA/dt| (or |dA|) per wavelength.

    Returns (ridge_wl (M,), ridge_t (M,)) or (None, None) if too few points.
    """
    wl = np.asarray(wavelength, float).ravel()
    t = np.asarray(delay, float).ravel()
    D = np.asarray(data, float)
    m = (t >= t_min) & (t <= t_max)
    if np.count_nonzero(m) < 3:
        return None, None
    t_sub = t[m]
    sub = D[:, m]
    if method == 'max|dA/dt|':
        dA = np.diff(sub, axis=1)
        dt = np.diff(t_sub)
        rate = dA / dt[np.newaxis, :]
        t_axis = (t_sub[:-1] + t_sub[1:]) / 2.0
        idx = np.argmax(np.abs(rate), axis=1)
    elif method == 'max|deltaA|':
        t_axis = t_sub
        idx = np.argmax(np.abs(sub), axis=1)
    else:
        return None, None
    r_t = t_axis[idx]
    if smooth_n > 1 and len(r_t) > smooth_n:
        # Moving average via scipy's uniform_filter1d — meaningfully
        # faster than np.convolve for the typical M ~ 2k case, and
        # handles the boundary by reflection so we don't need to pad
        # manually.  ``mode='nearest'`` matches the previous edge-clamp
        # behavior (the old code padded with the first / last value).
        try:
            from scipy.ndimage import uniform_filter1d
            r_t = uniform_filter1d(r_t.astype(float), size=int(smooth_n),
                                   mode='nearest')
        except Exception:
            # Fallback to numpy if scipy is unavailable for any reason
            kernel = np.ones(smooth_n) / smooth_n
            pad = smooth_n // 2
            padded = np.concatenate([
                np.full(pad, r_t[0]), r_t,
                np.full(smooth_n - pad - 1, r_t[-1])])
            r_t = np.convolve(padded, kernel, mode='valid')
    return wl.copy(), r_t


# =====================================================================
# Global analysis: IRF-convolved exponential, fitting, EADS
# =====================================================================
def exp_irf_conv(t: np.ndarray, tau: float, t0: float, fwhm: float) -> np.ndarray:
    """Exponential decay convolved with a Gaussian IRF (FWHM).

    Piecewise evaluation to avoid numerical overflow.
        far past t0 (dt > threshold)   -> asymptotic exp form
        near t0     (|dt| <= threshold) -> full formula via erfcx
        far before t0 (dt < -threshold) -> ~0

    tau = +inf gives the step-response (used for the constant offset).
    """
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    t_arr = np.asarray(t, float).ravel()
    dt = t_arr - t0
    c = np.zeros_like(dt)

    if np.isinf(tau):
        with np.errstate(invalid='ignore'):
            c = 0.5 * erfc(-dt / (sigma * np.sqrt(2.0)))
        c[~np.isfinite(c)] = 0.0
        return c

    thresh = 5.0 * sigma + sigma ** 2 / tau
    near = np.abs(dt) <= thresh
    far_pos = dt > thresh

    if np.any(far_pos):
        dtf = dt[far_pos]
        c[far_pos] = np.exp(sigma ** 2 / (2.0 * tau ** 2) - dtf / tau)

    if np.any(near):
        dtn = dt[near]
        B = (sigma / tau - dtn / sigma) / np.sqrt(2.0)
        with np.errstate(invalid='ignore', over='ignore'):
            c[near] = 0.5 * np.exp(-dtn ** 2 / (2.0 * sigma ** 2)) * erfcx(B)

    c[~np.isfinite(c)] = 0.0
    return c


def build_ga_basis(t: np.ndarray, tau_vec, t0: float, fwhm: float,
                   has_inf: bool) -> np.ndarray:
    """Build the (N_t x k) basis matrix of IRF-convolved exponentials."""
    tau_vec = np.asarray(tau_vec, float).ravel()
    k = len(tau_vec) + (1 if has_inf else 0)
    t_arr = np.asarray(t, float).ravel()
    C = np.zeros((len(t_arr), k))
    for j, tau in enumerate(tau_vec):
        C[:, j] = exp_irf_conv(t_arr, tau, t0, fwhm)
    if has_inf:
        C[:, -1] = exp_irf_conv(t_arr, np.inf, t0, fwhm)
    return C


def _lsqminnorm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """MATLAB lsqminnorm-style solve.  Minimum-norm least-squares.

    Works for full/rank-deficient systems; A (m x n), B (m x k) -> X (n x k).
    """
    X, *_ = np.linalg.lstsq(A, B, rcond=None)
    return X


class GlobalAnalysisStopped(Exception):
    """Raised from inside the GA objective when the caller-supplied
    ``stop_check`` reports that the user cancelled the fit.

    Caught in the outer scope of ``fit_global_analysis`` and re-raised so
    the caller (typically the GA dialog's worker) can distinguish a
    user-requested abort from a real numerical failure.
    """


def fit_global_analysis(D: np.ndarray, t: np.ndarray,
                        tau_init, t0_init: float, fwhm_init: float,
                        tau_fixed, t0_fixed: bool, fwhm_fixed: bool,
                        has_inf: bool,
                        beta_init=None, beta_fixed=None, stretch_on=None,
                        irf_mode: str = 'numerical',
                        backend=None, stop_check=None,
                        method: str = 'trf'):
    """VARPRO global fit.

    Nonlinear params (tau, beta, t0, FWHM) via Nelder-Mead; amplitudes A
    via linear least-squares per-call.

    Stretched-exponential support:
        ``stretch_on[j] = True``  → component j uses
            exp(-((t-t0)/τ_j)^β_j) ⊗ IRF (numerical convolution if
            ``irf_mode='numerical'``, or no IRF if ``'skip'``).
        ``stretch_on[j] = False`` → component j is the standard exp.

    β_j is globally shared across all wavelengths.  Per-component β
    fixed-flags are honoured: ``beta_fixed[j] = True`` keeps β_j at its
    initial value (only meaningful when ``stretch_on[j] = True``).
    When ``stretch_on`` is None or all-False (default), the call
    behaves identically to the pre-stretched implementation.

    ``backend`` (optional): a ``ta_device.TFBackend`` instance.  When
    supplied, the two heavy linear-algebra kernels — the amplitude
    least-squares solve and the ``A @ C.T`` reconstruction — run
    through the backend's device (CPU or GPU via TensorFlow).  Passing
    ``None`` keeps the original NumPy path so existing callers and unit
    tests are unaffected.

    ``stop_check`` (optional): a zero-arg callable polled at the start
    of every objective evaluation.  Returning True raises
    ``GlobalAnalysisStopped`` and aborts the fit; the caller is
    responsible for catching it.

    ``method`` (optional, default ``'trf'``):
        ``'trf'``  Trust-Region-Reflective Levenberg-Marquardt via
                   ``scipy.optimize.least_squares``.  Uses the
                   least-squares residual vector to build Gauss-Newton
                   steps; typically converges in 10-100× fewer
                   objective evaluations than Nelder-Mead for
                   well-conditioned GA problems.  This is the default.
        ``'nm'``   Legacy Nelder-Mead (derivative-free scalar
                   minimisation).  Kept as a fallback for datasets
                   where the loss surface has sharp discontinuities
                   TRF struggles with (rare).

    Returns dict with keys:
        tau, beta, t0, fwhm, A (M x k), fit (M x N), info, stretch_on
    """
    tau_init = np.asarray(tau_init, float).ravel()
    tau_fixed = np.asarray(tau_fixed, bool).ravel()
    t_arr = np.asarray(t, float).ravel()
    D = np.asarray(D, float)
    M, N = D.shape

    n_nl = len(tau_init)
    if beta_init is None:
        beta_init = np.ones(n_nl)
    if stretch_on is None:
        stretch_on = np.zeros(n_nl, dtype=bool)
    if beta_fixed is None:
        beta_fixed = np.zeros(n_nl, dtype=bool)
    beta_init = np.asarray(beta_init, dtype=float).ravel()
    stretch_on = np.asarray(stretch_on, dtype=bool).ravel()
    beta_fixed = np.asarray(beta_fixed, dtype=bool).ravel()
    if (beta_init.size != n_nl or stretch_on.size != n_nl
            or beta_fixed.size != n_nl):
        raise ValueError(
            'beta_init / stretch_on / beta_fixed length must match tau_init')
    any_stretched = bool(stretch_on.any())

    tau_cur = tau_init.copy()
    beta_cur = beta_init.copy()
    t0_cur = float(t0_init)
    fwhm_cur = float(fwhm_init)
    idx_free_tau = np.where(~tau_fixed)[0]
    n_free_tau = len(idx_free_tau)
    # All stretched betas are free (matches the MATLAB convention)
    idx_free_beta = np.where(stretch_on & ~beta_fixed)[0]
    n_free_beta = len(idx_free_beta)

    # Pack free params (log transform for positivity)
    x0_list = []
    if n_free_tau > 0:
        x0_list.extend(np.log(tau_init[idx_free_tau]).tolist())
    if n_free_beta > 0:
        x0_list.extend(np.log(np.maximum(beta_init[idx_free_beta], 1e-3)).tolist())
    if not t0_fixed:
        x0_list.append(float(t0_init))
    if not fwhm_fixed:
        x0_list.append(float(np.log(fwhm_init)))
    x0 = np.array(x0_list, float)

    # When 'skip' mode and any stretched component, drop points within
    # ~3σ of t0 from the residual
    skip_mask_active = any_stretched and irf_mode.lower() == 'skip'

    def build_basis_local(tau_v, beta_v, t0_v, fwhm_v):
        """Build (N, n_cols) basis with mixed exp / stretched columns."""
        if not any_stretched:
            return build_ga_basis(t_arr, tau_v, t0_v, fwhm_v, has_inf)
        n_cols = n_nl + (1 if has_inf else 0)
        C = np.zeros((N, n_cols))
        for j in range(n_nl):
            if stretch_on[j]:
                C[:, j] = stretched_irf_conv(
                    t_arr, tau_v[j], beta_v[j], t0_v, fwhm_v, irf_mode)
            else:
                C[:, j] = exp_irf_conv(t_arr, tau_v[j], t0_v, fwhm_v)
        if has_inf:
            C[:, -1] = exp_irf_conv(t_arr, np.inf, t0_v, fwhm_v)
        return C

    # Route the two hot linear-algebra kernels (lstsq + reconstruction
    # matmul) through the user-selected backend when one is supplied.
    # Local aliases keep the inner loop readable and let us drop back to
    # NumPy in one place if ``backend`` is None.
    if backend is not None:
        _solve = backend.lstsq
        _matmul = backend.matmul
    else:
        _solve = _lsqminnorm
        _matmul = lambda a, b: a @ b  # noqa: E731

    def objective(x):
        nonlocal tau_cur, beta_cur, t0_cur, fwhm_cur
        # User-requested cancellation: check before doing any work so
        # Stop responds quickly even on a very tight IRF grid.
        if stop_check is not None and stop_check():
            raise GlobalAnalysisStopped()
        idx = 0
        for j in range(n_free_tau):
            tau_cur[idx_free_tau[j]] = np.exp(x[idx]); idx += 1
        for j in range(n_free_beta):
            beta_cur[idx_free_beta[j]] = np.exp(x[idx]); idx += 1
        if not t0_fixed:
            t0_cur = float(x[idx]); idx += 1
        if not fwhm_fixed:
            fwhm_cur = float(np.exp(x[idx])); idx += 1

        sigma_cur = fwhm_cur / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        t_span = t_arr.max() - t_arr.min()
        if (np.any(tau_cur < sigma_cur / 4.0) or
                np.any(tau_cur > 100.0 * t_span) or
                fwhm_cur <= 0 or fwhm_cur > t_span or
                (any_stretched and (np.any(beta_cur[stretch_on] <= 0)
                                    or np.any(beta_cur[stretch_on] > 2)))):
            k_cols = n_nl + (1 if has_inf else 0)
            return 1e30, np.zeros((M, k_cols)), np.zeros((M, N))

        C = build_basis_local(tau_cur, beta_cur, t0_cur, fwhm_cur)
        col_norms = np.sqrt(np.nansum(C ** 2, axis=0))
        if np.any(col_norms < 1e-12) or not np.all(np.isfinite(C)):
            return 1e30, np.zeros((M, C.shape[1])), np.zeros((M, N))

        # Optional skip mask (drop IRF region from the LS solve)
        col_mask = np.ones(N, dtype=bool)
        if skip_mask_active:
            col_mask = t_arr > (t0_cur + 3.0 * sigma_cur)
            if int(col_mask.sum()) <= C.shape[1]:
                col_mask = np.ones(N, dtype=bool)

        k_cols = C.shape[1]
        As = np.zeros((M, k_cols))
        fit_M = np.zeros((M, N))

        if np.all(np.isfinite(D)):
            Cs = C[col_mask, :]
            Ds = D[:, col_mask]
            try:
                At = _solve(Cs, Ds.T)  # (k x M)
            except GlobalAnalysisStopped:
                raise
            except Exception:
                return 1e30, As, fit_M
            if not np.all(np.isfinite(At)) or np.max(np.abs(At)) > 1e10:
                return 1e30, As, fit_M
            As = At.T
            fit_M = _matmul(As, C.T)
        else:
            for ii in range(M):
                row = D[ii, :]
                mm = np.isfinite(row) & col_mask
                if np.count_nonzero(mm) <= k_cols:
                    continue
                Ci = C[mm, :]
                di = row[mm]
                try:
                    ai = _solve(Ci, di)
                except GlobalAnalysisStopped:
                    raise
                except Exception:
                    continue
                if not np.all(np.isfinite(ai)) or np.max(np.abs(ai)) > 1e8:
                    continue
                As[ii, :] = ai
                fit_M[ii, :] = _matmul(ai.reshape(1, -1), C.T).ravel()

        # Residual: respect skip mask in the loss too
        R = D - fit_M
        if skip_mask_active:
            R = R[:, col_mask]
        mfin = np.isfinite(R)
        if not np.any(mfin):
            loss = 1e30
        else:
            loss = float(np.sum(R[mfin] ** 2))
        return loss, As, fit_M

    method_lc = str(method).lower()
    if method_lc in ('trf', 'levenberg-marquardt', 'lm', 'least_squares'):
        method_used = 'trf'
    elif method_lc in ('nm', 'nelder-mead', 'neldermead', 'nelder_mead'):
        method_used = 'nm'
    else:
        raise ValueError(
            f"fit_global_analysis: unknown method {method!r}. "
            f"Expected 'trf' or 'nm'.")

    if x0.size == 0:
        # All nonlinear params are fixed → linear-only problem, one
        # objective call suffices.
        loss, A_out, fit_out = objective(np.array([]))
        iters = 0
        n_fev = 1
        init_loss = loss
    elif method_used == 'trf':
        # Trust-Region-Reflective Levenberg-Marquardt.  Feeds the
        # per-element residual vector to ``least_squares`` so it can
        # build Gauss-Newton steps from a numerical Jacobian, which
        # converges 10-100× faster than the derivative-free Nelder-
        # Mead loop for our nonlinear least-squares structure.
        init_loss, _, _ = objective(x0)

        def _residuals(x):
            _, _, fit_M = objective(x)
            # Residual (M, N) → 1D.  NaN entries in D produce non-
            # finite residuals which we replace with 0 so the vector
            # length stays constant across evaluations — a hard
            # requirement of scipy.optimize.least_squares.  A zero
            # residual contributes nothing to the sum-of-squares, so
            # this is equivalent to masking those cells out.
            R = D - fit_M
            R = np.where(np.isfinite(R), R, 0.0)
            # Same for the (optional) skip mask on stretched-skip mode:
            # we zero-out the excluded region rather than shrink the
            # vector, again to keep the length constant.
            if skip_mask_active:
                cm = t_arr > (t0_cur + 3.0 * fwhm_cur / (2.0
                                * np.sqrt(2.0 * np.log(2.0))))
                if cm.any() and cm.sum() < N:
                    R[:, ~cm] = 0.0
            return R.ravel()

        try:
            res = least_squares(
                _residuals, x0, method='trf',
                # xtol/ftol chosen to match Nelder-Mead's convergence
                # tightness in practice; gtol relaxed slightly because
                # the numerical Jacobian is only forward-differenced.
                xtol=1e-8, ftol=1e-10, gtol=1e-8,
                max_nfev=5000)
        except GlobalAnalysisStopped:
            raise
        loss, A_out, fit_out = objective(res.x)
        iters = int(getattr(res, 'nfev', 0))
        n_fev = int(getattr(res, 'nfev', 0))
    else:
        # method_used == 'nm' — legacy Nelder-Mead path.
        init_loss, _, _ = objective(x0)
        res = minimize(lambda x: objective(x)[0], x0, method='Nelder-Mead',
                       options={'xatol': 1e-8, 'fatol': 1e-10,
                                'maxiter': 5000, 'maxfev': 20000,
                                'disp': False})
        loss, A_out, fit_out = objective(res.x)
        iters = int(res.nit)
        n_fev = int(getattr(res, 'nfev', iters))

    info = {
        'rss': loss,
        'iters': iters,
        'nfev': n_fev,
        'method': method_used,
        'rms': float(np.sqrt(loss / D.size)),
        'initialLoss': init_loss,
        'initialRMS': float(np.sqrt(init_loss / D.size)),
        'irf_mode': irf_mode if any_stretched else 'closed-form',
    }
    return {
        'tau': tau_cur.copy(),
        'beta': beta_cur.copy(),
        'stretch_on': stretch_on.copy(),
        't0': t0_cur,
        'fwhm': fwhm_cur,
        'A': A_out,
        'fit': fit_out,
        'info': info,
    }


def compute_eads_from_dads(DADS: np.ndarray, tau_vec, has_inf: bool):
    """Convert parallel DADS to sequential EADS.

    Sequential model: species 1 -> 2 -> ... -> N (+ optional terminal
    species when has_inf=True).  Species are ordered by ascending tau.

    Returns (EADS (M, n), tau_sorted, Bmat).
    """
    tau_vec = np.asarray(tau_vec, float).ravel()
    sort_idx = np.argsort(tau_vec)
    tau_sorted = tau_vec[sort_idx]
    n_decay = len(tau_sorted)

    if has_inf:
        k_vec = np.concatenate([1.0 / tau_sorted, [0.0]])
        perm = np.concatenate([sort_idx, [n_decay]])
    else:
        k_vec = 1.0 / tau_sorted
        perm = sort_idx

    n = len(k_vec)
    Bmat = np.zeros((n, n))
    for ii in range(n):
        prod_k = 1.0 if ii == 0 else float(np.prod(k_vec[:ii]))
        for jj in range(ii + 1):
            denom = 1.0
            for mm in range(ii + 1):
                if mm != jj:
                    denom *= (k_vec[mm] - k_vec[jj])
            Bmat[ii, jj] = prod_k / denom

    DADS_perm = DADS[:, perm]
    try:
        # EADS' = B' \ DADS'  ->  solve B^T x = DADS^T, then transpose
        EADS = np.linalg.solve(Bmat.T, DADS_perm.T).T
        if not np.all(np.isfinite(EADS)):
            raise np.linalg.LinAlgError('non-finite')
    except np.linalg.LinAlgError:
        EADS = DADS_perm @ np.linalg.pinv(Bmat)

    return EADS, tau_sorted, Bmat

# =====================================================================
# Stretched exponential (Kohlrausch) and IRF convolution
# =====================================================================
def stretched_irf_conv(t: np.ndarray, tau: float, beta: float,
                       t0: float, fwhm: float,
                       mode: str = 'numerical') -> np.ndarray:
    """Stretched-exponential f(t) = exp(-((t-t0)/tau)^beta), with optional
    Gaussian-IRF convolution.

    Parameters
    ----------
    mode : {'skip', 'numerical'}
        ``'skip'`` ignores the IRF entirely (caller must mask data within
        ~3σ of t0).  ``'numerical'`` does a direct convolution on a fine
        uniform grid.
    """
    t = np.asarray(t, dtype=float).ravel()
    dt = t - t0
    out = np.zeros_like(dt)

    if (not np.isfinite(tau)) or tau <= 0 or \
       (not np.isfinite(beta)) or beta <= 0:
        return out

    def bare(x):
        # Stretched exp for x >= 0; zero before t0
        with np.errstate(invalid='ignore'):
            xx = np.maximum(x, 0.0)
            return np.exp(-(xx / tau) ** beta) * (x >= 0).astype(float)

    mode = mode.lower()
    if mode == 'skip':
        out = bare(dt)
    elif mode == 'numerical':
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        if not np.isfinite(sigma) or sigma <= 0:
            return bare(dt)
        # Fine uniform grid spanning [-5σ + min, 5σ + max], step small
        # enough to resolve both the IRF and the fastest stretched
        # variation.
        step = min(0.2 * sigma, 0.05 * tau)
        if not np.isfinite(step) or step <= 0:
            step = 0.01
        t_lo = float(t.min()) - 5.0 * sigma
        t_hi = float(t.max()) + 5.0 * sigma
        # Cap the number of fine-grid points to keep memory sane on
        # very wide delay axes.
        n_grid = int(np.ceil((t_hi - t_lo) / step)) + 1
        if n_grid > 200_000:
            step = (t_hi - t_lo) / 200_000.0
            n_grid = 200_001
        t_grid = t_lo + np.arange(n_grid) * step
        f_grid = bare(t_grid - t0)
        # Gaussian IRF kernel (unit-area)
        n_k = int(np.ceil(5.0 * sigma / step))
        kx = np.arange(-n_k, n_k + 1) * step
        irf = np.exp(-kx ** 2 / (2.0 * sigma ** 2))
        irf /= irf.sum()
        c_grid = np.convolve(f_grid, irf, mode='same')
        out = np.interp(t, t_grid, c_grid, left=0.0, right=0.0)
    else:
        raise ValueError(f'stretched_irf_conv: unknown mode {mode!r}')

    out[~np.isfinite(out)] = 0.0
    return out


# =====================================================================
# Single-trace fit (used by the Kinetic Fit dialog)
# =====================================================================
def fit_single_trace(t: np.ndarray, y: np.ndarray, *,
                     tau_init, tau_fixed,
                     beta_init=None, beta_fixed=None, stretch_on=None,
                     t0_init: float = 0.0, t0_fixed: bool = True,
                     fwhm_init: float = 0.15, fwhm_fixed: bool = True,
                     has_inf: bool = False,
                     irf_mode: str = 'skip'):
    """Fit a single kinetic trace y(t) with sum of (possibly stretched)
    exponentials convolved with a Gaussian IRF.

    Linear amplitudes A_i are solved per non-linear step (VARPRO).
    Non-linear parameters (tau, beta, t0, FWHM) are optimized over a
    log-transformed parameterization via Nelder-Mead.

    Returns dict with keys: tau, beta, t0, fwhm, A, fit, residual, info.
    """
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if t.shape != y.shape:
        raise ValueError(f'shape mismatch: t {t.shape} vs y {y.shape}')

    n_nl = int(len(tau_init))
    tau_init = np.asarray(tau_init, dtype=float).ravel()
    tau_fixed = np.asarray(tau_fixed, dtype=bool).ravel()
    if beta_init is None:
        beta_init = np.ones(n_nl)
    if beta_fixed is None:
        beta_fixed = np.zeros(n_nl, dtype=bool)
    if stretch_on is None:
        stretch_on = np.zeros(n_nl, dtype=bool)
    beta_init = np.asarray(beta_init, dtype=float).ravel()
    beta_fixed = np.asarray(beta_fixed, dtype=bool).ravel()
    stretch_on = np.asarray(stretch_on, dtype=bool).ravel()

    # Mask of valid points (and IRF region if 'skip' + stretched)
    mask = np.isfinite(y)
    if irf_mode.lower() == 'skip' and stretch_on.any():
        sig = fwhm_init / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        mask = mask & (t > t0_init + 3.0 * sig)

    n_min = n_nl + (1 if has_inf else 0) + 1
    if int(mask.sum()) < n_min:
        raise ValueError(
            f'Not enough data points for the fit '
            f'(need ≥ {n_min}, have {int(mask.sum())} after masking).')

    free_tau_idx = np.where(~tau_fixed)[0]
    free_beta_idx = np.where(stretch_on & ~beta_fixed)[0]

    # Pack initial free-parameter vector
    x0 = []
    if free_tau_idx.size:
        x0.extend(np.log(np.maximum(tau_init[free_tau_idx], 1e-12)))
    if free_beta_idx.size:
        x0.extend(np.log(np.maximum(beta_init[free_beta_idx], 1e-3)))
    if not t0_fixed:
        x0.append(float(t0_init))
    if not fwhm_fixed:
        x0.append(float(np.log(max(fwhm_init, 1e-12))))
    x0 = np.asarray(x0, dtype=float)

    cur = {
        'tau': tau_init.copy(),
        'beta': beta_init.copy(),
        't0': float(t0_init),
        'fwhm': float(fwhm_init),
    }
    t_span = float(t.max() - t.min()) if t.size else 1.0
    n_cols = n_nl + (1 if has_inf else 0)

    def unpack(x):
        i = 0
        if free_tau_idx.size:
            cur['tau'][free_tau_idx] = np.exp(x[i:i + free_tau_idx.size])
            i += free_tau_idx.size
        if free_beta_idx.size:
            cur['beta'][free_beta_idx] = np.exp(x[i:i + free_beta_idx.size])
            i += free_beta_idx.size
        if not t0_fixed:
            cur['t0'] = float(x[i]); i += 1
        if not fwhm_fixed:
            cur['fwhm'] = float(np.exp(x[i])); i += 1

        # Reject pathological regions cheaply
        if (not np.all(np.isfinite(cur['tau'])) or np.any(cur['tau'] <= 0)
                or not np.all(np.isfinite(cur['beta']))
                or np.any(cur['beta'] <= 0) or np.any(cur['beta'] > 2)
                or not np.isfinite(cur['fwhm']) or cur['fwhm'] <= 0
                or cur['fwhm'] > t_span):
            return (np.zeros(n_cols), np.zeros_like(t))

        # Build basis
        M = np.zeros((t.size, n_cols))
        for j in range(n_nl):
            if stretch_on[j]:
                M[:, j] = stretched_irf_conv(
                    t, cur['tau'][j], cur['beta'][j],
                    cur['t0'], cur['fwhm'], irf_mode)
            else:
                M[:, j] = exp_irf_conv(
                    t, cur['tau'][j], cur['t0'], cur['fwhm'])
        if has_inf:
            M[:, -1] = exp_irf_conv(t, np.inf, cur['t0'], cur['fwhm'])

        # Solve amplitudes on the masked points
        Mm = M[mask, :]
        ym = y[mask]
        try:
            A = _lsqminnorm(Mm, ym)
        except Exception:
            A = np.zeros(n_cols)
        if not np.all(np.isfinite(A)):
            A = np.zeros(n_cols)
        return A, M @ A

    def loss(x):
        _, fv = unpack(x)
        r = (y - fv)[mask]
        L = float(np.sum(r ** 2))
        return L if np.isfinite(L) else 1e30

    if x0.size:
        res = minimize(loss, x0, method='Nelder-Mead',
                       options={'xatol': 1e-8, 'fatol': 1e-10,
                                'maxiter': 5000, 'maxfev': 20000,
                                'disp': False})
        A_final, fit_v = unpack(res.x)
        iters = int(res.nit)
        fval = float(res.fun)
    else:
        A_final, fit_v = unpack(np.array([]))
        iters = 0
        fval = float(np.sum(((y - fit_v)[mask]) ** 2))

    res_v = y - fit_v
    res_v[~mask] = np.nan
    rms = float(np.sqrt(np.nanmean(res_v[mask] ** 2))) if mask.any() else 0.0

    return {
        'tau': cur['tau'].copy(),
        'beta': cur['beta'].copy(),
        't0': cur['t0'],
        'fwhm': cur['fwhm'],
        'A': A_final,
        'fit': fit_v,
        'residual': res_v,
        'info': {'iters': iters, 'fval': fval, 'rms': rms,
                 'mask': mask, 'irf_mode': irf_mode},
    }


# =====================================================================
# Lifetime Distribution / Density Analysis (LDA)
# =====================================================================
def compute_lda(D: np.ndarray, t: np.ndarray, tau_grid: np.ndarray,
                t0: float, fwhm: float, alpha: float,
                reg_type: str = 'l2deriv'):
    """Lifetime Density Analysis with Tikhonov regularization.

    Decompose ``D[λ, t]`` ≈ Σ_k A[λ, k] · IRF*exp(-(t-t0)/τ_k) for a
    fixed log-spaced τ grid.  ``alpha`` controls the smoothness penalty.

    Parameters
    ----------
    D : (M, N) float, NaNs allowed (treated as 0 in the LS solve)
    t : (N,) delay axis
    tau_grid : (K,) lifetime grid (log-spaced typical)
    t0, fwhm : Gaussian-IRF parameters (same units as t)
    alpha : non-negative regularization strength
    reg_type : ``'l2'`` or ``'l2deriv'`` (1st-derivative on log τ)

    Returns
    -------
    dict with keys ``A_map (M, K)``, ``Drec (M, N)``, ``info``
    """
    D = np.asarray(D, dtype=float)
    t = np.asarray(t, dtype=float).ravel()
    tau_grid = np.asarray(tau_grid, dtype=float).ravel()
    M = D.shape[0]
    N = t.size
    K = tau_grid.size

    # Basis E (N × K)
    E = np.zeros((N, K))
    for k in range(K):
        E[:, k] = exp_irf_conv(t, float(tau_grid[k]), t0, fwhm)

    # Regularization matrix
    rt = reg_type.lower()
    if rt == 'l2':
        L = np.eye(K)
    elif rt == 'l2deriv' and K >= 2:
        d_log_tau = np.diff(np.log(np.maximum(tau_grid, 1e-30)))
        ws = float(np.mean(d_log_tau))
        if ws <= 0:
            ws = 1.0
        L = (-np.eye(K) + np.eye(K, k=1)) / ws
        L = L[:-1, :]  # drop last row (forward diff)
    else:
        L = np.eye(K)

    # Augmented LS: [E; α L] A^T = [D^T; 0]
    D_clean = np.where(np.isfinite(D), D, 0.0)
    E_aug = np.vstack([E, alpha * L])
    D_aug = np.vstack([D_clean.T, np.zeros((L.shape[0], M))])
    try:
        A_t = _lsqminnorm(E_aug, D_aug)  # (K, M)
    except Exception:
        A_t = np.linalg.pinv(E_aug) @ D_aug
    A_map = A_t.T
    A_map[~np.isfinite(A_map)] = 0.0

    Drec = A_map @ E.T
    R = D - Drec
    fin = np.isfinite(R)
    rms = float(np.sqrt(np.mean(R[fin] ** 2))) if fin.any() else float('nan')
    den = float(np.sqrt(np.mean(D[fin] ** 2))) if fin.any() else 0.0
    ratio = (rms / den) if den > 0 else float('nan')

    info = {'rms': rms, 'ratio': ratio, 'tau_grid': tau_grid,
            'alpha': float(alpha), 'reg_type': rt,
            'K': K, 'M': M, 'N': N}
    return {'A_map': A_map, 'Drec': Drec, 'info': info}


def compute_lcurve(D: np.ndarray, t: np.ndarray, tau_grid: np.ndarray,
                   t0: float, fwhm: float, reg_type: str = 'l2deriv',
                   alpha_list=None):
    """Sweep regularization parameter α over a range and return
    (residual norm, solution norm) pairs for an L-curve plot, plus an
    estimated corner via maximum log-log curvature.
    """
    if alpha_list is None:
        alpha_list = np.logspace(-6, 2, 25)
    alpha_list = np.asarray(alpha_list, dtype=float).ravel()
    n = alpha_list.size
    res_norm = np.zeros(n)
    sol_norm = np.zeros(n)
    for i, a in enumerate(alpha_list):
        out = compute_lda(D, t, tau_grid, t0, fwhm, float(a), reg_type)
        Drec = out['Drec']
        A = out['A_map']
        R = D - Drec
        res_norm[i] = float(np.sqrt(np.nansum(R ** 2)))
        sol_norm[i] = float(np.sqrt(np.nansum(A ** 2)))

    x = np.log(np.maximum(res_norm, 1e-30))
    y = np.log(np.maximum(sol_norm, 1e-30))
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        alpha_corner = alpha_list[n // 2]
    else:
        x = x[ok]; y = y[ok]; aa = alpha_list[ok]
        dx = np.gradient(x); dy = np.gradient(y)
        ddx = np.gradient(dx); ddy = np.gradient(dy)
        with np.errstate(invalid='ignore', divide='ignore'):
            curv = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
        curv[~np.isfinite(curv)] = 0.0
        alpha_corner = float(aa[int(np.argmax(curv))])
    return alpha_list, res_norm, sol_norm, alpha_corner


# =====================================================================
# MCR-ALS (Multivariate Curve Resolution by Alternating Least Squares)
# =====================================================================
def compute_mcr(D: np.ndarray, n_comp: int, *,
                nn_C: bool = True, uni_C: bool = False,
                nn_S: bool = False, tol: float = 1e-4,
                max_iter: int = 200, init_mode: str = 'svd',
                C_init: np.ndarray | None = None,
                S_init: np.ndarray | None = None):
    """MCR-ALS: factor ``D ≈ C · S^T``.

    Convention: ``D`` is shaped (Nt, Nwl).  ``C`` is (Nt, n_comp) and
    ``S`` is (Nwl, n_comp).  NaNs in ``D`` are treated as missing in the
    LS solves; LOF (lack of fit, %) is computed over valid entries only.

    Parameters
    ----------
    init_mode : ``'svd'``, ``'random'``, ``'custom'``
        ``'custom'`` requires ``C_init`` (Nt × n) or ``S_init`` (Nwl × n).
    """
    D = np.asarray(D, dtype=float)
    Nt, Nwl = D.shape
    valid = np.isfinite(D)
    D_clean = np.where(valid, D, 0.0)
    n = int(n_comp)

    # ---- initial estimate ----
    init_mode = init_mode.lower()
    C = None
    S = None
    if init_mode == 'custom':
        if C_init is not None and C_init.shape[1] == n:
            C = np.asarray(C_init, dtype=float).copy()
        elif S_init is not None and S_init.shape[1] == n:
            S = np.asarray(S_init, dtype=float).copy()
            try:
                C, *_ = np.linalg.lstsq(S.T, D_clean.T, rcond=None)  # n × Nt
                C = C.T
            except Exception:
                C = np.abs(np.random.default_rng(0).standard_normal((Nt, n)))
        else:
            C = _mcr_init_svd(D_clean, n)
    elif init_mode == 'random':
        rng = np.random.default_rng(0)
        C = np.abs(rng.standard_normal((Nt, n)))
    else:  # 'svd'
        C = _mcr_init_svd(D_clean, n)
    if S is None:
        S = _mcr_solve_S(D_clean, C)

    # ---- ALS loop ----
    lof_history = []
    prev_lof = float('inf')
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        S = _mcr_solve_S(D_clean, C)
        if nn_S:
            S = np.where(S < 0, 0.0, S)
        C = _mcr_solve_C(D_clean, S)
        if nn_C:
            C = np.where(C < 0, 0.0, C)
        if uni_C:
            C = _mcr_apply_unimodality(C)

        Drec = C @ S.T
        R = D - Drec
        Rv = R[valid]; Dv = D[valid]
        denom = float(np.sum(Dv ** 2))
        if denom <= 0:
            lof = 0.0
        else:
            lof = 100.0 * float(np.sqrt(np.sum(Rv ** 2) / denom))
        lof_history.append(lof)
        if abs(prev_lof - lof) / max(prev_lof, 1e-30) < tol:
            converged = True
            break
        prev_lof = lof

    info = {'lof': lof_history[-1] if lof_history else float('nan'),
            'iter': it, 'lof_history': np.asarray(lof_history),
            'converged': converged,
            'options': {'nn_C': nn_C, 'uni_C': uni_C, 'nn_S': nn_S,
                        'tol': tol, 'max_iter': max_iter,
                        'init_mode': init_mode}}
    return {'C': C, 'S': S, 'info': info}


def _mcr_init_svd(D0: np.ndarray, n: int) -> np.ndarray:
    """SVD-based initial estimate of C (Nt × n) for MCR-ALS."""
    try:
        # Truncated SVD via numpy on possibly large matrices
        U, _, _ = np.linalg.svd(D0, full_matrices=False)
        U = U[:, :n]
    except Exception:
        rng = np.random.default_rng(0)
        return np.abs(rng.standard_normal((D0.shape[0], n)))
    # Make the dominant entry of each column positive
    for j in range(n):
        idx = int(np.argmax(np.abs(U[:, j])))
        if U[idx, j] < 0:
            U[:, j] = -U[:, j]
    C0 = np.abs(U)
    if not np.all(np.isfinite(C0)):
        rng = np.random.default_rng(0)
        C0 = np.abs(rng.standard_normal((D0.shape[0], n)))
    return C0


def _mcr_solve_S(D: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Return S such that D ≈ C · S^T  (S is Nwl × n)."""
    try:
        X, *_ = np.linalg.lstsq(C, D, rcond=None)  # (n × Nwl)
    except Exception:
        X = np.linalg.pinv(C) @ D
    S = X.T
    S[~np.isfinite(S)] = 0.0
    return S


def _mcr_solve_C(D: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Return C such that D ≈ C · S^T  (C is Nt × n)."""
    try:
        X, *_ = np.linalg.lstsq(S, D.T, rcond=None)  # (n × Nt)
    except Exception:
        X = np.linalg.pinv(S) @ D.T
    C = X.T
    C[~np.isfinite(C)] = 0.0
    return C


def _mcr_apply_unimodality(C: np.ndarray) -> np.ndarray:
    """Force each column of C to have a single peak (cheap horizontal
    unimodality: scan from the global max outward and impose
    monotonicity)."""
    out = C.copy()
    Nt, n = out.shape
    for j in range(n):
        col = out[:, j].copy()
        ipk = int(np.argmax(col))
        for k in range(ipk - 1, -1, -1):
            if col[k] > col[k + 1]:
                col[k] = col[k + 1]
        for k in range(ipk + 1, Nt):
            if col[k] > col[k - 1]:
                col[k] = col[k - 1]
        out[:, j] = col
    return out


# =====================================================================
# Vibrational coherence — 2D oscillation map (|FFT|^2 of residual)
# =====================================================================
def compute_coherence(R: np.ndarray, t: np.ndarray, *,
                      t_min: float | None = None,
                      t_max: float | None = None,
                      apod: str = 'hann', zero_pad: int = 2,
                      detrend: str = 'linear',
                      freq_unit: str = 'cm-1',
                      norm_mode: str = 'peak',
                      time_unit: str = 'ps'):
    """2D oscillation map (|FFT|^2 of the residual matrix).

    Parameters
    ----------
    R : (Nwl, Nt) residual matrix (kinetic component already subtracted)
    t : (Nt,) delay axis
    apod : 'rect' | 'hann' | 'hamming' | 'blackman'
    zero_pad : integer factor ≥ 1 (1 = none, 2 = 2x, …)
    detrend : 'none' | 'mean' | 'linear'  (per row)
    freq_unit : 'cm-1' | 'THz' | 'Hz'
    norm_mode : 'peak' | 'none' | 'perWl'
    time_unit : the unit of ``t`` ('fs', 'ps', 'ns', 'us')
    """
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).ravel()
    if t_min is None:
        t_min = float(t.min())
    if t_max is None:
        t_max = float(t.max())
    mask = (t >= t_min) & (t <= t_max)
    tw = t[mask]
    Rw = R[:, mask]
    if tw.size < 8:
        raise ValueError(
            f'Time window is too short (need ≥ 8 points, got {tw.size}).')

    # Resample to uniform grid (median dt of the masked window)
    dts = np.diff(tw)
    dt_uni = float(np.median(dts))
    if dt_uni <= 0:
        raise ValueError('Non-positive median dt.')
    n_uni = int(np.floor((tw[-1] - tw[0]) / dt_uni)) + 1
    if n_uni < 8:
        raise ValueError(f'Resampled grid too short (n={n_uni}).')
    t_uni = tw[0] + np.arange(n_uni) * dt_uni

    # Resample each row (NaN-safe)
    Ru = np.zeros((Rw.shape[0], n_uni))
    for i in range(Rw.shape[0]):
        row = Rw[i, :]
        m = np.isfinite(row)
        if int(m.sum()) < 4:
            continue
        Ru[i, :] = np.interp(t_uni, tw[m], row[m], left=0.0, right=0.0)

    # Detrend per row
    detrend = detrend.lower()
    if detrend == 'mean':
        Ru = Ru - np.mean(Ru, axis=1, keepdims=True)
    elif detrend == 'linear':
        A = np.column_stack([t_uni, np.ones(n_uni)])
        for i in range(Ru.shape[0]):
            y = Ru[i, :]
            m = np.isfinite(y)
            if int(m.sum()) < 2:
                continue
            try:
                coef, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
            except Exception:
                continue
            Ru[i, :] = y - A @ coef
    Ru[~np.isfinite(Ru)] = 0.0

    # Apodization
    n = n_uni
    apod = apod.lower()
    if apod == 'rect':
        w = np.ones(n)
    elif apod == 'hann':
        w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))
    elif apod == 'hamming':
        w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    elif apod == 'blackman':
        k = np.arange(n) / (n - 1)
        w = 0.42 - 0.5 * np.cos(2 * np.pi * k) + 0.08 * np.cos(4 * np.pi * k)
    else:
        w = np.ones(n)
    Ru = Ru * w[None, :]

    # Zero-pad
    zpad = max(1, int(round(zero_pad)))
    Nfft = max(int(2 ** np.ceil(np.log2(n * zpad))), n)

    # FFT (one-sided power)
    X = np.fft.fft(Ru, n=Nfft, axis=1)
    n_half = Nfft // 2 + 1
    P = np.abs(X[:, :n_half]) ** 2

    # Frequency axis
    f_nyq = 1.0 / (2.0 * dt_uni)        # cycles per time_unit
    f_per_tu = np.linspace(0.0, f_nyq, n_half)
    tu2s = {'fs': 1e-15, 'ps': 1e-12, 'ns': 1e-9, 'us': 1e-6}.get(
        time_unit.lower(), 1e-12)
    f_hz = f_per_tu / tu2s
    fu = freq_unit.lower()
    if fu == 'cm-1':
        c_cm = 2.99792458e10
        freq = f_hz / c_cm
    elif fu == 'thz':
        freq = f_hz / 1e12
    else:
        freq = f_hz

    # Normalization
    nm = norm_mode.lower()
    if nm == 'peak':
        pk = float(np.nanmax(P))
        if np.isfinite(pk) and pk > 0:
            P = P / pk
    elif nm == 'perwl':
        pkW = np.nanmax(P, axis=1, keepdims=True)
        pkW = np.where(pkW <= 0, 1.0, pkW)
        P = P / pkW
    P[~np.isfinite(P)] = 0.0

    info = {'dt_uni': dt_uni, 'Nfft': Nfft, 't_uni': t_uni, 'Nwin': n,
            'apod': apod, 'zero_pad': zpad, 'detrend': detrend,
            'freq_unit': fu, 'norm_mode': nm}
    return {'P': P, 'freq': freq, 'info': info}
