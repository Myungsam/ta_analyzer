"""
Regression tests for the moving-average mode of Crop Data resampling.

Verifies:

  1. ta_core.moving_average_wavelength / resample_by_spec / apply_resample_info
        - same length, unchanged λ axis, 2-D / 1-D / empty inputs
        - box / triangular / gaussian (σ = k/6) weights, edge and NaN
          renormalisation, invalid window sizes and kernels
        - the bin-only resample_wavelength refuses 'moving'

  2. App state: status label / title / export suffix, solvent smoothed with
     the same kernel, requests without window keys, rejected windows

  3. CropDialog: widgets and the single enable rule, live preview,
     invalid window fallback, persistence, hidden-parameter changes

  4. Apply timing with the widest window
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

import numpy as np  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402

import ta_core  # noqa: E402
import ta_main  # noqa: E402
import ta_dialogs_a  # noqa: E402


def _synthetic(seed=0, n_wl=1200, lo=400.0):
    """Non-uniform λ grid (0.27–0.38 nm steps) and a noisy two-band map."""
    rng = np.random.default_rng(seed)
    wl = lo + np.cumsum(rng.uniform(0.27, 0.38, n_wl))
    t = np.r_[np.arange(-1.0, 1.0, 0.1), np.logspace(0, 2.5, 40)]
    tp = np.clip(t, 0, None)[None, :]
    A = (np.exp(-tp / 5.0) * np.exp(-((wl[:, None] - 500) / 30) ** 2)
         - 0.5 * np.exp(-tp / 80.0) * np.exp(-((wl[:, None] - 560) / 25) ** 2))
    A = A + rng.normal(scale=0.02, size=A.shape)
    return wl, t, A


def _window():
    win = ta_main.TAAnalyzer()
    wl, t, A = _synthetic()
    win.set_loaded_data(wl, t, A, 'Loaded: synthetic.csv')
    return win, wl, t, A


def _moving(k=5, kernel='box', enabled=True):
    return {'enabled': enabled, 'mode': 'moving', 'window': k,
            'kernel': kernel}


def _expected(A, i, w):
    """Hand-computed weighted mean at row i (window centred on i)."""
    h = len(w) // 2
    num = den = 0.0
    for j, wj in enumerate(w):
        r = i + j - h
        if 0 <= r < len(A) and np.isfinite(A[r]):
            num += wj * A[r]
            den += wj
    return num / den if den > 0 else np.nan


# =====================================================================
# 1) Kernel
# =====================================================================
def test_kernel_api():
    wl, _, A = _synthetic()
    w_out, a_out, info = ta_core.moving_average_wavelength(wl, A, 5, 'box')
    assert info['applied'] and info['mode'] == 'moving'
    assert info['k'] == 5 and info['kernel'] == 'box'
    assert info['n_in'] == info['n_out'] == len(wl)
    assert ta_core.ALL_RESAMPLE_MODES == ('average', 'decimate', 'moving')
    assert ta_core.RESAMPLE_MODES == ('average', 'decimate')
    assert ta_core.MOVING_KERNELS == ('box', 'triangular', 'gaussian')
    try:
        ta_core.resample_wavelength(wl, A, 1.0, 'moving')
    except ValueError:
        pass
    else:
        raise AssertionError("resample_wavelength must refuse 'moving'")
    # dispatcher
    _, a2, info2 = ta_core.resample_by_spec(wl, A, _moving(5, 'box'))
    assert np.array_equal(a2, a_out, equal_nan=True) and info2 == info
    _, _, info3 = ta_core.resample_by_spec(
        wl, A, {'mode': 'average', 'dx': 1.0})
    assert info3['mode'] == 'average' and info3['n_out'] < len(wl)
    print('[kernel] API / modes / dispatcher OK')


def test_shapes():
    wl, _, A = _synthetic()
    for arr in (A, A[:, 3], np.zeros((len(wl), 0))):
        w_out, a_out, info = ta_core.moving_average_wavelength(
            wl, arr, 7, 'triangular')
        assert info['applied']
        assert a_out.shape == arr.shape, (a_out.shape, arr.shape)
        assert np.array_equal(w_out, wl)
    print('[kernel] 2-D / 1-D / empty shapes, λ unchanged OK')


def test_interior_weights():
    rng = np.random.default_rng(3)
    wl = np.arange(50, dtype=float)
    y = rng.normal(size=50)
    weights = {
        'box': np.ones(5),
        'triangular': np.array([1, 2, 3, 2, 1], float),
        'gaussian': np.exp(-np.arange(-2, 3) ** 2 / (2 * (5 / 6) ** 2)),
    }
    for kernel, w in weights.items():
        assert np.allclose(ta_core.moving_kernel_weights(5, kernel), w)
        _, out, _ = ta_core.moving_average_wavelength(wl, y, 5, kernel)
        for i in range(2, 48):
            assert np.isclose(out[i], np.dot(w, y[i - 2:i + 3]) / w.sum()), \
                (kernel, i)
    print('[kernel] interior weights (box / triangular / gaussian) OK')


def test_edges():
    rng = np.random.default_rng(4)
    wl = np.arange(30, dtype=float)
    y = rng.normal(size=30)
    for kernel in ta_core.MOVING_KERNELS:
        w = ta_core.moving_kernel_weights(5, kernel)
        _, out, _ = ta_core.moving_average_wavelength(wl, y, 5, kernel)
        for i in (0, 1, 28, 29):
            assert np.isclose(out[i], _expected(y, i, w)), (kernel, i)
    print('[kernel] edge renormalisation OK')


def test_nan():
    wl = np.arange(20, dtype=float)
    y = np.arange(20, dtype=float)
    y[3] = np.nan
    y[10:16] = np.nan                       # 6 NaNs: window 5 at 12/13 all NaN
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _, out, _ = ta_core.moving_average_wavelength(wl, y, 5, 'box')
    w = np.ones(5)
    assert np.isclose(out[3], _expected(y, 3, w))      # NaN point itself
    assert np.isclose(out[4], _expected(y, 4, w))
    assert np.isnan(out[12]) and np.isnan(out[13])
    assert np.isfinite(out[11]) and np.isfinite(out[16])
    print('[kernel] NaN renormalisation / all-NaN → NaN OK')


def test_constant_linear():
    wl = np.arange(40, dtype=float)
    const = np.full(40, 3.5)
    lin = 2.0 * wl - 7.0
    for kernel in ta_core.MOVING_KERNELS:
        _, c, _ = ta_core.moving_average_wavelength(wl, const, 9, kernel)
        assert np.allclose(c, 3.5), kernel
        _, l, _ = ta_core.moving_average_wavelength(wl, lin, 9, kernel)
        assert np.allclose(l[4:-4], lin[4:-4]), kernel
    print('[kernel] constant / linear preservation OK')


def test_invalid_k():
    wl = np.arange(7, dtype=float)
    y = np.arange(7, dtype=float)
    bad = [4, 1, 103, 7, 9, 2.5, True, np.float64(3.2)]
    for k in bad:
        w_out, a_out, info = ta_core.moving_average_wavelength(wl, y, k)
        assert not info['applied'], k
        assert np.array_equal(a_out, y) and a_out is not y
    _, _, info = ta_core.moving_average_wavelength(wl, y, 5, 'foo')
    assert not info['applied']
    for k in (5, np.int64(5), 5.0):
        _, _, info = ta_core.moving_average_wavelength(wl, y, k)
        assert info['applied'] and info['k'] == 5 and type(info['k']) is int
    print('[kernel] invalid windows / kernels rejected, int-valued k OK')


# =====================================================================
# 2) App
# =====================================================================
def test_solvent_same_kernel():
    win, wl, t, A = _window()
    win.sub_irf_solv_wl = wl.copy()
    win.sub_irf_solv_t = t.copy()
    win.sub_irf_solv_data = A.copy()
    win.sub_irf_scale = 1.0
    win.apply_crop_by_range(430, 700, -1, 400,
                            resample=_moving(7, 'gaussian'))
    win.sub_irf_applied = True
    win.update_all()
    assert win.resample_enabled and win.resample_info['mode'] == 'moving'
    assert np.allclose(win.sub_irf_aligned, win.deltaA_raw, equal_nan=True)
    assert np.nanmax(np.abs(win.deltaA)) < 1e-12
    # mismatched rows must raise, never pass through silently
    try:
        ta_core.apply_resample_info(np.zeros((5, 2)), win.resample_info)
    except ValueError:
        pass
    else:
        raise AssertionError('row-count mismatch must raise')
    assert ta_core.apply_bin_groups is ta_core.apply_resample_info
    win.close()
    print('[solvent] smoothed with the same kernel OK')


def test_labels():
    win, wl, _, _ = _window()
    win.apply_crop_by_range(wl[0], wl[-1], -1.0, 400.0, resample=_moving())
    txt = win.lbl_status.text()
    assert '[smoothed 5 px, box]' in txt and '[cropped]' not in txt, txt
    assert len(win.wavelength) == len(wl)
    assert np.array_equal(win.wavelength, wl)
    assert not win.is_cropped() and win.is_modified()
    assert 'smoothed' in win._map_title()
    assert win._resample_suffix() == 'ma5box'
    win.apply_crop_by_range(wl[0], wl[-1], -1.0, 400.0,
                            resample={'enabled': True, 'mode': 'average',
                                      'dx': 1.0})
    assert win._resample_suffix() == 'rs1nm'
    assert 'resampled Δλ=1 nm, avg' in win.lbl_status.text()
    win.close()
    print('[app] status / title / export suffix OK')


def test_apply_crop_defaults():
    win, wl, _, _ = _window()
    win.apply_crop_by_range(wl[0], wl[-1], -1, 400,
                            resample={'enabled': True, 'mode': 'moving'})
    assert win.resample_enabled
    assert (win.resample_window, win.resample_kernel) == (5, 'box')
    for bad in (2.5, 6):
        win.apply_crop_by_range(wl[0], wl[-1], -1, 400,
                                resample=_moving(bad, 'box'))
        assert not win.resample_enabled, bad
        assert win.resample_window == 5 and type(win.resample_window) is int
    dlg = ta_dialogs_a.CropDialog(win, win)       # must not raise
    assert dlg.ed_window.value() == 5
    dlg.close(); win.close()
    print('[app] missing keys use defaults, rejected window not stored OK')


# =====================================================================
# 3) Dialog
# =====================================================================
def _states(dlg):
    return tuple(w.isEnabled() for w in (
        dlg.rb_avg, dlg.rb_dec, dlg.rb_mov,
        dlg.ed_resample_dx, dlg.ed_window, dlg.dd_kernel))


def _flush(dlg):
    dlg._resample_timer.stop()
    dlg._on_resample_changed()


def test_dialog_widgets():
    win, wl, _, _ = _window()
    dlg = ta_dialogs_a.CropDialog(win, win)
    assert dlg.ed_window.value() == 5
    assert (dlg.ed_window.minimum(), dlg.ed_window.maximum(),
            dlg.ed_window.singleStep()) == (3, 101, 2)
    assert [dlg.dd_kernel.itemData(i) for i in range(dlg.dd_kernel.count())] \
        == list(ta_core.MOVING_KERNELS)
    assert dlg.dd_kernel.currentData() == 'box'
    _flush(dlg)
    assert _states(dlg) == (False,) * 6            # checkbox off
    dlg.cb_resample.setChecked(True); _flush(dlg)
    assert _states(dlg) == (True, True, True, True, False, False)
    dlg.rb_mov.setChecked(True); _flush(dlg)
    assert _states(dlg) == (True, True, True, False, True, True)
    assert dlg.lbl_resample_info.text() == \
        f'{len(wl)} → {len(wl)} λ points (window 5 px, box)', \
        dlg.lbl_resample_info.text()
    dlg.rb_avg.setChecked(True); _flush(dlg)
    assert _states(dlg) == (True, True, True, True, False, False)
    dlg.close(); win.close()
    print('[dialog] widgets + single enable rule (round trip) OK')


def test_dialog_preview():
    win, wl, _, _ = _window()
    dlg = ta_dialogs_a.CropDialog(win, win)
    dlg.cb_resample.setChecked(True)
    dlg.rb_dec.setChecked(True)
    dlg.rb_mov.setChecked(True)                    # decimate → moving
    dlg.ed_window.setValue(9)
    dlg.dd_kernel.setCurrentIndex(ta_core.MOVING_KERNELS.index('gaussian'))
    _flush(dlg)
    lines = [l for l in dlg.canvas_spec.ax.get_lines()
             if l.get_label().startswith('smoothed')]
    assert len(lines) == 1 and len(lines[0].get_xdata()) == len(wl)
    idx = dlg._sel_t_idx
    _, expect, _ = ta_core.moving_average_wavelength(
        wl, dlg._display_deltaA()[:, idx], 9, 'gaussian')
    assert np.allclose(lines[0].get_ydata(), expect, equal_nan=True)
    assert np.array_equal(lines[0].get_xdata(), wl)
    dlg.close(); win.close()
    print('[dialog] live smoothed preview OK')


def test_dialog_invalid_k():
    win, wl, _, _ = _window()
    dlg = ta_dialogs_a.CropDialog(win, win)
    dlg.cb_resample.setChecked(True)
    dlg.rb_mov.setChecked(True)
    dlg.ed_wl_min.setValue(float(wl[100]))         # 4 original points
    dlg.ed_wl_max.setValue(float(wl[103]))
    dlg.ed_window.setValue(5)
    _flush(dlg)
    assert 'window must be odd' in dlg.lbl_resample_info.text()
    calls = []
    real = ta_dialogs_a.warn_box
    ta_dialogs_a.warn_box = lambda *a, **k: calls.append(a)
    try:
        dlg._apply()
    finally:
        ta_dialogs_a.warn_box = real
    assert len(calls) == 1 and not win.resample_enabled
    assert win.crop_bounds is not None and len(win.wavelength) == 4
    # an even value typed in is rejected too
    dlg = ta_dialogs_a.CropDialog(win, win)
    dlg._set_full_wl()
    dlg.cb_resample.setChecked(True)
    dlg.rb_mov.setChecked(True)
    dlg.ed_window.setValue(6)
    calls.clear()
    ta_dialogs_a.warn_box = lambda *a, **k: calls.append(a)
    assert dlg.ed_window.value() == 6          # typed-in even value is kept
    try:
        dlg._apply()
        assert len(calls) == 1 and not win.resample_enabled
    finally:
        ta_dialogs_a.warn_box = real
    win.close()
    print('[dialog] invalid window → warning + crop only OK')


def test_persist_idempotent():
    win, wl, _, _ = _window()
    dlg = ta_dialogs_a.CropDialog(win, win)
    dlg.cb_resample.setChecked(True)
    dlg.rb_mov.setChecked(True)
    dlg.ed_window.setValue(7)
    dlg.dd_kernel.setCurrentIndex(ta_core.MOVING_KERNELS.index('gaussian'))
    dlg._apply()
    assert (win.resample_mode, win.resample_window, win.resample_kernel) == \
        ('moving', 7, 'gaussian')
    dlg = ta_dialogs_a.CropDialog(win, win)
    assert dlg.rb_mov.isChecked() and dlg.ed_window.value() == 7
    assert dlg.dd_kernel.currentData() == 'gaussian'
    win.bg_applied = True
    win.bg_spectrum = np.zeros(len(win.wavelength))
    dlg._apply()                                   # unchanged
    assert win.bg_applied
    dlg = ta_dialogs_a.CropDialog(win, win)
    dlg.dd_kernel.setCurrentIndex(ta_core.MOVING_KERNELS.index('box'))
    dlg._apply()                                   # kernel changed
    assert not win.bg_applied and win.resample_kernel == 'box'
    w2, t2, A2 = _synthetic(seed=5)
    win.set_loaded_data(w2, t2, A2, 'Loaded: other.csv')
    assert (win.resample_enabled, win.resample_window,
            win.resample_kernel) == (False, 5, 'box')
    win.close()
    print('[dialog] persistence / idempotent re-apply OK')


def test_hidden_params_ignored():
    win, wl, _, _ = _window()
    win.apply_crop_by_range(wl[0], wl[-1], -1, 400, resample=_moving(7))
    win.bg_applied = True
    win.bg_spectrum = np.zeros(len(win.wavelength))
    dlg = ta_dialogs_a.CropDialog(win, win)
    dlg.ed_resample_dx.setValue(3.0)               # inactive parameter
    dlg._apply()
    assert win.bg_applied, 'Δλ change must not matter in moving mode'
    win.apply_crop_by_range(wl[0], wl[-1], -1, 400,
                            resample={'enabled': True, 'mode': 'average',
                                      'dx': 1.0})
    win.bg_applied = True
    win.bg_spectrum = np.zeros(len(win.wavelength))
    dlg = ta_dialogs_a.CropDialog(win, win)
    dlg.ed_window.setValue(11)                     # inactive parameters
    dlg.dd_kernel.setCurrentIndex(2)
    dlg._apply()
    assert win.bg_applied, 'window/kernel change must not matter in bin mode'
    win.close()
    print('[dialog] inactive-mode parameters ignored OK')


def test_apply_timing():
    win = ta_main.TAAnalyzer()
    rng = np.random.default_rng(1)
    wl = 317 + np.cumsum(rng.uniform(0.27, 0.38, 2136))
    t = np.r_[np.linspace(-1, 1, 40), np.logspace(0.01, 3.5, 152)]
    win.set_loaded_data(wl, t, rng.normal(size=(2136, 192)), 'Loaded: perf')
    t0 = time.perf_counter()
    win.apply_crop_by_range(wl[0], wl[-1], t[0], t[-1],
                            resample=_moving(101, 'gaussian'))
    dt = time.perf_counter() - t0
    assert win.resample_enabled
    print(f'[perf] apply window 101 gaussian on 2136x192: {dt * 1e3:.0f} ms')
    assert dt < 2.0, dt
    win.close()


TESTS = [
    test_kernel_api, test_shapes, test_interior_weights, test_edges,
    test_nan, test_constant_linear, test_invalid_k,
    test_solvent_same_kernel, test_labels, test_apply_crop_defaults,
    test_dialog_widgets, test_dialog_preview, test_dialog_invalid_k,
    test_persist_idempotent, test_hidden_params_ignored, test_apply_timing,
]


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    for fn in TESTS:
        fn()
    print(f'\nAll {len(TESTS)} moving-average checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
