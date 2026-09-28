"""
Regression tests for wavelength resampling in the Crop Data dialog.

Verifies:

  1. ta_core.resample_wavelength / apply_bin_groups
        - uniform bins from origin, boundary points go to the upper bin
        - empty bins are dropped
        - average = nanmean (all-NaN bin → NaN, no RuntimeWarning)
        - decimate = row closest to the bin centre (ties → lower index)
        - Δλ ≤ mean spacing → no-op (applied=False)
        - apply_bin_groups matches a hand-computed nanmean

  2. App state (ta_main.TAAnalyzer)
        - status label / map title tags, is_cropped vs is_modified
        - Revert and "full range + unchecked" restore the original grid
        - pinned kinetics / selected λ survive a resample
        - an empty crop range changes no state

  3. Solvent IRF: the solvent is binned exactly like the sample
     (independent oracle: solvent == sample ⇒ aligned == resampled sample)

  4. CropDialog
        - widgets + defaults, live spectrum preview, invalid Δλ fallback,
          re-apply idempotence, settings persistence across re-opens
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
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
    """Non-uniform λ grid (0.27–0.38 nm steps, like the real spectrometer)
    and a two-component decay on a mixed linear/log delay axis."""
    rng = np.random.default_rng(seed)
    wl = lo + np.cumsum(rng.uniform(0.27, 0.38, n_wl))
    t = np.r_[np.arange(-1.0, 1.0, 0.1), np.logspace(0, 2.5, 40)]
    A = (np.exp(-np.clip(t, 0, None)[None, :] / 5.0)
         * np.exp(-((wl[:, None] - 500) / 30) ** 2)
         - 0.5 * np.exp(-np.clip(t, 0, None)[None, :] / 80.0)
         * np.exp(-((wl[:, None] - 560) / 25) ** 2))
    return wl, t, A


def _window():
    win = ta_main.TAAnalyzer()
    wl, t, A = _synthetic()
    win.set_loaded_data(wl, t, A, 'Loaded: synthetic.csv')
    return win, wl, t, A


# =====================================================================
# 1) Kernel
# =====================================================================
def test_kernel_origin():
    wl = np.round(np.arange(500.0, 510.0, 0.1), 10)           # 100 points
    A = np.vstack([wl, 2 * wl]).T
    w, a, info = ta_core.resample_wavelength(wl, A, 1.0)
    assert info['applied'] and info['origin'] == 500.0
    assert info['n_out'] == 10
    # bin k holds wl in [500+k, 501+k): mean is 500.45 + k
    assert np.allclose(w, 500.45 + np.arange(10))
    assert np.allclose(a[:, 1], 2 * w)
    # boundary rule: with Δλ = 0.5 the point 500.5 opens the second bin
    w, _, info = ta_core.resample_wavelength(wl, A, 0.5)
    assert info['n_out'] == 20
    assert np.isclose(w[0], np.mean(wl[:5])) and np.isclose(w[1], np.mean(wl[5:10]))
    # explicit origin shifts the bins
    w, _, info = ta_core.resample_wavelength(wl, A, 1.0, origin=499.5)
    assert info['n_out'] == 11 and np.isclose(w[0], np.mean(wl[:5]))
    assert np.all(np.diff(w) > 0)
    print('[kernel] origin / boundaries OK')


def test_kernel_empty_bins():
    wl = np.round(np.arange(500.0, 520.0, 0.1), 10)
    keep = (wl < 505.0) | (wl >= 510.0)                        # 5 nm gap
    wl = wl[keep]
    w, _, info = ta_core.resample_wavelength(wl, np.ones((wl.size, 3)), 1.0)
    assert info['n_out'] == 15, info['n_out']
    assert not np.any((w > 505.0) & (w < 510.0))
    print('[kernel] empty bins dropped OK')


def test_kernel_average_nan():
    wl = np.round(np.arange(500.0, 504.0, 0.1), 10)            # 4 bins × 10
    A = np.tile(np.arange(wl.size, dtype=float)[:, None], (1, 2))
    A[0:10, :] = np.nan                                         # bin 0: all NaN
    A[10:12, 0] = np.nan                                        # bin 1: partial
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _, a, _ = ta_core.resample_wavelength(wl, A, 1.0)
    assert np.all(np.isnan(a[0]))
    assert np.isclose(a[1, 0], np.mean(np.arange(12, 20)))
    assert np.isclose(a[1, 1], np.mean(np.arange(10, 20)))
    print('[kernel] average nanmean OK')


def test_kernel_decimate():
    wl, _, A = _synthetic(seed=3)
    w, a, info = ta_core.resample_wavelength(wl, A, 2.0, 'decimate')
    idx = info['idx']
    assert set(w).issubset(set(wl)) and np.array_equal(a, A[idx])
    k = np.floor((wl - wl[0]) / 2.0 + 1e-9).astype(int)
    for j, i in enumerate(idx):
        members = np.flatnonzero(k == k[i])
        centre = wl[0] + (k[i] + 0.5) * 2.0
        d = np.abs(wl[members] - centre)
        assert i == members[np.argmin(d)], (j, i)
    # bin [0, 2): 0.5 and 1.5 are equally close to the centre 1.0 → lower
    # index wins; bin [2, 4): 3.0 sits on its centre.
    wl_t = np.array([0.0, 0.5, 1.5, 2.0, 3.0])
    _, _, info = ta_core.resample_wavelength(wl_t, np.zeros((5, 1)), 2.0,
                                             'decimate')
    assert list(info['idx']) == [1, 4], list(info['idx'])
    print('[kernel] decimate nearest-centre OK')


def test_kernel_too_small():
    wl, _, A = _synthetic()
    mean = (wl[-1] - wl[0]) / (wl.size - 1)
    for dx in (mean, 0.9 * mean, 0.05):
        w, a, info = ta_core.resample_wavelength(wl, A, dx)
        assert not info['applied']
        assert np.array_equal(w, wl) and np.array_equal(a, A)
    _, _, info = ta_core.resample_wavelength(wl, A, 1.01 * mean)
    assert info['applied']
    print('[kernel] Δλ ≤ mean spacing → no-op OK')


def test_apply_bin_groups():
    rng = np.random.default_rng(7)
    wl, _, A = _synthetic(seed=7)
    B = rng.normal(size=A.shape)
    B[rng.random(B.shape) < 0.2] = np.nan
    _, _, info = ta_core.resample_wavelength(wl, A, 1.5)
    out = ta_core.apply_bin_groups(B, info)
    k = np.floor((wl - wl[0]) / 1.5 + 1e-9).astype(int)
    for j, kk in enumerate(np.unique(k)):
        rows = B[k == kk]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            expect = np.nanmean(rows, axis=0)
        assert np.allclose(out[j], expect, equal_nan=True)
    _, _, info_d = ta_core.resample_wavelength(wl, A, 1.5, 'decimate')
    assert np.array_equal(ta_core.apply_bin_groups(B, info_d),
                          B[info_d['idx']], equal_nan=True)
    print('[kernel] apply_bin_groups OK')


# =====================================================================
# 2) App state
# =====================================================================
def test_status_label():
    win, wl, _, _ = _window()
    win.bg_spectrum = np.zeros(len(wl)); win.bg_applied = True; win.bg_n = 3
    win.apply_crop_by_range(wl[0], wl[-1], -1.0, 400.0,
                            resample={'enabled': True, 'dx': 1.0,
                                      'mode': 'average'})
    assert win.resample_enabled and len(win.wavelength) < len(wl)
    assert not win.bg_applied, 'resampling must reset corrections'
    txt = win.lbl_status.text()
    assert 'resampled Δλ=1 nm, avg' in txt and '[cropped]' not in txt, txt
    assert not win.is_cropped() and win.is_modified()
    assert 'resampled' in win._map_title()
    win.apply_crop_by_range(450, 600, -1.0, 400.0,
                            resample={'enabled': True, 'dx': 2.0,
                                      'mode': 'decimate'})
    txt = win.lbl_status.text()
    assert '[cropped]' in txt and 'Δλ=2 nm, decimate' in txt, txt
    win.close()
    print('[app] status label / title / is_cropped OK')


def test_revert():
    win, wl, t, _ = _window()
    win.apply_crop_by_range(450, 600, 0, 100,
                            resample={'enabled': True, 'dx': 1.0,
                                      'mode': 'average'})
    dlg = ta_dialogs_a.CropDialog(win, win)
    assert dlg.btn_revert.isEnabled()
    dlg._revert()
    assert len(win.wavelength) == len(wl) and len(win.delay) == len(t)
    assert not win.resample_enabled and win.crop_bounds is None
    assert not win.is_modified()
    win.close()
    print('[app] revert OK')


def test_full_range_uncheck():
    win, wl, t, A = _window()
    win.apply_crop_by_range(450, 600, 0, 100,
                            resample={'enabled': True, 'dx': 1.0,
                                      'mode': 'average'})
    dlg = ta_dialogs_a.CropDialog(win, win)
    dlg._set_full_wl()
    dlg._set_full_t()
    dlg.cb_resample.setChecked(False)
    dlg._apply()
    assert len(win.wavelength) == len(wl) and len(win.delay) == len(t)
    assert np.array_equal(win.deltaA_raw, A)
    assert not win.is_modified()
    win.close()
    print('[app] full range + unchecked restores original OK')


def test_clip():
    win, wl, _, _ = _window()
    win.kinOverlays = [460.0, 520.0]
    win.selWL = 500.0
    win.apply_crop_by_range(wl[0], wl[-1], -1.0, 400.0,
                            resample={'enabled': True, 'dx': 2.0,
                                      'mode': 'average'})
    assert win.kinOverlays == [460.0, 520.0]
    assert win.wavelength.min() <= win.selWL <= win.wavelength.max()
    win.close()
    print('[app] overlays / selection survive resample OK')


def test_empty_range_no_state_change():
    win, wl, _, _ = _window()
    win.apply_crop_by_range(450, 600, -1, 400,
                            resample={'enabled': True, 'dx': 1.0,
                                      'mode': 'average'})
    before = (win.crop_bounds, win.resample_enabled, win.resample_dx,
              len(win.wavelength))
    calls = []
    real = ta_main.warn_box
    ta_main.warn_box = lambda *a, **k: calls.append(a)
    try:
        win.apply_crop_by_range(2000, 3000, -1, 400,
                                resample={'enabled': True, 'dx': 5.0,
                                          'mode': 'decimate'})
    finally:
        ta_main.warn_box = real
    assert calls
    after = (win.crop_bounds, win.resample_enabled, win.resample_dx,
             len(win.wavelength))
    assert before == after, (before, after)
    win.close()
    print('[app] empty range leaves state untouched OK')


# =====================================================================
# 3) Solvent
# =====================================================================
def test_solvent_same_bins():
    for mode in ('average', 'decimate'):
        win, wl, t, A = _window()
        # solvent == sample on the same grid → after identical binning the
        # aligned solvent must equal the resampled sample exactly.
        win.sub_irf_solv_wl = wl.copy()
        win.sub_irf_solv_t = t.copy()
        win.sub_irf_solv_data = A.copy()
        win.sub_irf_applied = True
        win.sub_irf_scale = 1.0
        win.apply_crop_by_range(430, 640, -1, 400,
                                resample={'enabled': True, 'dx': 1.5,
                                          'mode': mode})
        win.sub_irf_applied = True
        win.update_all()
        assert win.resample_enabled
        assert win.sub_irf_aligned.shape == win.deltaA_raw.shape
        assert np.allclose(win.sub_irf_aligned, win.deltaA_raw,
                           equal_nan=True), mode
        assert np.nanmax(np.abs(win.deltaA)) < 1e-12, mode
        win.close()
    print('[solvent] binned like the sample (average + decimate) OK')


# =====================================================================
# 4) Dialog
# =====================================================================
def test_dialog_widgets():
    win, _, _, _ = _window()
    dlg = ta_dialogs_a.CropDialog(win, win)
    assert not dlg.cb_resample.isChecked()
    assert abs(dlg.ed_resample_dx.value() - 1.0) < 1e-12
    assert dlg.rb_avg.isChecked() and not dlg.rb_dec.isChecked()
    assert dlg.lbl_resample_info.text() == ''
    assert not dlg.btn_revert.isEnabled()
    dlg.close(); win.close()
    print('[dialog] widgets + defaults OK')


def test_dialog_preview():
    win, wl, _, _ = _window()
    dlg = ta_dialogs_a.CropDialog(win, win)
    # pin one wavelength so the preview matrix differs from the original
    dlg._kin_overlay_wl = [float(wl[300])]
    dlg._recompute_preview()
    dlg.cb_resample.setChecked(True)
    dlg.ed_resample_dx.setValue(2.0)
    dlg._resample_timer.stop()
    dlg._on_resample_changed()
    txt = dlg.lbl_resample_info.text()
    assert txt.startswith(f'{len(wl)} → ') and 'avg' in txt, txt
    m = int(txt.split('→')[1].split()[0])
    lines = [l for l in dlg.canvas_spec.ax.get_lines()
             if l.get_label().startswith('resampled')]
    assert len(lines) == 1 and len(lines[0].get_xdata()) == m
    # the overlay is built from the drop-interpolated preview data
    idx = dlg._sel_t_idx
    _, expect, _ = ta_core.resample_wavelength(
        wl, dlg._display_deltaA()[:, idx], 2.0, 'average')
    assert np.allclose(lines[0].get_ydata(), expect, equal_nan=True)
    dlg.cb_resample.setChecked(False)
    dlg._resample_timer.stop()
    dlg._on_resample_changed()
    assert not [l for l in dlg.canvas_spec.ax.get_lines()
                if l.get_label().startswith('resampled')]
    dlg.close(); win.close()
    print('[dialog] live spectrum preview OK')


def test_dialog_invalid_dx():
    win, wl, _, _ = _window()
    win.apply_crop_by_range(wl[0], wl[-1], -1, 400,
                            resample={'enabled': True, 'dx': 2.0,
                                      'mode': 'average'})
    dlg = ta_dialogs_a.CropDialog(win, win)
    dlg.ed_resample_dx.setValue(0.1)             # below the ~0.33 nm spacing
    dlg.ed_wl_min.setValue(450.0)                # synthetic λ spans ~400–795
    dlg.ed_wl_max.setValue(580.0)
    calls = []
    real = ta_dialogs_a.warn_box
    ta_dialogs_a.warn_box = lambda *a, **k: calls.append(a)
    try:
        dlg._apply()
    finally:
        ta_dialogs_a.warn_box = real
    assert len(calls) == 1
    assert not win.resample_enabled
    assert win.crop_bounds[:2] == (450.0, 580.0), win.crop_bounds
    assert win.wavelength.min() >= 450.0 and win.wavelength.max() <= 580.0
    win.close()
    print('[dialog] invalid Δλ → warning + crop only OK')


def test_reapply_idempotent():
    win, wl, _, _ = _window()
    dlg = ta_dialogs_a.CropDialog(win, win)
    dlg.ed_wl_min.setValue(450.0)
    dlg.ed_wl_max.setValue(620.0)
    dlg.cb_resample.setChecked(True)
    dlg.ed_resample_dx.setValue(2.0)
    dlg._apply()
    bounds, origin = win.crop_bounds, win.resample_info['origin']
    # re-open, switch the mode only
    dlg = ta_dialogs_a.CropDialog(win, win)
    assert (dlg.ed_wl_min.value(), dlg.ed_wl_max.value()) == (450.0, 620.0)
    dlg.rb_dec.setChecked(True)
    dlg._apply()
    assert win.crop_bounds == bounds
    assert win.resample_info['origin'] == origin
    assert win.resample_mode == 'decimate'
    # re-open and apply unchanged → no rebuild (corrections survive)
    win.bg_applied = True
    win.bg_spectrum = np.zeros(len(win.wavelength))
    dlg = ta_dialogs_a.CropDialog(win, win)
    dlg._apply()
    assert win.bg_applied, 'unchanged apply must not reset corrections'
    win.close()
    print('[dialog] re-apply idempotent OK')


def test_settings_persist():
    win, _, _, _ = _window()
    dlg = ta_dialogs_a.CropDialog(win, win)
    dlg.cb_resample.setChecked(True)
    dlg.ed_resample_dx.setValue(3.0)
    dlg.rb_dec.setChecked(True)
    dlg._apply()
    dlg = ta_dialogs_a.CropDialog(win, win)
    assert dlg.cb_resample.isChecked() and dlg.rb_dec.isChecked()
    assert abs(dlg.ed_resample_dx.value() - 3.0) < 1e-12
    assert dlg.btn_revert.isEnabled()
    dlg.close()
    wl, t, A = _synthetic(seed=5)
    win.set_loaded_data(wl, t, A, 'Loaded: other.csv')
    assert not win.resample_enabled and win.crop_bounds is None
    assert win.resample_dx == 1.0 and win.resample_mode == 'average'
    win.close()
    print('[dialog] settings persist, reset on new load OK')


TESTS = [
    test_kernel_origin, test_kernel_empty_bins, test_kernel_average_nan,
    test_kernel_decimate, test_kernel_too_small, test_apply_bin_groups,
    test_status_label, test_revert, test_full_range_uncheck, test_clip,
    test_empty_range_no_state_change,
    test_solvent_same_bins,
    test_dialog_widgets, test_dialog_preview, test_dialog_invalid_dx,
    test_reapply_idempotent, test_settings_persist,
]


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    for fn in TESTS:
        fn()
    # test_solvent_same_bins covers both modes → 18 checks in 17 functions
    print(f'\nAll {len(TESTS) + 1} resampling checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
