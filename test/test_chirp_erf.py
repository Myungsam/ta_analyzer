"""
Tests for the per-wavelength erf (IRF⊗exp) t0 helper in the Chirp dialog.

Verifies:
    1) ``ta_core.compute_erf_t0`` recovers a known chirp curve within
       20 fs and beats the argmax ridge on noisy data.
    2) Rows near a sign change (no signal) are rejected by R² min, and
       ``erf_t0_accept`` re-filters cached fits without refitting.
    3) The dialog: option buttons in the left column, erf row directly
       below the ridge row, helper switch updates app state, Auto-place in
       erf mode places points on the true chirp, cache reuse on R² change.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys

import numpy as np
from PyQt5 import QtWidgets

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import ta_core  # noqa: E402


def _synthetic(n_wl=300, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    wl = np.linspace(400.0, 700.0, n_wl)
    t = np.concatenate([np.arange(-1.0, 2.0, 0.02),
                        np.geomspace(2.0, 1000.0, 40)])
    t0 = 0.6 * ((550.0 / wl) ** 2 - 1.0)
    amp = (np.exp(-((wl - 520) / 80) ** 2)
           - 0.6 * np.exp(-((wl - 640) / 40) ** 2))
    D = np.empty((wl.size, t.size))
    for i in range(wl.size):
        D[i] = amp[i] * (0.7 * ta_core.exp_irf_conv(t, 0.5, t0[i], 0.12)
                         + 0.3 * ta_core.exp_irf_conv(t, 200, t0[i], 0.12))
    D += rng.normal(0, noise, D.shape)
    return wl, t, D, t0


def test_core_accuracy():
    wl, t, D, t0 = _synthetic()
    res = ta_core.compute_erf_t0(wl, t, D, -0.8, 1.5, smooth_n=5,
                                 stride=0, r2_min=0.9)
    assert res is not None
    assert res['ok'].sum() > 0.8 * res['ok'].size
    err = res['t0'] - np.interp(res['wl'], wl, t0)
    assert np.abs(err).max() < 0.02, np.abs(err).max()
    fw = np.nanmedian(res['fwhm_all'][res['ok']])
    assert 0.08 < fw < 0.16, fw
    _, tr = ta_core.compute_ridge(wl, t, D, 'max|dA/dt|', -0.8, 1.5, 5)
    assert np.abs(tr - t0).max() > np.abs(err).max()
    print(f'[1] erf max err {np.abs(err).max() * 1e3:.1f} fs, '
          f'ridge max err {np.abs(tr - t0).max() * 1e3:.0f} fs, '
          f'FWHM {fw:.3f}')


def test_rejection_and_refilter():
    wl, t, D, _ = _synthetic(n_wl=200)
    fits = ta_core.fit_erf_rows(wl, t, D, -0.8, 1.5, stride=1)
    a = ta_core.erf_t0_accept(fits, 0.9, 5)
    # amp changes sign near 600 nm -> no signal there -> rejected
    near0 = np.abs(fits['wl_all'] - 602) < 4
    assert not a['ok'][near0].all()
    b = ta_core.erf_t0_accept(fits, 0.0, 1)
    assert b['ok'].sum() >= a['ok'].sum()
    assert ta_core.compute_erf_t0(wl, t[:3], D[:, :3], -1, 2) is None
    # Median smoothing stays inside runs of accepted rows: an outlier run
    # across a rejected gap must not be pulled toward the other side.
    fake = {'wl_all': np.arange(10.0), 't_lo': -1.0, 't_hi': 1.0,
            't0_all': np.array([0, 0, 0, 0, np.nan, .5, .5, .5, .5, .5]),
            'r2_all': np.ones(10), 'fwhm_all': np.ones(10)}
    c = ta_core.erf_t0_accept(fake, 0.9, 5)
    np.testing.assert_allclose(c['t0'], [0, 0, 0, 0, .5, .5, .5, .5, .5])
    assert ta_core.erf_stride(1000, 0) == 5
    assert ta_core.erf_stride(1000, 3) == 3
    print(f'[2] accepted {a["ok"].sum()}/{a["ok"].size} at R²≥0.9, '
          f'{b["ok"].sum()} at R²≥0')


def test_dialog():
    # Keep a reference: an unreferenced QApplication is collected at once.
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    assert qapp is not None
    import ta_main
    import ta_chirp
    from ta_chirp import ChirpDialog
    # A real modal box crashes offscreen Qt; record warnings instead.
    warned = []
    ta_chirp.warn_box = ta_main.warn_box = \
        lambda *a, **kw: warned.append(a[1:])
    wl, t, D, t0 = _synthetic(n_wl=150)
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, D, 'chirp_erf_test.csv')
    win.ridge_tmin, win.ridge_tmax = -0.8, 1.5
    dlg = ChirpDialog(win, win)
    lay = dlg.rb_ridge.parentWidget().layout()

    def pos(w):
        r, c, _, _ = lay.getItemPosition(lay.indexOf(w))
        return r, c
    assert pos(dlg.rb_ridge) == (0, 0)
    assert pos(dlg.rb_erf) == (1, 0)
    assert pos(dlg.dd_method)[0] == 0 and pos(dlg.ed_r2min)[0] == 1
    assert pos(dlg.btn_auto)[0] == 2
    assert dlg.rb_ridge.isChecked() and not dlg.ed_r2min.isEnabled()

    dlg.rb_erf.setChecked(True)
    assert win.t0_helper == 'erf'
    assert dlg.ed_r2min.isEnabled() and not dlg.dd_method.isEnabled()
    dlg.auto_place(8)
    pts = win.chirp_pts
    assert pts.shape[0] >= 6, pts.shape
    err = pts[:, 1] - np.interp(pts[:, 0], wl, t0)
    assert np.abs(err).max() < 0.03, err
    assert 'accepted' in dlg.lbl_erf.text()
    assert dlg._ridge_line is not None

    assert dlg.cb_show_ridge.isChecked()   # Auto-place turned overlay on
    cache = dlg._erf_cache
    dlg.ed_r2min.setValue(0.5)          # re-filter only, no refit
    assert dlg._erf_cache is cache
    dlg.ed_stride.setValue(3)           # different rows -> debounced refit
    assert dlg._refit_timer.isActive() and dlg._erf_cache is cache
    dlg._refit_timer.timeout.emit()
    assert dlg._erf_cache is not cache and win.ridge_t is not None

    # t0 shift (same shape) while open: content key forces a refit in the
    # new time frame instead of reusing the old curve.
    before = np.interp(560.0, win.ridge_wl, win.ridge_t)
    win.delay = win.delay - 0.2
    cache = dlg._erf_cache
    dlg.recompute_ridge()
    after = np.interp(560.0, win.ridge_wl, win.ridge_t)
    assert dlg._erf_cache is not cache
    assert abs((before - after) - 0.2) < 0.03, (before, after)
    win.delay = win.delay + 0.2

    # Auto-place guard: an impossible R² keeps the user's points.
    keep = win.chirp_pts.copy()
    dlg.ed_r2min.setValue(0.999)
    warned.clear()
    dlg.auto_place(8)
    assert warned and np.array_equal(win.chirp_pts, keep), warned
    dlg.ed_r2min.setValue(0.9)

    # Crop while the (modeless) dialog is open must not crash.
    n_axes = len(dlg.canvas_map.fig.axes)
    assert win.apply_crop_by_range(450.0, 650.0, -1.0, 1000.0)
    dlg.recompute_ridge()
    assert win.apply_crop_by_range(460.0, 640.0, -1.0, 1000.0)
    dlg.recompute_ridge()
    assert len(dlg.canvas_map.fig.axes) == n_axes   # colorbar not stacked
    assert dlg._base_data.shape == (win.wavelength.size, win.delay.size)
    assert win.ridge_wl.min() >= 450.0 - 1e-9

    dlg.rb_ridge.setChecked(True)
    assert win.t0_helper == 'ridge' and dlg.lbl_erf.text() == ''
    dlg.auto_place(8)
    assert win.chirp_pts.shape[0] == 8      # ridge: every target placed
    dlg.close()
    # Choice persists for the next dialog
    win.t0_helper = 'erf'
    dlg2 = ChirpDialog(win, win)
    assert dlg2.rb_erf.isChecked()
    dlg2.close()
    print(f'[3] dialog: Auto-place {pts.shape[0]} pts, '
          f'max err {np.abs(err).max() * 1e3:.1f} fs')


if __name__ == '__main__':
    test_core_accuracy()
    test_rejection_and_refilter()
    test_dialog()
    print('\nAll chirp erf tests passed.')
