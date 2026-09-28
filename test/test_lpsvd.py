"""
Tests for the LPSVD module (ta_lpsvd).

Covers:
    1) preprocess_signal: window crop, baseline removal, tiling, dt.
    2) apply_window: Kaiser & Gaussian shapes and edge cases.
    3) lpsvd_modes: recovery of known Lorentzian-damped oscillators.
    4) gaussian_fit_modes: recovery of known Gaussian-damped oscillators.
    5) run_lpsvd_analysis end-to-end (LPSVD + Gaussian models, both windows).
    6) End-to-end Coherence dialog 'Run LPSVD' button on synthetic dataset.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import numpy as np
from PyQt5 import QtWidgets

import ta_lpsvd as lp


C_CM = 2.99792458e10


def cm_to_hz(wn_cm):
    return wn_cm * C_CM


# =====================================================================
# 1) Preprocessing
# =====================================================================
def test_preprocess_basic():
    t = np.linspace(0.0, 5.0, 501)        # ps
    y = np.cos(2 * np.pi * 3.0 * t) + 0.05 * t + 1.0
    out = lp.preprocess_signal(t, y, t_start=0.5, t_end=4.5,
                               do_polyfit=True, polyfit_deg=4,
                               do_detrend=True, do_center=True,
                               append_count=0)
    assert out['t_late'][0] >= 0.5 - 1e-9
    assert out['t_late'][-1] <= 4.5 + 1e-9
    # zero-mean after centering
    assert abs(np.mean(out['y_centered'])) < 1e-9
    # raw = centered + bg
    assert np.allclose(out['y_raw'], out['y_centered'] + out['bg'])
    # uniform dt
    assert abs(out['dt'] - 0.01) < 1e-6
    print('[1.preprocess] OK')


def test_preprocess_tiling_extends_length():
    t = np.linspace(0.0, 2.0, 201)
    y = np.cos(2 * np.pi * 1.0 * t)
    out0 = lp.preprocess_signal(t, y, t_start=0.0, t_end=2.0, append_count=0)
    out2 = lp.preprocess_signal(t, y, t_start=0.0, t_end=2.0, append_count=2)
    assert out2['y_centered'].size == 3 * out0['y_centered'].size
    # tiled timeline starts at t_start and uses uniform dt
    assert abs(out2['t_late'][0] - 0.0) < 1e-9
    assert abs(np.diff(out2['t_late']).mean() - out0['dt']) < 1e-9
    print('[1.tile]      OK')


def test_preprocess_rejects_short_window():
    t = np.linspace(0.0, 1.0, 100)
    y = np.zeros_like(t)
    try:
        lp.preprocess_signal(t, y, t_start=0.0, t_end=0.02)
    except ValueError:
        print('[1.short]     OK')
        return
    raise AssertionError('Expected ValueError on too-short window.')


# =====================================================================
# 2) Windowing
# =====================================================================
def test_window_shapes():
    t = np.linspace(0.0, 4.0, 401)
    w_k = lp.apply_window(t, window_type='kaiser', beta=8.0)
    w_g = lp.apply_window(t, window_type='gaussian', sigma=1.5)
    w_n = lp.apply_window(t, window_type='none')
    # Kaiser is symmetric about t/2 only when t starts at 0 and is full
    # period - but it should peak at t=0 and decay smoothly.
    assert np.isclose(w_k[0], 1.0)
    assert w_k[-1] < w_k[0]
    assert np.all(w_k >= 0)
    # Gaussian peaks at t=0 too
    assert np.isclose(w_g[0], 1.0)
    assert w_g[-1] < w_g[0]
    # None is all-ones
    assert np.allclose(w_n, 1.0)
    # Zero-sigma fallback
    w_g0 = lp.apply_window(t, window_type='gaussian', sigma=0.0)
    assert np.allclose(w_g0, 1.0)
    print('[2.window]    OK')


# =====================================================================
# 3) LPSVD recovers known Lorentzian modes
# =====================================================================
def _build_lorentzian_signal(t_ps, freqs_THz, dampings_per_ps, amps, phases):
    """Synthesise sum_k A_k exp(-Γ_k t) cos(2π f_k t + φ_k)."""
    y = np.zeros_like(t_ps)
    for A, f, g, p in zip(amps, freqs_THz, dampings_per_ps, phases):
        y += A * np.exp(-g * t_ps) * np.cos(2 * np.pi * f * t_ps + p)
    return y


def test_lpsvd_recovers_lorentzian():
    # Lightly-damped record - representative of vibrational coherences in TA.
    t_ps = np.linspace(0.0, 15.0, 1501)
    freqs_THz = [1.2, 3.5]
    damps_per_ps = [0.03, 0.05]
    amps = [1.0, 0.6]
    phases = [0.0, 0.7]
    y = _build_lorentzian_signal(t_ps, freqs_THz, damps_per_ps, amps, phases)
    dt_s = (t_ps[1] - t_ps[0]) * 1e-12

    modes = lp.lpsvd_modes(y, dt_s, num_modes=2)
    assert len(modes) == 2
    rec = sorted([(m['freq_Hz'] * 1e-12, m['amp'], m['damp_Hz'] * 1e-12)
                  for m in modes], key=lambda x: x[0])
    # Frequencies should be recovered to better than 0.5 %.
    # Dampings within ~10 % (intrinsically harder to recover).
    for (f_hat, _a, g_hat), f_true, g_true in zip(rec, freqs_THz, damps_per_ps):
        assert abs(f_hat - f_true) / f_true < 5e-3, f'freq {f_hat} vs {f_true}'
        assert abs(g_hat - g_true) / g_true < 0.15, f'damp {g_hat} vs {g_true}'

    # Variance reduction: > 99 % of the signal variance is captured.
    t_s = t_ps * 1e-12
    _recons, total = lp.reconstruct_modes(t_s, modes, 'lorentzian')
    var_in = float(np.var(y))
    var_res = float(np.var(y - total))
    assert var_res < 0.01 * var_in, (
        f'reconstruction left {var_res:.4g} of {var_in:.4g} variance')
    print('[3.lpsvd]     OK')


# =====================================================================
# 4) Gaussian curve_fit recovers known modes
# =====================================================================
def _build_gaussian_signal(t_ps, freqs_THz, gammas_per_ps2, amps, phases):
    y = np.zeros_like(t_ps)
    for A, f, g, p in zip(amps, freqs_THz, gammas_per_ps2, phases):
        y += A * np.exp(-g * t_ps ** 2) * np.cos(2 * np.pi * f * t_ps + p)
    return y


def test_gaussian_fit_recovers_modes():
    t_ps = np.linspace(0.0, 4.0, 401)
    freqs_THz = [1.0, 3.0]
    gammas = [0.05, 0.10]
    amps = [1.0, 0.6]
    phases = [0.2, -0.4]
    y = _build_gaussian_signal(t_ps, freqs_THz, gammas, amps, phases)

    t_s = t_ps * 1e-12
    dt_s = (t_ps[1] - t_ps[0]) * 1e-12
    modes = lp.gaussian_fit_modes(y, t_s, dt_s, num_modes=2)
    assert len(modes) == 2
    rec = [(m['freq_Hz'] * 1e-12, m['amp'],
            m['damp_per_s2'] * 1e-24) for m in modes]
    rec.sort(key=lambda x: x[0])
    for (f_hat, A_hat, g_hat), f_true, A_true, g_true in zip(
            rec, freqs_THz, amps, gammas):
        assert abs(f_hat - f_true) / f_true < 5e-2, f'freq {f_hat} vs {f_true}'
        assert abs(A_hat - A_true) / A_true < 1e-1, f'amp {A_hat} vs {A_true}'
        assert abs(g_hat - g_true) / g_true < 3e-1, f'gamma {g_hat} vs {g_true}'
    print('[4.gauss]     OK')


# =====================================================================
# 5) Driver: run_lpsvd_analysis (LPSVD + Gaussian; Kaiser + Gaussian window)
# =====================================================================
def test_run_lpsvd_analysis_lorentzian():
    t_ps = np.linspace(0.0, 8.0, 801)
    y = _build_lorentzian_signal(
        t_ps, [1.5, 4.2], [0.20, 0.30], [1.0, 0.5], [0.1, -0.2])
    y = y + 0.01 * np.random.default_rng(0).standard_normal(y.size)
    res = lp.run_lpsvd_analysis(
        t_ps, y, time_unit='ps', t_start=0.3, t_end=7.0,
        do_polyfit=True, polyfit_deg=4, do_detrend=True, do_center=True,
        append_count=0, model='lorentzian', num_modes=2,
        window_type='kaiser', beta=8.0)
    assert len(res['modes']) == 2
    assert res['fft_orig'].shape == res['fft_fit'].shape
    assert res['residual'].size == res['y_raw'].size
    assert res['std_residual'] >= 0
    # The fit should explain most of the variance
    var_in = np.var(res['y_centered'])
    var_res = np.var(res['y_centered'] - res['total_centered'])
    assert var_res < 0.2 * var_in, 'LPSVD fit didn’t reduce variance enough.'
    print('[5.runL]      OK')


def test_run_lpsvd_analysis_gaussian():
    t_ps = np.linspace(0.0, 4.0, 401)
    y = _build_gaussian_signal(
        t_ps, [1.0, 2.5], [0.05, 0.08], [1.0, 0.6], [0.2, -0.4])
    res = lp.run_lpsvd_analysis(
        t_ps, y, time_unit='ps', t_start=0.0, t_end=4.0,
        do_polyfit=False, do_detrend=True, do_center=True,
        model='gaussian', num_modes=2,
        window_type='gaussian', gaussian_sigma=2.0)
    assert len(res['modes']) == 2
    assert res['fft_modes'][0].shape == res['fft_orig'].shape
    print('[5.runG]      OK')


# =====================================================================
# 6) Coherence dialog end-to-end (synthetic dataset, run LPSVD button)
# =====================================================================
def _build_synthetic_dataset():
    """Make a (Nwl, Nt) ΔA dataset whose residual has known oscillations."""
    rng = np.random.default_rng(7)
    wl = np.linspace(450.0, 700.0, 50)
    t = np.linspace(-0.5, 5.0, 351)
    # No kinetic component - just an oscillating residual with a Lorentzian
    # spectral envelope peaked around 550 nm.
    env = np.exp(-((wl - 550.0) / 30.0) ** 2)
    osc = (1.0 * np.exp(-0.2 * t) * np.cos(2 * np.pi * 1.5 * t)
           + 0.4 * np.exp(-0.3 * t) * np.cos(2 * np.pi * 4.0 * t + 0.5))
    A = np.outer(env, osc)
    A = A + 0.01 * rng.standard_normal(A.shape)
    return wl, t, A


def test_coherence_dialog_runs_lpsvd():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    import ta_main
    from ta_coherence import CoherenceDialog

    wl, t, A = _build_synthetic_dataset()
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'lpsvd_test')

    dlg = CoherenceDialog(win, win)
    # Pick raw-data path so no GA/LDA prerequisites are needed
    dlg.rb_use_data.setChecked(True)
    dlg.ed_t_min.setValue(0.3)
    dlg.ed_t_max.setValue(5.0)
    # Configure LPSVD options
    dlg.rb_lps_lor.setChecked(True)
    dlg.ed_lps_modes.setValue(2)
    dlg.chk_polyfit.setChecked(True)
    dlg.ed_poly_deg.setValue(4)
    dlg.chk_detrend.setChecked(True)
    dlg.chk_center.setChecked(True)
    dlg.ed_append.setValue(0)
    dlg.chk_window.setChecked(True)
    dlg.dd_win_type.setCurrentIndex(dlg.dd_win_type.findData('kaiser'))
    dlg.ed_win_beta.setValue(8.0)
    dlg.cmb_wl_mode.setCurrentIndex(dlg.cmb_wl_mode.findData('mean'))
    dlg.do_run_lpsvd()
    assert dlg._last_lp is not None, 'LPSVD did not produce a result.'
    modes = dlg._last_lp['modes']
    assert len(modes) == 2
    # Both injected modes are below 10 THz so they're inside the default band.
    rec_THz = sorted(m['freq_Hz'] * 1e-12 for m in modes)
    # Expected (≈) 1.5 THz and 4.0 THz
    assert abs(rec_THz[0] - 1.5) < 0.2, f'low mode off: {rec_THz[0]}'
    assert abs(rec_THz[1] - 4.0) < 0.3, f'high mode off: {rec_THz[1]}'

    # Switch to Gaussian model and re-run; should still produce 2 modes
    dlg.rb_lps_gauss.setChecked(True)
    dlg.dd_win_type.setCurrentIndex(dlg.dd_win_type.findData('gaussian'))
    dlg.ed_win_sigma.setValue(2.0)
    dlg.do_run_lpsvd()
    assert dlg._last_lp is not None
    assert len(dlg._last_lp['modes']) == 2

    dlg.close()
    print('[6.dialog]    OK')


def test_dialog_uses_loaded_data_not_file():
    """The LPSVD code path must operate on data passed through
    ``set_loaded_data`` (in memory), not by re-reading any external file.
    Smuggle a recognisable signature into ΔA, load it, run LPSVD on raw
    data, and confirm the recovered frequency matches.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    import ta_main, ta_lpsvd
    from ta_coherence import CoherenceDialog

    rng = np.random.default_rng(123)
    wl = np.linspace(500.0, 650.0, 40)
    t = np.linspace(-0.2, 6.0, 401)
    # Unique fingerprint frequency that's unlikely to appear by chance:
    f_unique_THz = 2.347
    env = np.exp(-((wl - 575.0) / 25.0) ** 2)
    osc = np.exp(-0.05 * t) * np.cos(2 * np.pi * f_unique_THz * t)
    A = np.outer(env, osc) + 0.005 * rng.standard_normal((40, 401))

    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'Loaded: synthetic_fingerprint.csv')

    dlg = CoherenceDialog(win, win)
    dlg.rb_use_data.setChecked(True)
    dlg.ed_t_min.setValue(0.5)
    dlg.ed_t_max.setValue(6.0)
    dlg.rb_lps_lor.setChecked(True)
    dlg.ed_lps_modes.setValue(1)
    dlg.cmb_wl_mode.setCurrentIndex(dlg.cmb_wl_mode.findData('single'))
    dlg.ed_wl_one.setValue(575.0)
    dlg.chk_window.setChecked(False)        # don't window before FFT
    dlg.do_run_lpsvd()

    assert dlg._last_lp is not None, 'LPSVD did not run.'
    modes = dlg._last_lp['modes']
    assert len(modes) == 1, f'Expected 1 mode, got {len(modes)}'
    f_THz = modes[0]['freq_Hz'] * 1e-12
    assert abs(f_THz - f_unique_THz) < 0.05, (
        f'Recovered freq {f_THz:.3f} != injected {f_unique_THz} - '
        'LPSVD is not seeing the loaded data!')

    # The info panel should advertise the loaded-data source.
    info = dlg.txt_info.toPlainText()
    assert 'synthetic_fingerprint.csv' in info, (
        'Info panel should name the loaded dataset.')
    assert 'raw' in info.lower(), 'Residual source not reported.'
    dlg.close()
    print('[7.loaded]    OK')


def main():
    test_preprocess_basic()
    test_preprocess_tiling_extends_length()
    test_preprocess_rejects_short_window()
    test_window_shapes()
    test_lpsvd_recovers_lorentzian()
    test_gaussian_fit_recovers_modes()
    test_run_lpsvd_analysis_lorentzian()
    test_run_lpsvd_analysis_gaussian()
    test_coherence_dialog_runs_lpsvd()
    test_dialog_uses_loaded_data_not_file()
    print('\nAll LPSVD tests passed.')


if __name__ == '__main__':
    main()
