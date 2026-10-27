"""
Regression tests for fit statistics and the Kinetic Fit Import / global fit.

Verifies:

  1. ta_core.varpro_stats / population_ratio / format_fit_stats
        - noise-free R² ≈ 1, reduced χ² ≈ σ² for known noise
        - predicted 1σ of τ and A agree with a Monte-Carlo spread
        - |Aᵢ| / Σ|Aⱼ| (offset included) sums to 100 %

  2. Global Analysis dialog shows τ ± σ and the statistics block

  3. Kinetic Fit dialog
        - opens empty; Run without Import is refused
        - "Import pinned kinetics" defaults to on when pins exist
        - pinned import (BW band kept) → global fit with shared τ,
          per-trace A ± σ, ratios, R² per trace
        - unchecked / no pins → single trace from the λ window
        - exports (params, trace, per-trace residual files)
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import tempfile
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

import numpy as np  # noqa: E402
from PyQt5 import QtWidgets, QtCore  # noqa: E402

import ta_core  # noqa: E402
import ta_main  # noqa: E402
import ta_kfit  # noqa: E402
import ta_ga  # noqa: E402

TAUS = (2.0, 30.0)
T = np.r_[np.linspace(-1, 2, 40), np.logspace(0.4, 2.5, 80)]


def _model(A, taus=TAUS, t=T):
    return sum(a * ta_core.exp_irf_conv(t, tau, 0.0, 0.2)
               for a, tau in zip(A, taus))


def _dataset(seed=0, noise=0.01):
    """λ-resolved map whose kinetics are A₁(λ)·e^{-t/2} + A₂(λ)·e^{-t/30}."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(450.0, 650.0, 101)                 # 2 nm pixels
    a1 = np.exp(-((wl - 500) / 80) ** 2)
    a2 = -0.6 * np.exp(-((wl - 600) / 80) ** 2)
    D = np.outer(a1, _model([1, 0])) + np.outer(a2, _model([0, 1]))
    return wl, T.copy(), D + noise * rng.normal(size=D.shape)


def test_core_stats():
    warnings.simplefilter('ignore')
    rng = np.random.default_rng(1)
    # Noise-free: R² = 1 within numerics
    r = ta_core.fit_single_trace(T, _model([1.0, 0.5]), tau_init=[1.5, 20],
                                 tau_fixed=[0, 0], fwhm_init=0.2)
    st = r['info']['stats']
    assert st['r2'] > 1 - 1e-8, st['r2']
    # Known noise σ = 0.02 → χ²_red ≈ 4e-4; σ(τ) vs Monte Carlo
    taus, As, pred = [], [], None
    for _ in range(60):
        D = np.vstack([_model([1.0, 0.5]), _model([-0.4, 0.8])])
        D += 0.02 * rng.normal(size=D.shape)
        g = ta_core.fit_global_analysis(D, T, [1.5, 20], 0.0, 0.2, [0, 0],
                                        True, True, False)
        taus.append(g['tau'])
        As.append(g['A'])
        pred = pred or g['info']['stats']
    assert 2.5e-4 < pred['chi2_red'] < 6e-4, pred['chi2_red']
    mc_tau, mc_A = np.std(taus, 0), np.std(As, 0)
    assert np.all(np.abs(pred['tau_std'] / mc_tau - 1) < 0.4), \
        (pred['tau_std'], mc_tau)
    assert np.all(np.abs(pred['A_std'] / mc_A - 1) < 0.4), \
        (pred['A_std'], mc_A)
    assert np.isnan(pred['t0_std']) and np.isnan(pred['fwhm_std'])  # fixed
    ratio = ta_core.population_ratio([[1.0, -3.0], [0.0, 0.0]])
    assert np.allclose(ratio[0], [25.0, 75.0]) and np.all(np.isnan(ratio[1]))
    text = '\n'.join(ta_core.format_fit_stats(pred))
    for key in ('R²', 'Adjusted R²', 'Reduced χ²', 'AIC', 'dof'):
        assert key in text, key
    assert ta_core.format_fit_stats(None) == ['Fit statistics: not available']
    print(f'[ok] core stats (σ_τ pred {pred["tau_std"]} vs MC {mc_tau})')


def test_ga_dialog_report():
    wl, t, D = _dataset()
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, D, 'Loaded: stats')
    ga = ta_ga.GlobalAnalysisDialog(win, win)
    ga.dd_N.setCurrentText('2')
    ga.tau_table.cellChanged.disconnect()
    for r, v in enumerate([1.5, 20.0]):
        ga.tau_table.item(r, 0).setText(str(v))
    ga.tau_table.cellChanged.connect(ga._on_tau_edit)
    ga.read_table()
    ga.do_run()
    st = win.ga_result_stats
    assert st is not None and st['r2'] > 0.99, st and st['r2']
    assert np.all(np.isfinite(st['tau_std'])) and np.all(st['tau_std'] > 0)
    txt = ga.txt_result.toPlainText()
    assert 'tau_1 =' in txt and '±' in txt and 'R²' in txt, txt
    assert 'Reduced χ²' in txt and 'AIC / BIC' in txt
    QtWidgets.QApplication.processEvents()   # flush queued redraws
    ga.close()
    win.close()
    print('[ok] GA report: τ ± σ + statistics')


def _kfit(win):
    kd = ta_kfit.KineticFitDialog(win, win)
    kd.tbl.item(0, 0).setText('1.5')
    kd.tbl.item(1, 0).setText('20')
    kd.ed_fw.setValue(0.2)
    return kd


def test_kfit_import_and_global():
    wl, t, D = _dataset()
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, D, 'Loaded: kfit')

    # No pins → checkbox off, opens empty, Run refused
    kd = _kfit(win)
    assert not kd.cb_pinned.isChecked() and kd._traces == []
    warned = []
    orig_warn = ta_kfit.warn_box
    ta_kfit.warn_box = lambda *a, **k: warned.append(a[1])
    try:
        kd.do_run()
    finally:
        ta_kfit.warn_box = orig_warn
    assert warned == ['No data'] and kd._last_fit is None
    # Checked without pins → falls back to the λ window, single fit
    kd.cb_pinned.setChecked(True)
    kd.ed_wl.setValue(550.0)
    kd.do_import()
    assert len(kd._traces) == 1 and 'no pinned' in kd.lbl_import.text()
    kd.do_run()
    r = kd._last_fit
    assert not r['_global'] and r['A'].shape == (1, 2)
    assert np.allclose(sorted(r['tau']), TAUS, rtol=0.1), r['tau']
    assert np.isclose(r['ratio'].sum(), 100.0)
    assert np.all(np.isfinite(r['stats']['A_std']))
    kd.close()

    # Pins (one with a BW band) → checkbox on, global fit
    win.set_sel_wl(500.0)
    win.pin_kin()
    win.ed_kin_int_bw.setValue(10.0)
    win.chk_kin_int.setChecked(True)
    win.set_sel_wl(600.0)
    win.pin_kin()
    kd = _kfit(win)
    kd.cb_inf.setChecked(True)
    assert kd.cb_pinned.isChecked()
    kd.ed_wl.setValue(550.0)             # ignored while importing pins
    kd.do_import()
    assert len(kd._traces) == 2 and 'global fit' in kd.lbl_import.text()
    i600 = int(np.argmin(np.abs(win.wavelength - 600.0)))
    band = ta_core.wavelength_gate(win.wavelength, win.wavelength[i600], 10.0)
    assert np.allclose(kd._traces[1]['y'], D[band].mean(axis=0))
    assert kd._traces[1]['n_avg'] == int(band.sum())
    # Changing λ after Import does not touch the imported data
    y0 = kd._traces[0]['y'].copy()
    kd.ed_wl.setValue(620.0)
    assert np.array_equal(kd._traces[0]['y'], y0)
    kd.do_run()
    r = kd._last_fit
    assert r['_global'] and r['A'].shape == (2, 3)          # A1, A2, A∞
    assert np.allclose(sorted(r['tau']), TAUS, rtol=0.1), r['tau']
    assert np.allclose(r['ratio'].sum(axis=1), 100.0)
    st = r['stats']
    assert st['r2_rows'].shape == (2,) and np.all(st['r2_rows'] > 0.9)
    txt = kd.txt_result.toPlainText()
    for key in ('Global fit: 2 traces', '±', '%)', 'A∞', 'R²', 'BW 10 nm'):
        assert key in txt, (key, txt)

    # Exports
    with tempfile.TemporaryDirectory() as tmp:
        paths = iter([os.path.join(tmp, 'params.csv'),
                      os.path.join(tmp, 'trace.csv'),
                      os.path.join(tmp, 'resid.csv')])
        orig_ask, orig_info = ta_kfit.ask_save_path, ta_kfit.info_box
        ta_kfit.ask_save_path = lambda *a, **k: (next(paths), '')
        ta_kfit.info_box = lambda *a, **k: None
        try:
            kd._export_params()
            kd._export_trace()
            kd.save_residual()
        finally:
            ta_kfit.ask_save_path, ta_kfit.info_box = orig_ask, orig_info
        with open(os.path.join(tmp, 'params.csv'), encoding='utf-8') as f:
            params = dict(line.strip().split(',', 1) for line in f)
        for key in ('fit_mode', 'r2', 'tau_1_std', 'T2_A_inf_std',
                    'T1_ratio_1_pct', 'T2_r2'):
            assert key in params, key
        assert params['fit_mode'] == 'global'
        hdr = open(os.path.join(tmp, 'trace.csv'),
                   encoding='utf-8').readline()
        assert 'fit_500.0nm' in hdr and 'residual_600.0nm' in hdr, hdr
        files = sorted(os.listdir(tmp))
        assert 'kfit_500.0nm_resid.csv' in files, files
        assert 'kfit_600.0nm_resid.csv' in files, files
        import ta_residual_store
        meta = ta_residual_store.load_residual(
            os.path.join(tmp, 'kfit_600.0nm_resid.csv'))['meta']
        assert abs(float(meta['wavelength_nm']) - 600.0) < 0.1, meta
        assert 'resid.csv' not in files

    # Unchecked → back to the λ window
    kd.cb_pinned.setChecked(False)
    kd.do_import()
    assert len(kd._traces) == 1 and kd._last_fit is None
    assert abs(kd._traces[0]['wl'] - 620.0) < 1.5
    kd.close()
    win.close()
    print('[ok] kinetic fit import + global fit')


def test_edge_cases():
    warnings.simplefilter('ignore')
    rng = np.random.default_rng(5)
    # A trace with ≤ k usable points carries no parameters (A_std NaN)
    D = np.vstack([_model([1.0, 0.5]), _model([-0.4, 0.8])])
    D += 0.01 * rng.normal(size=D.shape)
    D[1, 2:] = np.nan
    g = ta_core.fit_global_analysis(D, T, [1.5, 20], 0.0, 0.2, [0, 0],
                                    True, True, False)
    st = g['info']['stats']
    assert np.all(np.isnan(st['A_std'][1])), st['A_std']
    assert st['n_par'] == 2 + 2 and st['n_obs'] == T.size, st

    # Global stretched + skip: unfitted IRF region is NaN in the residual
    wl, t, Dm = _dataset()
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, Dm, 'Loaded: skip')
    for lam in (500.0, 600.0):
        win.set_sel_wl(lam)
        win.pin_kin()
    kd = _kfit(win)
    kd.tbl.item(1, 2).setCheckState(QtCore.Qt.Checked)      # τ₂ stretched
    kd.dd_irf.setCurrentIndex(kd.dd_irf.findData('skip'))
    kd.do_import()
    kd.do_run()
    r = kd._last_fit
    sig = r['fwhm'] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    skip = r['_t_fit'] <= r['t0'] + 3.0 * sig
    assert skip.any() and np.all(np.isnan(r['residual'][:, skip]))
    assert np.all(np.isfinite(r['residual'][:, ~skip]))
    kd.close()
    win.close()
    print('[ok] edge cases (sparse trace, skip-mode residual)')


TESTS = [test_core_stats, test_ga_dialog_report, test_kfit_import_and_global,
         test_edge_cases]


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    for fn in TESTS:
        fn()
    print(f'\nAll {len(TESTS)} fit-stats / kfit-import checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
