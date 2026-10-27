"""
Regression tests for the time panels next to DADS / EADS in Global Analysis.

Verifies:

  1. ta_core.build_ga_basis (exp / stretched / τ = ∞ columns) and
     sequential_populations: DADS·Cᵀ = EADS·Pᵀ exactly, populations start
     in S₁ and sum to the total decay
  2. GA dialog: the DADS row shows C_j(t), the EADS row c_S(t);
     DADS·C(t) at the fitted delays reproduces the global fit; component
     check-boxes hide curves; the Scale selector switches log / linear
  3. Export DADS / EADS writes spectra and time files side by side
     (DADS_spectra + DADS_kinetics, EADS_spectra + EADS_populations)
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

import numpy as np  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402

import ta_core  # noqa: E402
import ta_main  # noqa: E402
import ta_ga  # noqa: E402


def test_core():
    t = np.linspace(-1.0, 200.0, 500)
    tau = [30.0, 2.0]
    for has_inf in (False, True):
        C = ta_core.build_ga_basis(t, tau, 0.0, 0.2, has_inf)
        D = np.random.default_rng(0).normal(size=(6, C.shape[1]))
        E, tau_sorted, _ = ta_core.compute_eads_from_dads(D, tau, has_inf)
        P = ta_core.sequential_populations(C, tau, has_inf)
        assert np.allclose(D @ C.T, E @ P.T, atol=1e-12)
        late = t > 1.0
        assert P[late][0, 0] > 0.5            # S₁ (fastest) populated first
        if has_inf:                           # nothing is lost: Σ = step
            assert np.allclose(P[late].sum(axis=1), C[late, -1], atol=1e-9)
    # Stretched columns match stretched_irf_conv; others stay exp ⊗ IRF
    C = ta_core.build_ga_basis(t, [5.0, 40.0], 0.0, 0.2, False,
                               beta=[1.0, 0.6], stretch_on=[False, True],
                               irf_mode='numerical')
    assert np.allclose(C[:, 0], ta_core.exp_irf_conv(t, 5.0, 0.0, 0.2))
    assert np.allclose(C[:, 1], ta_core.stretched_irf_conv(
        t, 40.0, 0.6, 0.0, 0.2, 'numerical'))
    print('[ok] core basis + sequential populations')


def _dataset():
    rng = np.random.default_rng(2)
    wl = np.linspace(450.0, 650.0, 60)
    t = np.r_[np.linspace(-1, 2, 30), np.logspace(0.4, 2.7, 70)]
    s1 = np.exp(-((wl - 500) / 40) ** 2)
    s2 = -0.6 * np.exp(-((wl - 600) / 40) ** 2)
    D = (np.outer(s1, ta_core.exp_irf_conv(t, 2.0, 0.0, 0.15))
         + np.outer(s2, ta_core.exp_irf_conv(t, 40.0, 0.0, 0.15)))
    return wl, t, D + 0.003 * rng.normal(size=D.shape)


def test_dialog():
    wl, t, D = _dataset()
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, D, 'Loaded: time panels')
    ga = ta_ga.GlobalAnalysisDialog(win, win)
    ga.dd_N.setCurrentText('2')
    ga.tau_table.cellChanged.disconnect()
    for r, v in enumerate([1.0, 30.0]):
        ga.tau_table.item(r, 0).setText(str(v))
    ga.tau_table.cellChanged.connect(ga._on_tau_edit)
    ga.cb_inf.setChecked(True)
    ga.read_table()
    ga.do_run()
    QtWidgets.QApplication.processEvents()

    assert len(ga.ax_dads_t.get_lines()) == 3 + 1          # 3 comps + y=0
    assert len(ga.ax_eads_t.get_lines()) == 3 + 1
    assert ga.ax_dads_t.get_xscale() == 'log'
    t_pos = win.ga_result_delay[win.ga_result_delay > 0]
    assert np.isclose(ga.ax_dads_t.get_xlim()[0], t_pos.min())
    # DADS · C(t) reproduces the fitted matrix at the fitted delays
    t_dense, C = ga._time_profiles()
    t_fit = win.ga_result_delay
    assert np.all(np.isin(t_fit, t_dense))                 # fit delays kept
    C_fit = C[np.searchsorted(t_dense, t_fit)]
    rise = ((t_dense > win.ga_result_t0 - win.ga_result_fwhm)
            & (t_dense < win.ga_result_t0 + win.ga_result_fwhm))
    assert rise.sum() >= 30                                # IRF resolved
    exact = ta_core.build_ga_basis(t_fit, win.ga_result_tau,
                                   win.ga_result_t0, win.ga_result_fwhm, True)
    assert np.allclose(win.ga_result_dads @ exact.T, win.ga_result_fit,
                       atol=1e-10)
    assert np.allclose(C_fit, exact, atol=1e-12)
    # Populations: the EADS-row curves equal sequential_populations
    P = ta_core.sequential_populations(C, win.ga_result_tau, True)
    y = np.asarray(ga.ax_eads_t.get_lines()[0].get_ydata())
    assert np.allclose(y, P[:, 0])

    _check_exports(ga, win)

    ga.cb_comps[0].setChecked(False)                          # hide τ₁
    assert len(ga.ax_dads_t.get_lines()) == 2 + 1
    ga.dd_kin_scale.setCurrentText('Linear')
    assert ga.ax_dads_t.get_xscale() == 'linear'
    assert ga.ax_eads_t.get_xscale() == 'linear'
    # Cleared results (e.g. new data): EADS row resets, no stale draw
    win._clear_ga_state()
    ga.plot_eads()
    ga.plot_dads()
    assert ga.ax_eads_t.get_lines() == []
    QtWidgets.QApplication.processEvents()
    ga.close()
    win.close()
    print('[ok] GA dialog time panels')


def _export(ga, fn, path):
    orig_ask, orig_info = ta_ga.ask_save_path, ta_ga.info_box
    ta_ga.ask_save_path = lambda *a, **k: (path, '')
    ta_ga.info_box = lambda *a, **k: None
    try:
        fn()
    finally:
        ta_ga.ask_save_path, ta_ga.info_box = orig_ask, orig_info


def _load(path):
    with open(path, encoding='utf-8') as f:
        header = f.readline().strip().split(',')
    return header, np.loadtxt(path, delimiter=',', skiprows=1)


def _check_exports(ga, win):
    """Both buttons write a spectra file and its time-domain companion
    (all components, independent of the Show-components boxes)."""
    t, C = ga._time_profiles()
    P = ta_core.sequential_populations(C, win.ga_result_tau, True)
    with tempfile.TemporaryDirectory() as tmp:
        _export(ga, ga.export_dads, os.path.join(tmp, 'DADS_spectra.csv'))
        _export(ga, ga.export_eads, os.path.join(tmp, 'EADS_spectra.csv'))
        assert sorted(os.listdir(tmp)) == [
            'DADS_kinetics.csv', 'DADS_spectra.csv',
            'EADS_populations.csv', 'EADS_spectra.csv']
        hdr, M = _load(os.path.join(tmp, 'DADS_spectra.csv'))
        assert hdr[0] == 'wavelength_nm' and hdr[1].startswith('DADS_tau=')
        assert hdr[-1] == 'DADS_tau=inf'
        assert np.allclose(M[:, 1:], win.ga_result_dads, rtol=1e-8)
        hdr, M = _load(os.path.join(tmp, 'DADS_kinetics.csv'))
        assert hdr[0] == 'delay_ps' and hdr[1].startswith('C_tau=')
        assert len(hdr) == 1 + C.shape[1] and hdr[-1] == 'C_tau=inf'
        assert np.allclose(M[:, 0], t) and np.allclose(M[:, 1:], C, atol=1e-9)
        hdr, M = _load(os.path.join(tmp, 'EADS_spectra.csv'))
        assert hdr[0] == 'wavelength_nm' and hdr[1].startswith('EADS_S1_tau=')
        assert np.allclose(M[:, 1:], win.ga_result_eads, rtol=1e-8)
        hdr, M = _load(os.path.join(tmp, 'EADS_populations.csv'))
        assert hdr[0] == 'delay_ps' and hdr[1].startswith('pop_S1_tau=')
        assert hdr[-1] == 'pop_S3_tau=inf'
        assert np.allclose(M[:, 1:], P, atol=1e-9)
        # A renamed spectra file keeps its companion next to it
        _export(ga, ga.export_dads, os.path.join(tmp, 'run2_DADS_spectra.csv'))
        _export(ga, ga.export_eads, os.path.join(tmp, 'other.csv'))
        names = set(os.listdir(tmp))
        assert {'run2_DADS_kinetics.csv', 'other_populations.csv'} <= names
    assert ga._companion_path('x/A.tsv', 'kinetics') == 'x/A_kinetics.tsv'

    # Existing companion of a NEW spectra file: asked; "No" writes nothing
    with tempfile.TemporaryDirectory() as tmp:
        comp = os.path.join(tmp, 'DADS_kinetics.csv')
        open(comp, 'w').write('keep')
        asked = []
        ga._confirm_overwrite = lambda p: asked.append(p) or False
        _export(ga, ga.export_dads, os.path.join(tmp, 'DADS_spectra.csv'))
        assert asked == [comp] and os.listdir(tmp) == ['DADS_kinetics.csv']
        assert open(comp).read() == 'keep'
        # Re-export over an existing pair (dialog confirmed): no question
        ga._confirm_overwrite = lambda p: True
        _export(ga, ga.export_dads, os.path.join(tmp, 'DADS_spectra.csv'))
        asked.clear()
        ga._confirm_overwrite = lambda p: asked.append(p) or False
        _export(ga, ga.export_dads, os.path.join(tmp, 'DADS_spectra.csv'))
        assert asked == [] and open(comp).read() != 'keep'
        # A failing time table reports the spectra file already written
        warned = []
        orig_warn, orig_tp = ta_ga.warn_box, ga._time_profiles
        ta_ga.warn_box = lambda *a, **k: warned.append(a[2])
        ga._time_profiles = lambda: (_ for _ in ()).throw(RuntimeError('x'))
        try:
            _export(ga, ga.export_eads, os.path.join(tmp, 'EADS_spectra.csv'))
        finally:
            ta_ga.warn_box, ga._time_profiles = orig_warn, orig_tp
        assert len(warned) == 1 and 'EADS_spectra.csv' in warned[0]
    del ga._confirm_overwrite


TESTS = [test_core, test_dialog]


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    for fn in TESTS:
        fn()
    print(f'\nAll {len(TESTS)} GA time-panel checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
