"""
Tests for the coherent-artifact controls in the analysis dialogs.

Plan .omc/plans/artifact-ga-kfit-plan.md §4 (with the §0 decisions):

  Global Analysis
    AC6)  Artifact combo (off / xpm / full) with a tooltip on bias and
          chirp, separate-FWHM controls, From Chirp, inline skip warning.
    AC7)  DADS raw / normalised panels get grey dashed G′ / G″ lines; the
          C(t) panel and EADS are unchanged.
    AC8)  fit = exponential part + artifact_fit; kinetics show the
          artifact as a dashed line.
    AC9)  (CSV part) Off exports unchanged; xpm adds
          ``<companion>_artifact_spectra.csv`` with its overwrite check;
          kinetics export gains the artifact columns only when On.
    AC13) "Show artifact-subtracted" changes the plots, not app.deltaA.
    AC14) Art.-sub export: subtracted + artifact = data (1e-9 · max|D|).
    AC17) Art.-sub → Residual store loads back with artifact_mode meta.
          Outside-window case: warning + status note, nothing to apply.
  From Chirp
    AC18) No erf result → notice, nothing changes; with a result the IRF
          FWHM and the combo are seeded (separate FWHM untouched); the
          Chirp dialog's erf run stores app.erf_fwhm_med; load resets it.
    AC19) Opening the dialog never seeds anything.

The Kinetic Fit cases are added to this file as further ``test_kfit_*``
functions listed in ``TESTS`` (they can reuse ``_win`` / ``_patch_boxes``).

Synthetic data only.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import glob
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')

import numpy as np
from PyQt5 import QtWidgets

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
for _p in (_PROJ, _HERE, os.path.join(_HERE, 'fixtures')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Keep a reference: an unreferenced QApplication is collected at once.
qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

import ta_core  # noqa: E402
import ta_main  # noqa: E402
import ta_ga  # noqa: E402
import ta_opju_export  # noqa: E402
import ta_residual_store  # noqa: E402
import make_artifact_off_golden as gold  # noqa: E402

DESC = 'Loaded: artifact dialogs'
BOXES = []          # (kind, title, text) of patched message boxes


def _patch_boxes():
    """Record info / warn boxes instead of opening modal dialogs."""
    BOXES.clear()
    ta_ga.info_box = lambda *a, **k: BOXES.append(('info', a[1], a[2]))
    ta_ga.warn_box = lambda *a, **k: BOXES.append(('warn', a[1], a[2]))


def _win():
    wl, t, D, Bp, Bpp = gold._art_synth(ta_core, seed=0)
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, D, DESC)
    return win, (wl, t, D, Bp, Bpp)


def _ga(win, mode='xpm', t_lo=None, t_hi=None, run=True):
    """GA dialog set up for the synthetic recipe (2 τ, t0 / FWHM free)."""
    ga = ta_ga.GlobalAnalysisDialog(win, win)
    ga.dd_N.setCurrentText('2')
    ga.tau_table.cellChanged.disconnect()
    for r, v in enumerate([0.5, 30.0]):
        ga.tau_table.item(r, 0).setText(str(v))
    ga.tau_table.cellChanged.connect(ga._on_tau_edit)
    ga.ed_t0.setValue(0.0)
    ga.cb_t0_fix.setChecked(False)
    ga.ed_fw.setValue(0.1)
    ga.cb_fw_fix.setChecked(False)
    ga.dd_artifact.setCurrentIndex(ga.dd_artifact.findData(mode))
    if t_lo is not None:
        ga.ed_t_min.setValue(t_lo)
    if t_hi is not None:
        ga.ed_t_max.setValue(t_hi)
    if run:
        ga.do_run()
    return ga


def _export(ga, fn, path):
    orig = ta_ga.ask_save_path
    ta_ga.ask_save_path = lambda *a, **k: (path, '')
    try:
        fn()
    finally:
        ta_ga.ask_save_path = orig


def _lines(ax, style=None):
    return [ln for ln in ax.get_lines()
            if style is None or ln.get_linestyle() == style]


# =====================================================================
# Global Analysis
# =====================================================================
def test_ga_controls():
    """AC6 + AC19 (GA part)."""
    _patch_boxes()
    win, _ = _win()
    win.erf_artifact = 'xpm'
    win.erf_fwhm_med = 0.2
    ga = ta_ga.GlobalAnalysisDialog(win, win)
    dd = ga.dd_artifact
    assert [dd.itemData(i) for i in range(dd.count())] == \
        ['off', 'xpm', 'full']
    tip = dd.toolTip()
    assert 'bias' in tip and 'chirp' in tip
    for w in (ga.cb_art_sep, ga.ed_art_fw, ga.cb_art_fw_fix,
              ga.btn_from_chirp):
        assert w is not None
    # AC19: the constructor reads app state only, never the chirp result
    assert dd.currentData() == 'off' and not ga.cb_art_sep.isChecked()
    assert ga.ed_fw.value() == win.ga_fwhm
    assert not ga.ed_art_fw.isEnabled() and not ga.cb_art_fw_fix.isEnabled()
    ga.cb_art_sep.setChecked(True)
    assert ga.ed_art_fw.isEnabled() and ga.cb_art_fw_fix.isEnabled()
    ga.cb_art_sep.setChecked(False)
    # Artifact actions are disabled without an artifact result
    for b in (ga.btn_exp_artsub, ga.btn_artsub_resid, ga.cb_show_artsub,
              ga.btn_apply_art, ga.btn_revert_art):
        assert not b.isEnabled()
    # Inline warning: artifact On + stretched row + 'skip'
    assert ga.lbl_art_warn.isHidden()
    dd.setCurrentIndex(dd.findData('xpm'))
    ga.dd_irf_mode.setCurrentText('skip')
    assert ga.lbl_art_warn.isHidden()
    it = ga.tau_table.item(0, 2)
    it.setCheckState(2)                       # Qt.Checked → _on_tau_edit
    assert not ga.lbl_art_warn.isHidden()
    assert 'numerical' in ga.lbl_art_warn.text()
    dd.setCurrentIndex(dd.findData('off'))
    assert ga.lbl_art_warn.isHidden()
    it.setCheckState(0)
    ga.dd_irf_mode.setCurrentText('numerical')
    ga.close()
    win.close()
    print('[ok] AC6/AC19 GA controls')


def test_ga_off_and_xpm():
    """AC7, AC8, AC9 (CSV), AC13, AC14, AC17, persistence (AC19)."""
    _patch_boxes()
    win, (wl, t, D, Bp, Bpp) = _win()

    # ---- Off run: reference outputs ---------------------------------
    ga = _ga(win, 'off')
    assert win.ga_result_B is None and win.ga_result_artifact == 'off'
    assert win.ga_result_pipeline_sig is None       # no hash when Off
    status_off = ga.lbl_status.text()
    assert status_off.startswith('Done. RMS') and '\n' not in status_off
    assert 'rtifact' not in ga.txt_result.toPlainText()
    n_dads_off = len(ga.ax_dads.get_lines())
    n_dads_n_off = len(ga.ax_dads_n.get_lines())
    n_dads_t_off = len(ga.ax_dads_t.get_lines())
    assert not _lines(ga.ax_kin_fit, '--')
    for b in (ga.btn_exp_artsub, ga.btn_artsub_resid, ga.cb_show_artsub,
              ga.btn_apply_art):
        assert not b.isEnabled()
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'DADS_spectra.csv')
        _export(ga, ga.export_dads, p)
        assert sorted(os.listdir(tmp)) == ['DADS_kinetics.csv',
                                           'DADS_spectra.csv']
        # Same bytes as the pre-artifact writer (_write_table of A only)
        ref = os.path.join(tmp, 'ref.csv')
        names = [f'tau={tv:.4g}_ps' for tv in win.ga_result_tau]
        ga._write_table(ref, [wl, win.ga_result_dads],
                        ['wavelength_nm'] + [f'DADS_{n}' for n in names])
        assert open(ref, 'rb').read() == open(p, 'rb').read()
        kp = os.path.join(tmp, 'kin.csv')
        _export(ga, ga.export_kinetics, kp)
        hdr = open(kp, encoding='utf-8').readline().strip().split(',')
        assert hdr == ['delay_ps', 'data_dA', 'fit_dA', 'residual_dA']
    assert set(win.opju_stage.entries) <= (set(ta_opju_export.SLOT_KEYS)
            | set(ta_opju_export.OPTIONAL_KEYS))
    ga.close()

    # ---- xpm run ----------------------------------------------------
    deltaA0 = win.deltaA.copy()
    ga = _ga(win, 'xpm')
    assert win.ga_artifact == 'xpm'                  # do_run persists
    B, art = win.ga_result_B, win.ga_result_artifact_fit
    assert B.shape == (wl.size, 2) and art.shape == win.ga_result_fit.shape
    assert win.ga_result_art_ignored is None
    assert win.ga_result_pipeline_sig == win._compute_art_pre_hash()
    assert win.ga_result_on_artsub is False
    assert np.corrcoef(B[:, 0], Bp)[0, 1] > 0.9
    txt = ga.txt_result.toPlainText()
    assert u'Artifact : G′+G″ (sample)' in txt and 'FWHM_art' in txt
    # chirp not applied → pre-run note kept in the final status
    assert 'chirp not applied' in ga.lbl_status.text()

    # AC7: two grey dashed DADS lines, labelled G′ / G″, both panels
    for ax, n_off in ((ga.ax_dads, n_dads_off), (ga.ax_dads_n, n_dads_n_off)):
        assert len(ax.get_lines()) == n_off + 2
        dashed = _lines(ax, '--')
        assert [ln.get_label() for ln in dashed] == [u'G′', u'G″']
        assert all(ln.get_color() == 'grey' for ln in dashed)
    assert np.allclose(_lines(ga.ax_dads, '--')[0].get_ydata(), B[:, 0])
    assert len(ga.ax_dads_t.get_lines()) == n_dads_t_off
    E, _, _ = ta_core.compute_eads_from_dads(
        win.ga_result_dads, win.ga_result_tau, False)
    assert win.ga_result_eads.shape == (wl.size, 2)
    assert np.allclose(win.ga_result_eads, E)

    # AC8: fit = exp part + artifact; dashed Artifact kinetics line
    tf = win.ga_result_delay
    C = ta_core.build_ga_basis(tf, win.ga_result_tau, win.ga_result_t0,
                               win.ga_result_fwhm, False)
    assert np.abs(win.ga_result_fit - win.ga_result_dads @ C.T
                  - art).max() < 1e-10
    iw = int(np.argmin(np.abs(wl - ga.cur_kin_wl)))
    kin_art = [ln for ln in _lines(ga.ax_kin_fit, '--')
               if ln.get_label() == 'Artifact']
    assert len(kin_art) == 1
    assert np.allclose(kin_art[0].get_ydata(), art[iw])
    for b in (ga.btn_exp_artsub, ga.btn_artsub_resid, ga.cb_show_artsub,
              ga.btn_apply_art):
        assert b.isEnabled()

    # AC13: the toggle changes the plotted data, never app.deltaA
    data_ln = [ln for ln in ga.ax_kin_fit.get_lines()
               if ln.get_label() == 'Data'][0]
    y0 = np.asarray(data_ln.get_ydata()).copy()
    m0 = np.asarray(ga.ax_exp.collections[0].get_array()).copy()
    ga.cb_show_artsub.setChecked(True)
    data_ln = [ln for ln in ga.ax_kin_fit.get_lines()
               if ln.get_label().startswith('Data')][0]
    y1 = np.asarray(data_ln.get_ydata())
    assert np.allclose(y1, y0 - art[iw]) and not np.allclose(y1, y0)
    m1 = np.asarray(ga.ax_exp.collections[0].get_array())
    assert not np.allclose(m0, m1)
    assert np.array_equal(win.deltaA, deltaA0, equal_nan=True)
    ga.cb_show_artsub.setChecked(False)
    assert np.array_equal(win.deltaA, deltaA0, equal_nan=True)

    _check_xpm_exports(ga, win)

    # AC17: Art.-sub → Residual store
    with tempfile.TemporaryDirectory() as tmp:
        orig = ta_residual_store.get_residual_dir
        ta_residual_store.get_residual_dir = lambda base_dir=None: tmp
        try:
            ga.save_artsub_residual()
        finally:
            ta_residual_store.get_residual_dir = orig
        files = glob.glob(os.path.join(tmp, '*.xlsx'))
        assert len(files) == 1, files
        r = ta_residual_store.load_residual(files[0])
        assert r['kind'] == '2D'
        assert r['meta'].get('artifact_mode') == 'xpm'
        assert np.allclose(r['t'], tf) and np.allclose(r['wl'], wl)
        assert np.allclose(r['R'], win.ga_result_data - art, atol=1e-12)
    ga.close()

    # Persistence: a new dialog shows the last run's artifact choice
    ga2 = ta_ga.GlobalAnalysisDialog(win, win)
    assert ga2.dd_artifact.currentData() == 'xpm'
    ga2.close()
    win.close()
    print('[ok] AC7/AC8/AC9/AC13/AC14/AC17 GA xpm run')


def _check_xpm_exports(ga, win):
    wl = win.wavelength
    B, art = win.ga_result_B, win.ga_result_artifact_fit
    D_fit, tf = win.ga_result_data, win.ga_result_delay
    with tempfile.TemporaryDirectory() as tmp:
        # AC9: DADS file unchanged in shape; artifact companion added
        p = os.path.join(tmp, 'DADS_spectra.csv')
        _export(ga, ga.export_dads, p)
        assert sorted(os.listdir(tmp)) == [
            'DADS_artifact_spectra.csv', 'DADS_kinetics.csv',
            'DADS_spectra.csv']
        hdr = open(p, encoding='utf-8').readline().strip().split(',')
        assert hdr[0] == 'wavelength_nm' and hdr[1].startswith('DADS_tau=')
        assert len(hdr) == 1 + win.ga_result_dads.shape[1]
        pa = os.path.join(tmp, 'DADS_artifact_spectra.csv')
        hdr = open(pa, encoding='utf-8').readline().strip()
        assert hdr == u'wavelength_nm,G1 (σG′),G2 (σ²G″)', hdr
        assert 'tau=' not in hdr
        M = np.loadtxt(pa, delimiter=',', skiprows=1, encoding='utf-8')
        assert np.allclose(M[:, 0], wl) and np.allclose(M[:, 1:], B,
                                                        rtol=1e-8)
        # Kinetics export: artifact columns appended when On
        kp = os.path.join(tmp, 'kin.csv')
        _export(ga, ga.export_kinetics, kp)
        hdr = open(kp, encoding='utf-8').readline().strip().split(',')
        assert hdr[-2:] == ['artifact_dA', 'data_minus_artifact_dA']
        K = np.loadtxt(kp, delimiter=',', skiprows=1)
        assert np.allclose(K[:, 1] - K[:, 4], K[:, 5])

        # AC14: Art.-sub 2D export + _artifact companion
        ps = os.path.join(tmp, 'GA_artifact_subtracted.csv')
        _export(ga, ga.export_artifact_subtracted, ps)
        pc = os.path.join(tmp, 'GA_artifact_subtracted_artifact.csv')
        assert os.path.exists(pc)
        w1, t1, S = ta_core.parse_data_file(ps)
        w2, t2, A = ta_core.parse_data_file(pc)
        assert np.allclose(t1, tf) and np.allclose(w2, wl)
        tol = 1e-9 * np.nanmax(np.abs(D_fit))
        assert np.nanmax(np.abs(S + A - D_fit)) <= tol
    assert set(win.opju_stage.entries) <= (set(ta_opju_export.SLOT_KEYS)
            | set(ta_opju_export.OPTIONAL_KEYS))

    # AC9: an existing artifact companion of a NEW DADS file is asked
    # about (QMessageBox.question); "No" writes nothing.
    with tempfile.TemporaryDirectory() as tmp:
        comp = os.path.join(tmp, 'DADS_artifact_spectra.csv')
        open(comp, 'w').write('keep')
        asked = []
        orig_q = QtWidgets.QMessageBox.question
        QtWidgets.QMessageBox.question = staticmethod(
            lambda *a, **k: asked.append(a[2]) or QtWidgets.QMessageBox.No)
        try:
            _export(ga, ga.export_dads, os.path.join(tmp, 'DADS_spectra.csv'))
            assert len(asked) == 1 and 'DADS_artifact_spectra.csv' in asked[0]
            assert os.listdir(tmp) == ['DADS_artifact_spectra.csv']
            assert open(comp).read() == 'keep'
            # Same rule for the Art.-sub companion
            comp2 = os.path.join(tmp, 'GA_artifact_subtracted_artifact.csv')
            open(comp2, 'w').write('keep')
            asked.clear()
            _export(ga, ga.export_artifact_subtracted,
                    os.path.join(tmp, 'GA_artifact_subtracted.csv'))
            assert len(asked) == 1 and open(comp2).read() == 'keep'
            assert not os.path.exists(
                os.path.join(tmp, 'GA_artifact_subtracted.csv'))
        finally:
            QtWidgets.QMessageBox.question = orig_q
    tpe = os.path.join(_PROJ, 'Data', 'ta_plot_export.py')
    if not os.path.exists(tpe):
        print('    SKIP ta_plot_export detection (Data/ta_plot_export.py '
              'not present)')


def test_ga_outside_window():
    """artifact_ignored ('outside-window'): warning, status, no Apply."""
    _patch_boxes()
    win, _ = _win()
    ga = _ga(win, 'xpm', t_lo=0.3)
    assert win.ga_result_art_ignored == 'outside-window'
    assert np.all(win.ga_result_B == 0)
    assert any(k == 'warn' and 'Artifact ignored' in title
               for k, title, _ in BOXES), BOXES
    assert 'artifact ignored (outside fit window)' in ga.lbl_status.text()
    assert 'Artifact ignored (outside fit window)' in \
        ga.txt_result.toPlainText()
    BOXES.clear()
    ga.apply_artifact_to_main()
    assert win.art_main is None
    assert BOXES and 'ignored' in BOXES[0][2]
    ga.close()
    win.close()
    print('[ok] GA outside-window artifact ignored')


# =====================================================================
# From Chirp (shared helper, GA dialog)
# =====================================================================
def test_from_chirp():
    """AC18 (D2: seeds the IRF FWHM and the combo only)."""
    _patch_boxes()
    win, _ = _win()
    assert win.erf_fwhm_med is None
    ga = ta_ga.GlobalAnalysisDialog(win, win)
    fw0 = ga.ed_fw.value()
    win.erf_artifact = 'full'
    ga.btn_from_chirp.click()                 # no erf result → notice
    assert len(BOXES) == 1 and BOXES[0][0] == 'info'
    assert 'erf' in BOXES[0][2]
    assert ga.ed_fw.value() == fw0 and ga.dd_artifact.currentData() == 'off'

    win.erf_fwhm_med = 0.18
    art_fw0 = ga.ed_art_fw.value()
    for sep in (False, True):
        ga.dd_artifact.setCurrentIndex(0)
        ga.ed_fw.setValue(fw0)
        ga.cb_art_sep.setChecked(sep)
        BOXES.clear()
        ga.btn_from_chirp.click()
        assert not BOXES
        assert ga.ed_fw.value() == 0.18
        assert ga.dd_artifact.currentData() == 'full'
        assert ga.cb_art_sep.isChecked() == sep        # untouched
        assert ga.ed_art_fw.value() == art_fw0         # untouched
    # 'off' in the chirp helper leaves the combo alone
    ga.dd_artifact.setCurrentIndex(ga.dd_artifact.findData('xpm'))
    win.erf_artifact = 'off'
    assert ta_ga.seed_artifact_from_chirp(win, ga.dd_artifact, ga.ed_fw, ga)
    assert ga.dd_artifact.currentData() == 'xpm'
    ga.close()

    # The Chirp dialog's erf run stores the median FWHM; load resets it
    import ta_chirp
    from test_chirp_erf import _synthetic
    ta_chirp.warn_box = lambda *a, **k: None
    wl, t, D, _ = _synthetic(n_wl=150)
    win.set_loaded_data(wl, t, D, 'chirp_fwhm.csv')
    assert win.erf_fwhm_med is None
    win.ridge_tmin, win.ridge_tmax = -0.8, 1.5
    dlg = ta_chirp.ChirpDialog(win, win)
    dlg.rb_erf.setChecked(True)
    dlg.recompute_ridge()
    assert win.erf_fwhm_med is not None
    assert 0.08 < win.erf_fwhm_med < 0.16, win.erf_fwhm_med
    dlg.close()
    win.set_loaded_data(wl, t, D, 'chirp_fwhm.csv')
    assert win.erf_fwhm_med is None
    win.close()
    print('[ok] AC18 From Chirp')


# Kinetic Fit cases are appended here (test_kfit_* functions).
TESTS = [
    test_ga_controls,
    test_ga_off_and_xpm,
    test_ga_outside_window,
    test_from_chirp,
]


if __name__ == '__main__':
    for fn in TESTS:
        fn()
    print(f'\nAll {len(TESTS)} artifact dialog checks passed.')
