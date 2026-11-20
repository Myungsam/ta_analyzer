"""
Kinetic Fit dialog: coherent-artifact terms (plan
.omc/plans/artifact-ga-kfit-plan.md §3 Step 3, §4 AC10–AC14/AC17/AC19).

Verifies (synthetic data only):
    1) AC10/AC19: Artifact combo / separate-FWHM / Fixed / From Chirp
       controls, default Off and not auto-seeded on open; From Chirp (D2)
       seeds ed_fw + combo and leaves the separate-FWHM controls alone; no
       erf result → notice, nothing changes; stretched + skip inline warning.
    2) AC11/AC12/AC13/AC14/AC17 single trace: B rows ± σ (no %), ratio over
       the exponentials only, dashed artifact line, params / trace exports,
       "Show artifact-subtracted", Residual store file with artifact_mode.
    3) Global fit: per-trace B rows, ratio per trace, one residual per trace.
    4) Off: params / trace exports identical to the pre-change dialog.
    5) Stretched + skip + artifact On: Run asks to switch to numerical
       (Yes / No / Cancel); No → artifact ignored (skip-mask) in the status.
    6) Import while app.art_main is active adds a notice.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import subprocess
import sys
import tempfile
import types
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import matplotlib  # noqa: E402
matplotlib.use('Agg')

import numpy as np  # noqa: E402
from PyQt5 import QtWidgets, QtCore  # noqa: E402

qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

import ta_core  # noqa: E402
import ta_main  # noqa: E402
import ta_kfit  # noqa: E402
import ta_ga  # noqa: E402
import ta_residual_store  # noqa: E402

TAUS = (2.0, 30.0)
FW = 0.2
T = np.r_[np.linspace(-1, 2, 151), np.logspace(0.4, 2.5, 60)]
B_TRUE = np.array([0.3, -0.2])          # xpm: G′, G″


def _dataset(seed=0, noise=0.003, art=True):
    rng = np.random.default_rng(seed)
    wl = np.linspace(450.0, 650.0, 101)
    a1 = np.exp(-((wl - 500) / 80) ** 2)
    a2 = -0.6 * np.exp(-((wl - 600) / 80) ** 2)
    e1 = ta_core.exp_irf_conv(T, TAUS[0], 0.0, FW)
    e2 = ta_core.exp_irf_conv(T, TAUS[1], 0.0, FW)
    D = np.outer(a1, e1) + np.outer(a2, e2)
    if art:
        g = np.exp(-((wl - 550) / 120) ** 2)
        D += np.outer(g, ta_core.artifact_basis(T, 0.0, FW, 'xpm') @ B_TRUE)
    return wl, T.copy(), D + noise * rng.normal(size=D.shape)


def _win(art=True):
    wl, t, D = _dataset(art=art)
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, D, 'Loaded: kfit artifact')
    return win


def _kfit(win, mod=ta_kfit):
    kd = mod.KineticFitDialog(win, win)
    kd.tbl.item(0, 0).setText('1.5')
    kd.tbl.item(1, 0).setText('20')
    kd.ed_fw.setValue(FW)
    return kd


def _set_art(kd, mode):
    kd.dd_artifact.setCurrentIndex(kd.dd_artifact.findData(mode))


def _exports(kd, mod, tmp, tag):
    """Params + trace exports into tmp; returns their paths."""
    p_par = os.path.join(tmp, f'{tag}_params.csv')
    p_tr = os.path.join(tmp, f'{tag}_trace.csv')
    paths = iter([p_par, p_tr])
    orig_ask, orig_info = mod.ask_save_path, mod.info_box
    mod.ask_save_path = lambda *a, **k: (next(paths), '')
    mod.info_box = lambda *a, **k: None
    try:
        kd._export_params()
        kd._export_trace()
    finally:
        mod.ask_save_path, mod.info_box = orig_ask, orig_info
    return p_par, p_tr


def _params(path):
    with open(path, encoding='utf-8') as f:
        return dict(line.rstrip('\n').split(',', 1) for line in f)


def _line_styles(ax):
    return [ln.get_linestyle() for ln in ax.get_lines()]


def test_controls_and_from_chirp():
    win = _win()
    win.erf_artifact = 'xpm'
    win.erf_fwhm_med = 0.2
    kd = _kfit(win)
    kd.ed_fw.setValue(0.15)
    # AC19: opening does not seed anything
    assert kd.dd_artifact.currentData() == 'off'
    assert not kd.cb_art_sep.isChecked() and kd.cb_art_fw_fix.isChecked()
    assert not kd.ed_art_fw.isEnabled()
    assert abs(kd.ed_fw.value() - 0.15) < 1e-12
    assert [kd.dd_artifact.itemData(i) for i in
            range(kd.dd_artifact.count())] == ['off', 'xpm', 'full']
    assert 'numerical' in kd.dd_artifact.toolTip().lower()
    kd.cb_art_sep.setChecked(True)
    assert kd.ed_art_fw.isEnabled() and kd.cb_art_fw_fix.isEnabled()
    kd.cb_art_sep.setChecked(False)
    art_fw0 = kd.ed_art_fw.value()

    # No erf result → notice, nothing changes
    win.erf_fwhm_med = None
    shown = []
    orig = ta_ga.info_box
    ta_ga.info_box = lambda *a, **k: shown.append(a[1])
    try:
        kd.btn_from_chirp.click()
    finally:
        ta_ga.info_box = orig
    assert shown == ['From Chirp'], shown
    assert kd.dd_artifact.currentData() == 'off'
    assert abs(kd.ed_fw.value() - 0.15) < 1e-12

    # D2: seeds ed_fw + combo, separate-FWHM controls untouched
    win.erf_fwhm_med = 0.18
    win.erf_artifact = 'full'
    kd.btn_from_chirp.click()
    assert abs(kd.ed_fw.value() - 0.18) < 1e-9, kd.ed_fw.value()
    assert kd.dd_artifact.currentData() == 'full'
    assert not kd.cb_art_sep.isChecked()
    assert kd.ed_art_fw.value() == art_fw0

    # Inline stretched + skip warning
    assert kd.lbl_art_warn.isHidden()
    kd.tbl.item(0, 2).setCheckState(QtCore.Qt.Checked)
    assert not kd.lbl_art_warn.isHidden()
    assert kd.lbl_art_warn.text() == ta_ga.ARTIFACT_SKIP_WARN
    kd.dd_irf.setCurrentIndex(kd.dd_irf.findData('numerical'))
    assert kd.lbl_art_warn.isHidden()
    kd.close()
    win.close()
    print('[1] controls, From Chirp (D2), AC19 no auto-seed, skip warning')


def test_single_trace_artifact():
    warnings.simplefilter('ignore')
    win = _win()
    D0 = win.deltaA.copy()
    kd = _kfit(win)
    kd.ed_wl.setValue(550.0)
    kd.do_import()
    _set_art(kd, 'xpm')
    kd.do_run()
    r = kd._last_fit
    assert not r['_global'] and r['A'].shape == (1, 2), r['A'].shape
    assert r['B'].shape == (1, 2) and r['artifact_fit'].shape == (1, r['_t_fit'].size)
    assert r['art_ignored'] is None
    assert np.allclose(sorted(r['tau']), TAUS, rtol=0.1), r['tau']
    assert np.allclose(r['B'][0], B_TRUE, atol=0.05), r['B']
    # AC11: ratio from the exponentials only
    assert r['ratio'].shape == (1, 2)
    assert abs(r['ratio'].sum() - 100.0) < 1e-9
    txt = kd.txt_result.toPlainText()
    b_lines = [ln for ln in txt.splitlines() if 'B_' in ln]
    assert len(b_lines) == 2, txt
    assert any(u'B_G′ = ' in ln for ln in b_lines), b_lines
    assert any(u'B_G″ = ' in ln for ln in b_lines), b_lines
    for ln in b_lines:
        assert '±' in ln and '%' not in ln, ln
    assert 'FWHM_art' in txt and 'Artifact:' in txt
    assert kd.cb_artsub.isEnabled() and kd.btn_artsub_resid.isEnabled()

    # AC12: dashed artifact line
    assert '--' in _line_styles(kd.ax_main), _line_styles(kd.ax_main)

    # AC13: toggle changes the plotted data, main data untouched
    y_plain = kd.ax_main.get_lines()[0].get_ydata().copy()
    kd.cb_artsub.setChecked(True)
    y_sub = kd.ax_main.get_lines()[0].get_ydata()
    assert y_sub.shape != y_plain.shape or not np.allclose(y_sub, y_plain)
    assert np.allclose(y_sub, r['_Y_fit'][0] - r['artifact_fit'][0])
    assert '--' in _line_styles(kd.ax_main)
    kd.cb_artsub.setChecked(False)
    assert np.array_equal(win.deltaA, D0)

    with tempfile.TemporaryDirectory() as tmp:
        p_par, p_tr = _exports(kd, ta_kfit, tmp, 'on')
        par = _params(p_par)
        for key in ('artifact_mode', 'art_fwhm', 'art_fwhm_std',
                    'B_dG', 'B_dG_std', 'B_d2G', 'B_d2G_std',
                    'A_1', 'ratio_2_pct'):
            assert key in par, key
        assert par['artifact_mode'] == 'xpm' and 'B_G' not in par
        hdr = open(p_tr, encoding='utf-8').readline().strip().split(',')
        assert hdr[1:] == ['data', 'fit', 'residual', 'artifact',
                           'data_minus_artifact'], hdr
        # AC14: sub + art = data
        M = np.loadtxt(p_tr, delimiter=',', skiprows=1)
        data = r['_Y_fit'][0]
        tol = 1e-9 * np.nanmax(np.abs(data))
        assert np.nanmax(np.abs(M[:, 4] + M[:, 5] - M[:, 1])) <= tol
        assert np.nanmax(np.abs(M[:, 1] - data)) <= tol

        # AC17: Residual store (patched folder)
        orig_dir = ta_residual_store.get_residual_dir
        orig_info = ta_kfit.info_box
        ta_residual_store.get_residual_dir = lambda base_dir=None: tmp
        ta_kfit.info_box = lambda *a, **k: None
        try:
            kd.btn_artsub_resid.click()
        finally:
            ta_residual_store.get_residual_dir = orig_dir
            ta_kfit.info_box = orig_info
        files = [f for f in os.listdir(tmp) if f.endswith('.xlsx')]
        assert len(files) == 1, files
        rr = ta_residual_store.load_residual(os.path.join(tmp, files[0]))
        assert rr['kind'] == '1D' and rr['meta']['artifact_mode'] == 'xpm'
        want = r['_Y_fit'][0] - r['artifact_fit'][0]
        assert np.allclose(np.ravel(rr['t']), r['_t_fit'])
        assert np.allclose(np.ravel(rr['R']), want, atol=1e-12), rr.keys()
    kd.close()
    win.close()
    print('[2] single trace: B ± σ rows, ratio, dashed line, exports, '
          'art-sub toggle, residual store')


def test_global_artifact():
    warnings.simplefilter('ignore')
    win = _win()
    for w in (500.0, 550.0, 600.0):
        win.set_sel_wl(w)
        win.pin_kin()
    kd = _kfit(win)
    assert kd.cb_pinned.isChecked()
    kd.do_import()
    assert len(kd._traces) == 3
    _set_art(kd, 'xpm')
    kd.do_run()
    r = kd._last_fit
    assert r['_global'] and r['A'].shape == (3, 2) and r['B'].shape == (3, 2)
    assert r['ratio'].shape[1] == 2
    assert np.all(np.abs(r['ratio'].sum(axis=1) - 100.0) < 1e-9)
    txt = kd.txt_result.toPlainText()
    assert sum(u'B_G′ = ' in ln for ln in txt.splitlines()) == 3, txt
    assert '--' in _line_styles(kd.ax_main)
    with tempfile.TemporaryDirectory() as tmp:
        p_par, p_tr = _exports(kd, ta_kfit, tmp, 'glob')
        par = _params(p_par)
        for j in (1, 2, 3):
            assert f'T{j}_B_dG' in par and f'T{j}_B_d2G_std' in par
        hdr = open(p_tr, encoding='utf-8').readline()
        assert 'artifact_550.0nm' in hdr and 'data_minus_artifact_600.0nm' \
            in hdr, hdr
        sub = os.path.join(tmp, 'res')
        os.makedirs(sub)
        orig_dir = ta_residual_store.get_residual_dir
        orig_info = ta_kfit.info_box
        ta_residual_store.get_residual_dir = lambda base_dir=None: sub
        ta_kfit.info_box = lambda *a, **k: None
        try:
            kd.save_artsub_residual()
        finally:
            ta_residual_store.get_residual_dir = orig_dir
            ta_kfit.info_box = orig_info
        assert len(os.listdir(sub)) == 3, os.listdir(sub)
    kd.close()
    win.close()
    print('[3] global: per-trace B rows, ratios, one residual per trace')


def _head_module():
    """The pre-change ta_kfit.py (git HEAD) as a module, or None."""
    try:
        src = subprocess.run(
            ['git', 'show', 'HEAD:ta_kfit.py'], cwd=_PROJ,
            capture_output=True, check=True).stdout.decode('utf-8')
    except Exception:
        return None
    mod = types.ModuleType('ta_kfit_head')
    mod.__file__ = os.path.join(_PROJ, 'ta_kfit_head.py')
    exec(compile(src, mod.__file__, 'exec'), mod.__dict__)
    return mod


PRE_KEYS_SINGLE = (
    ['fit_mode', 'n_traces', 'rms', 'r2', 'adj_r2', 'chi2_red', 'rss',
     'aic', 'bic', 'n_obs', 'n_par', 'dof', 't0', 't0_std', 'fwhm',
     'fwhm_std', 'irf_mode']
    + [f'{k}_{i}{s}' for i in (1, 2)
       for k, s in (('tau', ''), ('tau', '_std'), ('beta', ''),
                    ('beta', '_std'), ('type', ''))]
    + ['lambda_nm', 'n_avg_px']
    + [f'{k}' for i in (1, 2)
       for k in (f'A_{i}', f'A_{i}_std', f'ratio_{i}_pct')])


def test_off_identity():
    warnings.simplefilter('ignore')
    win = _win()
    head = _head_module()
    with tempfile.TemporaryDirectory() as tmp:
        out = {}
        for tag, mod in (('new', ta_kfit), ('head', head)):
            if mod is None:
                continue
            kd = _kfit(win, mod)
            kd.ed_wl.setValue(550.0)
            kd.do_import()
            kd.do_run()
            assert kd._last_fit is not None
            if tag == 'new':
                assert kd._last_fit['B'] is None
                assert not kd.cb_artsub.isEnabled()
                assert '--' not in _line_styles(kd.ax_main)
                txt = kd.txt_result.toPlainText()
                assert 'B_' not in txt and 'Artifact' not in txt
                assert 'Artifact' not in kd.lbl_status.text()
            out[tag] = [open(p, 'rb').read()
                        for p in _exports(kd, mod, tmp, tag)]
            out[tag + '_status'] = kd.lbl_status.text()
            out[tag + '_report'] = kd.txt_result.toPlainText()
            kd.close()
        keys = [ln.split(b',', 1)[0].decode() for ln in
                out['new'][0].splitlines()]
        assert keys == PRE_KEYS_SINGLE, keys
        hdr = out['new'][1].splitlines()[0].decode()
        assert hdr == f'delay_{win.t_unit_hdr()},data,fit,residual', hdr
        if head is not None:
            assert out['new'] == out['head'], 'Off export differs from HEAD'
            assert out['new_status'] == out['head_status']
            assert out['new_report'] == out['head_report']
    win.close()
    print('[4] Off: params / trace exports '
          + ('byte-identical to HEAD' if head is not None
             else 'match the pre-change key list (git unavailable)'))


def test_skip_prompt():
    warnings.simplefilter('ignore')
    win = _win()
    kd = _kfit(win)
    kd.ed_wl.setValue(550.0)
    kd.do_import()
    _set_art(kd, 'xpm')
    kd.tbl.item(1, 2).setCheckState(QtCore.Qt.Checked)      # stretched
    assert kd.dd_irf.currentData() == 'skip'
    asked = []
    MB = QtWidgets.QMessageBox
    orig_q = MB.question

    def answer(ans):
        def q(*a, **k):
            asked.append(a[1])
            return ans
        return q
    try:
        MB.question = staticmethod(answer(MB.Cancel))
        kd.do_run()
        assert len(asked) == 1 and kd._last_fit is None
        MB.question = staticmethod(answer(MB.No))
        kd.do_run()
        r = kd._last_fit
        assert len(asked) == 2 and kd.dd_irf.currentData() == 'skip'
        assert r['art_ignored'] == 'skip-mask', r['art_ignored']
        assert 'Artifact ignored (skip-mask)' in kd.lbl_status.text()
        assert 'Artifact ignored (skip-mask)' in kd.txt_result.toPlainText()
        assert np.all(r['B'] == 0)
        MB.question = staticmethod(answer(MB.Yes))
        kd.do_run()
        assert len(asked) == 3
        assert kd.dd_irf.currentData() == 'numerical'
        assert kd._last_fit['art_ignored'] is None
        assert 'ignored' not in kd.lbl_status.text()
        kd.do_run()                                   # no conflict now
        assert len(asked) == 3
    finally:
        MB.question = orig_q
    kd.close()
    win.close()
    print('[5] stretched + skip: prompt Cancel / No (ignored) / Yes '
          '(numerical)')


def test_import_notice_art_main():
    win = _win()
    kd = _kfit(win)
    kd.ed_wl.setValue(550.0)
    kd.do_import()
    assert 'artifact-subtracted' not in kd.lbl_import.text()
    win.art_main = {'mode': 'xpm'}
    try:
        kd.do_import()
        assert 'main data is artifact-subtracted' in kd.lbl_import.text()
    finally:
        win.art_main = None
    kd.close()
    win.close()
    print('[6] import notice while app.art_main is active')


if __name__ == '__main__':
    test_controls_and_from_chirp()
    test_single_trace_artifact()
    test_global_artifact()
    test_off_identity()
    test_skip_prompt()
    test_import_notice_art_main()
    assert qapp is not None
    print('\nAll Kinetic Fit artifact tests passed.')
