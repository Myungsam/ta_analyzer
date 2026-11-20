"""
Tests for the artifact-subtraction stage of the main pipeline
(``TAAnalyzer.apply_artifact_main`` / ``revert_artifact_main``, applied
from Global Analysis).

Plan .omc/plans/artifact-ga-kfit-plan.md §4 (with the §0 decisions):

  AC15) GA fit (xpm, sub-window) → Apply: deltaA = pre − B·basis(delay)ᵀ on
        the full grid, map title / layout signature / status tag; the GA
        dialog result is reset (D3); Revert restores the data.  Window
        coverage confirmation; guards (fit on artifact-subtracted data,
        data changed since the fit).
  AC16) Starting from no chirp / solvent / BG, each of crop, resample,
        t₀ shift (both paths), chirp apply, chirp revert, BG apply,
        solvent on, solvent off, solvent scale, load and Revert clears the
        stage with a visible notice that keeps the data description; the
        notice is gone one action later.  Off mode leaves lbl_status
        alone (crop_info suffix, t₀-shift text).  The solvent preview
        sequence and a mask change keep the stage.

Synthetic data only.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys

import matplotlib
matplotlib.use('Agg')

import numpy as np
from PyQt5 import QtWidgets

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
for _p in (_PROJ, os.path.join(_HERE, 'fixtures')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Keep a reference: an unreferenced QApplication is collected at once.
qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

import ta_core  # noqa: E402
import ta_main  # noqa: E402
import ta_ga  # noqa: E402
import ta_dialogs_a  # noqa: E402
import make_artifact_off_golden as gold  # noqa: E402

DESC = 'Loaded: artifact stage'
NOTICE = 'artifact subtraction cleared'
WL, T, D, BP, BPP = gold._art_synth(ta_core, seed=0)
B_SYN = np.column_stack([BP, BPP])
BOXES = []


def _patch_boxes():
    BOXES.clear()
    ta_ga.info_box = lambda *a, **k: BOXES.append(('info', a[1], a[2]))
    ta_ga.warn_box = lambda *a, **k: BOXES.append(('warn', a[1], a[2]))
    ta_main.warn_box = lambda *a, **k: BOXES.append(('warn', a[1], a[2]))


def _load(win):
    win.set_loaded_data(WL, T, D, DESC)


def _ga(win, t_lo, t_hi):
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
    ga.dd_artifact.setCurrentIndex(ga.dd_artifact.findData('xpm'))
    ga.ed_t_min.setValue(t_lo)
    ga.ed_t_max.setValue(t_hi)
    ga.do_run()
    return ga


class _Question:
    """Patched QMessageBox.question: records the text, answers ``ans``."""

    def __init__(self, ans):
        self.ans, self.asked = ans, []
        self._orig = QtWidgets.QMessageBox.question

    def __enter__(self):
        QtWidgets.QMessageBox.question = staticmethod(
            lambda *a, **k: self.asked.append(a[2]) or self.ans)
        return self

    def __exit__(self, *exc):
        QtWidgets.QMessageBox.question = self._orig


# =====================================================================
# AC15: Apply from GA / Revert / guards
# =====================================================================
def test_apply_revert():
    _patch_boxes()
    win = ta_main.TAAnalyzer()
    _load(win)
    assert win.chirp_params is None and win.bg_spectrum is None
    assert win.sub_irf_aligned is None
    ga = _ga(win, -0.5, 100.0)               # GA with all None: no raise
    assert win.ga_result_B is not None and win.ga_result_art_ignored is None
    pre = win._pre_stage_data()
    B, t0 = win.ga_result_B.copy(), float(win.ga_result_t0)
    fw, mode = float(win.ga_result_art_fwhm), win.ga_result_artifact
    sig0 = win._compute_map_layout_sig()
    title0 = win._map_title()
    assert 'artifact-sub' not in title0

    with _Question(QtWidgets.QMessageBox.Yes) as q:
        ga.apply_artifact_to_main()
    assert q.asked == []                       # window covers t0 ± 3σ
    assert win.art_main is not None
    assert win.art_main['matrix'].shape == (WL.size, T.size)
    expect = pre - B @ ta_core.artifact_basis(win.delay, t0, fw, mode).T
    assert np.abs(win.deltaA - expect).max() <= 1e-12
    assert 'artifact-sub' in win._map_title()
    assert win._compute_map_layout_sig() != sig0
    assert '[artifact-sub]' in win.lbl_status.text()
    assert win.lbl_status.text().startswith(DESC)
    # D3: the GA dialog result is reset, exports disabled
    assert win.ga_result_fit is None and win.ga_result_B is None
    assert win.ga_result_dads is None
    for b in (ga.btn_exp_dads, ga.btn_exp_eads, ga.btn_exp_fit,
              ga.btn_exp_res, ga.btn_exp_kin, ga.btn_save_resid,
              ga.btn_exp_artsub, ga.btn_artsub_resid, ga.btn_apply_art):
        assert not b.isEnabled()
    assert ga.lbl_status.text() == \
        u'Artifact applied to main data — re-run Fit'
    assert ga.ax_dads.get_lines() == [] and ga.ax_kin_fit.get_lines() == []
    assert ga.btn_revert_art.isEnabled()
    # A mask change keeps the stage (masks follow it)
    win.masked_regions.append((500.0, 520.0, 'nan'))
    win.update_all()
    assert win.art_main is not None and not win._art_stale
    m = (WL >= 500) & (WL <= 520)
    assert np.isnan(win.deltaA[m]).all()
    assert np.abs(win.deltaA[~m] - expect[~m]).max() <= 1e-12
    win.masked_regions = []
    win.update_all()

    # Guard: a re-run on the subtracted data cannot be applied again
    ga.do_run()
    assert win.ga_result_on_artsub is True
    assert ga.btn_exp_dads.isEnabled()          # re-enabled by the run
    BOXES.clear()
    ga.apply_artifact_to_main()
    assert BOXES and 'Revert artifact first' in BOXES[0][2]

    # Revert restores the pre-stage data exactly; tag gone
    ga.revert_artifact_main()
    assert win.art_main is None
    assert np.array_equal(win.deltaA, pre)
    assert '[artifact-sub]' not in win.lbl_status.text()
    assert NOTICE not in win.lbl_status.text()
    assert 'artifact-sub' not in win._map_title()
    assert not ga.btn_revert_art.isEnabled()
    BOXES.clear()
    ga.apply_artifact_to_main()                 # fit was on art-sub data
    assert win.art_main is None and 're-run GA' in BOXES[0][2]

    # Guard: data changed since the fit
    ga.do_run()
    assert win.ga_result_on_artsub is False
    win.ga_result_pipeline_sig = b'other'
    BOXES.clear()
    ga.apply_artifact_to_main()
    assert win.art_main is None and 'Data changed' in BOXES[0][2]

    # Window not covering t0 ± 3σ_art: confirmation; "No" applies nothing
    ga.close()
    ga = _ga(win, 0.0, 100.0)
    assert win.ga_result_art_ignored is None
    with _Question(QtWidgets.QMessageBox.No) as q:
        ga.apply_artifact_to_main()
    assert len(q.asked) == 1 and 'extrapolated' in q.asked[0]
    assert win.art_main is None and win.ga_result_B is not None
    with _Question(QtWidgets.QMessageBox.Yes) as q:
        ga.apply_artifact_to_main()
    assert win.art_main is not None
    ga.close()
    win.close()
    print('[ok] AC15 apply / revert / guards / D3 reset')


# =====================================================================
# AC16: auto-clear with notice
# =====================================================================
def _chirp_params():
    wl4 = np.linspace(470.0, 630.0, 6)
    pts = np.column_stack([wl4, 0.1 * ((550.0 / wl4) ** 2 - 1.0)])
    return ta_core.fit_chirp_params(pts)[0]


def _chirp_on(win):
    win.chirp_params = _chirp_params()
    win.chirp_applied = True
    win.update_all()


def _solvent_load(win, on):
    win.sub_irf_solv_wl = WL.copy()
    win.sub_irf_solv_t = T.copy()
    win.sub_irf_solv_data = 0.3 * (B_SYN @ ta_core.artifact_basis(
        T, 0.05, 0.12, 'xpm').T)
    win.sub_irf_scale = 1.0
    win.sub_irf_applied = on
    win.update_all()
    assert win.sub_irf_aligned is not None


def _bg_toggle(win):
    """An action that only calls update_all."""
    if win.bg_spectrum is None:
        win.bg_spectrum = np.nanmean(win.deltaA_raw[:, :5], axis=1)
        win.bg_n = 5
    win.bg_applied = not win.bg_applied
    win.update_all()


def _arm(win, prep=None):
    _load(win)
    if prep is not None:
        prep(win)
    win.apply_artifact_main(B_SYN, 0.05, 0.12, 'xpm')
    assert win.art_main is not None and not win._art_stale
    assert '[artifact-sub]' in win.lbl_status.text()


def _crosshair(win):
    win.selT = float(win.delay[60])
    assert win.selT != 0
    win.apply_zero_time_at_crosshair()


def _revert_dialog(win):
    dlg = ta_dialogs_a.CropDialog(win, win)
    dlg._revert()
    dlg.close()


def _bg_apply(win):
    win.bg_spectrum = np.nanmean(win.deltaA_raw[:, :5], axis=1)
    win.bg_n = 5
    win.bg_applied = True
    win.update_all()


def _set(attr, val):
    def op(win):
        setattr(win, attr, val)
        win.update_all()
    return op


OPS = [
    ('crop', None,
     lambda w: w.apply_crop_by_range(460.0, 640.0, -1.0, 500.0)),
    ('resample', None,
     lambda w: w.apply_crop_by_range(
         float(WL[0]), float(WL[-1]), float(T[0]), float(T[-1]),
         resample={'enabled': True, 'dx': 5.0, 'mode': 'average'})),
    ('t0 shift', None, lambda w: w.apply_zero_time_shift(0.01)),
    ('t0 at crosshair', None, _crosshair),
    ('chirp apply', None, _chirp_on),
    ('chirp revert', _chirp_on, lambda w: w.reset_corrections()),
    ('BG apply', None, _bg_apply),
    ('solvent on', lambda w: _solvent_load(w, False),
     _set('sub_irf_applied', True)),
    ('solvent off', lambda w: _solvent_load(w, True),
     _set('sub_irf_applied', False)),
    ('solvent scale', lambda w: _solvent_load(w, True),
     _set('sub_irf_scale', 0.5)),
    ('load', None, _load),
    ('Revert', None, _revert_dialog),
]


def test_auto_clear():
    _patch_boxes()
    win = ta_main.TAAnalyzer()
    for name, prep, op in OPS:
        _arm(win, prep)
        op(win)
        txt = win.lbl_status.text()
        assert win.art_main is None, name
        assert NOTICE in txt, (name, txt)
        if name == 't0 at crosshair':
            assert txt.startswith('t_0 shift applied'), txt
        else:
            assert DESC in txt, (name, txt)
        assert win.deltaA.shape == (len(win.wavelength), len(win.delay))
        assert np.array_equal(win.deltaA, win._pre_stage_data(),
                              equal_nan=True), name
        # Notice lifetime: gone after the next update_all-only action
        _bg_toggle(win)
        txt = win.lbl_status.text()
        assert NOTICE not in txt and DESC in txt, (name, txt)
        assert '[artifact-sub]' not in txt
        print(f'    {name}: cleared with notice, notice expired')
    win.close()
    print('[ok] AC16 auto-clear on', len(OPS), 'operations')


def test_off_mode_status_and_negatives():
    _patch_boxes()
    win = ta_main.TAAnalyzer()
    _load(win)
    win.update_all()
    # Off mode: other writers' lbl_status text survives update_all
    win.lbl_status.setText('X · saved crop_info')
    _bg_toggle(win)
    assert win.lbl_status.text() == 'X · saved crop_info'
    win.selT = float(win.delay[60])
    win.apply_zero_time_at_crosshair()
    msg = win.lbl_status.text()
    assert msg.startswith('t_0 shift applied') and '[' not in msg
    _bg_toggle(win)
    assert win.lbl_status.text() == msg
    assert win._art_status_dirty is False

    # Solvent preview sequence (ta_solvent_irf) keeps the stage
    _arm(win, lambda w: _solvent_load(w, True))
    sub = win.deltaA.copy()
    was_on = win.sub_irf_applied
    win.sub_irf_applied = False
    win.recompute()
    assert win._art_stale                       # preview skips the stage
    win.sub_irf_applied = was_on
    win.recompute()
    assert win.art_main is not None and not win._art_stale
    assert np.array_equal(win.deltaA, sub, equal_nan=True)
    win.update_all()
    assert win.art_main is not None
    win.close()
    print('[ok] AC16 Off-mode status preserved, solvent preview keeps stage')


if __name__ == '__main__':
    test_apply_revert()
    test_auto_clear()
    test_off_mode_status_and_negatives()
    print('\nAll artifact main-stage checks passed.')
