"""
Tests for the new ``Export 2D data_original...`` button.

Verifies:
    1) ``export_2d_data_original`` writes ``deltaA_raw`` (cropped,
       pre-processing data) — *not* the corrected ``deltaA``.
    2) Even when BG subtraction is active, the original export still
       reflects ``deltaA_raw``.
    3) Refuses to open a save dialog when no data is loaded.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import tempfile

import numpy as np
from PyQt5 import QtWidgets


_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)


def _make_window():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    import ta_main
    rng = np.random.default_rng(11)
    wl = np.linspace(400.0, 700.0, 30)
    t = np.linspace(-0.5, 5.0, 50)
    env = np.exp(-((wl - 525.0) / 60.0) ** 2)
    osc = np.exp(-0.4 * np.clip(t, 0, None))
    A = np.outer(env, osc) + 0.005 * rng.standard_normal((30, 50))
    w = ta_main.TAAnalyzer()
    w.set_loaded_data(wl, t, A, 'export_orig_test.csv')
    return w


def test_export_original_writes_raw_matrix():
    import ta_core, ta_main
    w = _make_window()

    # Apply a BG correction so deltaA differs from deltaA_raw.
    bg = 0.1 * np.cos(0.05 * w.wavelength)
    w.bg_spectrum = bg
    w.bg_applied = True
    w.bg_n = 3
    w.recompute()
    # Sanity: corrected matrix must differ from the raw one.
    assert not np.allclose(w.deltaA, w.deltaA_raw)

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'orig.csv')
        orig_dlg = QtWidgets.QFileDialog.getSaveFileName
        orig_info = ta_main.info_box
        QtWidgets.QFileDialog.getSaveFileName = staticmethod(
            lambda *a, **kw: (out, 'CSV (*.csv)'))
        ta_main.info_box = lambda *a, **kw: None
        try:
            w.export_2d_data_original()
        finally:
            QtWidgets.QFileDialog.getSaveFileName = orig_dlg
            ta_main.info_box = orig_info

        assert os.path.isfile(out), 'export did not write the file'
        wl_r, t_r, A_r = ta_core.parse_data_file(out)
        np.testing.assert_allclose(wl_r, w.wavelength)
        np.testing.assert_allclose(t_r, w.delay)
        # The file must reflect deltaA_raw, NOT deltaA.
        np.testing.assert_allclose(A_r, w.deltaA_raw, rtol=1e-5, atol=1e-6)
        # And explicitly must NOT match the BG-corrected matrix.
        assert not np.allclose(A_r, w.deltaA, atol=1e-6)

    w.close()
    print('[orig vs corrected]  OK')


def test_export_original_uses_post_crop_axes():
    """If the user has cropped, the export uses the post-crop axes
    (``self.wavelength`` / ``self.delay``), not the original ones."""
    import ta_core, ta_main
    w = _make_window()

    # Simulate a crop by trimming the working axes + deltaA_raw.
    wl_mask = (w.wavelength >= 450) & (w.wavelength <= 650)
    t_mask = (w.delay >= 0.0)
    w.wavelength = w.wavelength[wl_mask].copy()
    w.delay = w.delay[t_mask].copy()
    w.deltaA_raw = w.deltaA_raw[np.ix_(wl_mask, t_mask)].copy()
    w.recompute()
    assert w.is_cropped(), 'crop helper should report cropped state'

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'orig_crop.csv')
        orig_dlg = QtWidgets.QFileDialog.getSaveFileName
        orig_info = ta_main.info_box
        QtWidgets.QFileDialog.getSaveFileName = staticmethod(
            lambda *a, **kw: (out, 'CSV (*.csv)'))
        ta_main.info_box = lambda *a, **kw: None
        try:
            w.export_2d_data_original()
        finally:
            QtWidgets.QFileDialog.getSaveFileName = orig_dlg
            ta_main.info_box = orig_info

        assert os.path.isfile(out)
        wl_r, t_r, A_r = ta_core.parse_data_file(out)
        np.testing.assert_allclose(wl_r, w.wavelength)
        np.testing.assert_allclose(t_r, w.delay)
        np.testing.assert_allclose(A_r, w.deltaA_raw, rtol=1e-5, atol=1e-6)
    w.close()
    print('[cropped axes]       OK')


def test_export_original_refuses_without_data():
    import ta_main
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = ta_main.TAAnalyzer()
    # No set_loaded_data() — deltaA_raw is None.
    assert w.deltaA_raw is None

    called = {'asked': False}

    def _ask(*a, **kw):
        called['asked'] = True
        return ('/should/not/run.csv', '')

    orig_dlg = QtWidgets.QFileDialog.getSaveFileName
    orig_warn = ta_main.warn_box
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(_ask)
    ta_main.warn_box = lambda *a, **kw: None
    try:
        w.export_2d_data_original()
    finally:
        QtWidgets.QFileDialog.getSaveFileName = orig_dlg
        ta_main.warn_box = orig_warn

    assert called['asked'] is False, \
        'No data -> no save dialog should be opened'
    w.close()
    print('[no-data]            OK')


def main():
    test_export_original_writes_raw_matrix()
    test_export_original_uses_post_crop_axes()
    test_export_original_refuses_without_data()
    print('\nAll Export-2D-Original tests passed.')


if __name__ == '__main__':
    main()
