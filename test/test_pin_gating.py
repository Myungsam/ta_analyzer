"""
Regression tests for spectrum / kinetics gating ("Integrate" rows).

Verifies:

  1. ta_core.delay_gate / wavelength_gate / gated_mean
        - clipped ±idx window, BW/2 band with edge pixels kept,
          narrow BW falling back to the nearest pixel, NaN-aware mean

  2. Main window
        - current spectrum = mean over idx ± N, legend 't = mean ± std'
        - current kinetics = mean over |λ - λc| <= BW/2
        - indicators show window / center / point count, 'off' when off
        - pins keep their own window; unchecked pins stay single traces
        - Export pins writes the integrated columns with tagged headers
        - crop drops the windows of clipped pins
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


def _window():
    rng = np.random.default_rng(3)
    wl = np.linspace(400.0, 700.0, 151)          # 2 nm pixels
    t = np.linspace(-1.0, 10.0, 56)
    A = rng.normal(size=(151, 56))
    w = ta_main.TAAnalyzer()
    w.set_loaded_data(wl, t, A, 'Loaded: gating')
    return w


def _current_spec(w):
    return np.asarray(w._spec_line_current.get_ydata())


def _current_kin(w):
    return np.asarray(w._kin_line_current[0].get_ydata())


def _export(w, fn):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'pins.csv')
        orig_ask, orig_info = ta_main.ask_save_path, ta_main.info_box
        ta_main.ask_save_path = lambda *a, **kw: (out, '')
        ta_main.info_box = lambda *a, **kw: None
        try:
            fn()
        finally:
            ta_main.ask_save_path, ta_main.info_box = orig_ask, orig_info
        with open(out, encoding='utf-8') as f:
            header = f.readline().strip().split(',')
        return header, np.loadtxt(out, delimiter=',', skiprows=1)


def test_core_helpers():
    assert ta_core.delay_gate(10, 5, 2) == (3, 7)
    assert ta_core.delay_gate(10, 1, 3) == (0, 4)
    assert ta_core.delay_gate(10, 9, 3) == (6, 9)
    assert ta_core.delay_gate(10, 4, 0) == (4, 4)
    wl = np.arange(400.0, 420.0, 2.0)            # 400 … 418
    sel = ta_core.wavelength_gate(wl, 410.0, 4.0)
    assert list(wl[sel]) == [408.0, 410.0, 412.0], wl[sel]
    sel = ta_core.wavelength_gate(wl, 410.0, 0.5)
    assert list(wl[sel]) == [410.0]
    sel = ta_core.wavelength_gate(wl, 411.0, 0.5)   # between pixels
    assert sel.sum() == 1
    A = np.array([[1.0, np.nan], [3.0, np.nan]])
    m = ta_core.gated_mean(A, slice(0, 2), axis=0)
    assert m[0] == 2.0 and np.isnan(m[1])
    print('[ok] core helpers')


def test_spectrum_gating():
    w = _window()
    w.set_sel_t(w.delay[20])
    single = w.deltaA[:, 20].copy()
    assert np.allclose(_current_spec(w), single)
    assert w.lbl_spec_int.text() == 'off'
    w.ed_spec_int_half.setValue(3)
    w.chk_spec_int.setChecked(True)
    assert np.allclose(_current_spec(w), w.deltaA[:, 17:24].mean(axis=1))
    t = w.delay[17:24]
    lbl = w._spec_line_current.get_label()
    assert f'{t.mean():.3g} ± {t.std():.2g}' in lbl and 'idx 17–23' in lbl, lbl
    txt = w.lbl_spec_int.text()
    assert 'idx 17–23' in txt and '7 pts' in txt and 'center' in txt, txt
    # Window clipped at the start of the delay array
    w.ed_sel_t_idx.setValue(1)
    assert np.allclose(_current_spec(w), w.deltaA[:, 0:5].mean(axis=1))
    assert 'idx 0–4' in w.lbl_spec_int.text()
    w.chk_spec_int.setChecked(False)
    assert np.allclose(_current_spec(w), w.deltaA[:, 1])
    assert w.lbl_spec_int.text() == 'off'
    w.close()
    print('[ok] spectrum gating')


def test_kinetics_gating():
    w = _window()
    w.set_sel_wl(550.0)
    i = int(np.argmin(np.abs(w.wavelength - 550.0)))
    assert np.allclose(_current_kin(w), w.deltaA[i, :])
    w.ed_kin_int_bw.setValue(10.0)
    w.chk_kin_int.setChecked(True)
    band = np.abs(w.wavelength - 550.0) <= 5.0 + 1e-9
    assert band.sum() == 5                    # 546 … 554 nm
    assert np.allclose(_current_kin(w), w.deltaA[band, :].mean(axis=0))
    txt = w.lbl_kin_int.text()
    assert 'center 550.0 nm' in txt and f'{band.sum()} pts' in txt, txt
    assert 'BW 10 nm' in w._kin_line_current[0].get_label()
    # Split mode slices the same integrated trace
    w.delay_scale_mode = 'split'
    w.split_threshold = 1.0
    w._draw_kinetics()
    full = w.deltaA[band, :].mean(axis=0)
    got = np.concatenate([np.asarray(l.get_ydata())
                          for l in w._kin_line_current])
    want = np.concatenate([full[w.delay <= 1.0], full[w.delay >= 1.0]])
    assert np.allclose(got, want)
    w.close()
    print('[ok] kinetics gating')


def test_pins_and_export():
    w = _window()
    # Spectrum: one single pin, one integrated pin
    w.set_sel_t(w.delay[10])
    w.pin_spec()
    w.ed_spec_int_half.setValue(2)
    w.chk_spec_int.setChecked(True)
    w.set_sel_t(w.delay[30])
    w.pin_spec()
    assert w.specOverlayHalf == {float(w.delay[30]): 2}
    ov = [np.asarray(h.get_ydata()) for h in w._spec_lines_overlay]
    assert np.allclose(ov[0], w.deltaA[:, 10])
    assert np.allclose(ov[1], w.deltaA[:, 28:33].mean(axis=1))
    header, data = _export(w, w.export_spec_overlays)
    assert header[1] == f'dA_t={w.delay[10]:.6g}ps', header
    assert header[2] == f'dA_t={w.delay[30]:.6g}ps_avg_idx28-32', header
    assert np.allclose(data[:, 2], w.deltaA[:, 28:33].mean(axis=1),
                       atol=1e-5)
    # Kinetics: integrated pin
    w.ed_kin_int_bw.setValue(6.0)
    w.chk_kin_int.setChecked(True)
    w.set_sel_wl(500.0)
    w.pin_kin()
    band = np.abs(w.wavelength - 500.0) <= 3.0 + 1e-9
    assert np.allclose(np.asarray(w._kin_lines_overlay[0][0].get_ydata()),
                       w.deltaA[band, :].mean(axis=0))
    header, data = _export(w, w.export_kin_overlays)
    assert header[1] == 'dA_wl=500nm_BW6nm', header
    assert np.allclose(data[:, 1], w.deltaA[band, :].mean(axis=0), atol=1e-5)
    # Crop that removes the integrated spectrum pin drops its window
    w.apply_crop_by_range(w.wavelength[0], w.wavelength[-1],
                          w.delay[0], w.delay[20])
    assert w.specOverlayHalf == {}
    assert w.kinOverlayBW == {500.0: 6.0}     # λ pin untouched
    w.clear_kin_overlays()
    assert w.kinOverlayBW == {}
    w.close()
    print('[ok] pins + export')


TESTS = [test_core_helpers, test_spectrum_gating, test_kinetics_gating,
         test_pins_and_export]


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    for fn in TESTS:
        fn()
    print(f'\nAll {len(TESTS)} gating checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
