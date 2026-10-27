"""
Regression tests for the Crop dialog pin memory and Crop_information.csv.

Verifies:

  1. Pinned delays / wavelengths and the interpolation method survive
     closing and re-opening the Crop dialog; only loading new data
     clears them
  2. Removing an already-applied pin (or changing the method) rebuilds
     from the original, so the old interpolation does not stay baked in
  3. Apply writes Crop_information.csv next to the loaded data (λ / t
     range, resample or moving-average settings, method, pins with their
     original-grid index), overwriting it each time; data without a
     folder writes nothing
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import csv
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

import numpy as np  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402

import ta_main  # noqa: E402
import ta_dialogs_a  # noqa: E402

CSV_NAME = 'Crop_information.csv'


def _data(seed=0):
    rng = np.random.default_rng(seed)
    wl = np.linspace(400.0, 700.0, 151)
    t = np.linspace(-1.0, 20.0, 80)
    A = (np.outer(np.exp(-((wl - 550) / 60) ** 2), np.exp(-np.clip(t, 0, None) / 5))
         + 0.01 * rng.normal(size=(151, 80)))
    return wl, t, A


def _read(folder):
    with open(os.path.join(folder, CSV_NAME), encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))
    assert rows[0] == ['parameter', 'value', 'unit', 'index'], rows[0]
    return {r[0]: r[1:] for r in rows[1:]}


def _pin(dlg, t_idx=None, wl_idx=None):
    if t_idx is not None:
        dlg._set_selection(t_idx=t_idx)
        dlg._pin_current_as_drop()
    if wl_idx is not None:
        dlg._set_selection(wl_idx=wl_idx)
        dlg._pin_current_kin_overlay()


def test_memory_and_csv():
    wl, t, A = _data()
    with tempfile.TemporaryDirectory() as tmp:
        win = ta_main.TAAnalyzer()
        win.set_loaded_data(wl, t, A, 'Loaded: mem.csv', source_dir=tmp)

        # 1. Pins + method survive close / re-open
        dlg = ta_dialogs_a.CropDialog(win, win)
        assert dlg._drop_t_values == [] and dlg._kin_overlay_wl == []
        _pin(dlg, t_idx=40, wl_idx=60)
        dlg.dd_interp_method.setCurrentText('pchip')
        dlg.close()
        assert not os.path.exists(os.path.join(tmp, CSV_NAME))  # no Apply
        dlg = ta_dialogs_a.CropDialog(win, win)
        assert dlg._drop_t_values == [float(t[40])]
        assert dlg._kin_overlay_wl == [float(wl[60])]
        assert dlg.dd_interp_method.currentText() == 'pchip'
        assert dlg.tbl_drops.rowCount() == 1
        assert dlg.tbl_drops_wl.rowCount() == 1
        assert len(dlg._drop_lines) == 1 and len(dlg._kin_overlay_lines) == 1
        assert dlg._preview_deltaA is not None                # preview rebuilt

        # 3. Apply (crop + moving average) writes the CSV
        dlg.ed_wl_min.setValue(450.0)
        dlg.ed_wl_max.setValue(650.0)
        dlg.ed_t_min.setValue(-0.5)
        dlg.ed_t_max.setValue(15.0)
        dlg.cb_resample.setChecked(True)
        dlg.rb_mov.setChecked(True)
        dlg.ed_window.setValue(7)
        dlg.dd_kernel.setCurrentIndex(dlg.dd_kernel.findData('gaussian'))
        dlg._apply()
        info = _read(tmp)
        assert info['source'][0] == 'Loaded: mem.csv'
        assert float(info['wavelength_min'][0]) == 450.0
        assert info['wavelength_max'][:2] == ['650', 'nm']
        assert float(info['t_min'][0]) == -0.5 and info['t_max'][1] == 'ps'
        assert info['resample_smooth_enabled'][0] == 'True'
        assert info['resample_mode'][0] == 'moving'
        assert info['moving_window'][:2] == ['7', 'px']
        assert info['moving_kernel'][0] == 'gaussian'
        assert info['resample_applied'][0] == 'True'
        assert 'resample_dx' not in info
        assert info['interpolation_method'][0] == 'pchip'
        assert info['n_pinned_delays'][0] == '1'
        assert float(info['pinned_delay_1'][0]) == float(f'{t[40]:.6g}')
        assert info['pinned_delay_1'][2] == '40'
        assert info['pinned_wavelength_1'][1:] == ['nm', '60']
        assert info['original_points'][0] == '151 x 80'
        assert info['result_points'][0] == \
            f'{len(win.wavelength)} x {len(win.delay)}'
        assert win.lbl_status.text().count(CSV_NAME) == 1

        # Rejected crop (range between two grid points): nothing recorded
        before = (win.crop_applied, open(os.path.join(tmp, CSV_NAME),
                                         encoding='utf-8-sig').read())
        dlg = ta_dialogs_a.CropDialog(win, win)
        dlg.ed_wl_min.setValue(500.5)
        dlg.ed_wl_max.setValue(501.0)
        warned = []
        mocks = (ta_main, ta_dialogs_a)
        origs = [m.warn_box for m in mocks]
        for m in mocks:
            m.warn_box = lambda *a, **k: warned.append(a[1])
        try:
            dlg._apply()
        finally:
            for m, o in zip(mocks, origs):
                m.warn_box = o
        assert 'Invalid range' in warned, warned
        after = (win.crop_applied, open(os.path.join(tmp, CSV_NAME),
                                        encoding='utf-8-sig').read())
        assert after == before
        dlg.close()

        # Pins persist after Apply as well
        dlg = ta_dialogs_a.CropDialog(win, win)
        assert dlg._drop_t_values == [float(t[40])]

        # Bin resampling + a too-small Δλ: falls back, recorded as such
        dlg.rb_avg.setChecked(True)
        dlg.ed_resample_dx.setValue(0.5)                 # < 2 nm spacing
        orig_warn = ta_dialogs_a.warn_box
        ta_dialogs_a.warn_box = lambda *a, **k: None
        try:
            dlg._apply()
        finally:
            ta_dialogs_a.warn_box = orig_warn
        info = _read(tmp)                                # overwritten
        assert info['resample_mode'][0] == 'average'
        assert info['resample_dx'][:2] == ['0.5', 'nm']
        assert info['resample_applied'][0] == 'False'
        assert 'moving_window' not in info

        # 2. Removing an applied pin restores the original values
        dlg = ta_dialogs_a.CropDialog(win, win)
        dlg.cb_resample.setChecked(False)
        dlg._apply()
        i_t = int(np.argmin(np.abs(win.delay - t[40])))
        i_w = int(np.argmin(np.abs(win.wavelength - wl[100])))
        assert not np.isclose(win.deltaA[i_w, i_t], A[100, 40])  # interpolated
        dlg = ta_dialogs_a.CropDialog(win, win)
        dlg.tbl_drops.selectRow(0)
        dlg._drop_remove()
        assert win.crop_drop_t == []                      # stored at once
        dlg._apply()
        assert np.isclose(win.deltaA[i_w, i_t], A[100, 40])
        assert _read(tmp)['n_pinned_delays'][0] == '0'
        # Method change of an applied λ pin also rebuilds
        i_w60 = int(np.argmin(np.abs(win.wavelength - wl[60])))
        before = win.deltaA[i_w60, :].copy()
        dlg = ta_dialogs_a.CropDialog(win, win)
        dlg.dd_interp_method.setCurrentText('linear')
        dlg._apply()
        assert not np.allclose(win.deltaA[i_w60, :], before)
        assert win.crop_applied[2] == 'linear'

        # 1b. New data clears the memory
        win.set_loaded_data(wl, t, A, 'Loaded: other.csv', source_dir=tmp)
        assert win.crop_drop_t == [] and win.crop_drop_wl == []
        dlg = ta_dialogs_a.CropDialog(win, win)
        assert dlg._drop_t_values == [] and dlg._kin_overlay_wl == []
        assert dlg.tbl_drops.rowCount() == 0
        dlg.close()
        win.close()
    print('[ok] pin memory + Crop_information.csv')


def test_no_folder_no_file():
    wl, t, A = _data(1)
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            win = ta_main.TAAnalyzer()
            win.set_loaded_data(wl, t, A, 'synthetic')     # no source_dir
            dlg = ta_dialogs_a.CropDialog(win, win)
            dlg.ed_wl_min.setValue(450.0)
            dlg._apply()
            assert os.listdir(tmp) == []
            win.close()
        finally:
            os.chdir(cwd)
    print('[ok] no data folder → nothing written')


TESTS = [test_memory_and_csv, test_no_folder_no_file]


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    for fn in TESTS:
        fn()
    print(f'\nAll {len(TESTS)} crop memory / info checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
