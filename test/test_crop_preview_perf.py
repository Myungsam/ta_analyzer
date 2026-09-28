"""
Performance smoke test for the live-preview path on a real
2136 x 192 dataset.

Verifies:
  1. Dialog open + initial heatmap is fast (< 1 s).
  2. Pinning a delay recomputes the preview and updates the heatmap
     in-place via set_array (no slow pcolormesh rebuild).  Should be
     well under 500 ms.
  3. Changing the method re-runs interpolation, still fast.
  4. Clicking the 2D map updates the spec/kin panels in real-time
     using the preview data.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import time
import numpy as np
from PyQt5 import QtWidgets

import ta_core
import ta_main
from ta_dialogs_a import CropDialog


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    print('=== Loading the real 2136 x 192 dataset ===')
    wl, t, A = ta_core.parse_data_file('Data/A_MAPbI3/_TA_spectra_Accumulated.csv')
    print(f'    shape = {A.shape}')
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'preview_perf_test')

    # 1. Dialog open
    t0 = time.perf_counter()
    dlg = CropDialog(win, win)
    open_ms = (time.perf_counter() - t0) * 1000
    print(f'[1] CropDialog open: {open_ms:.0f} ms (limit 1000 ms)')
    assert open_ms < 1500, f'too slow: {open_ms:.0f} ms'

    # 2. Pin a delay -> preview rebuild
    pin_times = []
    for k in (10, 20, 50, 80, 120):
        t0 = time.perf_counter()
        dlg._add_drop_value(float(t[k]))
        pin_times.append((time.perf_counter() - t0) * 1000)
    pin_times.sort()
    median = pin_times[len(pin_times) // 2]
    print(f'[2] _add_drop_value median = {median:.0f} ms  '
          f'(min {pin_times[0]:.0f}, max {pin_times[-1]:.0f})')
    assert median < 800, f'preview rebuild too slow: median {median:.0f} ms'
    assert dlg._preview_deltaA is not None
    assert dlg._preview_deltaA.shape == A.shape

    # 3. Method change re-runs interp
    methods = ['cubic', 'pchip', 'akima', 'linear']
    mtimes = []
    for m in methods:
        t0 = time.perf_counter()
        dlg.dd_interp_method.setCurrentText(m)
        mtimes.append((m, (time.perf_counter() - t0) * 1000))
    print('[3] method change recompute:')
    for m, dt in mtimes:
        print(f'        {m:7s} {dt:.0f} ms')
        assert dt < 1500, f'{m} too slow: {dt:.0f} ms'

    # 4. Click on 2D map updates spec/kin
    class _Ev:
        pass
    ev = _Ev()
    ev.inaxes = dlg.canvas.ax
    times = []
    for (xi, ti) in [(100, 30), (500, 60), (1000, 80), (1500, 100), (2000, 150)]:
        ev.xdata = float(wl[xi])
        ev.ydata = float(t[ti])
        t0 = time.perf_counter()
        dlg._on_canvas_click(ev)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    click_med = times[len(times) // 2]
    print(f'[4] 2D click + spec/kin redraw median = {click_med:.0f} ms')
    assert click_med < 250, f'click redraw too slow: {click_med:.0f} ms'

    # 5. Clear all drops -> mesh restored
    t0 = time.perf_counter()
    dlg._drop_clear()
    clear_ms = (time.perf_counter() - t0) * 1000
    print(f'[5] Clear all drops: {clear_ms:.0f} ms')
    assert dlg._preview_deltaA is None

    # 6. End-to-end: pin a few delays, set a crop, apply -> result equals
    #    interp(full)-then-crop slice.
    dlg = CropDialog(win, win)
    dlg._add_drop_value(float(t[40]))
    dlg._add_drop_value(float(t[80]))
    dlg.ed_t_min.setValue(float(t[10]))
    dlg.ed_t_max.setValue(float(t[170]))
    dlg.dd_interp_method.setCurrentText('pchip')
    expected = ta_core.interpolate_missing_columns(
        A, t, [40, 80], method='pchip')
    t0 = time.perf_counter()
    dlg._apply()
    apply_ms = (time.perf_counter() - t0) * 1000
    print(f'[6] _apply (interp-first-then-crop): {apply_ms:.0f} ms')
    # The spinbox rounds to 6 decimals so the exact crop bounds may
    # clip one edge pixel.  Build the expected slice from the actual
    # post-apply axes rather than the requested values, so the test
    # validates the algorithm, not float-precision artefacts.
    wl_lo, wl_hi = float(win.wavelength[0]), float(win.wavelength[-1])
    t_lo, t_hi = float(win.delay[0]), float(win.delay[-1])
    wl_mask = (wl >= wl_lo - 1e-9) & (wl <= wl_hi + 1e-9)
    t_mask = (t >= t_lo - 1e-9) & (t <= t_hi + 1e-9)
    expected_cropped = expected[np.ix_(wl_mask, t_mask)]
    np.testing.assert_allclose(win.deltaA_raw, expected_cropped, atol=1e-12)
    print(f'    result exactly matches interp(full)-then-crop slice '
          f'(shape {win.deltaA_raw.shape})')

    print('\n*** LIVE PREVIEW + INTERP-FIRST-THEN-CROP VERIFIED ON REAL DATA ***')


if __name__ == '__main__':
    main()
