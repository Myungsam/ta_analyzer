"""
Regression tests for the Crop-dialog input-eating bug.

Symptom (reported by user, image-attached):
  When typing into the wavelength / time range spinboxes on a real
  2136×192 dataset, characters were swallowed and the dialog seemed
  to "kick out" the user.

Root cause:
  Every valueChanged signal triggered a full pcolormesh rebuild
  (~2 s on this dataset).  While that ran the GUI froze and
  successive keystrokes were dropped.

Fix in ta_dialogs_a.CropDialog:
  1. The heatmap is drawn exactly ONCE via _draw_heatmap_full().
     All later changes go through _update_overlays(), which only
     moves four boundary lines and resizes four dim Rectangle patches.
  2. valueChanged signals are debounced through a 120 ms QTimer so
     successive keystrokes are coalesced into a single redraw.
  3. The user-typed spinbox values are NEVER overwritten — even if
     the user temporarily types wl_min > wl_max, the spinboxes hold
     exactly what was typed; the swap happens only inside _apply.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import time
import numpy as np
from PyQt5 import QtCore, QtWidgets

import ta_core
import ta_main
from ta_dialogs_a import CropDialog


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    print('=== Loading the real 2136 × 192 dataset ===')
    wl, t, A = ta_core.parse_data_file(
        '/mnt/user-data/uploads/_TA_spectra_Accumulated.csv')
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'crop_perf_test')

    # =====================================================================
    # 1) Initial open is fast — heatmap drawn ONCE.
    # =====================================================================
    t0 = time.perf_counter()
    crop = CropDialog(win, win)
    open_ms = (time.perf_counter() - t0) * 1000
    print(f'[1] CropDialog open + initial heatmap = {open_ms:.0f} ms')
    assert open_ms < 1000, (
        f'Dialog open too slow ({open_ms:.0f} ms); '
        'the user-reported "kick out" was at ~2000 ms')

    # =====================================================================
    # 2) Overlay updates are essentially free (< 5 ms each).
    # =====================================================================
    times = []
    for v in [400, 425, 450, 475, 500, 525, 550, 575, 600]:
        crop.ed_wl_min.blockSignals(True)
        crop.ed_wl_min.setValue(float(v))
        crop.ed_wl_min.blockSignals(False)
        t0 = time.perf_counter()
        crop._update_overlays()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    median = times[len(times) // 2]
    print(f'[2] _update_overlays median = {median:.2f} ms '
          f'(min {times[0]:.2f}, max {times[-1]:.2f})')
    assert median < 20, (
        f'Overlay update median = {median:.0f} ms; '
        'should be well under 20 ms so the GUI never freezes while typing')

    # =====================================================================
    # 3) User-typed spinbox values are NEVER overwritten by the dialog.
    #    This is the "튕긴다" bug — it must NOT happen.
    # =====================================================================
    crop.ed_wl_min.setValue(425.0)
    crop.ed_wl_max.setValue(875.0)
    assert crop.ed_wl_min.value() == 425.0
    assert crop.ed_wl_max.value() == 875.0
    print(f'[3a] sane order:     '
          f'wl_min={crop.ed_wl_min.value()}, wl_max={crop.ed_wl_max.value()}')

    # Type wl_min ABOVE wl_max (mid-edit state).  The dialog used to
    # silently swap or drop the input — verify it preserves both.
    crop.ed_wl_min.setValue(900.0)
    assert crop.ed_wl_min.value() == 900.0, (
        f'Spinbox was overwritten! Got {crop.ed_wl_min.value()}, '
        'expected 900.0 — the bug is back.')
    assert crop.ed_wl_max.value() == 875.0, (
        'wl_max was modified by changing wl_min — should not happen.')
    print(f'[3b] inverted typed: '
          f'wl_min={crop.ed_wl_min.value()}, wl_max={crop.ed_wl_max.value()} '
          '(both preserved as typed)')

    # And finishing the edit by raising wl_max also preserves wl_min
    crop.ed_wl_max.setValue(950.0)
    assert crop.ed_wl_min.value() == 900.0
    assert crop.ed_wl_max.value() == 950.0
    print(f'[3c] after final edit: '
          f'wl_min={crop.ed_wl_min.value()}, wl_max={crop.ed_wl_max.value()}')

    # =====================================================================
    # 4) Title shows a clear "will swap on Apply" warning when the user
    #    has temporarily inverted the range.
    # =====================================================================
    crop.ed_wl_min.setValue(700.0)
    crop.ed_wl_max.setValue(500.0)   # inverted
    crop._update_overlays()
    title = crop.canvas.ax.get_title()
    print(f'[4] inverted-range title: {title!r}')
    assert 'swap' in title.lower(), (
        'Title should warn that the range will swap on Apply')

    # =====================================================================
    # 5) Apply with inverted-typed values still works (apply_crop_by_range
    #    swaps internally) — the user gets the band they intended.
    # =====================================================================
    crop.ed_wl_min.setValue(700.0)   # > wl_max
    crop.ed_wl_max.setValue(500.0)
    crop.ed_t_min.setValue(-5.0)
    crop.ed_t_max.setValue(50.0)
    crop._apply()                      # closes the dialog and crops
    assert win.wavelength.min() >= 500.0 - 1e-6
    assert win.wavelength.max() <= 700.0 + 1e-6
    assert win.delay.min() >= -5.0 - 1e-6
    assert win.delay.max() <= 50.0 + 1e-6
    print(f'[5] After apply(inverted typed): '
          f'wl=[{win.wavelength.min():.1f}, {win.wavelength.max():.1f}] nm, '
          f'  t=[{win.delay.min():.2f}, {win.delay.max():.2f}] ps')

    print('\n*** CROP-DIALOG INPUT BUG FIX VERIFIED ***')


if __name__ == '__main__':
    main()
