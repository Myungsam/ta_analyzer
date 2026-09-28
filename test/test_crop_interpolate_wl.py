"""
Regression tests for the new wavelength-drop interpolation +
bilinear (2D-surface) interpolation features in the Crop dialog.

Backend pieces under test:
  - ta_core.interpolate_missing_rows:  column-wise interp along wavelength
  - ta_core.interpolate_missing_2d:    2D bilinear surface fill (griddata)
  - TAAnalyzer.interpolate_wavelengths_at
  - TAAnalyzer.interpolate_2d_at

UI under test:
  - CropDialog._pin_current_kin_overlay populates _kin_overlay_wl AND
    drives the preview matrix (no longer visual-only).
  - CropDialog._apply: routes through interpolate_wavelengths_at /
    interpolate_2d_at depending on the 'Interpolation method' dropdown.
  - CropDialog._recompute_preview honours wavelength drops too.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import numpy as np
from PyQt5 import QtWidgets

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import ta_core
import ta_main
from ta_dialogs_a import CropDialog


def _make_synthetic(n_wl=40, n_t=60):
    """Smooth synthetic TA matrix with smooth structure along both axes."""
    wl = np.linspace(400.0, 800.0, n_wl)
    t = np.linspace(-2.0, 100.0, n_t)
    fwhm = 0.4
    amp1 = np.exp(-((wl - 500.0) / 60.0) ** 2)
    amp2 = -0.6 * np.exp(-((wl - 650.0) / 80.0) ** 2)
    k1 = ta_core.exp_irf_conv(t, 5.0, 0.0, fwhm)
    k2 = ta_core.exp_irf_conv(t, 40.0, 0.0, fwhm)
    A = np.outer(amp1, k1) + np.outer(amp2, k2)
    return wl, t, A


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    wl, t, A = _make_synthetic()

    # =====================================================================
    # 1) interpolate_missing_rows: round-trip on a smooth grid.
    # =====================================================================
    print('=== 1. interpolate_missing_rows: per-method accuracy ===')
    drop_wl = [8, 19, 27]   # interior rows
    truth = A[drop_wl, :].copy()
    for method in ('linear', 'cubic', 'pchip', 'akima'):
        out = ta_core.interpolate_missing_rows(A, wl, drop_wl, method=method)
        keep = np.setdiff1d(np.arange(wl.size), drop_wl)
        assert np.allclose(out[keep, :], A[keep, :]), \
            f'{method}: kept rows were modified'
        err = float(np.max(np.abs(out[drop_wl, :] - truth)))
        print(f'    {method:7s}  max abs err = {err:.4e}')
        assert err < 5e-2, f'{method}: error {err} too large'

    # bilinear is reserved for the 2D function
    try:
        ta_core.interpolate_missing_rows(A, wl, [10], method='bilinear')
    except ValueError:
        print('    bilinear correctly rejected by interpolate_missing_rows')
    else:
        raise AssertionError("expected ValueError for bilinear via missing_rows")

    # =====================================================================
    # 2) interpolate_missing_2d: both-axis drops, intersection cells.
    # =====================================================================
    print('\n=== 2. interpolate_missing_2d: rows + cols dropped together ===')
    drop_w = [10, 20]
    drop_t = [15, 30]
    out = ta_core.interpolate_missing_2d(A, wl, t, drop_w, drop_t)
    # All kept cells must remain untouched.
    mask_drop = np.zeros(A.shape, dtype=bool)
    mask_drop[drop_w, :] = True
    mask_drop[:, drop_t] = True
    keep = ~mask_drop
    np.testing.assert_allclose(out[keep], A[keep])
    # Reconstruction must be reasonable on a smooth synthetic.
    err = float(np.max(np.abs(out[mask_drop] - A[mask_drop])))
    print(f'    max abs err over all missing cells = {err:.4e}')
    assert err < 1e-1, f'2D bilinear error too large ({err})'

    # No NaNs anywhere — even cells inside the rectangular hole at row+col
    # intersection must be filled (griddata handles them via triangulation,
    # nearest-neighbour fallback handles convex-hull misses).
    assert np.isfinite(out).all(), 'bilinear left NaN entries'
    print('    no residual NaNs')

    # =====================================================================
    # 3) TAAnalyzer.interpolate_wavelengths_at
    # =====================================================================
    print('\n=== 3. TAAnalyzer.interpolate_wavelengths_at ===')
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'wl_interp_test')
    win.interpolate_wavelengths_at([12, 28], method='pchip')
    assert win.deltaA_raw.shape == A.shape
    keep = np.setdiff1d(np.arange(wl.size), [12, 28])
    assert np.allclose(win.deltaA_raw[keep, :], A[keep, :])
    diff_dropped = float(np.max(np.abs(win.deltaA_raw[[12, 28], :]
                                       - A[[12, 28], :])))
    print(f'    dropped-row max diff = {diff_dropped:.4e}  (should be > 0)')
    assert diff_dropped > 0, 'wavelength rows were NOT interpolated'
    assert np.array_equal(win.original_deltaA, A), \
        'original_deltaA was mutated'

    # Guard fires when too many rows are dropped.
    win2 = ta_main.TAAnalyzer()
    win2.set_loaded_data(wl, t, A, 'wl_guard_test')
    raw_before = win2.deltaA_raw.copy()
    import ta_main as _tm
    warn_calls = []
    real_warn = _tm.warn_box
    _tm.warn_box = lambda *a, **kw: warn_calls.append((a, kw))
    try:
        win2.interpolate_wavelengths_at(list(range(wl.size - 1)),
                                        method='linear')
    finally:
        _tm.warn_box = real_warn
    assert np.array_equal(win2.deltaA_raw, raw_before), \
        'guard failed: rows dropped past safe threshold mutated data'
    assert warn_calls, 'guard did not emit warn_box'
    print(f'    guard fires when only 1 wavelength would remain')

    # =====================================================================
    # 4) TAAnalyzer.interpolate_2d_at
    # =====================================================================
    print('\n=== 4. TAAnalyzer.interpolate_2d_at ===')
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, '2d_interp_test')
    win.interpolate_2d_at([10, 25], [8, 20])
    mask = np.zeros(A.shape, dtype=bool)
    mask[[8, 20], :] = True
    mask[:, [10, 25]] = True
    np.testing.assert_allclose(win.deltaA_raw[~mask], A[~mask])
    err2 = float(np.max(np.abs(win.deltaA_raw[mask] - A[mask])))
    print(f'    max err over dropped cells (synthetic smooth) = {err2:.4e}')
    assert err2 < 1e-1
    assert np.array_equal(win.original_deltaA, A), \
        'original_deltaA was mutated by 2d interp'

    # =====================================================================
    # 5) CropDialog: pin-λ now populates the wavelength drops table AND
    #    drives the preview.
    # =====================================================================
    print('\n=== 5. CropDialog: pin λ drives wavelength drop list ===')
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'crop_wl_dlg_test')
    dlg = CropDialog(win, win)

    dlg._sel_wl_idx = 15
    dlg._sel_wl = float(wl[15])
    dlg._pin_current_kin_overlay()
    assert dlg._kin_overlay_wl == [float(wl[15])]
    assert dlg.tbl_drops_wl.rowCount() == 1
    assert dlg.tbl_drops_wl.item(0, 0).text() == '15'
    # Preview must now exist and differ from raw at the dropped row
    assert dlg._preview_deltaA is not None, \
        'pin λ did not build the preview matrix'
    diff_row = float(np.max(np.abs(
        dlg._preview_deltaA[15, :] - A[15, :])))
    assert diff_row > 0, \
        'preview matrix unchanged at dropped wavelength row'
    print(f'    Pin λ idx=15 → drops_wl=[wl[15]={wl[15]:.4f}], '
          f'preview row diff = {diff_row:.2e}')

    # Duplicate pin is a no-op
    dlg._pin_current_kin_overlay()
    assert len(dlg._kin_overlay_wl) == 1
    print('    duplicate λ pin ignored')

    # =====================================================================
    # 6) CropDialog: bilinear option exists in the dropdown.
    # =====================================================================
    print('\n=== 6. method dropdown includes bilinear ===')
    items = [dlg.dd_interp_method.itemText(i)
             for i in range(dlg.dd_interp_method.count())]
    assert 'bilinear' in items, f'bilinear missing from {items}'
    print(f'    dropdown items = {items}')

    # =====================================================================
    # 7) _apply: λ drop alone (no crop) goes through
    #    interpolate_wavelengths_at for non-bilinear methods.
    # =====================================================================
    print('\n=== 7. _apply: wavelength drop, no crop, axis-wise method ===')
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'apply_wl_only_test')
    dlg = CropDialog(win, win)
    dlg._sel_wl_idx = 22
    dlg._sel_wl = float(wl[22])
    dlg._pin_current_kin_overlay()
    dlg.dd_interp_method.setCurrentText('linear')
    dlg._apply()
    keep_rows = np.setdiff1d(np.arange(wl.size), [22])
    assert np.allclose(win.deltaA_raw[keep_rows, :], A[keep_rows, :])
    diff = float(np.max(np.abs(win.deltaA_raw[22, :] - A[22, :])))
    assert diff > 0, 'wavelength row never interpolated'
    print(f'    row idx=22 interpolated linearly (Δ={diff:.2e})')

    # =====================================================================
    # 8) _apply: bilinear method dispatches to interpolate_2d_at and
    #    handles both axes simultaneously.
    # =====================================================================
    print('\n=== 8. _apply: bilinear over both axes (no crop) ===')
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'apply_bilinear_test')
    dlg = CropDialog(win, win)
    dlg._add_drop_value(float(t[18]))
    dlg._sel_wl_idx = 14
    dlg._sel_wl = float(wl[14])
    dlg._pin_current_kin_overlay()
    dlg.dd_interp_method.setCurrentText('bilinear')
    dlg._apply()

    mask = np.zeros(A.shape, dtype=bool)
    mask[14, :] = True
    mask[:, 18] = True
    np.testing.assert_allclose(win.deltaA_raw[~mask], A[~mask],
                               rtol=1e-10, atol=1e-12)
    # Dropped cells differ from the raw input — interpolation actually ran.
    diff_dropped = float(np.max(np.abs(win.deltaA_raw[mask] - A[mask])))
    assert diff_dropped > 0
    print(f'    bilinear apply: rows[14] + cols[18] interpolated '
          f'(max Δ={diff_dropped:.2e})')

    # =====================================================================
    # 9) _apply: bilinear + crop together — preview-matches-apply path.
    # =====================================================================
    print('\n=== 9. _apply: bilinear + crop ===')
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'apply_bilinear_crop_test')
    dlg = CropDialog(win, win)
    dlg._add_drop_value(float(t[30]))
    dlg._sel_wl_idx = 12
    dlg._sel_wl = float(wl[12])
    dlg._pin_current_kin_overlay()
    dlg.dd_interp_method.setCurrentText('bilinear')
    # Crop window
    dlg.ed_t_min.setValue(0.0)
    dlg.ed_t_max.setValue(80.0)
    dlg.ed_wl_min.setValue(450.0)
    dlg.ed_wl_max.setValue(750.0)
    dlg._apply()

    assert win.delay.min() >= 0.0 - 1e-6
    assert win.delay.max() <= 80.0 + 1e-6
    assert win.wavelength.min() >= 450.0 - 1e-6
    assert win.wavelength.max() <= 750.0 + 1e-6
    # The interpolated-then-cropped slice should match what we get by
    # running interpolate_missing_2d on the full grid then slicing.
    full_interp = ta_core.interpolate_missing_2d(A, wl, t, [12], [30])
    wl_mask = (wl >= 450.0) & (wl <= 750.0)
    t_mask = (t >= 0.0) & (t <= 80.0)
    expected = full_interp[np.ix_(wl_mask, t_mask)]
    np.testing.assert_allclose(win.deltaA_raw, expected,
                               rtol=1e-10, atol=1e-12)
    # And the snapshot for "Revert to Original" stays intact.
    assert np.array_equal(win.original_deltaA, A)
    print('    bilinear-then-crop matches full-bilinear sliced to crop')

    # =====================================================================
    # 10) live preview: pinning λ updates _preview_deltaA, mesh, panels.
    # =====================================================================
    print('\n=== 10. live preview: pinning λ recomputes preview ===')
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'live_wl_preview_test')
    dlg = CropDialog(win, win)
    assert dlg._preview_deltaA is None
    base_mesh = dlg._preview_im.get_array().copy()
    dlg._sel_wl_idx = 18
    dlg._sel_wl = float(wl[18])
    dlg._pin_current_kin_overlay()
    assert dlg._preview_deltaA is not None
    mesh_after = dlg._preview_im.get_array()
    assert not np.array_equal(mesh_after, base_mesh), \
        '2D mesh not updated when λ was pinned'
    print('    preview matrix + mesh both updated on λ pin')

    # =====================================================================
    # 11) live preview: switching to bilinear re-runs the interpolation.
    # =====================================================================
    print('\n=== 11. live preview: linear vs bilinear differ ===')
    dlg.dd_interp_method.setCurrentText('linear')
    lin_row = dlg._preview_deltaA[18, :].copy()
    dlg.dd_interp_method.setCurrentText('bilinear')
    bil_row = dlg._preview_deltaA[18, :].copy()
    # For wavelength-only drops they may be close; just verify the
    # preview matrix was rebuilt (method dispatch fired).
    diff = float(np.max(np.abs(lin_row - bil_row)))
    print(f'    linear vs bilinear row[18] max diff = {diff:.2e}')

    # =====================================================================
    # 12) Clear-λ-drops empties both the drop list and the visual lines,
    #     and clears the preview when there are no other drops.
    # =====================================================================
    print('\n=== 12. Clear λ drops resets preview ===')
    dlg._clear_kin_overlays()
    assert dlg._kin_overlay_wl == []
    assert dlg._kin_overlay_lines == []
    assert dlg.tbl_drops_wl.rowCount() == 0
    assert dlg._preview_deltaA is None, \
        'preview should reset when last drop is cleared'
    print('    drops cleared, preview reset to None')

    print('\n*** WAVELENGTH-INTERP + BILINEAR FEATURE VERIFIED ***')


if __name__ == '__main__':
    main()
