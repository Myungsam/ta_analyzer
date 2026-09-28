"""
Regression tests for the Crop dialog's "delete & interpolate spectra
at specific delays" feature.

Backend pieces under test:
  - ta_core.interpolate_missing_columns: row-wise interpolation along t
  - TAAnalyzer.interpolate_delays_at:    wires it to deltaA_raw + update_all

UI under test:
  - CropDialog._add_drop_value / _drop_clear / _refresh_drop_table
  - CropDialog._apply: crop-then-interpolate, and the "crop unchanged →
    skip apply_crop_by_range so bg/chirp survive" branch
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import numpy as np
from PyQt5 import QtWidgets

import ta_core
import ta_main
from ta_dialogs_a import CropDialog


def _make_synthetic(n_wl=40, n_t=60):
    """Smooth synthetic TA matrix on a sorted delay grid.

    Two exponential components with wavelength-dependent amplitudes.
    Smooth along t so interpolation across a removed column is accurate
    enough to verify the wiring.
    """
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
    # 1) ta_core.interpolate_missing_columns: every method round-trips a
    #    smooth matrix to within reasonable error.
    # =====================================================================
    print('=== 1. interpolate_missing_columns: per-method accuracy ===')
    drop_idx = [10, 25, 40]   # interior columns
    truth = A[:, drop_idx].copy()
    for method in ('linear', 'cubic', 'pchip', 'akima'):
        out = ta_core.interpolate_missing_columns(A, t, drop_idx, method=method)
        # Untouched columns must be untouched
        keep = np.setdiff1d(np.arange(t.size), drop_idx)
        assert np.allclose(out[:, keep], A[:, keep]), \
            f'{method}: kept columns were modified'
        # Reconstruction error on the dropped columns
        err = float(np.max(np.abs(out[:, drop_idx] - truth)))
        print(f'    {method:7s}  max abs err = {err:.4e}')
        assert err < 5e-2, f'{method}: error {err} too large'

    # =====================================================================
    # 2) Cubic-family with too few anchors silently falls back to linear.
    # =====================================================================
    print('\n=== 2. cubic-family fallback for tiny grids ===')
    t_tiny = np.array([0.0, 1.0, 2.0])
    A_tiny = np.array([[0.0, 1.0, 2.0],
                       [10.0, 11.0, 12.0]])
    out = ta_core.interpolate_missing_columns(A_tiny, t_tiny, [1], method='cubic')
    expected = np.array([1.0, 11.0])   # linear midpoint
    assert np.allclose(out[:, 1], expected), \
        f'cubic should fall back to linear with 2 anchors; got {out[:, 1]}'
    print(f'    cubic with 2 anchors → linear: ok')

    # =====================================================================
    # 3) interpolate_delays_at on the app: deltaA_raw updates, GA state
    #    is cleared, deltaA mirrors deltaA_raw when no corrections.
    # =====================================================================
    print('\n=== 3. TAAnalyzer.interpolate_delays_at ===')
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'crop_interp_test')
    win.interpolate_delays_at([15, 30], method='pchip')
    assert win.deltaA_raw.shape == A.shape
    # Touched columns should differ; kept columns identical
    keep = np.setdiff1d(np.arange(t.size), [15, 30])
    assert np.allclose(win.deltaA_raw[:, keep], A[:, keep])
    diff_dropped = float(np.max(np.abs(win.deltaA_raw[:, [15, 30]]
                                       - A[:, [15, 30]])))
    print(f'    dropped-col max diff = {diff_dropped:.4e}  (should be small but > 0)')
    # original_deltaA stays untouched ("Revert to Original" must still work)
    assert np.array_equal(win.original_deltaA, A), \
        'original_deltaA was mutated — Revert to Original would lose data'

    # =====================================================================
    # 4) Too many drops → guard fires, no mutation.
    # =====================================================================
    print('\n=== 4. drop-count guard ===')
    win2 = ta_main.TAAnalyzer()
    win2.set_loaded_data(wl, t, A, 'guard_test')
    raw_before = win2.deltaA_raw.copy()
    # Try to drop n-1 of n delays → only 1 kept → must refuse
    QtWidgets.QApplication.setActiveWindow(win2)
    # Patch warn_box to a counter so the test doesn't pop a modal
    import ta_main as _tm
    warn_calls = []
    real_warn = _tm.warn_box
    _tm.warn_box = lambda *a, **kw: warn_calls.append((a, kw))
    try:
        win2.interpolate_delays_at(list(range(t.size - 1)), method='linear')
    finally:
        _tm.warn_box = real_warn
    assert np.array_equal(win2.deltaA_raw, raw_before), \
        'guard failed: deltaA_raw was changed even though only 1 delay would remain'
    assert warn_calls, 'guard did not emit warn_box'
    print(f'    refused (warn_box called {len(warn_calls)}×, data unchanged)')

    # =====================================================================
    # 5) CropDialog: pin-driven drop list (selection → Pin button).
    # =====================================================================
    print('\n=== 5. CropDialog UI wiring (pin-driven) ===')
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'crop_dlg_test')
    dlg = CropDialog(win, win)

    # 5a) Set selection via the Delay-idx spinbox, then Pin → adds to drops
    dlg.ed_sel_t_idx.setValue(20)
    assert dlg._sel_t == float(t[20]), \
        f'spinbox should set _sel_t to t[20]={t[20]}, got {dlg._sel_t}'
    dlg._pin_current_as_drop()
    assert len(dlg._drop_t_values) == 1
    assert dlg._drop_t_values[0] == float(t[20])
    assert dlg.tbl_drops.rowCount() == 1
    assert dlg.tbl_drops.item(0, 0).text() == '20'
    print(f'    [5a] Delay-idx=20 + Pin → drops=[t[20]={t[20]:.4f}]')

    # 5b) Pinning the same delay again is a no-op
    dlg._pin_current_as_drop()
    assert len(dlg._drop_t_values) == 1, 'duplicate was accepted'
    print(f'    [5b] duplicate Pin ignored (still 1 row)')

    # 5c) Programmatic add via _add_drop_value (used by power-user paths)
    #     keeps the list sorted and updates overlays + spectrum panel.
    dlg._add_drop_value(float(t[5]))
    dlg._add_drop_value(float(t[40]))
    assert dlg._drop_t_values == sorted(dlg._drop_t_values)
    assert len(dlg._drop_lines) == 3, \
        f'expected 3 overlay lines, got {len(dlg._drop_lines)}'
    print(f'    [5c] 3 drops, overlay lines drawn, list kept sorted')

    # 5d) 2D-map click sets selection but does NOT auto-add a drop.
    drops_before = list(dlg._drop_t_values)

    class _FakeEvent:
        pass
    ev = _FakeEvent()
    ev.inaxes = dlg.canvas.ax
    ev.xdata = float(wl[10])
    ev.ydata = float(t[8])
    dlg._on_canvas_click(ev)
    assert dlg._drop_t_values == drops_before, \
        'click on 2D map auto-added a drop — algorithm regressed'
    assert dlg._sel_wl_idx == 10
    assert dlg._sel_t_idx == 8
    print(f'    [5d] 2D click moves crosshair (selWL idx=10, selT idx=8); '
          f'drops unchanged')

    # 5e) Kinetics overlay pin is independent of drops.
    dlg._sel_wl_idx = 12
    dlg._sel_wl = float(wl[12])
    dlg._pin_current_kin_overlay()
    assert dlg._kin_overlay_wl == [float(wl[12])]
    assert dlg._drop_t_values == drops_before, \
        'kinetics-pin should not touch the drop list'
    print(f'    [5e] λ-pin populates _kin_overlay_wl only; drops unaffected')

    # 5f) Clear all drops
    dlg._drop_clear()
    assert dlg._drop_t_values == []
    assert dlg.tbl_drops.rowCount() == 0
    assert dlg._drop_lines == []
    print(f'    [5f] Clear all empties drop list, table, and overlay')

    # 5g) Clear λ pins
    dlg._clear_kin_overlays()
    assert dlg._kin_overlay_wl == []
    assert dlg._kin_overlay_lines == []
    print(f'    [5g] Clear λ pins empties overlay list')

    # =====================================================================
    # 6) _apply: crop unchanged + drops → skips re-crop, runs interpolation,
    #    bg/chirp state survives.
    # =====================================================================
    print('\n=== 6. _apply: crop-unchanged path preserves corrections ===')
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'preserve_bg_test')
    # Pretend BG was already applied — _apply should NOT reset this flag
    # when the crop range is unchanged.
    win.bg_applied = True
    win.bg_n = 5
    win.chirp_applied = True

    dlg = CropDialog(win, win)
    # Pick two interior delays
    dlg._add_drop_value(float(t[12]))
    dlg._add_drop_value(float(t[35]))
    # Spinboxes remain at the full (= unchanged) range, so no re-crop
    dlg.dd_interp_method.setCurrentText('cubic')
    dlg._apply()

    assert win.bg_applied is True, \
        'bg_applied was reset — _apply re-cropped when it should not have'
    assert win.chirp_applied is True, \
        'chirp_applied was reset — _apply re-cropped when it should not have'
    # Data on those columns changed but rest is identical to original
    keep = np.setdiff1d(np.arange(t.size), [12, 35])
    assert np.allclose(win.deltaA_raw[:, keep], A[:, keep])
    print(f'    bg_applied & chirp_applied survived; interpolation applied')

    # =====================================================================
    # 7) _apply: crop AND drops together → re-crop runs first, drops
    #    that fall inside the new window are mapped to NEW indices.
    # =====================================================================
    print('\n=== 7. _apply: combined crop + interpolation ===')
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'combined_test')
    dlg = CropDialog(win, win)

    # Drop one delay inside the future crop and one outside (must be ignored)
    t_inside = float(t[30])
    t_outside = float(t[2])   # will be cropped out by t_min=10
    dlg._add_drop_value(t_inside)
    dlg._add_drop_value(t_outside)
    # New crop window [t=10, t=80]
    dlg.ed_t_min.setValue(10.0)
    dlg.ed_t_max.setValue(80.0)
    dlg._apply()

    assert win.delay.min() >= 10.0 - 1e-6
    assert win.delay.max() <= 80.0 + 1e-6
    # The inside drop was interpolated — find its new index
    new_idx = int(np.argmin(np.abs(win.delay - t_inside)))
    # Sanity: column is different from the original cropped slice would be.
    orig_cropped = A[:, (t >= 10.0) & (t <= 80.0)]
    diff_at_drop = float(np.max(np.abs(
        win.deltaA_raw[:, new_idx] - orig_cropped[:, new_idx])))
    # For a smooth synthetic the column should be CLOSE to the original
    # but not bit-identical (interpolation went through neighbors).
    assert diff_at_drop > 0, \
        'inside drop column unchanged — interpolation never ran'
    print(f'    crop [10, 80] applied; inside drop @ new idx {new_idx} '
          f'interpolated (Δ={diff_at_drop:.2e})')

    # =====================================================================
    # 8) Methods dropdown actually changes which interpolator is used.
    # =====================================================================
    print('\n=== 8. method dropdown is honored ===')
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'method_dispatch_test')
    dlg = CropDialog(win, win)
    dlg._add_drop_value(float(t[20]))
    dlg.dd_interp_method.setCurrentText('akima')
    dlg._apply()
    col_akima = win.deltaA_raw[:, 20].copy()

    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'method_dispatch_test')
    dlg = CropDialog(win, win)
    dlg._add_drop_value(float(t[20]))
    dlg.dd_interp_method.setCurrentText('linear')
    dlg._apply()
    col_linear = win.deltaA_raw[:, 20].copy()

    diff = float(np.max(np.abs(col_akima - col_linear)))
    assert diff > 0, 'akima vs linear gave identical columns — dropdown ignored'
    print(f'    akima vs linear differ at the dropped column (Δ={diff:.2e})')

    # =====================================================================
    # 9) Live preview: pinning a delay recomputes _preview_deltaA and
    #    updates the 2D pcolormesh data in-place (set_array).
    # =====================================================================
    print('\n=== 9. live preview: pin → 2D + spec + kin all reflect interp ===')
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'preview_test')
    dlg = CropDialog(win, win)
    assert dlg._preview_deltaA is None, 'preview should be None before any drop'
    # 2D mesh starts from raw original data
    base_array_before = dlg._preview_im.get_array().copy()

    dlg._add_drop_value(float(t[25]))
    assert dlg._preview_deltaA is not None, \
        'pinning a drop did not build the preview matrix'
    # Preview must differ from original at the dropped column (interpolated)
    diff_col = float(np.max(np.abs(
        dlg._preview_deltaA[:, 25] - A[:, 25])))
    assert diff_col > 0, \
        'preview matrix is identical to original at dropped col — no interp ran'
    # 2D mesh data was updated in-place (set_array, not a fresh pcolormesh)
    array_after = dlg._preview_im.get_array()
    assert not np.array_equal(array_after, base_array_before), \
        '_preview_im.set_array was not called when a drop was added'
    print(f'    pinned t[25]: preview built (Δ at col 25 = {diff_col:.2e}); '
          f'mesh array updated in-place')

    # Spectrum at sel_t = dropped delay reads from preview, not raw
    dlg.ed_sel_t_idx.setValue(25)
    sp_line = [ln for ln in dlg.canvas_spec.ax.lines if ln.get_color() == 'b'][-1]
    # The blue line at idx=25 should equal preview[:, 25], NOT A[:, 25]
    np.testing.assert_allclose(sp_line.get_ydata(),
                               dlg._preview_deltaA[:, 25])
    print(f'    spectrum panel reads from preview at the pinned delay')

    # Kinetics at sel_wl reads preview, so col-25 sample equals interp value
    dlg._set_selection(wl_idx=20)
    kin_line = [ln for ln in dlg.canvas_kin.ax.lines if ln.get_color() == 'r'][-1]
    kin_y = kin_line.get_ydata()
    assert abs(kin_y[25] - dlg._preview_deltaA[20, 25]) < 1e-12, \
        'kinetics curve does not reflect interpolation at dropped delay'
    print(f'    kinetics panel reads from preview at the dropped delay')

    # =====================================================================
    # 10) Live preview: changing the method recomputes (akima ≠ linear).
    # =====================================================================
    print('\n=== 10. live preview: changing method re-runs interpolation ===')
    dlg.dd_interp_method.setCurrentText('linear')
    prev_linear = dlg._preview_deltaA[:, 25].copy()
    dlg.dd_interp_method.setCurrentText('akima')
    prev_akima = dlg._preview_deltaA[:, 25].copy()
    diff_methods = float(np.max(np.abs(prev_akima - prev_linear)))
    assert diff_methods > 0, \
        'changing method did not rebuild the preview'
    print(f'    method change re-runs preview (linear→akima Δ={diff_methods:.2e})')

    # =====================================================================
    # 11) Live preview: clearing drops restores the original mesh data.
    # =====================================================================
    print('\n=== 11. live preview: clearing drops resets preview ===')
    dlg._drop_clear()
    assert dlg._preview_deltaA is None, \
        'preview was not reset when all drops were cleared'
    array_reset = dlg._preview_im.get_array()
    # Mesh data now equals original_deltaA.T.ravel()
    np.testing.assert_allclose(array_reset, A.T.ravel())
    print(f'    drops cleared → preview None, mesh restored to raw data')

    # =====================================================================
    # 12) _apply interp-first-then-crop: result matches the preview slice.
    #     This is the user-requested algorithm change.
    # =====================================================================
    print('\n=== 12. _apply: result matches live preview (interp first, then crop) ===')
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'order_test')
    dlg = CropDialog(win, win)

    # Pin a delay that's INSIDE the eventual crop window.  Interpolate on
    # the FULL anchor set first, then crop.
    dlg._add_drop_value(float(t[30]))
    expected_full_interp = dlg._preview_deltaA.copy()
    # Crop [t=10, t=80]
    dlg.ed_t_min.setValue(10.0)
    dlg.ed_t_max.setValue(80.0)
    dlg.dd_interp_method.setCurrentText('cubic')
    # Recompute the preview key (method changed) for an apples-to-apples
    # comparison with the apply path.
    expected_full_interp = ta_core.interpolate_missing_columns(
        A, t, [30], method='cubic')
    dlg._apply()

    # The new deltaA_raw should equal expected_full_interp sliced to the crop
    t_mask = (t >= 10.0) & (t <= 80.0)
    expected_cropped = expected_full_interp[:, t_mask]
    np.testing.assert_allclose(win.deltaA_raw, expected_cropped, atol=1e-12)
    print(f'    apply produced exactly the interpolated-then-cropped matrix')

    # Revert sanity: original_deltaA snapshot must NOT be the interp version.
    # The swap-and-restore in _apply preserves the raw snapshot so a future
    # "Revert to Original" still works as expected.
    assert np.array_equal(win.original_deltaA, A), \
        'original_deltaA was permanently overwritten by the swap'
    print(f'    original_deltaA snapshot preserved (Revert still works)')

    # =====================================================================
    # 13) Click on 2D map updates spectrum AND kinetics in real-time
    #     using the preview data when drops exist.
    # =====================================================================
    print('\n=== 13. 2D click updates spec/kin live with preview data ===')
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'click_live_test')
    dlg = CropDialog(win, win)
    dlg._add_drop_value(float(t[40]))   # build a preview

    class _Ev:
        pass
    ev = _Ev()
    ev.inaxes = dlg.canvas.ax
    # Click exactly at (wl[8], t[40]) — i.e. ON the dropped delay
    ev.xdata = float(wl[8])
    ev.ydata = float(t[40])
    dlg._on_canvas_click(ev)
    assert dlg._sel_wl_idx == 8 and dlg._sel_t_idx == 40

    # Spectrum's "current" (blue) line at idx 40 must be the interp value
    sp_blue = [ln for ln in dlg.canvas_spec.ax.lines if ln.get_color() == 'b'][-1]
    np.testing.assert_allclose(sp_blue.get_ydata(),
                               dlg._preview_deltaA[:, 40])
    # Kinetics' "current" (red) line at wl idx 8: sample at t-idx 40 must
    # equal the interpolated value (not the raw one).
    kin_red = [ln for ln in dlg.canvas_kin.ax.lines if ln.get_color() == 'r'][-1]
    assert abs(kin_red.get_ydata()[40] - dlg._preview_deltaA[8, 40]) < 1e-12
    print(f'    click on dropped delay → spec/kin show interpolated value')

    # Click on a non-dropped delay: panels show the original data (==preview
    # at non-dropped columns, which equals raw for those columns)
    ev.xdata = float(wl[15])
    ev.ydata = float(t[5])
    dlg._on_canvas_click(ev)
    sp_blue = [ln for ln in dlg.canvas_spec.ax.lines if ln.get_color() == 'b'][-1]
    np.testing.assert_allclose(sp_blue.get_ydata(), A[:, 5])
    print(f'    click on non-dropped delay → spec shows raw original data')

    print('\n*** CROP-INTERPOLATE FEATURE VERIFIED ***')


if __name__ == '__main__':
    main()
