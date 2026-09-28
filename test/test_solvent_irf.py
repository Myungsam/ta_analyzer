"""
Regression tests for the pure-solvent IRF subtraction feature.

Verifies:

  1. ta_core.align_solvent_to_sample numerically correct
     (same-grid case round-trips exactly; different-grid case
     produces NaN outside the solvent's extent).

  2. ta_core.apply_solvent_subtraction:
        - identical grids → numerical residual ≈ 0
        - partial-overlap (NaN regions) → those cells unchanged

  3. App state machinery:
        - cb_sub_irf toolbar checkbox exists
        - sub_irf_* attributes initialized
        - update_all() respects sub_irf_applied flag

  4. Pipeline integration:
        - subtraction happens AFTER BG and chirp
        - crop triggers realign automatically (via update_all)
        - load_data clears solvent state

  5. SolventIRFDialog basic construction + close.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import numpy as np
from PyQt5 import QtWidgets

import ta_core
import ta_main


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # =====================================================================
    # 1) align_solvent_to_sample numerical sanity
    # =====================================================================
    print('=== 1. align_solvent_to_sample ===')

    # Same grid → exact round-trip
    wl = np.linspace(400, 700, 50)
    t = np.linspace(-1, 5, 40)
    A_solv = (np.exp(-((wl - 500) / 100) ** 2)[:, None]
              * np.exp(-(t / 0.2) ** 2)[None, :])
    aligned = ta_core.align_solvent_to_sample(wl, t, A_solv, wl, t)
    max_diff = float(np.max(np.abs(aligned - A_solv)))
    print(f'    same-grid max diff = {max_diff:.2e}')
    assert max_diff < 1e-12, 'same-grid alignment should be exact'

    # Solvent grid smaller than sample grid → NaN outside
    wl_solv_small = wl[10:40]  # 410..600 nm subset
    t_solv_small = t[5:30]
    A_small = A_solv[10:40, 5:30]
    aligned = ta_core.align_solvent_to_sample(
        wl_solv_small, t_solv_small, A_small, wl, t)
    assert aligned.shape == (50, 40)
    n_nan = int(np.sum(~np.isfinite(aligned)))
    print(f'    sample grid extends beyond solvent: {n_nan} NaN cells '
          '(expected > 0)')
    assert n_nan > 0

    # =====================================================================
    # 2) apply_solvent_subtraction correctness
    # =====================================================================
    print('\n=== 2. apply_solvent_subtraction ===')
    sample = A_solv + 0.3 * np.outer(np.exp(-((wl - 450) / 30) ** 2),
                                      ta_core.exp_irf_conv(t, 5, 0, 0.15))
    # scale=1 fully removes the IRF on the same grid
    aligned_full = ta_core.align_solvent_to_sample(wl, t, A_solv, wl, t)
    corrected = ta_core.apply_solvent_subtraction(sample, aligned_full, 1.0)
    diff_at_irf_peak = float(np.abs(corrected[25, 20] - 0.3 *
                                    np.exp(-((wl[25] - 450) / 30) ** 2)
                                    * ta_core.exp_irf_conv(
                                        np.array([t[20]]), 5, 0, 0.15)[0]))
    print(f'    same-grid: residual at IRF peak = {diff_at_irf_peak:.2e}')
    assert diff_at_irf_peak < 1e-10

    # scale=0.5 partial removal
    half = ta_core.apply_solvent_subtraction(sample, aligned_full, 0.5)
    # Cells where aligned is NaN must remain unchanged
    aligned_partial = ta_core.align_solvent_to_sample(
        wl_solv_small, t_solv_small, A_small, wl, t)
    out_nan = ta_core.apply_solvent_subtraction(
        sample, aligned_partial, 1.0)
    nan_cells = ~np.isfinite(aligned_partial)
    same = float(np.max(np.abs(out_nan[nan_cells] - sample[nan_cells])))
    print(f'    NaN regions left unchanged: max diff = {same:.2e}')
    assert same < 1e-15

    # =====================================================================
    # 3) App-level state machinery
    # =====================================================================
    print('\n=== 3. App state ===')
    wl2 = np.linspace(400, 700, 40)
    t2 = np.linspace(-1, 5, 35)
    A2 = (np.exp(-((wl2 - 500) / 100) ** 2)[:, None]
          * np.exp(-(t2 / 0.2) ** 2)[None, :])
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl2, t2, A2, 'sample')

    assert hasattr(win, 'cb_sub_irf'), 'toolbar checkbox missing'
    assert win.sub_irf_applied is False
    assert win.sub_irf_scale == 1.0
    assert win.sub_irf_aligned is None
    print('    cb_sub_irf exists, defaults OK')

    # =====================================================================
    # 4) End-to-end correction with IRF subtraction enabled
    # =====================================================================
    print('\n=== 4. End-to-end correction ===')
    sample_real = (np.exp(-((wl2 - 450) / 30) ** 2)[:, None]
                   * ta_core.exp_irf_conv(t2, 50, 0, 0.15)[None, :])
    irf_truth = (np.exp(-((wl2 - 500) / 100) ** 2)[:, None]
                 * np.exp(-(t2 / 0.2) ** 2)[None, :])
    SAMPLE = sample_real + 0.8 * irf_truth
    win.set_loaded_data(wl2, t2, SAMPLE, 'sample_with_irf')

    # Simulate the user having loaded a pure solvent
    win.sub_irf_solv_wl = wl2.copy()
    win.sub_irf_solv_t = t2.copy()
    win.sub_irf_solv_data = irf_truth.copy()
    win.sub_irf_applied = True
    win.sub_irf_scale = 0.8
    win.realign_solvent()
    assert win.sub_irf_aligned is not None
    win.recompute()
    resid_rms = float(np.sqrt(np.mean((win.deltaA - sample_real) ** 2)))
    print(f'    with scale=0.8 (true 0.8): RMS residual vs truth = '
          f'{resid_rms:.2e}')
    assert resid_rms < 1e-10, 'subtraction should be numerically exact'

    # Toggle OFF → recompute restores raw (+BG/mask/etc) without subtract
    win.sub_irf_applied = False
    win.recompute()
    resid_off = float(np.sqrt(np.mean((win.deltaA - SAMPLE) ** 2)))
    print(f'    OFF: RMS vs raw = {resid_off:.2e}')
    assert resid_off < 1e-15

    # =====================================================================
    # 5) Crop triggers realign
    # =====================================================================
    print('\n=== 5. Crop preserves solvent realignment ===')
    win.sub_irf_applied = True
    win.update_all()
    win.apply_crop_by_range(420, 680, -0.5, 4.0)
    assert win.sub_irf_aligned is not None, \
        'solvent should be re-aligned after crop'
    assert win.sub_irf_aligned.shape == win.deltaA.shape, \
        ('aligned shape must match cropped data: '
         f'{win.sub_irf_aligned.shape} vs {win.deltaA.shape}')
    print(f'    after crop: data {win.deltaA.shape}, '
          f'solvent aligned {win.sub_irf_aligned.shape} — match ✓')
    # Subtraction state — crop turns it off until next user-toggle
    print(f'    sub_irf_applied after crop: {win.sub_irf_applied} '
          '(deliberately reset)')

    # =====================================================================
    # 6) New data load clears solvent state
    # =====================================================================
    print('\n=== 6. set_loaded_data clears solvent ===')
    win.sub_irf_solv_data = irf_truth.copy()
    win.sub_irf_applied = True
    win.set_loaded_data(wl2, t2, SAMPLE, 'fresh')
    assert win.sub_irf_solv_data is None
    assert win.sub_irf_applied is False
    print('    solvent state cleared on new load ✓')

    # =====================================================================
    # 7) Dialog construct + close
    # =====================================================================
    print('\n=== 7. SolventIRFDialog construct + close ===')
    from ta_solvent_irf import SolventIRFDialog
    dlg = SolventIRFDialog(win, win)
    assert dlg is not None
    # Simulate solvent load
    win.sub_irf_solv_wl = wl2.copy()
    win.sub_irf_solv_t = t2.copy()
    win.sub_irf_solv_data = irf_truth.copy()
    win.realign_solvent()
    dlg._reflect_state_in_ui()
    dlg._refresh_now()
    # Slider sync
    dlg.sl_scale.setValue(150)
    QtWidgets.QApplication.processEvents()
    assert abs(win.sub_irf_scale - 1.5) < 1e-9, \
        f'slider 150 should map to 1.5, got {win.sub_irf_scale}'
    print(f'    slider → app scale: {win.sub_irf_scale}')
    dlg.close()
    print('    dialog construct + close ✓')

    # =====================================================================
    # 8) Loading a solvent file from a DIFFERENT directory
    # =====================================================================
    # Regression for the "다른 경로에 있는 파일은 가져오지 못하는 문제":
    # ask_open_paths returns a *string* by default, not a list.  The
    # original do_load did ``paths = ask_open_paths(...)`` then
    # ``path = paths[0]``, which took the first *character* of the
    # path — so any absolute path got truncated to ``/`` or ``C`` and
    # parse_data_file failed.
    print('\n=== 8. Load solvent from a different directory ===')
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix='solvent_regression_')
    solv_path = os.path.join(tmpdir, 'pure_solvent_run.csv')
    ta_core.write_data_file(solv_path, wl2, t2, irf_truth, ',')
    cwd = os.getcwd()
    print(f'    cwd:           {cwd}')
    print(f'    solvent path:  {solv_path}')
    assert os.path.dirname(solv_path) != cwd

    # Reset state so we exercise the load fresh
    win.sub_irf_solv_data = None
    win.sub_irf_solv_path = ''
    win.sub_irf_aligned = None

    # Monkey-patch the file dialog + the modal info box for headless run
    import ta_solvent_irf
    orig_ask = ta_solvent_irf.ask_open_paths
    orig_info = ta_solvent_irf.info_box
    ta_solvent_irf.ask_open_paths = lambda parent, caption, **kw: (
        [solv_path] if kw.get('multi') else solv_path)
    ta_solvent_irf.info_box = lambda *a, **kw: None
    try:
        dlg2 = SolventIRFDialog(win, win)
        dlg2.do_load()
        assert win.sub_irf_solv_data is not None, \
            'do_load failed silently — the file was not parsed'
        assert win.sub_irf_solv_path == solv_path, (
            f'wrong path stored: {win.sub_irf_solv_path!r} '
            f'(expected {solv_path!r}) — the path was truncated again')
        assert win.sub_irf_aligned is not None, \
            'realign_solvent did not produce an aligned grid'
        print(f'    loaded: {win.sub_irf_solv_data.shape}, '
              f'path stored verbatim')
        dlg2.close()
    finally:
        ta_solvent_irf.ask_open_paths = orig_ask
        ta_solvent_irf.info_box = orig_info
        os.unlink(solv_path)
        os.rmdir(tmpdir)

    # =====================================================================
    # 9) Chirp correction is mirrored onto the solvent reference
    # =====================================================================
    # When the user has chirp correction applied to the sample, the
    # pure-solvent reference must be put into the SAME chirp-corrected
    # frame before subtraction.  Otherwise the IRF in the solvent
    # would still sit at t₀(λ) ≠ 0 while the sample's IRF has already
    # been shifted to t=0, and the subtraction would actually *add*
    # artifacts at both positions instead of removing them.
    print('\n=== 9. Chirp also applied to solvent reference ===')
    wl3 = np.linspace(400, 700, 60)
    t3 = np.linspace(-1, 5, 80)

    def chirped_irf(wl, t, sigma=0.15, slope=0.5):
        """A coherent-artifact IRF whose centre depends linearly on λ."""
        t0 = slope * (wl - 550.0) / 100.0
        arg = (t[None, :] - t0[:, None]) / sigma
        return np.exp(-arg ** 2)

    irf_raw = chirped_irf(wl3, t3)
    sample_real = (np.exp(-((wl3 - 450) / 30) ** 2)[:, None]
                   * ta_core.exp_irf_conv(t3, 50, 0, 0.15)[None, :])
    SAMPLE_C = sample_real + 0.8 * irf_raw
    SOLVENT_C = irf_raw.copy()

    # Fit the chirp on dense truth points so it's effectively exact
    pts = np.column_stack([wl3, 0.5 * (wl3 - 550.0) / 100.0])
    chirp_params, _ = ta_core.fit_chirp_params(pts)

    win.set_loaded_data(wl3, t3, SAMPLE_C, 'sample_chirped')
    win.chirp_params = chirp_params
    win.chirp_applied = True
    win.sub_irf_solv_wl = wl3.copy()
    win.sub_irf_solv_t = t3.copy()
    win.sub_irf_solv_data = SOLVENT_C.copy()
    win.sub_irf_applied = True
    win.sub_irf_scale = 0.8

    # Truth: what the (chirp-corrected) signal-only sample looks like
    real_chirped = ta_core.apply_chirp_shift(
        sample_real, t3.astype(float), wl3.astype(float), chirp_params)

    # Old behaviour: solvent_aligned was the resampled-but-NOT-chirped
    # solvent.  Subtracting this from a chirp-corrected sample leaves
    # a clearly visible curved residual.
    win.sub_irf_aligned = ta_core.align_solvent_to_sample(
        wl3, t3, SOLVENT_C, win.wavelength, win.delay)
    win.recompute()
    mask_fin = np.isfinite(real_chirped) & np.isfinite(win.deltaA)
    rms_no_chirp_on_solvent = float(np.sqrt(
        np.mean((win.deltaA[mask_fin] - real_chirped[mask_fin]) ** 2)))

    # New behaviour: realign_solvent mirrors the chirp.
    win.realign_solvent()
    win.recompute()
    mask_fin = np.isfinite(real_chirped) & np.isfinite(win.deltaA)
    rms_with_chirp = float(np.sqrt(
        np.mean((win.deltaA[mask_fin] - real_chirped[mask_fin]) ** 2)))

    print(f'    RMS residual (solvent without chirp): '
          f'{rms_no_chirp_on_solvent:.4e}')
    print(f'    RMS residual (solvent WITH    chirp): '
          f'{rms_with_chirp:.4e}')
    print(f'    improvement: '
          f'{rms_no_chirp_on_solvent / max(rms_with_chirp, 1e-30):.1e}×')
    # The chirped solvent path must produce numerical-noise-level
    # residual (≤ 1e-12), while the un-chirped path is dominated by
    # the curved IRF artifact (≥ 1e-2 here).
    assert rms_with_chirp < 1e-10, (
        f'Even with the fix, residual is {rms_with_chirp:.2e} — '
        'chirp not applied to solvent?')
    assert rms_no_chirp_on_solvent > 100 * rms_with_chirp, (
        'Sanity check: the fix should dramatically improve over '
        'subtracting an un-chirped solvent.')

    # Toggling chirp OFF should re-align the solvent back to its
    # un-chirped form on the next update_all().
    win.chirp_applied = False
    win.update_all()
    # The solvent reference (un-chirped now) should match the
    # original 2-D linear interpolation exactly.
    expected = ta_core.align_solvent_to_sample(
        wl3, t3, SOLVENT_C, win.wavelength, win.delay)
    same = float(np.nanmax(np.abs(win.sub_irf_aligned - expected)))
    print(f'    after chirp OFF: aligned matches non-chirped resample '
          f'(max diff {same:.2e})')
    assert same < 1e-12, 'realign_solvent did not drop chirp when chirp_applied=False'

    print('\n*** SOLVENT IRF TESTS ALL PASSED ***')


if __name__ == '__main__':
    main()
