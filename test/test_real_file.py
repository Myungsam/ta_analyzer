"""
Smoke test using the user's real TA CSV file.

Exercises the new format-B parser, GUI loading, basic corrections,
and quickly runs a small global fit.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import numpy as np
from PyQt5 import QtWidgets

import ta_core
import ta_main

PATH = '/mnt/user-data/uploads/_TA_spectra_Accumulated.csv'


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # ---- 1. Parse ----
    wl, t, A = ta_core.parse_data_file(PATH)
    print(f'[1] parse OK: M={len(wl)} λ, N={len(t)} t, '
          f'A.shape={A.shape}, finite={np.isfinite(A).mean():.3f}')
    print(f'    λ range: {wl.min():.2f} .. {wl.max():.2f} nm')
    print(f'    t range: {t.min():.2f} .. {t.max():.2f} ps')
    print(f'    ΔA (finite) range: [{np.nanmin(A):.2f}, {np.nanmax(A):.2f}]')
    assert len(wl) == 2136
    assert len(t) == 192
    assert np.all(np.isfinite(wl))
    assert np.all(np.isfinite(t))
    # inf must be gone
    assert not np.any(np.isinf(A))
    print('    inf values successfully converted to NaN')

    # ---- 2. Load into the GUI ----
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, f'Loaded: {os.path.basename(PATH)}')
    assert win.wavelength is not None
    assert win.deltaA.shape == A.shape
    print(f'[2] GUI loaded: status="{win.lbl_status.text()}"')

    # ---- 3. Drawing (all scale modes) ----
    for mode in ('linear', 'log', 'split'):
        win.delay_scale_mode = mode
        win.update_all()
    print('[3] 2D map / spectrum / kinetics drew in all 3 scale modes')

    # ---- 4. Crop to a reasonable wavelength / time window ----
    win.apply_crop_by_range(450, 750, -1, 200)
    print(f'[4] crop applied: {len(win.wavelength)} λ × {len(win.delay)} t')
    assert 450 <= win.wavelength.min() and win.wavelength.max() <= 750
    assert -1 <= win.delay.min() and win.delay.max() <= 200

    # ---- 5. BG correction on first 3 negative-delay points ----
    # Pick N from the pre-t0 region
    n_neg = int(np.sum(win.delay < 0))
    bg_n = max(1, min(3, n_neg))
    win.bg_applied = True
    win.bg_n = bg_n
    win.bg_spectrum = np.nanmean(win.deltaA_raw[:, :bg_n], axis=1)
    win.update_all()
    print(f'[5] BG correction with N={bg_n} applied; '
          f'NaN fraction in ΔA = {np.isnan(win.deltaA).mean():.3f}')

    # ---- 6. Selection + pinning ----
    win.set_sel_wl(600.0)
    win.set_sel_t(10.0)
    win.pin_spec()
    win.pin_kin()
    assert abs(win.selWL - 600) < 2
    assert len(win.specOverlays) == 1 and len(win.kinOverlays) == 1
    print(f'[6] selection: λ={win.selWL:.2f} nm, t={win.selT:.3f} ps')

    # ---- 7. Quick global analysis on a SMALL wavelength slice ----
    # Full fit on 2136 λ × 192 t would be slow; use the cropped, down-sampled data
    # Sub-sample every 10 wavelengths for a fast sanity check
    sub = slice(None, None, 10)
    D_small = win.deltaA[sub, :]
    wl_small = win.wavelength[sub]
    print(f'[7] running global fit on {D_small.shape[0]} λ × '
          f'{D_small.shape[1]} t subset ...')
    res = ta_core.fit_global_analysis(
        D_small, win.delay,
        tau_init=[1, 10, 100], t0_init=0, fwhm_init=0.15,
        tau_fixed=[False, False, False],
        t0_fixed=True, fwhm_fixed=True,
        has_inf=False)
    print(f'    τ = {np.sort(res["tau"]).round(3).tolist()}')
    print(f'    RMS = {res["info"]["rms"]:.4g}, iters={res["info"]["iters"]}')
    assert np.all(np.isfinite(res['tau']))
    assert res['info']['rms'] < np.nanmax(np.abs(D_small))  # sanity

    # ---- 8. EADS ----
    EADS, tau_sorted, B = ta_core.compute_eads_from_dads(
        res['A'], res['tau'], False)
    assert EADS.shape == res['A'].shape
    print(f'[8] EADS computed: shape={EADS.shape}')

    # ---- 9. Round-trip export (new format is compatible with Format A on save) ----
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
        out_path = f.name
    try:
        ta_core.write_data_file(out_path, win.wavelength, win.delay,
                                win.deltaA, ',')
        wl2, t2, A2 = ta_core.parse_data_file(out_path)
        assert np.allclose(wl2, win.wavelength)
        assert np.allclose(t2, win.delay)
        finite = np.isfinite(A2) & np.isfinite(win.deltaA)
        assert np.allclose(A2[finite], win.deltaA[finite])
        print(f'[9] exported + reloaded: round-trip consistent '
              f'({wl2.shape[0]} × {t2.shape[0]})')
    finally:
        os.unlink(out_path)

    print('\n*** ALL REAL-FILE TESTS PASSED ***')


if __name__ == '__main__':
    main()
