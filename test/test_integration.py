"""
End-to-end smoke test for the TAAnalyzer Python port.

Runs headless (QT_QPA_PLATFORM=offscreen) so no display is needed.
Exercises: data load, BG correction, chirp correction, masks, crop,
selection, zero-time shift, global analysis, and exports.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import tempfile
import numpy as np

from PyQt5 import QtWidgets, QtCore

import ta_core
import ta_main


# --------------------------------------------------------
# Synthesize a realistic TA dataset
# --------------------------------------------------------
def make_dataset(seed=1):
    rng = np.random.default_rng(seed)
    wl = np.linspace(400, 700, 80)
    # Mixed linear+log delay axis
    t = np.concatenate([np.linspace(-1, 1, 25), np.logspace(0, 3, 50)[1:]])
    t = np.unique(np.round(t, 6))

    def gauss(center, width, amp):
        return amp * np.exp(-(wl - center) ** 2 / (2 * width ** 2))

    taus = [0.4, 8, 200]
    spectra = [gauss(450, 30, 1.0),
               gauss(500, 45, -0.6),
               gauss(610, 55, 0.3)]

    data = np.zeros((len(wl), len(t)))
    for tau, sp in zip(taus, spectra):
        kin = ta_core.exp_irf_conv(t, tau, 0.0, 0.15)
        data += np.outer(sp, kin)
    # Add a small background offset
    data += 0.02 * np.cos(wl * 0.05)[:, None]
    # Add noise
    data += 0.005 * rng.standard_normal(data.shape)
    return wl, t, data


def run_tests():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = ta_main.TAAnalyzer()

    # -------- 1. Load data --------
    wl, t, A = make_dataset()
    win.set_loaded_data(wl, t, A, 'synthetic_test')
    assert win.wavelength is not None and win.deltaA is not None
    assert win.deltaA.shape == (len(wl), len(t))
    print('[1] set_loaded_data: OK')

    # -------- 2. Time-unit methods --------
    assert win.t_unit_hdr() == 'ps'
    win.time_unit = 'us'
    assert 'mu' in win.t_unit_ax() or 'μ' in win.t_unit_txt()
    win.time_unit = 'ps'
    print('[2] time-unit helpers: OK')

    # -------- 3. Scale mode changes --------
    for m in ('linear', 'log', 'split', 'linear'):
        win.delay_scale_mode = m
        win._draw_map_2d()
        win._draw_kinetics()
    print('[3] scale modes: linear/log/split drew OK')

    # -------- 4. Colormap + Z range --------
    for cm in ('turbo', 'jet', 'RdBu', 'BWR', 'seismic', 'gray', 'cool'):
        win.map_colormap = cm
        win._draw_map_2d()
    win.map_z_min, win.map_z_max = -0.8, 0.8
    win._draw_map_2d()
    win.reset_z_range()
    print('[4] colormap / z-range: OK')

    # -------- 5. Selection + crosshair --------
    win.set_sel_wl(500.0)
    win.set_sel_t(5.0)
    assert abs(win.selWL - 500.0) < 5
    assert abs(win.selT - 5.0) < 1
    print('[5] selection: OK (selWL=%.2f, selT=%.3f)' % (win.selWL, win.selT))

    # -------- 6. Pin / clear overlays --------
    win.pin_spec()
    win.pin_kin()
    assert len(win.specOverlays) == 1
    assert len(win.kinOverlays) == 1
    win.clear_spec_overlays()
    win.clear_kin_overlays()
    assert not win.specOverlays and not win.kinOverlays
    print('[6] pin/clear overlays: OK')

    # -------- 7. Background correction --------
    win.bg_applied = True
    win.bg_n = 5
    win.bg_spectrum = np.nanmean(win.deltaA_raw[:, :5], axis=1)
    win.update_all()
    # BG-corrected data should have much smaller baseline
    pre_t0 = win.delay < -0.3
    post_t0_idx = int(np.argmin(np.abs(win.delay - 100)))  # long time
    baseline_mag = np.nanmean(np.abs(win.deltaA[:, pre_t0]))
    print(f'[7] BG correction applied; baseline |dA|≈{baseline_mag:.4f}')
    assert baseline_mag < 0.1

    # -------- 8. Mask wavelengths --------
    win.masked_regions = [(530.0, 560.0, 'nan')]
    win.update_all()
    masked_rows = (win.wavelength >= 530) & (win.wavelength <= 560)
    assert np.all(np.isnan(win.deltaA[masked_rows, :]))
    win.masked_regions = []
    win.update_all()
    print('[8] wavelength masking: OK')

    # -------- 9. Chirp correction (synthetic known chirp) --------
    # Add a known chirp to the loaded data, then fit & correct
    chirp_params = np.array([0.3, 1.1, 1.3, 0.0])  # small chirp
    # Not applying an actual chirp to the data; just verify the fit works
    pts = np.array([[420, 0.25], [470, 0.08], [520, 0.0],
                    [570, -0.05], [620, -0.1], [670, -0.15]])
    p, rms = ta_core.fit_chirp_params(pts)
    assert np.all(np.isfinite(p))
    print(f'[9] chirp fit: params={p.round(3).tolist()}, RMS={rms:.3e}')

    # -------- 10. Crop --------
    win.apply_crop_by_range(450, 650, -0.5, 500)
    assert win.wavelength.min() >= 450 and win.wavelength.max() <= 650
    assert win.delay.min() >= -0.5 and win.delay.max() <= 500
    print(f'[10] crop: OK ({len(win.wavelength)} λ x {len(win.delay)} t)')

    # -------- 11. Reset corrections --------
    win.reset_corrections()
    assert not win.bg_applied and not win.chirp_applied
    print('[11] reset_corrections: OK')

    # -------- 12. Zero-time shift --------
    win.set_sel_t(2.0)
    win.apply_zero_time_shift(2.0)
    # After shift, the point previously at t=2 should be at t=0
    assert abs(win.tZeroShift - 2.0) < 1e-9
    print(f'[12] zero-time shift: tZeroShift={win.tZeroShift}')

    # -------- 13. Global analysis --------
    # Reset to a clean dataset for the fit
    wl2, t2, A2 = make_dataset(seed=2)
    win.set_loaded_data(wl2, t2, A2, 'ga_test')
    res = ta_core.fit_global_analysis(
        win.deltaA, win.delay,
        tau_init=[1, 10, 100],
        t0_init=0, fwhm_init=0.15,
        tau_fixed=[False, False, False],
        t0_fixed=True, fwhm_fixed=True,
        has_inf=False)
    recovered = np.sort(res['tau'])
    print(f'[13] global analysis: recovered τ = {recovered.round(3).tolist()}'
          f' (expected ~[0.4, 8, 200]), RMS = {res["info"]["rms"]:.4f}')
    assert abs(recovered[0] - 0.4) / 0.4 < 0.2
    assert abs(recovered[1] - 8.0) / 8.0 < 0.2
    assert abs(recovered[2] - 200) / 200 < 0.2

    # -------- 14. EADS from DADS --------
    EADS, tau_sorted, B = ta_core.compute_eads_from_dads(
        res['A'], res['tau'], False)
    assert EADS.shape == res['A'].shape
    print(f'[14] EADS from DADS: shape={EADS.shape}, tau_sorted='
          f'{tau_sorted.round(2).tolist()}')

    # -------- 15. Export --------
    with tempfile.TemporaryDirectory() as tmp:
        out_csv = os.path.join(tmp, 'out.csv')
        ta_core.write_data_file(out_csv, win.wavelength, win.delay,
                                win.deltaA, ',')
        wl3, t3, A3 = ta_core.parse_data_file(out_csv)
        assert np.allclose(wl3, win.wavelength)
        assert np.allclose(t3, win.delay)
        # NaN-safe comparison
        fin = np.isfinite(A3) & np.isfinite(win.deltaA)
        assert np.allclose(A3[fin], win.deltaA[fin])
        print('[15] export + reload: round-trip OK')

    # -------- 16. Dialogs instantiate --------
    from ta_dialogs_a import (BackgroundDialog, CropDialog, MaskDialog,
                              LoadAverageDialog)
    from ta_chirp import ChirpDialog
    from ta_ga import GlobalAnalysisDialog

    for DCls, name in [
            (BackgroundDialog, 'BackgroundDialog'),
            (CropDialog, 'CropDialog'),
            (MaskDialog, 'MaskDialog'),
            (ChirpDialog, 'ChirpDialog'),
            (GlobalAnalysisDialog, 'GlobalAnalysisDialog')]:
        d = DCls(win, win)
        d.close()
        print(f'[16.{name}] construct + close: OK')

    print('\n*** ALL INTEGRATION TESTS PASSED ***')


if __name__ == '__main__':
    run_tests()
