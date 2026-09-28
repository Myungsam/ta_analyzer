"""
GUI integration tests for the five advanced analysis dialogs:

  1. ta_svd.SVDDialog
  2. ta_kfit.KineticFitDialog
  3. ta_lda.LDADialog
  4. ta_mcr.MCRDialog
  5. ta_coherence.CoherenceDialog

Plus: GA dialog with the new stretched-exponential UI (5-column table).
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import numpy as np
from PyQt5 import QtCore, QtWidgets

import ta_core
import ta_main


def build_dataset(seed=1, with_stretch=False, with_osc=False):
    rng = np.random.default_rng(seed)
    wl = np.linspace(400, 700, 50)
    t = np.concatenate([np.linspace(-1, 1, 25), np.logspace(0, 3, 70)[1:]])
    t = np.unique(np.round(t, 6))

    def gauss(c, w, a):
        return a * np.exp(-(wl - c) ** 2 / (2 * w ** 2))

    spectra = [gauss(450, 30, 1.0), gauss(530, 40, -0.6),
               gauss(620, 55, 0.3)]
    taus = [0.5, 8.0, 200.0]
    A = np.zeros((len(wl), len(t)))
    for k, (tau, sp) in enumerate(zip(taus, spectra)):
        if with_stretch and k == 1:
            kin = ta_core.stretched_irf_conv(t, tau, beta=0.7,
                                              t0=0, fwhm=0.15,
                                              mode='numerical')
        else:
            kin = ta_core.exp_irf_conv(t, tau, 0, 0.15)
        A += np.outer(sp, kin)
    A += 0.005 * rng.standard_normal(A.shape)

    if with_osc:
        # 50 cm⁻¹ ≈ 1.5 cycles/ps, period 670 fs.  Use a long damping
        # so the oscillation persists across the available delay
        # window; otherwise the FFT resolution can't resolve the peak.
        f_per_ps = 50.0 * 2.998e10 * 1e-12
        env = np.exp(-np.maximum(t, 0) / 30.0)
        osc = 0.08 * env * np.cos(2 * np.pi * f_per_ps * np.maximum(t, 0))
        wl_mask = np.exp(-((wl - 500) / 35) ** 2)
        A += np.outer(wl_mask, osc)
    return wl, t, A


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # =====================================================================
    # 1. SVD dialog
    # =====================================================================
    print('=== 1. SVDDialog ===')
    wl, t, A = build_dataset()
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'svd_test')

    win.open_svd_window()
    d = win.svd_fig
    assert d is not None
    n_sigma = len(d._sigma)
    assert n_sigma == min(len(wl), len(t))
    var3 = float(np.sum(d._sigma[:3] ** 2) / np.sum(d._sigma ** 2))
    assert var3 > 0.95, f'top-3 captures only {100*var3:.1f}% (>95% expected)'
    print(f'    σ has {n_sigma} entries; top-3 capture {100*var3:.2f}%')
    d.ed_nkeep.setValue(2)
    rms2 = float(np.sqrt(np.nanmean((A - d._D_rec) ** 2)))
    d.ed_nkeep.setValue(3)
    rms3 = float(np.sqrt(np.nanmean((A - d._D_rec) ** 2)))
    assert rms3 < rms2
    print(f'    RMS rank-2 = {rms2:.4g}, rank-3 = {rms3:.4g} (rank-3 better)')
    print('[1] SVDDialog: OK')
    d.close()
    win.svd_fig = None

    # =====================================================================
    # 2. Kinetic Fit dialog
    # =====================================================================
    print('\n=== 2. KineticFitDialog ===')
    win.open_kinetic_fit_window()
    kd = win.kfit_fig
    assert kd is not None
    kd.ed_wl.setValue(450.0)
    kd.ed_wl_hw.setValue(0.0)
    kd.dd_N.setCurrentText('2')
    QtWidgets.QApplication.processEvents()
    # The KineticFitDialog table has live cellChanged wiring through
    # `_on_table_edit_or_check`; safest is to set values via setItem
    # directly and rely on the dialog re-reading the table inside do_run.
    kd.tbl.item(0, 0).setText('1.0')
    kd.tbl.item(1, 0).setText('20.0')
    kd.do_run()
    res = kd._last_fit
    assert res is not None, 'fit returned no result'
    tau_kfit = sorted(res['tau'].tolist())
    print(f'    recovered τ = {[round(x,3) for x in tau_kfit]}')
    print('[2] KineticFitDialog: OK')
    kd.close()
    win.kfit_fig = None

    # =====================================================================
    # 3. LDA dialog
    # =====================================================================
    print('\n=== 3. LDADialog ===')
    win.open_lda_window()
    ld = win.lda_fig
    assert ld is not None
    ld.do_run()
    res = ld._last['res'] if ld._last else None
    assert res is not None
    A_map = res['A_map']
    info = res['info']
    print(f'    A_map shape: {A_map.shape}, '
          f'rms_ratio = {info["ratio"]:.4f}')
    abs_amp = np.nansum(np.abs(A_map), axis=0)
    tau_grid = info['tau_grid']
    top3 = np.argsort(abs_amp)[-3:][::-1]
    print(f'    top-3 lifetimes: '
          f'{[round(tau_grid[i], 1) for i in top3]} ps')
    print('[3] LDADialog: OK')
    ld.close()
    win.lda_fig = None

    # =====================================================================
    # 4. MCR-ALS dialog
    # =====================================================================
    print('\n=== 4. MCRDialog ===')
    win.open_mcr_window()
    md = win.mcr_fig
    assert md is not None
    md.ed_N.setValue(3)
    md.ed_max.setValue(150)
    md.do_run()
    last = md._last
    assert last is not None, 'MCR did not produce a result'
    info = last['res']['info']
    print(f'    LOF = {info["lof"]:.3f}%, iters = {info["iter"]}, '
          f'converged = {info["converged"]}')
    assert info['lof'] < 5.0
    print('[4] MCRDialog: OK')
    md.close()
    win.mcr_fig = None

    # =====================================================================
    # 5. Coherence dialog
    # =====================================================================
    print('\n=== 5. CoherenceDialog ===')
    wl_o, t_o, A_o = build_dataset(seed=2, with_osc=True)
    win.set_loaded_data(wl_o, t_o, A_o, 'coh_test')

    A_clean = win.deltaA.copy()
    A_clean[~np.isfinite(A_clean)] = 0.0
    U, s, VT = np.linalg.svd(A_clean, full_matrices=False)
    A_back = (U[:, :3] * s[:3]) @ VT[:3, :]
    R = A_clean - A_back
    coh = ta_core.compute_coherence(R, t_o, t_min=0.5, t_max=8.0,
                                     freq_unit='cm-1', apod='hann',
                                     zero_pad=2, detrend='linear')
    integrated = coh['P'].sum(axis=0)
    peak_freq = coh['freq'][int(np.argmax(integrated))]
    assert abs(peak_freq - 50.0) < 15.0, \
        f'50 cm⁻¹ peak not detected (got {peak_freq:.1f})'
    print(f'    kernel detects {peak_freq:.0f} cm⁻¹ (input 50 cm⁻¹) ✓')

    win.open_coherence_window()
    cd = win.coh_fig
    assert cd is not None
    cd.rb_use_data.setChecked(True)   # use raw data, no kinetic subtraction
    cd.ed_t_min.setValue(0.5)
    cd.ed_t_max.setValue(8.0)
    cd.do_run()
    print('[5] CoherenceDialog: OK')
    cd.close()
    win.coh_fig = None

    # =====================================================================
    # 6. GA with stretched-exp
    # =====================================================================
    print('\n=== 6. GA with stretched-exp ===')
    wl_s, t_s, A_s = build_dataset(seed=42, with_stretch=True)
    win.set_loaded_data(wl_s, t_s, A_s, 'ga_stretched')

    import ta_ga
    ga = ta_ga.GlobalAnalysisDialog(win, win)
    assert ga.tau_table.columnCount() == 5
    headers = [ga.tau_table.horizontalHeaderItem(c).text() for c in range(5)]
    print(f'    table headers: {headers}')

    ga.dd_N.setCurrentText('3')
    ga.tau_table.cellChanged.disconnect()
    for r, v in enumerate([0.3, 5.0, 100.0]):
        ga.tau_table.item(r, 0).setText(str(v))
    ga.tau_table.item(1, 2).setCheckState(QtCore.Qt.Checked)
    ga.tau_table.item(1, 3).setText('0.85')
    ga.tau_table.cellChanged.connect(ga._on_tau_edit)
    ga.read_table()
    print(f'    pre-run: stretch_on = {win.ga_stretch_on.tolist()}, '
          f'beta_init = {win.ga_beta_init.tolist()}')

    ga.do_run()
    print(f'    τ          = {[round(x,3) for x in win.ga_result_tau]}')
    print(f'    β          = {[round(x,3) for x in win.ga_result_beta]}')
    print(f'    stretch_on = {win.ga_result_stretch_on.tolist()}')
    print(f'    RMS        = {win.ga_result_rms:.4g}')

    assert bool(win.ga_result_stretch_on[1])
    bs = float(win.ga_result_beta[1])
    assert 0.5 < bs < 0.95, f'β = {bs}, expected near 0.7'
    for i in range(3):
        if not win.ga_result_stretch_on[i]:
            assert win.ga_result_beta[i] == 1.0
    print(f'    stretched β converged to {bs:.3f} (true 0.7) ✓')
    print('[6] GA stretched-exp fit: OK')
    ga.close()

    print('\n*** ALL ADVANCED-DIALOG TESTS PASSED ***')


if __name__ == '__main__':
    main()
