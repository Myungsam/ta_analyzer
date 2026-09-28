"""
Tests for the consistency fixes around zoom-state preservation:

  * reset_corrections      → drops both spectrum and kinetics zoom
  * apply_zero_time_shift  → drops only kinetics zoom (delay axis moved)
                             but keeps spectrum zoom (wavelength axis is
                             unaffected by a delay translation)
  * apply_chirp_shift      → numerical correctness of the vectorized
                             shift (must match the per-row np.interp
                             result it replaced)
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import numpy as np
from PyQt5 import QtWidgets

import ta_core
import ta_main


def make_dataset(seed=1):
    rng = np.random.default_rng(seed)
    wl = np.linspace(400, 700, 60)
    t = np.concatenate([np.linspace(-1, 1, 20), np.logspace(0, 3, 50)[1:]])
    t = np.unique(np.round(t, 6))
    A = np.zeros((len(wl), len(t)))
    for tau, c, w in zip([0.4, 8, 200], [450, 500, 610],
                          [30, 45, 55]):
        sp = np.exp(-(wl - c) ** 2 / (2 * w ** 2))
        A += np.outer(sp, ta_core.exp_irf_conv(t, tau, 0.0, 0.15))
    A += 0.005 * rng.standard_normal(A.shape)
    return wl, t, A


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    wl, t, A = make_dataset()
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'consistency')

    # ====================================================================
    # 1) reset_corrections drops BOTH zoom states
    # ====================================================================
    win.bg_applied = True
    win.bg_n = 3
    win.bg_spectrum = np.nanmean(win.deltaA_raw[:, :3], axis=1)
    win.update_all()
    # Simulate user zoom in BOTH panels
    win.canvas_spec.ax.set_xlim(450, 600)
    win.canvas_spec.ax.set_ylim(-0.5, 0.5)
    assert win._spec_xlim is not None
    assert win._spec_ylim is not None
    # Zoom kinetics
    win.fig_kin.axes[0].set_xlim(0, 50)
    win.fig_kin.axes[0].set_ylim(-0.3, 0.4)
    assert win._kin_xlims[0] is not None
    assert win._kin_ylims[0] is not None
    print('[1] both zoom states captured')

    # Reset corrections — must clear all zoom state
    win.reset_corrections()
    assert win._spec_xlim is None, \
        f'reset_corrections should clear spectrum zoom; got {win._spec_xlim}'
    assert win._spec_ylim is None
    assert all(x is None for x in win._kin_xlims), \
        f'reset_corrections should clear kinetics zoom; got {win._kin_xlims}'
    assert all(y is None for y in win._kin_ylims)
    # Corrections themselves should also be off
    assert not win.bg_applied
    assert not win.chirp_applied
    print('[1] reset_corrections cleared zoom in BOTH panels — OK')

    # ====================================================================
    # 2) apply_zero_time_shift drops kinetics zoom but keeps spectrum
    # ====================================================================
    # Re-establish zoom in both panels
    win.update_all()
    win.canvas_spec.ax.set_xlim(450, 600)
    win.canvas_spec.ax.set_ylim(-0.5, 0.5)
    win.fig_kin.axes[0].set_xlim(5, 80)
    win.fig_kin.axes[0].set_ylim(-0.4, 0.6)
    spec_xlim_before = win._spec_xlim
    spec_ylim_before = win._spec_ylim
    assert spec_xlim_before is not None
    assert win._kin_xlims[0] is not None
    print('[2] both panels zoomed before shift: '
          f'spec={spec_xlim_before}, kin={win._kin_xlims[0]}')

    # Pick a non-zero shift
    win.set_sel_t(2.0)
    win.apply_zero_time_shift(2.0)

    # Spectrum (wavelength axis) must NOT change
    assert win._spec_xlim == spec_xlim_before, (
        f'apply_zero_time_shift should not touch spectrum zoom. '
        f'Was {spec_xlim_before}, now {win._spec_xlim}')
    assert win._spec_ylim == spec_ylim_before
    print(f'[2] spectrum zoom preserved: {win._spec_xlim}')

    # Kinetics (delay axis) MUST be cleared (axis was translated)
    assert all(x is None for x in win._kin_xlims), (
        f'apply_zero_time_shift should clear kinetics zoom; got '
        f'{win._kin_xlims}')
    assert all(y is None for y in win._kin_ylims)
    print('[2] kinetics zoom cleared — delay axis was translated')

    # tZeroShift bookkeeping
    assert abs(win.tZeroShift - 2.0) < 1e-9
    print(f'[2] tZeroShift = {win.tZeroShift}')

    # ====================================================================
    # 3) Numerical correctness of apply_chirp_shift
    #    (matches the original per-row np.interp loop, including NaN
    #    handling at the time-axis boundaries)
    # ====================================================================
    wl2, t2, A2 = make_dataset(seed=42)
    chirp_p = np.array([0.3, 1.1, 1.5, 0.0])  # mild chirp

    # Reference: original explicit per-row loop
    t0 = ta_core.chirp_model(chirp_p, wl2)
    ref = np.full_like(A2, np.nan)
    for i in range(len(wl2)):
        if not np.isfinite(t0[i]):
            continue
        new_t = t2 + t0[i]
        row = np.interp(new_t, t2, A2[i, :])
        bad = (new_t < t2[0]) | (new_t > t2[-1])
        row[bad] = np.nan
        ref[i, :] = row

    # New vectorized-friendly implementation
    out = ta_core.apply_chirp_shift(A2, t2, wl2, chirp_p)

    # Same shape
    assert out.shape == ref.shape
    # NaN pattern must be identical
    assert np.array_equal(np.isnan(out), np.isnan(ref))
    # Finite values must be numerically equal (np.interp is deterministic)
    fin = np.isfinite(ref)
    assert np.allclose(out[fin], ref[fin]), (
        f'apply_chirp_shift differs from reference loop! '
        f'max diff = {np.max(np.abs(out[fin] - ref[fin]))}')
    print(f'[3] apply_chirp_shift matches reference — '
          f'NaN cells: {np.sum(np.isnan(out))}/{out.size}, '
          f'max diff: 0')

    print('\n*** ALL CONSISTENCY TESTS PASSED ***')


if __name__ == '__main__':
    main()
