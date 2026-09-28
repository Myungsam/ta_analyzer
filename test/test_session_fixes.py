"""
Tests for the fixes requested by the user from the screenshots:

  1. CropDialog & MaskDialog: colorbar must not stack / shrink the
     main axes width across redraws.
  2. Display t_min / t_max: changing these spinboxes must update the
     y-axis limits of the 2D map (the in-place fast path was missing
     this update).
  3. Reset-zoom buttons removed from spectrum and kinetics panels;
     pressing Home in the matplotlib navigation toolbar must clear
     our application-side zoom cache too.
  4. Delay control rebuilt as an INDEX spinbox + value display label,
     stepping through the actually-recorded delay points one by one.
  5. Global Analysis dialog accepts a [t_min, t_max] fit window and
     restricts the fit / display to it.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import numpy as np
from PyQt5 import QtWidgets

import ta_core
import ta_main
from ta_dialogs_a import CropDialog, MaskDialog
from ta_ga import GlobalAnalysisDialog


def make_dataset(seed=1):
    rng = np.random.default_rng(seed)
    wl = np.linspace(400, 700, 80)
    t = np.concatenate([np.linspace(-1, 1, 25), np.logspace(0, 3, 50)[1:]])
    t = np.unique(np.round(t, 6))
    A = np.zeros((len(wl), len(t)))
    for tau, c, w, amp in zip([0.4, 8, 200], [450, 500, 610],
                                [30, 45, 55], [1.0, -0.6, 0.3]):
        sp = amp * np.exp(-(wl - c) ** 2 / (2 * w ** 2))
        A += np.outer(sp, ta_core.exp_irf_conv(t, tau, 0.0, 0.15))
    A += 0.005 * rng.standard_normal(A.shape)
    return wl, t, A


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # =====================================================================
    # 1. Crop / Mask dialogs: no colorbar stacking, no main-axes shrinkage
    # =====================================================================
    print('=== 1. Crop / Mask colorbar stability ===')
    wl, t, A = ta_core.parse_data_file(
        '/mnt/user-data/uploads/_TA_spectra_Accumulated.csv')
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'real')

    crop = CropDialog(win, win)
    bb0 = crop.canvas.ax.get_position()
    n0 = len(crop.canvas.fig.axes)
    for w_min in [400, 425, 450, 475, 500, 525, 550, 575, 600]:
        crop.ed_wl_min.setValue(float(w_min))
    bb1 = crop.canvas.ax.get_position()
    n1 = len(crop.canvas.fig.axes)
    assert n1 == n0 == 2, f'CropDialog axes count drifted: {n0}->{n1}'
    assert abs(bb0.width - bb1.width) < 0.005, (
        f'CropDialog main axes shrunk: {bb0.width:.3f}->{bb1.width:.3f}')
    print(f'[1.Crop] main width {bb1.width:.3f} stable across 9 redraws — OK')
    crop.close()

    mask = MaskDialog(win, win)
    bb0 = mask.canvas.ax.get_position()
    for _ in range(8):
        mask.draw_map()
    bb1 = mask.canvas.ax.get_position()
    assert abs(bb0.width - bb1.width) < 0.005
    print(f'[1.Mask] main width {bb1.width:.3f} stable across 8 redraws — OK')
    mask.close()

    # =====================================================================
    # 2. Display t_min / t_max actually updates the 2D map y-limits
    # =====================================================================
    print('\n=== 2. Display t_min / t_max applied to 2D map ===')
    win.delay_scale_mode = 'linear'
    win.update_all()
    win._on_main_view_change('min', -2.0)
    win._on_main_view_change('max', 10.0)
    ax = win._ax2d_list[0]
    ylo, yhi = ax.get_ylim()
    print(f'[2] requested t-range: [-2, 10] ps')
    print(f'    actual axes ylim:  [{ylo:.3g}, {yhi:.3g}]')
    assert ylo <= -1.5, f'y-min not applied (got {ylo})'
    assert yhi <= 12.0, f'y-max not applied (got {yhi})'
    win.reset_main_view()
    ylo2, yhi2 = win._ax2d_list[0].get_ylim()
    print(f'    after reset_main_view: [{ylo2:.3g}, {yhi2:.3g}]')
    assert (yhi2 - ylo2) > (yhi - ylo) * 5
    print('[2] in-place y-limit update works — OK')

    # =====================================================================
    # 3. Reset-zoom buttons removed; Home action clears zoom cache
    # =====================================================================
    print('\n=== 3. Reset-zoom buttons removed & Home hooks zoom cache ===')
    spec_buttons = win.canvas_spec.parentWidget().findChildren(
        QtWidgets.QPushButton)
    spec_btn_texts = [b.text() for b in spec_buttons]
    print(f'    Spectrum panel buttons: {spec_btn_texts}')
    assert 'Reset zoom' not in spec_btn_texts

    kin_buttons = win.canvas_kin.parentWidget().findChildren(
        QtWidgets.QPushButton)
    kin_btn_texts = [b.text() for b in kin_buttons]
    print(f'    Kinetics panel buttons: {kin_btn_texts}')
    assert 'Reset zoom' not in kin_btn_texts

    assert hasattr(win.canvas_spec, 'toolbar')
    assert hasattr(win.canvas_kin, 'toolbar')

    win._draw_spectrum()
    win.canvas_spec.ax.set_xlim(450, 600)
    win.canvas_spec.ax.set_ylim(-0.5, 0.5)
    assert win._spec_xlim is not None
    win.canvas_spec.toolbar.home()
    assert win._spec_xlim is None, (
        f'Home did not clear spec_xlim; got {win._spec_xlim}')
    assert win._spec_ylim is None
    print('[3] Spectrum Home action clears cached zoom — OK')

    win._draw_kinetics()
    win.fig_kin.axes[0].set_xlim(0, 50)
    win.fig_kin.axes[0].set_ylim(-0.3, 0.4)
    assert win._kin_xlims[0] is not None
    win.canvas_kin.toolbar.home()
    assert all(v is None for v in win._kin_xlims), (
        f'Home did not clear kin_xlims; got {win._kin_xlims}')
    assert all(v is None for v in win._kin_ylims)
    print('[3] Kinetics Home action clears cached zoom — OK')

    # =====================================================================
    # 4. Delay control: index spinbox + recorded delay value
    # =====================================================================
    print('\n=== 4. Delay control = INDEX spinbox + value label ===')
    assert hasattr(win, 'ed_sel_t_idx')
    assert not hasattr(win, 'ed_sel_t')
    assert win.ed_sel_t_idx.minimum() == 0
    assert win.ed_sel_t_idx.maximum() == len(win.delay) - 1
    print(f'    index range: 0 .. {win.ed_sel_t_idx.maximum()}')

    cur_idx = win._selT_idx
    cur_t = win.selT
    win.ed_sel_t_idx.setValue(cur_idx + 1)
    new_idx = win._selT_idx
    new_t = win.selT
    assert new_idx == cur_idx + 1
    assert new_t == win.delay[new_idx]
    assert new_t != cur_t
    assert new_t in win.delay.tolist()
    print(f'    idx {cur_idx} -> {new_idx}: t = {cur_t:.4g} -> {new_t:.4g} ps')
    print(f'    label text: "{win.lbl_sel_t_val.text()}"')
    assert 'pts' in win.lbl_sel_t_val.text()

    for k in range(10, 30, 3):
        win.ed_sel_t_idx.setValue(k)
        assert win.selT == win.delay[k], (
            f'Index {k} did not land on delay[{k}]: '
            f'got selT={win.selT}, expected {win.delay[k]}')
    print('[4] every index step lands on a recorded delay point — OK')

    # =====================================================================
    # 5. Global Analysis t_min / t_max fit window
    # =====================================================================
    print('\n=== 5. Global Analysis t-range window ===')
    wl2, t2, A2 = make_dataset(seed=42)
    win.set_loaded_data(wl2, t2, A2, 'synth_ga')

    ga = GlobalAnalysisDialog(win, win)
    assert hasattr(ga, 'ed_t_min')
    assert hasattr(ga, 'ed_t_max')
    assert hasattr(ga, 'btn_t_full')
    assert hasattr(ga, 'lbl_t_count')
    print(f'    Initial fit window: '
          f'[{ga.ed_t_min.value():.3g}, {ga.ed_t_max.value():.3g}] ps')
    print(f'    count label: "{ga.lbl_t_count.text()}"')

    ga.ed_t_min.setValue(-0.5)
    ga.ed_t_max.setValue(20.0)
    ga._update_t_count_label()
    print(f'    after narrowing: "{ga.lbl_t_count.text()}"')
    n_in = int(np.sum((win.delay >= -0.5) & (win.delay <= 20.0)))
    assert str(n_in) in ga.lbl_t_count.text(), (
        f'count label did not update: {ga.lbl_t_count.text()}')

    print('    running global fit on windowed data ...')
    ga.do_run()
    assert win.ga_result_delay is not None, 'fit did not store sliced delays'
    assert len(win.ga_result_delay) == n_in, (
        f'fit ran on {len(win.ga_result_delay)} points but window has {n_in}')
    assert win.ga_t_min == -0.5
    assert win.ga_t_max == 20.0
    assert win.ga_result_t_window == (-0.5, 20.0)
    expected_D = win.deltaA[:, (win.delay >= -0.5) & (win.delay <= 20.0)]
    assert np.array_equal(win.ga_result_data, expected_D)
    tau_sorted = np.sort(win.ga_result_tau)
    print(f'    recovered tau (windowed) = {tau_sorted.round(3).tolist()}')
    assert abs(tau_sorted[0] - 0.4) / 0.4 < 0.5, (
        f'fast tau {tau_sorted[0]} far from expected ~0.4 ps')

    ga.btn_t_full.click()
    assert ga.ed_t_min.value() == float(win.delay[0])
    assert ga.ed_t_max.value() == float(win.delay[-1])
    print('[5] t-range window applied & "Full" button restores — OK')
    ga.close()

    win.close()
    print('\n*** ALL FIVE FIXES VERIFIED ***')


if __name__ == '__main__':
    main()
