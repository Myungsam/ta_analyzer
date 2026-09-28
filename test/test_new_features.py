"""
Tests for the new features and bug fixes:
    1) 2D map data export to CSV / TSV / Excel
    2) Spectrum panel: zoom persists across delay changes
    3) Kinetics panel: zoom persists across wavelength changes
    4) Crop dialog: colorbar does not stack on repeated redraws
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import tempfile
import numpy as np
from PyQt5 import QtWidgets

import ta_core
import ta_main
from ta_dialogs_a import CropDialog, MaskDialog


def make_dataset(seed=1):
    rng = np.random.default_rng(seed)
    wl = np.linspace(400, 700, 60)
    t = np.concatenate([np.linspace(-1, 1, 20), np.logspace(0, 3, 50)[1:]])
    t = np.unique(np.round(t, 6))

    def gauss(c, w, a):
        return a * np.exp(-(wl - c) ** 2 / (2 * w ** 2))

    taus = [0.4, 8, 200]
    spectra = [gauss(450, 30, 1.0), gauss(500, 45, -0.6),
               gauss(610, 55, 0.3)]
    A = np.zeros((len(wl), len(t)))
    for tau, sp in zip(taus, spectra):
        A += np.outer(sp, ta_core.exp_irf_conv(t, tau, 0.0, 0.15))
    A += 0.005 * rng.standard_normal(A.shape)
    return wl, t, A


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # ----- Setup -----
    wl, t, A = make_dataset()
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'zoom_test')

    # ====================================================================
    # 1) 2D Export — CSV
    # ====================================================================
    tmpdir = tempfile.mkdtemp()
    csv_path = os.path.join(tmpdir, 'export_test.csv')
    ta_core.write_data_file(csv_path, win.wavelength, win.delay,
                            win.deltaA, ',')
    wl_b, t_b, A_b = ta_core.parse_data_file(csv_path)
    assert np.allclose(wl_b, win.wavelength)
    assert np.allclose(t_b, win.delay)
    fin = np.isfinite(A_b) & np.isfinite(win.deltaA)
    assert np.allclose(A_b[fin], win.deltaA[fin])
    print('[1.CSV]  export → reload round-trip OK')

    # ====================================================================
    # 2) 2D Export — TSV
    # ====================================================================
    tsv_path = os.path.join(tmpdir, 'export_test.tsv')
    ta_core.write_data_file(tsv_path, win.wavelength, win.delay,
                            win.deltaA, '\t')
    wl_b, t_b, A_b = ta_core.parse_data_file(tsv_path)
    assert np.allclose(wl_b, win.wavelength)
    assert np.allclose(t_b, win.delay)
    print('[2.TSV]  export → reload round-trip OK')

    # ====================================================================
    # 3) 2D Export — Excel (.xlsx)
    # ====================================================================
    xlsx_path = os.path.join(tmpdir, 'export_test.xlsx')
    ta_core.write_data_excel(xlsx_path, win.wavelength, win.delay,
                             win.deltaA)
    assert os.path.exists(xlsx_path) and os.path.getsize(xlsx_path) > 0
    # Re-read using openpyxl directly to verify structure
    from openpyxl import load_workbook
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    # Row 1: corner + delays
    assert rows[0][0] is None
    delays_back = np.array([float(v) for v in rows[0][1:]])
    assert np.allclose(delays_back, win.delay)
    # Row 2..M+1: wavelength + ΔA row
    wls_back = np.array([float(r[0]) for r in rows[1:]])
    assert np.allclose(wls_back, win.wavelength)
    print(f'[3.XLSX] export OK — {len(rows)} rows × {len(rows[0])} cols, '
          f'size={os.path.getsize(xlsx_path)} B')

    # ====================================================================
    # 4) Spectrum zoom is preserved across delay changes
    # ====================================================================
    win.set_sel_t(2.0)
    win._draw_spectrum()
    assert win._spec_xlim is None  # not zoomed yet

    # Simulate user pan/zoom: directly manipulate axes (this is what the
    # matplotlib navigation toolbar does internally), which fires the
    # xlim_changed / ylim_changed callbacks we registered.
    ax_s = win.canvas_spec.ax
    ax_s.set_xlim(450.0, 600.0)
    ax_s.set_ylim(-0.4, 0.7)
    # The callback should have stored the user's zoom
    assert win._spec_xlim is not None
    assert abs(win._spec_xlim[0] - 450.0) < 1e-6
    assert abs(win._spec_xlim[1] - 600.0) < 1e-6
    assert abs(win._spec_ylim[0] - (-0.4)) < 1e-6
    print(f'[4.SpecZoom] user zoom captured: x={win._spec_xlim}, '
          f'y={win._spec_ylim}')

    # Now change the selected delay and re-draw — zoom must be preserved
    win.set_sel_t(20.0)
    xl_after = ax_s.get_xlim()
    yl_after = ax_s.get_ylim()
    assert abs(xl_after[0] - 450.0) < 1e-6
    assert abs(xl_after[1] - 600.0) < 1e-6
    assert abs(yl_after[0] - (-0.4)) < 1e-6
    assert abs(yl_after[1] - 0.7) < 1e-6
    print(f'[4.SpecZoom] zoom preserved after delay change: '
          f'x={xl_after}, y={yl_after}')

    # Reset zoom
    win._reset_spec_zoom()
    assert win._spec_xlim is None
    assert win._spec_ylim is None
    print('[4.SpecZoom] reset works')

    # ====================================================================
    # 5) Kinetics zoom is preserved across wavelength changes
    # ====================================================================
    win.delay_scale_mode = 'linear'
    win.set_sel_wl(500.0)
    win._draw_kinetics()
    # All kinetics zoom slots start as None
    assert all(x is None for x in win._kin_xlims)

    # Simulate user zoom
    ax_k = win.fig_kin.axes[0]
    ax_k.set_xlim(0.0, 50.0)
    ax_k.set_ylim(-0.3, 0.5)
    assert win._kin_xlims[0] is not None
    assert abs(win._kin_xlims[0][0] - 0.0) < 1e-6
    assert abs(win._kin_xlims[0][1] - 50.0) < 1e-6
    print(f'[5.KinZoom] user zoom captured: x={win._kin_xlims[0]}')

    # Change wavelength and verify zoom holds
    win.set_sel_wl(600.0)
    ax_k = win.fig_kin.axes[0]
    xl = ax_k.get_xlim()
    yl = ax_k.get_ylim()
    assert abs(xl[0] - 0.0) < 1e-6
    assert abs(xl[1] - 50.0) < 1e-6
    assert abs(yl[0] - (-0.3)) < 1e-6
    print(f'[5.KinZoom] zoom preserved after λ change: x={xl}, y={yl}')

    # Test split-mode: 2 panels, each with own zoom
    win.delay_scale_mode = 'split'
    win.split_threshold = 1.0
    win._draw_kinetics()
    assert len(win.fig_kin.axes) == 2
    win.fig_kin.axes[0].set_xlim(-0.5, 0.5)
    win.fig_kin.axes[1].set_xlim(2.0, 100.0)
    assert win._kin_xlims[0] is not None
    assert win._kin_xlims[1] is not None
    print(f'[5.KinZoom-split] both panels captured zoom independently: '
          f'left={win._kin_xlims[0]}, right={win._kin_xlims[1]}')

    # Reset
    win._reset_kin_zoom()
    assert all(x is None for x in win._kin_xlims)
    print('[5.KinZoom] reset works')

    # ====================================================================
    # 6) Crop dialog does NOT stack colorbars
    # ====================================================================
    crop = CropDialog(win, win)
    initial_axes = len(crop.canvas.fig.axes)
    print(f'[6.Crop] axes after first draw: {initial_axes}')

    # Trigger many redraws by changing the wl/t edits (this is what the
    # user does when sliding wavelength/delay limits)
    for w_min in (410, 420, 430, 440, 450, 460, 470, 480):
        crop.ed_wl_min.setValue(float(w_min))   # fires draw_preview
        crop.ed_t_min.setValue(-0.5)
        crop.ed_t_max.setValue(50.0)
    final_axes = len(crop.canvas.fig.axes)
    print(f'[6.Crop] axes after 8 redraws: {final_axes}')
    # After the fix: should remain at 2 axes (main + colorbar) and NOT grow
    assert final_axes == initial_axes, (
        f'Colorbars are stacking! Started with {initial_axes} axes, '
        f'now have {final_axes}.')
    print('[6.Crop] colorbar NOT stacking — bug fixed')
    crop.close()

    # ====================================================================
    # 7) Mask dialog: same fix
    # ====================================================================
    mask = MaskDialog(win, win)
    n0 = len(mask.canvas.fig.axes)
    # add and remove regions, each triggers draw_map
    for _ in range(5):
        mask.draw_map()
    n1 = len(mask.canvas.fig.axes)
    print(f'[7.Mask] axes after 5 redraws: {n0} → {n1}')
    assert n1 == n0
    print('[7.Mask] colorbar NOT stacking — bug fixed')
    mask.close()

    # ====================================================================
    # Cleanup
    # ====================================================================
    for f in (csv_path, tsv_path, xlsx_path):
        if os.path.exists(f):
            os.unlink(f)
    os.rmdir(tmpdir)
    print('\n*** ALL NEW-FEATURE TESTS PASSED ***')


if __name__ == '__main__':
    main()
