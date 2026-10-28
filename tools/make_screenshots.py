"""
Render the README screenshots from synthetic data (no measurement data).

    python tools/make_screenshots.py                  # all scenes
    python tools/make_screenshots.py --scenes moving  # only some

Scenes: main → docs/images/main_window.png, crop → crop_resample.png,
moving → crop_moving_average.png.  Runs off-screen
(QT_QPA_PLATFORM=offscreen), so no window appears.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
# The offscreen plugin has no font database of its own; without this the
# Qt widgets render without any text.
if os.name == 'nt':
    os.environ.setdefault(
        'QT_QPA_FONTDIR',
        os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts'))

import argparse
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from PyQt5 import QtWidgets, QtGui  # noqa: E402

import ta_main  # noqa: E402
import ta_dialogs_a  # noqa: E402

OUT_DIR = os.path.join(ROOT, 'docs', 'images')


def synthetic_ta(seed: int = 0, noise: float = 2e-4):
    """Two-state TA map on a non-uniform λ grid (~0.33 nm steps).

    * ground-state bleach at 520 nm, recovering with τ = 300 ps
    * excited-state absorption at 650 nm, τ1 = 3 ps → τ2 = 300 ps
    * weak 150 cm⁻¹ coherent oscillation and measurement noise
    """
    rng = np.random.default_rng(seed)
    wl = 400.0 + np.cumsum(rng.uniform(0.27, 0.38, 1200))
    t = np.r_[np.linspace(-1.0, 1.0, 41), np.logspace(0.05, 3.0, 120)]
    tp = np.clip(t, 0, None)
    rise = 0.5 * (1 + np.tanh(t / 0.08))
    gsb = -np.exp(-((wl - 520) / 22) ** 2)[:, None] * np.exp(-tp / 300)[None, :]
    esa1 = 0.8 * np.exp(-((wl - 650) / 35) ** 2)[:, None] * np.exp(-tp / 3)[None, :]
    esa2 = 0.5 * np.exp(-((wl - 610) / 45) ** 2)[:, None] * (1 - np.exp(-tp / 3))[None, :] \
        * np.exp(-tp / 300)[None, :]
    osc = 0.03 * np.cos(2 * np.pi * 150 * 2.998e-2 * t)[None, :] * np.exp(-tp / 1.5)[None, :]
    A = (gsb + esa1 + esa2 + osc) * rise[None, :] * 0.02
    A += rng.normal(scale=noise, size=A.shape)
    return wl, t, A


def save(widget: QtWidgets.QWidget, name: str):
    widget.show()
    QtWidgets.QApplication.processEvents()
    path = os.path.join(OUT_DIR, name)
    pix: QtGui.QPixmap = widget.grab()
    if not pix.save(path, 'PNG'):
        raise RuntimeError(f'could not write {path}')
    print(f'wrote {path} ({pix.width()}x{pix.height()})')


SCENES = ('main', 'crop', 'moving', 'gating', 'kfit', 'ga', 'opju')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--scenes', default=','.join(SCENES),
                        help='comma-separated subset of ' + ', '.join(SCENES))
    args, _ = parser.parse_known_args(argv)
    scenes = {x.strip() for x in args.scenes.split(',') if x.strip()}
    unknown = scenes - set(SCENES)
    if unknown:
        parser.error(f'unknown scene(s): {", ".join(sorted(unknown))}')

    os.makedirs(OUT_DIR, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont('Segoe UI', 9))

    win = ta_main.TAAnalyzer()
    win.resize(1500, 950)
    wl, t, A = synthetic_ta()
    win.set_loaded_data(wl, t, A, 'Loaded: synthetic_TA.csv')
    win.set_sel_wl(650.0)
    win.set_sel_t(float(t[np.argmin(np.abs(t - 1.0))]))
    if 'main' in scenes:
        save(win, 'main_window.png')

    if 'crop' in scenes:
        dlg = ta_dialogs_a.CropDialog(win, win)
        dlg.resize(1400, 950)
        dlg.cb_resample.setChecked(True)
        dlg.ed_resample_dx.setValue(5.0)
        dlg.rb_avg.setChecked(True)
        dlg.ed_sel_t_idx.setValue(int(np.argmin(np.abs(t - 1.0))))
        dlg._resample_timer.stop()
        dlg._on_resample_changed()
        save(dlg, 'crop_resample.png')
        dlg.close()

    if 'moving' in scenes:
        # Noisier data so the smoothing is visible at a glance.
        wl_n, t_n, A_n = synthetic_ta(noise=1.5e-3)
        win.set_loaded_data(wl_n, t_n, A_n, 'Loaded: synthetic_TA_noisy.csv')
        dlg = ta_dialogs_a.CropDialog(win, win)
        dlg.resize(1400, 950)
        dlg.cb_resample.setChecked(True)
        dlg.rb_mov.setChecked(True)
        dlg.ed_window.setValue(15)
        dlg.dd_kernel.setCurrentIndex(dlg.dd_kernel.findData('gaussian'))
        dlg.ed_sel_t_idx.setValue(int(np.argmin(np.abs(t_n - 1.0))))
        dlg._resample_timer.stop()
        dlg._on_resample_changed()
        save(dlg, 'crop_moving_average.png')
        dlg.close()

    if 'gating' in scenes:
        # Noisy data: single vs integrated pins side by side.
        wl_n, t_n, A_n = synthetic_ta(noise=1.5e-3)
        win.set_loaded_data(wl_n, t_n, A_n, 'Loaded: synthetic_TA_noisy.csv')
        i1 = int(np.argmin(np.abs(t_n - 1.0)))
        win.set_sel_t(float(t_n[i1]))
        win.pin_spec()
        win.ed_spec_int_half.setValue(4)
        win.chk_spec_int.setChecked(True)
        win.set_sel_t(float(t_n[np.argmin(np.abs(t_n - 30.0))]))
        win.set_sel_wl(520.0)
        win.pin_kin()
        win.ed_kin_int_bw.setValue(20.0)
        win.chk_kin_int.setChecked(True)
        win.set_sel_wl(650.0)
        save(win, 'pin_gating.png')

    if 'kfit' in scenes:
        # Global kinetic fit of three pinned traces (one BW-integrated).
        import ta_kfit
        win.set_loaded_data(wl, t, A, 'Loaded: synthetic_TA.csv')
        for lam in (520.0, 610.0):
            win.set_sel_wl(lam)
            win.pin_kin()
        win.ed_kin_int_bw.setValue(10.0)
        win.chk_kin_int.setChecked(True)
        win.set_sel_wl(650.0)
        win.pin_kin()
        kd = ta_kfit.KineticFitDialog(win, win)
        kd.resize(1400, 900)
        kd.dd_N.setCurrentText('2')
        for r, v in enumerate((2.0, 200.0)):
            kd.tbl.item(r, 0).setText(str(v))
        kd.ed_fw.setValue(0.16)
        kd.do_import()
        kd.do_run()
        save(kd, 'kfit_global.png')
        kd.close()
        win.clear_kin_overlays()
        win.chk_kin_int.setChecked(False)

    if 'ga' in scenes:
        import ta_ga
        win.set_loaded_data(wl, t, A, 'Loaded: synthetic_TA.csv')
        ga = ta_ga.GlobalAnalysisDialog(win, win)
        ga.resize(1500, 950)
        ga.dd_N.setCurrentText('3')
        ga.tau_table.cellChanged.disconnect()
        for r, v in enumerate((1.0, 50.0, 500.0)):
            ga.tau_table.item(r, 0).setText(str(v))
        ga.tau_table.cellChanged.connect(ga._on_tau_edit)
        ga.read_table()
        ga.do_run()
        QtWidgets.QApplication.processEvents()
        save(ga, 'ga_stats.png')
        ga.close()

    if 'opju' in scenes:
        # opju dataset info with part of the sheets staged
        import tempfile
        import ta_opju_export
        win.set_loaded_data(wl, t, A, 'Loaded: synthetic_TA.csv')
        st = win.opju_stage
        st.stage_matrix('cropped_original', wl, t, A,
                        'synthetic_TA_2D_cropped_original.csv', 'cropped_original')
        st.stage_matrix('chirp_cropped', wl, t, A,
                        'synthetic_TA_2D_BG5_chirp_cropped.csv', 'BG5_chirp_cropped')
        tmp = tempfile.mkdtemp()
        for key, name, cols in (
                ('pinned_kinetics', 'synthetic_TA_pinned_kinetics.csv', 3),
                ('pinned_spectra', 'synthetic_TA_pinned_spectra.csv', 3),
                ('chirp', 'chirp_points.csv', 2)):
            p = os.path.join(tmp, name)
            np.savetxt(p, np.random.default_rng(0).normal(size=(10, cols)),
                       delimiter=',', header=','.join(['x'] * cols),
                       comments='')
            st.stage_file(key, p, f'{cols - 1} column(s)')
        dlg = ta_opju_export.OpjuDatasetDialog(win, win)
        # Offscreen Qt has no glyph fallback: use a font with Hangul so
        # the 저장 / 취소 buttons render (Windows itself falls back).
        dlg.setFont(QtGui.QFont('Malgun Gothic', 9))
        dlg.resize(1000, 480)
        save(dlg, 'opju_dataset_info.png')
        dlg.close()
        st.clear()
    win.close()


if __name__ == '__main__':
    main()
