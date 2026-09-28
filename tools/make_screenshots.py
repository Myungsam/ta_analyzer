"""
Render the README screenshots from synthetic data (no measurement data).

    python tools/make_screenshots.py

Writes docs/images/main_window.png and docs/images/crop_resample.png.
Runs off-screen (QT_QPA_PLATFORM=offscreen), so no window appears.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
# The offscreen plugin has no font database of its own; without this the
# Qt widgets render without any text.
if os.name == 'nt':
    os.environ.setdefault(
        'QT_QPA_FONTDIR',
        os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts'))

import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from PyQt5 import QtWidgets, QtGui  # noqa: E402

import ta_main  # noqa: E402
import ta_dialogs_a  # noqa: E402

OUT_DIR = os.path.join(ROOT, 'docs', 'images')


def synthetic_ta(seed: int = 0):
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
    A += rng.normal(scale=2e-4, size=A.shape)
    return wl, t, A


def save(widget: QtWidgets.QWidget, name: str):
    widget.show()
    QtWidgets.QApplication.processEvents()
    path = os.path.join(OUT_DIR, name)
    pix: QtGui.QPixmap = widget.grab()
    if not pix.save(path, 'PNG'):
        raise RuntimeError(f'could not write {path}')
    print(f'wrote {path} ({pix.width()}x{pix.height()})')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont('Segoe UI', 9))

    win = ta_main.TAAnalyzer()
    win.resize(1500, 950)
    wl, t, A = synthetic_ta()
    win.set_loaded_data(wl, t, A, 'Loaded: synthetic_TA.csv')
    win.set_sel_wl(650.0)
    win.set_sel_t(float(t[np.argmin(np.abs(t - 1.0))]))
    save(win, 'main_window.png')

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
    win.close()


if __name__ == '__main__':
    main()
