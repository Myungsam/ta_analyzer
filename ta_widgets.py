"""
TA Analyzer - GUI widgets (part 1): common matplotlib canvas and helpers.
"""
from __future__ import annotations

import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap

import ta_core


# =====================================================================
# Matplotlib canvas embedded in Qt
# =====================================================================
class MplCanvas(FigureCanvas):
    """Generic single-axes matplotlib canvas."""

    def __init__(self, parent=None, nrows=1, ncols=1,
                 figsize=(5.0, 4.0), tight=True):
        self.fig = Figure(figsize=figsize, constrained_layout=tight)
        super().__init__(self.fig)
        self.setParent(parent)
        self.axes_list = []
        for r in range(nrows):
            for c in range(ncols):
                ax = self.fig.add_subplot(nrows, ncols, r * ncols + c + 1)
                self.axes_list.append(ax)
        self.ax = self.axes_list[0]
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.updateGeometry()

    def clear_all(self):
        for ax in self.axes_list:
            ax.clear()

    def with_toolbar(self, parent=None) -> QtWidgets.QWidget:
        """Return a vertical widget containing the navigation toolbar + canvas.

        The created NavigationToolbar instance is also stored on
        ``self.toolbar`` so callers can hook ``home``/``back``/``forward``
        actions to clear application-side zoom state.
        """
        w = QtWidgets.QWidget(parent)
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        tb = NavigationToolbar(self, w)
        self.toolbar = tb
        lay.addWidget(tb)
        lay.addWidget(self)
        return w


# =====================================================================
# Small Qt helpers
# =====================================================================
def make_label(text, align='left'):
    lbl = QtWidgets.QLabel(text)
    if align == 'right':
        lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    elif align == 'center':
        lbl.setAlignment(QtCore.Qt.AlignCenter)
    else:
        lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
    return lbl


def make_double_edit(value=0.0, parent=None, decimals=6,
                     minv=-1e12, maxv=1e12) -> QtWidgets.QDoubleSpinBox:
    sb = QtWidgets.QDoubleSpinBox(parent)
    sb.setDecimals(decimals)
    sb.setRange(minv, maxv)
    sb.setValue(float(value))
    sb.setKeyboardTracking(False)
    return sb


def make_int_edit(value=1, parent=None, minv=1, maxv=1000000):
    sb = QtWidgets.QSpinBox(parent)
    sb.setRange(minv, maxv)
    sb.setValue(int(value))
    sb.setKeyboardTracking(False)
    return sb


def style_button(btn, bg=None, fg=None):
    """Apply a coloured look to a QPushButton."""
    parts = []
    if bg:
        parts.append(f"background-color: {bg}")
    if fg:
        parts.append(f"color: {fg}")
    if parts:
        btn.setStyleSheet("; ".join(parts) + "; font-weight: 600; padding: 4px 8px;")


def info_box(parent, title, text):
    QtWidgets.QMessageBox.information(parent, title, text)


def warn_box(parent, title, text):
    QtWidgets.QMessageBox.warning(parent, title, text)


def ask_save_path(parent, caption, default='out.csv',
                  filt='CSV (*.csv);;DAT (*.dat);;TXT (*.txt);;All files (*)'):
    path, sel = QtWidgets.QFileDialog.getSaveFileName(
        parent, caption, default, filt)
    return path, sel


def ask_open_paths(parent, caption,
                   filt='Data (*.csv *.dat *.txt);;All files (*)',
                   multi=False):
    if multi:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            parent, caption, '', filt)
        return paths
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent, caption, '', filt)
    return path


# =====================================================================
# Heatmap helper (draws TA 2D map)
# =====================================================================
def draw_heatmap(ax, wl, t, data, cmap_array, clim, y_scale='linear',
                 show_cbar=True, cbar_ax=None):
    """Draw the TA 2D map (wavelength on x, delay on y).

    data: (M, N) where M = wavelengths, N = delays
    Returns the image handle (pcolormesh / QuadMesh).
    """
    ax.clear()
    wl = np.asarray(wl).ravel()
    t = np.asarray(t).ravel()
    data = np.asarray(data)
    cmap = ListedColormap(cmap_array)

    # pcolormesh expects the data axes to match (Y, X) = (delay, wavelength)
    # Our data is (wavelength, delay); transpose before plotting.
    # Use shading='nearest' so each cell is centered on its (wl, t) point.
    Z = data.T
    im = ax.pcolormesh(wl, t, Z, cmap=cmap, vmin=clim[0], vmax=clim[1],
                       shading='nearest')
    ax.set_yscale(y_scale)
    ax.set_xlim(wl.min(), wl.max())
    # y-limits default to visible data (log-safe)
    if y_scale == 'log':
        pos = t[t > 0]
        if pos.size:
            ax.set_ylim(pos.min(), t.max())
    else:
        ax.set_ylim(t.min(), t.max())

    cb = None
    if show_cbar:
        cb = ax.figure.colorbar(im, ax=cbar_ax if cbar_ax is not None else ax)
        cb.set_label(r'$\Delta$A')
    return im, cb


def compute_zlim(data, z_min=None, z_max=None):
    """Symmetric auto-range when explicit z_min/z_max are not set."""
    if (z_min is not None and z_max is not None and z_min < z_max):
        return float(z_min), float(z_max)
    if data is None or data.size == 0:
        return -1.0, 1.0
    cl = np.nanmax(np.abs(data))
    if not np.isfinite(cl) or cl == 0:
        cl = 1.0
    return -float(cl), float(cl)
