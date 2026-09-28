"""
TA Analyzer GUI - sub-dialog windows (part 1):
    Background correction, Crop, Mask wavelengths, Load & Average.
"""
from __future__ import annotations

import os
import numpy as np
from PyQt5 import QtWidgets, QtCore

from ta_widgets import (
    MplCanvas, make_label, make_double_edit, make_int_edit,
    style_button, info_box, warn_box, ask_save_path, ask_open_paths,
    draw_heatmap, compute_zlim,
)
import ta_core


def _bounds_close(a: float, b: float) -> bool:
    """Equal within the 6-decimal rounding of the range spinboxes."""
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


# =====================================================================
# Background Correction window
# =====================================================================
class BackgroundDialog(QtWidgets.QDialog):
    """Dialog for averaging the first N delay points as a background spectrum."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.setWindowTitle('Background Correction')
        self.resize(720, 520)

        default_n = min(5, len(app.delay))
        if app.bg_applied:
            default_n = app.bg_n

        lay = QtWidgets.QGridLayout(self)

        lay.addWidget(make_label('# of initial delay points to average:',
                                 align='right'), 0, 0)
        self.ed_n = make_int_edit(default_n, minv=1, maxv=len(app.delay))
        lay.addWidget(self.ed_n, 0, 1)
        self.lbl_info = QtWidgets.QLabel('')
        lay.addWidget(self.lbl_info, 0, 2)

        self.canvas = MplCanvas(self)
        lay.addWidget(self.canvas.with_toolbar(self), 1, 0, 1, 3)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_apply = QtWidgets.QPushButton('Apply && Close')
        style_button(self.btn_apply, bg='#4fa35a', fg='white')
        self.btn_cancel = QtWidgets.QPushButton('Cancel')
        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_cancel)
        lay.addLayout(btn_row, 2, 0, 1, 3)

        self.ed_n.valueChanged.connect(self.draw_preview)
        self.btn_apply.clicked.connect(self.apply_bg)
        self.btn_cancel.clicked.connect(self.reject)

        lay.setRowStretch(1, 1)
        self.draw_preview()

    def draw_preview(self):
        app = self.app
        N = int(self.ed_n.value())
        if N < 1 or N > len(app.delay):
            return
        ax = self.canvas.ax
        ax.clear()
        cmap = np.linspace(0, 1, max(N, 2))
        for k in range(N):
            rgba = ta_core._mpl_cm.get_cmap('viridis')(cmap[k])
            ax.plot(app.wavelength, app.deltaA_raw[:, k], '-',
                    color=rgba, linewidth=0.7, alpha=0.6)
        bg = np.nanmean(app.deltaA_raw[:, :N], axis=1)
        ax.plot(app.wavelength, bg, 'k-', linewidth=2.0,
                label=f'Mean of first {N} point(s)')
        ax.axhline(0, color='k', linestyle=':')
        ax.grid(True)
        ax.legend(loc='best')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel(r'$\Delta$A')
        ax.set_title('Early-time spectra and their mean (background)')
        self.lbl_info.setText(
            f'Using t = {app.delay[0]:.4g} … {app.delay[N-1]:.4g} '
            f'{app.t_unit_txt()}')
        self.canvas.draw_idle()

    def apply_bg(self):
        app = self.app
        N = int(self.ed_n.value())
        if N < 1 or N > len(app.delay):
            return
        app.bg_spectrum = np.nanmean(app.deltaA_raw[:, :N], axis=1)
        app.bg_n = N
        app.bg_applied = True
        app.update_all()
        self.accept()


# =====================================================================
# Crop window
# =====================================================================
class CropDialog(QtWidgets.QDialog):
    """Dialog to select a rectangular (lambda, t) region of the original data."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.setWindowTitle('Crop Data')
        self.resize(1280, 820)

        self.wl_min_full = float(np.min(app.original_wavelength))
        self.wl_max_full = float(np.max(app.original_wavelength))
        self.t_min_full = float(np.min(app.original_delay))
        self.t_max_full = float(np.max(app.original_delay))

        wl_min0, wl_max0, t_min0, t_max0 = self._current_bounds()

        # ===== Pin / selection state =====
        # Drops are pinned spectra to delete+interpolate.  Stored as
        # t-VALUES (snapped to original_delay) so they survive any
        # subsequent crop without index aliasing.
        self._drop_t_values: list[float] = []
        self._drop_lines = []                 # red dashed h-lines on 2D map
        # Wavelength drops: same idea, but along the wavelength axis.
        # ``_kin_overlay_wl`` keeps its original attribute name (so
        # existing tests still see it) but is now the wavelength
        # drop-and-interpolate list, not just a visual overlay.
        self._kin_overlay_wl: list[float] = []
        self._kin_overlay_lines = []          # cyan dashed v-lines on 2D map
        # Cached preview: original_deltaA after interpolating the current
        # drops with the current method.  None when no drops are pinned
        # (then displays show original_deltaA directly).
        self._preview_deltaA = None
        self._preview_key = None              # (sorted drop tuple, method)
        self._preview_im = None               # pcolormesh handle (set_array)
        # Current selection (drives the spec/kin panels + 2D crosshair).
        # Start the crosshair at a sensible interior point.
        n_wl_full = int(len(app.original_wavelength))
        n_t_full = int(len(app.original_delay))
        self._sel_wl_idx = n_wl_full // 2
        self._sel_t_idx = n_t_full // 2
        self._sel_wl = float(app.original_wavelength[self._sel_wl_idx])
        self._sel_t = float(app.original_delay[self._sel_t_idx])

        outer = QtWidgets.QVBoxLayout(self)

        # ---- Crop range controls ----
        ctrl = QtWidgets.QGridLayout()
        ctrl.addWidget(make_label(u'λ range (nm):', align='right'), 0, 0)
        self.ed_wl_min = make_double_edit(wl_min0,
                                          minv=self.wl_min_full,
                                          maxv=self.wl_max_full)
        self.ed_wl_max = make_double_edit(wl_max0,
                                          minv=self.wl_min_full,
                                          maxv=self.wl_max_full)
        ctrl.addWidget(self.ed_wl_min, 0, 1)
        ctrl.addWidget(make_label('to', align='center'), 0, 2)
        ctrl.addWidget(self.ed_wl_max, 0, 3)
        self.btn_full_wl = QtWidgets.QPushButton(u'Full λ range')
        ctrl.addWidget(self.btn_full_wl, 0, 4)

        ctrl.addWidget(make_label(f't range ({app.t_unit_txt()}):', 'right'), 1, 0)
        self.ed_t_min = make_double_edit(t_min0,
                                         minv=self.t_min_full,
                                         maxv=self.t_max_full)
        self.ed_t_max = make_double_edit(t_max0,
                                         minv=self.t_min_full,
                                         maxv=self.t_max_full)
        ctrl.addWidget(self.ed_t_min, 1, 1)
        ctrl.addWidget(make_label('to', align='center'), 1, 2)
        ctrl.addWidget(self.ed_t_max, 1, 3)
        self.btn_full_t = QtWidgets.QPushButton('Full t range')
        ctrl.addWidget(self.btn_full_t, 1, 4)
        outer.addLayout(ctrl)

        # ---- Wavelength resampling (applied after the crop) ----
        gb_rs = QtWidgets.QGroupBox(u'Wavelength resampling')
        rs_row = QtWidgets.QHBoxLayout(gb_rs)
        self.cb_resample = QtWidgets.QCheckBox(u'Resample λ')
        self.cb_resample.setChecked(bool(app.resample_enabled))
        rs_row.addWidget(self.cb_resample)
        rs_row.addWidget(make_label(u'Δλ (nm):', 'right'))
        self.ed_resample_dx = make_double_edit(
            float(app.resample_dx), minv=0.001, maxv=1000.0, decimals=3)
        rs_row.addWidget(self.ed_resample_dx)
        self.rb_avg = QtWidgets.QRadioButton('Average')
        self.rb_dec = QtWidgets.QRadioButton('Decimate')
        (self.rb_dec if app.resample_mode == 'decimate'
         else self.rb_avg).setChecked(True)
        rs_row.addWidget(self.rb_avg)
        rs_row.addWidget(self.rb_dec)
        self.lbl_resample_info = QtWidgets.QLabel('')
        rs_row.addWidget(self.lbl_resample_info, stretch=1)
        rs_hint = QtWidgets.QLabel(
            'Bins of width Δλ start at the cropped λ_min.  Average: mean of '
            'the points in each bin.  Decimate: keep the point closest to '
            'each bin centre.  Like a crop, applying resets BG / chirp / '
            'masks.')
        rs_hint.setWordWrap(True)
        rs_hint.setStyleSheet('font-style: italic; color: #4d4d4d;')
        rs_col = QtWidgets.QVBoxLayout()
        rs_col.addWidget(gb_rs)
        rs_col.addWidget(rs_hint)
        outer.addLayout(rs_col)

        # ---- Preview split: 2D map | (Spectrum + Kinetics) ----
        preview_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # --- LEFT: 2D heatmap (existing canvas) ---
        self.canvas = MplCanvas(self)
        self.canvas.fig.clear()
        gs = self.canvas.fig.add_gridspec(1, 2, width_ratios=[1, 0.04],
                                          wspace=0.03)
        self.canvas.ax = self.canvas.fig.add_subplot(gs[0, 0])
        self._cax = self.canvas.fig.add_subplot(gs[0, 1])
        self.canvas.axes_list = [self.canvas.ax]
        self._cb = None
        preview_split.addWidget(self.canvas.with_toolbar(self))

        # --- RIGHT: Spectrum (top) | spec ctrl | Kinetics | kin ctrl ---
        right_widget = QtWidgets.QWidget(self)
        right_col = QtWidgets.QVBoxLayout(right_widget)
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(2)

        # Spectrum canvas
        self.canvas_spec = MplCanvas(self, figsize=(4.5, 3.0))
        right_col.addWidget(self.canvas_spec.with_toolbar(self), stretch=3)

        # Spectrum controls — Delay idx spinbox + Pin (mark for drop)
        spec_row = QtWidgets.QHBoxLayout()
        spec_row.addWidget(make_label('Delay idx:', 'right'))
        self.ed_sel_t_idx = make_int_edit(
            self._sel_t_idx, minv=0, maxv=n_t_full - 1)
        spec_row.addWidget(self.ed_sel_t_idx)
        self.lbl_sel_t = QtWidgets.QLabel(self._fmt_sel_t_label())
        self.lbl_sel_t.setStyleSheet('color: #444;')
        spec_row.addWidget(self.lbl_sel_t)
        spec_row.addStretch(1)
        self.btn_pin_spec = QtWidgets.QPushButton('Pin (mark for drop)')
        style_button(self.btn_pin_spec, bg='#d62728', fg='white')
        spec_row.addWidget(self.btn_pin_spec)
        right_col.addLayout(spec_row)

        # Kinetics canvas
        self.canvas_kin = MplCanvas(self, figsize=(4.5, 3.0))
        right_col.addWidget(self.canvas_kin.with_toolbar(self), stretch=3)

        # Kinetics controls — λ spinbox + Pin λ + Clear pins (visual only)
        kin_row = QtWidgets.QHBoxLayout()
        kin_row.addWidget(make_label(u'λ (nm):', 'right'))
        self.ed_sel_wl = make_double_edit(
            self._sel_wl,
            minv=self.wl_min_full, maxv=self.wl_max_full,
            decimals=3)
        kin_row.addWidget(self.ed_sel_wl)
        kin_row.addStretch(1)
        self.btn_pin_kin = QtWidgets.QPushButton(u'Pin λ (mark for drop)')
        style_button(self.btn_pin_kin, bg='#17becf', fg='white')
        self.btn_clear_kin = QtWidgets.QPushButton(u'Clear λ drops')
        kin_row.addWidget(self.btn_pin_kin)
        kin_row.addWidget(self.btn_clear_kin)
        right_col.addLayout(kin_row)

        preview_split.addWidget(right_widget)
        preview_split.setStretchFactor(0, 3)
        preview_split.setStretchFactor(1, 2)
        outer.addWidget(preview_split, stretch=1)

        # ---- Delete & interpolate group: hint + method + drops tables ----
        gb_interp = QtWidgets.QGroupBox(
            'Delete & interpolate at pinned delays / wavelengths')
        gbv = QtWidgets.QVBoxLayout(gb_interp)

        hint = QtWidgets.QLabel(
            'Click the 2D map to navigate the crosshair; inspect the '
            'spectrum / kinetics on the right.  Press "Pin (mark for drop)" '
            'on glitched delays (red dashed) or "Pin λ (mark for drop)" on '
            'glitched wavelengths (cyan dashed).  Pinned rows / columns are '
            'replaced by interpolation from the remaining data.  Choose '
            "'linear / cubic / pchip / akima' for axis-wise 1D interpolation, "
            "or 'bilinear' for a 2D surface fit considering both axes "
            'simultaneously.  Drops outside the current crop window are '
            'ignored on Apply.')
        hint.setWordWrap(True)
        hint.setStyleSheet('font-style: italic; color: #4d4d4d;')
        gbv.addWidget(hint)

        method_row = QtWidgets.QHBoxLayout()
        method_row.addWidget(make_label('Interpolation method:', 'right'))
        self.dd_interp_method = QtWidgets.QComboBox()
        for m in ('linear', 'cubic', 'pchip', 'akima', 'bilinear'):
            self.dd_interp_method.addItem(m)
        method_row.addWidget(self.dd_interp_method)
        method_row.addStretch(1)
        gbv.addLayout(method_row)

        bot = QtWidgets.QHBoxLayout()

        # -- Delay-drops side --
        delay_col = QtWidgets.QVBoxLayout()
        delay_col.addWidget(make_label('Pinned delays:', 'left'))
        drow1 = QtWidgets.QHBoxLayout()
        self.tbl_drops = QtWidgets.QTableWidget(0, 2)
        self.tbl_drops.setHorizontalHeaderLabels(
            ['Index', f'Delay ({app.t_unit_txt()})'])
        self.tbl_drops.horizontalHeader().setStretchLastSection(True)
        self.tbl_drops.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_drops.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_drops.setMaximumHeight(110)
        drow1.addWidget(self.tbl_drops, stretch=1)
        bcol = QtWidgets.QVBoxLayout()
        self.btn_drop_remove = QtWidgets.QPushButton('Remove selected')
        self.btn_drop_clear = QtWidgets.QPushButton('Clear all')
        bcol.addWidget(self.btn_drop_remove)
        bcol.addWidget(self.btn_drop_clear)
        bcol.addStretch(1)
        drow1.addLayout(bcol)
        delay_col.addLayout(drow1)
        bot.addLayout(delay_col, stretch=1)

        # -- Wavelength-drops side --
        wl_col = QtWidgets.QVBoxLayout()
        wl_col.addWidget(make_label(u'Pinned wavelengths:', 'left'))
        drow2 = QtWidgets.QHBoxLayout()
        self.tbl_drops_wl = QtWidgets.QTableWidget(0, 2)
        self.tbl_drops_wl.setHorizontalHeaderLabels(
            ['Index', u'λ (nm)'])
        self.tbl_drops_wl.horizontalHeader().setStretchLastSection(True)
        self.tbl_drops_wl.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_drops_wl.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_drops_wl.setMaximumHeight(110)
        drow2.addWidget(self.tbl_drops_wl, stretch=1)
        bcol2 = QtWidgets.QVBoxLayout()
        self.btn_drop_wl_remove = QtWidgets.QPushButton('Remove selected')
        self.btn_drop_wl_clear = QtWidgets.QPushButton('Clear all')
        bcol2.addWidget(self.btn_drop_wl_remove)
        bcol2.addWidget(self.btn_drop_wl_clear)
        bcol2.addStretch(1)
        drow2.addLayout(bcol2)
        wl_col.addLayout(drow2)
        bot.addLayout(wl_col, stretch=1)

        gbv.addLayout(bot)

        outer.addWidget(gb_interp)

        # ---- Action buttons ----
        br = QtWidgets.QHBoxLayout()
        br.addStretch(1)
        self.btn_revert = QtWidgets.QPushButton('Revert to Original')
        self.btn_revert.setEnabled(app.is_modified())
        self.btn_cancel = QtWidgets.QPushButton('Cancel')
        self.btn_apply = QtWidgets.QPushButton('Apply')
        style_button(self.btn_apply, bg='#4fa35a', fg='white')
        br.addWidget(self.btn_revert)
        br.addWidget(self.btn_cancel)
        br.addWidget(self.btn_apply)
        outer.addLayout(br)

        # Debounce: redrawing the full heatmap on every spinbox tick is
        # ~2 s on a 2136 × 192 dataset, which makes the dialog feel like
        # it's swallowing keystrokes ("튕긴다").  Coalesce successive
        # valueChanged signals into a single update_overlays() call after
        # the user stops typing for ~120 ms.  The heatmap itself is drawn
        # exactly once; spinbox edits only nudge the cheap overlay artists.
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(120)
        self._refresh_timer.timeout.connect(self._update_overlays)

        for ed in (self.ed_wl_min, self.ed_wl_max,
                   self.ed_t_min, self.ed_t_max):
            ed.valueChanged.connect(
                lambda _: self._refresh_timer.start())

        # Resample settings (and the λ range they apply to) refresh the
        # point-count label and the spectrum overlay on their own timer,
        # so the cheap overlay path above stays untouched.
        self._resample_timer = QtCore.QTimer(self)
        self._resample_timer.setSingleShot(True)
        self._resample_timer.setInterval(150)
        self._resample_timer.timeout.connect(self._on_resample_changed)
        for ed in (self.ed_wl_min, self.ed_wl_max, self.ed_resample_dx):
            ed.valueChanged.connect(
                lambda _: self._resample_timer.start())
        for w in (self.cb_resample, self.rb_avg):
            w.toggled.connect(lambda _: self._resample_timer.start())

        self.btn_full_wl.clicked.connect(self._set_full_wl)
        self.btn_full_t.clicked.connect(self._set_full_t)
        self.btn_revert.clicked.connect(self._revert)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_apply.clicked.connect(self._apply)

        # Pin / selection wiring
        self.btn_pin_spec.clicked.connect(self._pin_current_as_drop)
        self.btn_pin_kin.clicked.connect(self._pin_current_kin_overlay)
        self.btn_clear_kin.clicked.connect(self._clear_kin_overlays)
        self.btn_drop_remove.clicked.connect(self._drop_remove)
        self.btn_drop_clear.clicked.connect(self._drop_clear)
        self.btn_drop_wl_remove.clicked.connect(self._drop_wl_remove)
        self.btn_drop_wl_clear.clicked.connect(self._clear_kin_overlays)

        self.ed_sel_t_idx.valueChanged.connect(self._on_sel_t_idx_changed)
        self.ed_sel_wl.valueChanged.connect(self._on_sel_wl_changed)

        # Changing the interpolation method recomputes the live preview.
        self.dd_interp_method.currentTextChanged.connect(
            lambda _: self._recompute_preview())

        # 2D-map click now drives the crosshair / spec / kin panels.
        # Pin-as-drop is a separate explicit user action.
        self._click_cid = self.canvas.mpl_connect(
            'button_press_event', self._on_canvas_click)

        # ---- Initial draws ----
        self._draw_heatmap_full()
        self._update_overlays()
        self._update_resample_info()
        self._draw_spectrum()
        self._draw_kinetics()

    def _current_bounds(self):
        """(wl_min, wl_max, t_min, t_max) of the active crop request.

        Uses the stored request when there is one, so re-opening the
        dialog after resampling does not drift to the binned grid edges;
        otherwise the edges of the pre-resample working grid."""
        app = self.app
        if app.crop_bounds is not None:
            return tuple(float(v) for v in app.crop_bounds)
        wl = (app._crop_wl_pre_resample if app.resample_enabled
              else app.wavelength)
        return (float(np.min(wl)), float(np.max(wl)),
                float(np.min(app.delay)), float(np.max(app.delay)))

    # ------------------------------------------------------------------
    # Wavelength resampling
    # ------------------------------------------------------------------
    def _resample_spec(self) -> dict:
        return {'enabled': self.cb_resample.isChecked(),
                'dx': float(self.ed_resample_dx.value()),
                'mode': 'average' if self.rb_avg.isChecked() else 'decimate'}

    def _snap_bounds(self, wl1, wl2, t1, t2):
        """Spinbox values are rounded to 6 decimals, so a bound meant to
        equal the current request or a full-range edge is snapped back to
        that exact value — otherwise the rounding can cut off the
        outermost original point."""
        full = (self.wl_min_full, self.wl_max_full,
                self.t_min_full, self.t_max_full)
        out = []
        for new, old, edge in zip((wl1, wl2, t1, t2),
                                  self._current_bounds(), full):
            out.append(old if _bounds_close(new, old)
                       else edge if _bounds_close(new, edge) else new)
        return tuple(out)

    def _resample_window(self):
        """Original λ values inside the λ range currently typed in."""
        wl1, wl2 = sorted((self.ed_wl_min.value(), self.ed_wl_max.value()))
        wl1, wl2, _, _ = self._snap_bounds(wl1, wl2, self.t_min_full,
                                           self.t_max_full)
        wl = self.app.original_wavelength
        return (wl >= wl1) & (wl <= wl2)

    def _resample_probe(self):
        """Run the kernel on the in-range λ axis only (no data) to get the
        output size and whether Δλ is usable."""
        spec = self._resample_spec()
        wl = self.app.original_wavelength[self._resample_window()]
        if wl.size == 0:
            return None
        _, _, info = ta_core.resample_wavelength(
            wl, np.zeros((wl.size, 0)), spec['dx'], spec['mode'])
        return info

    def _resample_valid(self) -> bool:
        info = self._resample_probe()
        return bool(info and info['applied'])

    def _update_resample_info(self):
        spec = self._resample_spec()
        for w in (self.ed_resample_dx, self.rb_avg, self.rb_dec):
            w.setEnabled(spec['enabled'])
        info = self._resample_probe()
        if not spec['enabled'] or info is None:
            self.lbl_resample_info.setText('')
            return
        if not info['applied']:
            self.lbl_resample_info.setStyleSheet('color: #c0392b;')
            self.lbl_resample_info.setText(
                f"Δλ ≤ mean spacing {info['mean_spacing']:.3f} nm — "
                'Apply will crop without resampling')
            return
        self.lbl_resample_info.setStyleSheet('color: #333;')
        mode = 'avg' if spec['mode'] == 'average' else 'decimate'
        self.lbl_resample_info.setText(
            f"{info['n_in']} → {info['n_out']} λ points "
            f"(Δλ={spec['dx']:g} nm, {mode})")

    def _on_resample_changed(self):
        self._update_resample_info()
        self._draw_spectrum()

    def _set_full_wl(self):
        # Block valueChanged so we don't fire the debounce timer twice
        # — a single _update_overlays() at the end is enough.
        for ed in (self.ed_wl_min, self.ed_wl_max):
            ed.blockSignals(True)
        self.ed_wl_min.setValue(self.wl_min_full)
        self.ed_wl_max.setValue(self.wl_max_full)
        for ed in (self.ed_wl_min, self.ed_wl_max):
            ed.blockSignals(False)
        self._update_overlays()
        self._resample_timer.start()

    def _set_full_t(self):
        for ed in (self.ed_t_min, self.ed_t_max):
            ed.blockSignals(True)
        self.ed_t_min.setValue(self.t_min_full)
        self.ed_t_max.setValue(self.t_max_full)
        for ed in (self.ed_t_min, self.ed_t_max):
            ed.blockSignals(False)
        self._update_overlays()

    def _revert(self):
        self.app.apply_crop_by_range(
            self.wl_min_full, self.wl_max_full,
            self.t_min_full, self.t_max_full)
        self.accept()

    def _apply(self):
        wl1, wl2 = self.ed_wl_min.value(), self.ed_wl_max.value()
        t1, t2 = self.ed_t_min.value(), self.ed_t_max.value()
        if wl1 > wl2:
            wl1, wl2 = wl2, wl1
        if t1 > t2:
            t1, t2 = t2, t1

        # An unusable Δλ (not larger than the mean λ spacing) falls back to
        # a plain crop, after telling the user.
        spec = self._resample_spec()
        if spec['enabled'] and not self._resample_valid():
            warn_box(self, 'Resampling skipped',
                     u'Δλ must be larger than the mean wavelength spacing '
                     u'of the selected range.  The crop is applied without '
                     u'resampling.')
            spec = dict(spec, enabled=False)
        resample = spec if spec['enabled'] else None

        # Only re-crop when the range actually changed — otherwise running
        # apply_crop_by_range on identical bounds would needlessly reset
        # bg/chirp/solvent state.  We compare against the active crop
        # request (not the resampled grid edges); the tolerance absorbs
        # the spinboxes' 6-decimal rounding.
        wl1, wl2, t1, t2 = self._snap_bounds(wl1, wl2, t1, t2)
        crop_changed = any(
            not _bounds_close(new, old)
            for new, old in zip((wl1, wl2, t1, t2), self._current_bounds()))
        app = self.app
        resample_changed = (
            spec['enabled'] != bool(app.resample_enabled)
            or (spec['enabled']
                and (abs(spec['dx'] - app.resample_dx) > 1e-9
                     or spec['mode'] != app.resample_mode)))
        rebuild = crop_changed or resample_changed

        method = self.dd_interp_method.currentText()
        has_drops = bool(self._drop_t_values or self._kin_overlay_wl)

        # Order of operations follows the live preview: interpolate the
        # full original first, then crop (and resample) the processed
        # result.  When nothing about the grid changes we keep bg/chirp/etc
        # and just patch the current matrix in-place via interpolate_*_at
        # — only possible on an un-resampled grid, so with resampling on
        # the drops go through the rebuild path as well.
        if has_drops and (rebuild or spec['enabled']):
            t_full = self.app.original_delay
            w_full = self.app.original_wavelength
            drop_t_full = sorted({
                int(np.argmin(np.abs(t_full - t)))
                for t in self._drop_t_values
            })
            drop_w_full = sorted({
                int(np.argmin(np.abs(w_full - w)))
                for w in self._kin_overlay_wl
            })
            # Guards: drop axis-wise drop lists that would leave < 2 anchors.
            if drop_t_full and len(t_full) - len(drop_t_full) < 2:
                drop_t_full = []
            if drop_w_full and len(w_full) - len(drop_w_full) < 2:
                drop_w_full = []

            try:
                A_interp = self.app.original_deltaA
                if method == 'bilinear':
                    if drop_t_full or drop_w_full:
                        A_interp = ta_core.interpolate_missing_2d(
                            A_interp, w_full, t_full,
                            drop_w_full, drop_t_full)
                else:
                    if drop_t_full:
                        A_interp = ta_core.interpolate_missing_columns(
                            A_interp, t_full, drop_t_full, method=method)
                    if drop_w_full:
                        A_interp = ta_core.interpolate_missing_rows(
                            A_interp, w_full, drop_w_full, method=method)
            except Exception as e:
                warn_box(self, 'Interpolation failed', str(e))
                return

            # Swap original_deltaA so apply_crop_by_range slices the
            # interpolated matrix; restore afterwards so "Revert to
            # Original" still returns to the raw loaded data.
            saved = self.app.original_deltaA
            self.app.original_deltaA = A_interp
            try:
                self.app.apply_crop_by_range(wl1, wl2, t1, t2,
                                             resample=resample)
            finally:
                self.app.original_deltaA = saved
        elif rebuild:
            self.app.apply_crop_by_range(wl1, wl2, t1, t2, resample=resample)
        elif has_drops:
            assert not self.app.resample_enabled
            new_delay = self.app.delay
            new_wl = self.app.wavelength
            t_lo, t_hi = float(new_delay[0]), float(new_delay[-1])
            w_lo, w_hi = float(new_wl[0]),    float(new_wl[-1])
            drop_t_idx = sorted({
                int(np.argmin(np.abs(new_delay - t)))
                for t in self._drop_t_values
                if t_lo - 1e-9 <= t <= t_hi + 1e-9
            })
            drop_w_idx = sorted({
                int(np.argmin(np.abs(new_wl - w)))
                for w in self._kin_overlay_wl
                if w_lo - 1e-9 <= w <= w_hi + 1e-9
            })
            if method == 'bilinear' and (drop_t_idx or drop_w_idx):
                self.app.interpolate_2d_at(drop_t_idx, drop_w_idx)
            else:
                if drop_t_idx:
                    self.app.interpolate_delays_at(drop_t_idx, method=method)
                if drop_w_idx:
                    self.app.interpolate_wavelengths_at(
                        drop_w_idx, method=method)
        self.accept()

    # ------------------------------------------------------------------
    # Delete-and-interpolate helpers
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 2D-map interaction: click sets the crosshair, NOT the drop list.
    # ------------------------------------------------------------------
    def _on_canvas_click(self, ev):
        if ev.inaxes is not self.canvas.ax:
            return
        if ev.xdata is None or ev.ydata is None:
            return
        # Ignore clicks consumed by the toolbar's pan/zoom modes.
        tb = getattr(self.canvas, 'toolbar', None)
        if tb is not None and getattr(tb, 'mode', ''):
            return
        wl_full = self.app.original_wavelength
        t_full = self.app.original_delay
        wl_idx = int(np.argmin(np.abs(wl_full - float(ev.xdata))))
        t_idx = int(np.argmin(np.abs(t_full - float(ev.ydata))))
        self._set_selection(wl_idx=wl_idx, t_idx=t_idx)

    def _set_selection(self, *, wl_idx=None, t_idx=None):
        """Update the currently-inspected (selWL, selT), redraw everything
        that depends on it.  Caller passes either or both indices into the
        ORIGINAL (uncropped) arrays.
        """
        if wl_idx is not None:
            n = int(len(self.app.original_wavelength))
            self._sel_wl_idx = int(np.clip(wl_idx, 0, n - 1))
            self._sel_wl = float(
                self.app.original_wavelength[self._sel_wl_idx])
        if t_idx is not None:
            n = int(len(self.app.original_delay))
            self._sel_t_idx = int(np.clip(t_idx, 0, n - 1))
            self._sel_t = float(self.app.original_delay[self._sel_t_idx])

        # Mirror into the spinboxes without re-emitting valueChanged
        if t_idx is not None:
            self.ed_sel_t_idx.blockSignals(True)
            self.ed_sel_t_idx.setValue(self._sel_t_idx)
            self.ed_sel_t_idx.blockSignals(False)
            self.lbl_sel_t.setText(self._fmt_sel_t_label())
        if wl_idx is not None:
            self.ed_sel_wl.blockSignals(True)
            self.ed_sel_wl.setValue(self._sel_wl)
            self.ed_sel_wl.blockSignals(False)

        self._update_crosshair()
        if t_idx is not None:
            self._draw_spectrum()
        if wl_idx is not None:
            self._draw_kinetics()

    def _on_sel_t_idx_changed(self, v):
        self._set_selection(t_idx=int(v))

    def _on_sel_wl_changed(self, v):
        wl_full = self.app.original_wavelength
        idx = int(np.argmin(np.abs(wl_full - float(v))))
        self._set_selection(wl_idx=idx)

    def _fmt_sel_t_label(self):
        n = int(len(self.app.original_delay))
        return (f'{self._sel_t:.4g} {self.app.t_unit_txt()} '
                f'(of {n} pts)')

    # ------------------------------------------------------------------
    # Live interpolation preview
    # ------------------------------------------------------------------
    def _display_deltaA(self):
        """Return the current data the panels should display.

        Equals the interpolated preview when drops are pinned, otherwise
        the raw original snapshot.
        """
        if self._preview_deltaA is not None:
            return self._preview_deltaA
        return self.app.original_deltaA

    def _recompute_preview(self):
        """Rebuild the interpolation preview and refresh all three panels.

        Called whenever the drop list or interpolation method changes.
        Honours both delay drops (column interp) and wavelength drops
        (row interp).  When ``method == 'bilinear'`` the two are filled
        together as a 2D surface; otherwise the axis-wise 1D method is
        applied (columns first, then rows — intersection cells get
        filled by the column pass).
        """
        app = self.app
        t_full = app.original_delay
        w_full = app.original_wavelength
        A_full = app.original_deltaA
        method = self.dd_interp_method.currentText()

        drops_t_sorted = tuple(sorted(self._drop_t_values))
        drops_w_sorted = tuple(sorted(self._kin_overlay_wl))
        key = (drops_t_sorted, drops_w_sorted, method)
        if key == self._preview_key:
            return
        self._preview_key = key

        drop_t_idx = sorted({int(np.argmin(np.abs(t_full - t)))
                             for t in drops_t_sorted})
        drop_w_idx = sorted({int(np.argmin(np.abs(w_full - w)))
                             for w in drops_w_sorted})

        if not drop_t_idx and not drop_w_idx:
            self._preview_deltaA = None
        elif ((drop_t_idx and len(t_full) - len(drop_t_idx) < 2)
              or (drop_w_idx and len(w_full) - len(drop_w_idx) < 2)):
            # Too many drops along an axis — preview unavailable.
            self._preview_deltaA = None
        else:
            try:
                if method == 'bilinear':
                    self._preview_deltaA = ta_core.interpolate_missing_2d(
                        A_full, w_full, t_full, drop_w_idx, drop_t_idx)
                else:
                    tmp = A_full
                    if drop_t_idx:
                        tmp = ta_core.interpolate_missing_columns(
                            tmp, t_full, drop_t_idx, method=method)
                    if drop_w_idx:
                        tmp = ta_core.interpolate_missing_rows(
                            tmp, w_full, drop_w_idx, method=method)
                    self._preview_deltaA = tmp
            except Exception:
                self._preview_deltaA = None

        # 1) Update 2D heatmap data in-place.  pcolormesh with
        #    shading='nearest' stores the flattened (N_t, N_wl) matrix.
        display = self._display_deltaA()
        if self._preview_im is not None:
            try:
                self._preview_im.set_array(display.T.ravel())
            except Exception:
                pass
            self.canvas.draw_idle()

        # 2) Refresh the two side panels with the new data.
        self._draw_spectrum()
        self._draw_kinetics()

    # ------------------------------------------------------------------
    # Pin / drop list management
    # ------------------------------------------------------------------
    def _pin_current_as_drop(self):
        """Add the currently-selected delay to the drop+interpolate list."""
        self._add_drop_value(self._sel_t)

    def _add_drop_value(self, t_snap: float):
        # Treat values within one t-grid spacing as duplicates so a Pin
        # that happens to fall on an already-marked delay is a no-op.
        for existing in self._drop_t_values:
            if abs(existing - t_snap) < 1e-12:
                return
        self._drop_t_values.append(t_snap)
        self._drop_t_values.sort()
        self._refresh_drop_table()
        self._refresh_drop_lines()
        # Recomputing the preview also redraws the spec / kin panels.
        self._recompute_preview()

    def _drop_remove(self):
        rows = sorted(
            {idx.row() for idx in self.tbl_drops.selectedIndexes()},
            reverse=True)
        if not rows:
            return
        for r in rows:
            if 0 <= r < len(self._drop_t_values):
                del self._drop_t_values[r]
        self._refresh_drop_table()
        self._refresh_drop_lines()
        self._recompute_preview()

    def _drop_clear(self):
        if not self._drop_t_values:
            return
        self._drop_t_values.clear()
        self._refresh_drop_table()
        self._refresh_drop_lines()
        self._recompute_preview()

    def _refresh_drop_table(self):
        t_full = self.app.original_delay
        self.tbl_drops.setRowCount(len(self._drop_t_values))
        for r, t in enumerate(self._drop_t_values):
            idx = int(np.argmin(np.abs(t_full - t)))
            it_idx = QtWidgets.QTableWidgetItem(str(idx))
            it_t = QtWidgets.QTableWidgetItem(f'{t:.4g}')
            it_idx.setTextAlignment(QtCore.Qt.AlignCenter)
            it_t.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tbl_drops.setItem(r, 0, it_idx)
            self.tbl_drops.setItem(r, 1, it_t)

    def _refresh_drop_lines(self):
        for ln in self._drop_lines:
            try:
                ln.remove()
            except Exception:
                pass
        self._drop_lines.clear()
        for t in self._drop_t_values:
            ln = self.canvas.ax.axhline(
                t, color='#d62728', linewidth=1.1,
                linestyle='--', alpha=0.9, zorder=3.0)
            self._drop_lines.append(ln)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Wavelength drop pins (delete-and-interpolate along λ axis)
    # ------------------------------------------------------------------
    def _pin_current_kin_overlay(self):
        wl_snap = self._sel_wl
        for existing in self._kin_overlay_wl:
            if abs(existing - wl_snap) < 1e-9:
                return
        self._kin_overlay_wl.append(wl_snap)
        self._kin_overlay_wl.sort()
        self._refresh_drop_wl_table()
        self._refresh_kin_overlay_lines()
        self._recompute_preview()

    def _drop_wl_remove(self):
        rows = sorted(
            {idx.row() for idx in self.tbl_drops_wl.selectedIndexes()},
            reverse=True)
        if not rows:
            return
        for r in rows:
            if 0 <= r < len(self._kin_overlay_wl):
                del self._kin_overlay_wl[r]
        self._refresh_drop_wl_table()
        self._refresh_kin_overlay_lines()
        self._recompute_preview()

    def _clear_kin_overlays(self):
        if not self._kin_overlay_wl:
            return
        self._kin_overlay_wl.clear()
        self._refresh_drop_wl_table()
        self._refresh_kin_overlay_lines()
        self._recompute_preview()

    def _refresh_drop_wl_table(self):
        w_full = self.app.original_wavelength
        self.tbl_drops_wl.setRowCount(len(self._kin_overlay_wl))
        for r, w in enumerate(self._kin_overlay_wl):
            idx = int(np.argmin(np.abs(w_full - w)))
            it_idx = QtWidgets.QTableWidgetItem(str(idx))
            it_w = QtWidgets.QTableWidgetItem(f'{w:.3f}')
            it_idx.setTextAlignment(QtCore.Qt.AlignCenter)
            it_w.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tbl_drops_wl.setItem(r, 0, it_idx)
            self.tbl_drops_wl.setItem(r, 1, it_w)

    def _refresh_kin_overlay_lines(self):
        for ln in self._kin_overlay_lines:
            try:
                ln.remove()
            except Exception:
                pass
        self._kin_overlay_lines.clear()
        for w in self._kin_overlay_wl:
            ln = self.canvas.ax.axvline(
                w, color='#17becf', linewidth=1.1,
                linestyle='--', alpha=0.9, zorder=3.0)
            self._kin_overlay_lines.append(ln)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Crosshair on the 2D map (shows the current (selWL, selT))
    # ------------------------------------------------------------------
    def _update_crosshair(self):
        try:
            self._xhair_v.set_xdata([self._sel_wl, self._sel_wl])
            self._xhair_h.set_ydata([self._sel_t, self._sel_t])
        except Exception:
            pass
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Spectrum / kinetics drawing (always full redraw — cheap enough)
    # ------------------------------------------------------------------
    def _draw_spectrum(self):
        ax = self.canvas_spec.ax
        ax.clear()
        app = self.app
        wl_full = app.original_wavelength
        A_disp = self._display_deltaA()     # post-interp (preview)
        A_raw = app.original_deltaA         # pre-interp (for comparison)
        has_preview = self._preview_deltaA is not None

        import matplotlib.cm as _cm
        n_ov = len(self._drop_t_values)
        cmap = _cm.get_cmap('tab10', max(10, n_ov + 1))

        leg_h, leg_l = [], []
        for k, t_val in enumerate(self._drop_t_values):
            idx = int(np.argmin(np.abs(app.original_delay - t_val)))
            col = cmap(k % 10)
            # Pre-interpolation (raw) — faded dashed, for visual diff
            if has_preview:
                ax.plot(wl_full, A_raw[:, idx], '--', color=col,
                        linewidth=0.7, alpha=0.35)
            # Post-interpolation (preview) — solid line
            h, = ax.plot(wl_full, A_disp[:, idx], '-',
                         color=col, linewidth=1.0)
            tag = 'interp' if has_preview else 'pinned'
            label = (f'{tag}: t = {app.original_delay[idx]:.3g} '
                     f'{app.t_unit_ax()} (idx {idx})')
            h.set_label(label)
            leg_h.append(h)
            leg_l.append(label)

        idx = self._sel_t_idx
        cur_is_pinned = any(
            abs(app.original_delay[idx] - tv) < 1e-12
            for tv in self._drop_t_values)
        spec = self._resample_spec()
        rs_on = spec['enabled'] and self._resample_valid()
        h_cur, = ax.plot(wl_full, A_disp[:, idx],
                         color=('0.6' if rs_on else 'b'),
                         linewidth=(1.0 if rs_on else 1.6))
        cur_tag = ', interp' if (has_preview and cur_is_pinned) else ''
        label_cur = (f't = {app.original_delay[idx]:.3g} '
                     f'{app.t_unit_ax()} (current{cur_tag})')
        h_cur.set_label(label_cur)
        leg_h.append(h_cur)
        leg_l.append(label_cur)

        # Resampling preview: the current spectrum inside the λ range,
        # binned exactly as Apply will do it.
        if rs_on:
            win = self._resample_window()
            wl_rs, a_rs, info = ta_core.resample_wavelength(
                wl_full[win], A_disp[win, idx], spec['dx'], spec['mode'])
            mode = 'avg' if spec['mode'] == 'average' else 'decimate'
            label_rs = (f"resampled (Δλ={spec['dx']:g} nm, {mode}): "
                        f"{info['n_out']} pts")
            h_rs, = ax.plot(wl_rs, a_rs, 'o-', color='k',
                            markersize=3, linewidth=0.8, label=label_rs)
            leg_h.append(h_rs)
            leg_l.append(label_rs)

        # Mark dropped wavelengths as faint vertical lines so the user
        # sees which channels along this spectrum were interpolated.
        for w_val in self._kin_overlay_wl:
            ax.axvline(w_val, color='#17becf', linewidth=0.7,
                       linestyle=':', alpha=0.55)

        ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel(r'$\Delta$A')
        ax.grid(True, alpha=0.3)
        if len(leg_h) > 1:
            ax.legend(leg_h, leg_l, loc='best', fontsize=7)
        else:
            ax.set_title(label_cur, fontsize=9)
        self.canvas_spec.draw_idle()

    def _draw_kinetics(self):
        ax = self.canvas_kin.ax
        ax.clear()
        app = self.app
        t_full = app.original_delay
        A_disp = self._display_deltaA()
        A_raw = app.original_deltaA
        has_preview = self._preview_deltaA is not None

        import matplotlib.cm as _cm
        n_ov = len(self._kin_overlay_wl)
        cmap = _cm.get_cmap('tab10', max(10, n_ov + 1))

        leg_h, leg_l = [], []
        for k, wl_val in enumerate(self._kin_overlay_wl):
            idx = int(np.argmin(np.abs(app.original_wavelength - wl_val)))
            col = cmap(k % 10)
            # Pre-interpolation (raw) — faded dashed for visual diff
            if has_preview:
                ax.plot(t_full, A_raw[idx, :], '--', color=col,
                        linewidth=0.7, alpha=0.35)
            # Post-interpolation (preview) — solid line
            h, = ax.plot(t_full, A_disp[idx, :], '-',
                         color=col, linewidth=1.0)
            tag = 'interp' if has_preview else 'pinned'
            label = (f'{tag}: λ = '
                     f'{app.original_wavelength[idx]:.1f} nm (idx {idx})')
            h.set_label(label)
            leg_h.append(h)
            leg_l.append(label)

        idx = self._sel_wl_idx
        cur_is_pinned = any(
            abs(app.original_wavelength[idx] - wv) < 1e-12
            for wv in self._kin_overlay_wl)
        h_cur, = ax.plot(t_full, A_disp[idx, :], 'r-', linewidth=1.6)
        cur_tag = ', interp' if (has_preview and cur_is_pinned) else ''
        label_cur = (f'λ = {app.original_wavelength[idx]:.1f} nm '
                     f'(current{cur_tag})')
        h_cur.set_label(label_cur)
        leg_h.append(h_cur)
        leg_l.append(label_cur)

        # Mark the dropped delays as faint vertical lines so the user
        # sees which points along the kinetics curve were interpolated.
        for t_val in self._drop_t_values:
            ax.axvline(t_val, color='#d62728', linewidth=0.7,
                       linestyle=':', alpha=0.55)

        ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
        ax.set_xlabel(f'Delay time ({app.t_unit_ax()})')
        ax.set_ylabel(r'$\Delta$A')
        ax.grid(True, alpha=0.3)
        if len(leg_h) > 1:
            ax.legend(leg_h, leg_l, loc='best', fontsize=7)
        else:
            ax.set_title(label_cur, fontsize=9)
        self.canvas_kin.draw_idle()

    def _draw_heatmap_full(self):
        """Draw the full background heatmap exactly once.

        Subsequent crop-range changes only update the lightweight
        overlay (4 boundary lines + 4 dim-rectangles) via
        ``_update_overlays`` — they don't touch the (slow) pcolormesh.
        """
        app = self.app
        fig = self.canvas.fig
        # Always rebuild the figure so colorbars never accumulate.
        fig.clear()
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.04], wspace=0.03)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        self.canvas.ax = ax
        self._cax = cax
        self.canvas.axes_list = [ax]

        cl = float(np.nanmax(np.abs(app.original_deltaA)))
        if not np.isfinite(cl) or cl == 0:
            cl = 1.0
        wl_full = app.original_wavelength
        t_full = app.original_delay
        # Single full-resolution heatmap (no alpha — that's done with
        # the dim rectangles below for ~1000× cheaper updates).
        # We start from the live preview when drops were already pinned
        # (e.g. dialog re-opened); otherwise the raw original data.
        display = self._display_deltaA()
        im = ax.pcolormesh(wl_full, t_full, display.T,
                           cmap='turbo', vmin=-cl, vmax=cl,
                           shading='nearest')
        self._preview_im = im
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel(f'Delay time ({app.t_unit_ax()})')
        try:
            self._cb = self.canvas.fig.colorbar(im, cax=self._cax)
            self._cb.set_label(r'$\Delta$A')
        except Exception:
            self._cb = None

        # Pin the heatmap's view limits — they should never change
        # when the user merely changes the crop range.
        ax.set_xlim(float(wl_full[0]), float(wl_full[-1]))
        ax.set_ylim(float(t_full[0]), float(t_full[-1]))

        # Pre-create the four crop-boundary lines and four dim rectangles
        # so _update_overlays just edits their geometry.  Matplotlib
        # patches are cheap to mutate; pcolormesh isn't.
        from matplotlib.patches import Rectangle
        line_kw = dict(color='white', linewidth=1.4)
        self._ln_left  = ax.axvline(wl_full[0],  **line_kw)
        self._ln_right = ax.axvline(wl_full[-1], **line_kw)
        self._ln_bot   = ax.axhline(t_full[0],   **line_kw)
        self._ln_top   = ax.axhline(t_full[-1],  **line_kw)
        # Crosshair at the current (selWL, selT).  Yellow-ish so it stands
        # out against the turbo colormap as well as the white crop
        # boundaries.
        xhair_kw = dict(color='#ffd60a', linewidth=0.9,
                        linestyle='--', alpha=0.95, zorder=3.5)
        self._xhair_v = ax.axvline(self._sel_wl, **xhair_kw)
        self._xhair_h = ax.axhline(self._sel_t, **xhair_kw)
        # Dim rectangles for "outside" regions (left / right / bottom / top
        # of the kept band).  Their colour is white with 60% transparency
        # so the heatmap underneath is dimmed, not hidden.
        rect_kw = dict(facecolor='white', alpha=0.6,
                       edgecolor='none', zorder=2.5)
        self._rect_left   = Rectangle((0, 0), 0, 0, **rect_kw)
        self._rect_right  = Rectangle((0, 0), 0, 0, **rect_kw)
        self._rect_bottom = Rectangle((0, 0), 0, 0, **rect_kw)
        self._rect_top    = Rectangle((0, 0), 0, 0, **rect_kw)
        for rc in (self._rect_left, self._rect_right,
                   self._rect_bottom, self._rect_top):
            ax.add_patch(rc)

    def _update_overlays(self):
        """Cheap (<5 ms) overlay refresh: move lines + resize dim rects.

        Called after every spinbox change (debounced).  Does NOT redraw
        the heatmap.
        """
        app = self.app
        wl1, wl2 = self.ed_wl_min.value(), self.ed_wl_max.value()
        t1, t2 = self.ed_t_min.value(), self.ed_t_max.value()
        # Sort silently for display only; the spinbox values are NOT
        # changed, so the user's typed input is never overwritten.
        wlA, wlB = (wl1, wl2) if wl1 <= wl2 else (wl2, wl1)
        tA,  tB  = (t1, t2)  if t1 <= t2  else (t2, t1)
        # Same edge snapping as Apply, so the "Kept" count matches.
        wlA, wlB, tA, tB = self._snap_bounds(wlA, wlB, tA, tB)

        wl_full = app.original_wavelength
        t_full  = app.original_delay
        wl_lo, wl_hi = float(wl_full[0]), float(wl_full[-1])
        t_lo,  t_hi  = float(t_full[0]),  float(t_full[-1])

        # 1) Move the 4 boundary lines
        self._ln_left.set_xdata([wlA, wlA])
        self._ln_right.set_xdata([wlB, wlB])
        self._ln_bot.set_ydata([tA, tA])
        self._ln_top.set_ydata([tB, tB])

        # 2) Resize the 4 dim rectangles so the kept band stays bright
        #    and everything outside is whitened.
        full_t_span = t_hi - t_lo
        # Left  (wavelengths < wlA, full t range)
        self._rect_left.set_bounds(
            wl_lo, t_lo, max(wlA - wl_lo, 0.0), full_t_span)
        # Right (wavelengths > wlB, full t range)
        self._rect_right.set_bounds(
            wlB, t_lo, max(wl_hi - wlB, 0.0), full_t_span)
        # Bottom (within [wlA, wlB], t < tA)
        self._rect_bottom.set_bounds(
            wlA, t_lo, max(wlB - wlA, 0.0), max(tA - t_lo, 0.0))
        # Top    (within [wlA, wlB], t > tB)
        self._rect_top.set_bounds(
            wlA, tB, max(wlB - wlA, 0.0), max(t_hi - tB, 0.0))

        n_wl = int(np.sum((wl_full >= wlA) & (wl_full <= wlB)))
        n_t  = int(np.sum((t_full  >= tA)  & (t_full  <= tB)))
        # Show the (sorted) range in the title; the spinboxes themselves
        # remain untouched even if the user typed wl_min > wl_max while
        # in mid-edit.
        warn = ''
        if wl1 > wl2:
            warn += '  ⚠ λ_min > λ_max (will swap on Apply)'
        if t1 > t2:
            warn += '  ⚠ t_min > t_max (will swap on Apply)'
        self.canvas.ax.set_title(
            rf'Kept: $\lambda$=[{wlA:.1f}, {wlB:.1f}] nm ({n_wl} pts) | '
            rf't=[{tA:.3g}, {tB:.3g}] {app.t_unit_ax()} ({n_t} pts)'
            + warn)
        self.canvas.draw_idle()

    # Backwards-compatible alias for any caller that still uses the
    # old name.  Internally always goes through the fast path now.
    def draw_preview(self):
        self._update_overlays()


# =====================================================================
# Wavelength region mask window
# =====================================================================
class MaskDialog(QtWidgets.QDialog):
    """Add/remove wavelength regions that are set to NaN or zero."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.setWindowTitle('Mask Wavelength Regions')
        self.resize(960, 680)

        outer = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            'Mask wavelength regions to exclude from analysis. Click once '
            'on the 2D map to pick the start wavelength (From), click again '
            'for the end (To). Then choose a mode and press Add region. '
            '"Set to NaN" removes the region from fits and shows it as a '
            'transparent gap; "Set to 0" forces values to 0 while keeping '
            'them in the wavelength axis (not recommended for fitting).')
        info.setWordWrap(True)
        info.setStyleSheet('font-style: italic; color: #4d4d4d;')
        outer.addWidget(info)

        self.canvas = MplCanvas(self)
        outer.addWidget(self.canvas.with_toolbar(self), stretch=1)
        # Same fixed-gridspec strategy as CropDialog to keep the main
        # axes width stable across redraws.
        self.canvas.fig.clear()
        gs = self.canvas.fig.add_gridspec(1, 2, width_ratios=[1, 0.04],
                                          wspace=0.03)
        self.canvas.ax = self.canvas.fig.add_subplot(gs[0, 0])
        self._cax = self.canvas.fig.add_subplot(gs[0, 1])
        self.canvas.axes_list = [self.canvas.ax]
        self._cb = None  # colorbar handle (recreated each draw_map)

        # Add-region controls
        add_row = QtWidgets.QHBoxLayout()
        add_row.addWidget(make_label('From (nm):', 'right'))
        self.ed_wl1 = make_double_edit(0.0)
        add_row.addWidget(self.ed_wl1)
        add_row.addWidget(make_label('To (nm):', 'right'))
        self.ed_wl2 = make_double_edit(0.0)
        add_row.addWidget(self.ed_wl2)
        add_row.addWidget(make_label('Mode:', 'right'))
        self.dd_mode = QtWidgets.QComboBox()
        self.dd_mode.addItem('Set to NaN (mask out)', 'nan')
        self.dd_mode.addItem('Set to 0', 'zero')
        add_row.addWidget(self.dd_mode)
        self.btn_add = QtWidgets.QPushButton('Add region')
        style_button(self.btn_add, bg='#4c8cca', fg='white')
        add_row.addWidget(self.btn_add)
        add_row.addStretch(1)
        self.lbl_pick = QtWidgets.QLabel('Click map to pick From / To')
        self.lbl_pick.setStyleSheet('font-style: italic; color:#666;')
        add_row.addWidget(self.lbl_pick)
        outer.addLayout(add_row)

        # Table + row buttons
        bot = QtWidgets.QHBoxLayout()
        self.tbl = QtWidgets.QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(['From (nm)', 'To (nm)', 'Mode'])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        bot.addWidget(self.tbl, stretch=1)

        bcol = QtWidgets.QVBoxLayout()
        self.btn_remove = QtWidgets.QPushButton('Remove selected')
        self.btn_clear = QtWidgets.QPushButton('Clear all')
        bcol.addWidget(self.btn_remove)
        bcol.addWidget(self.btn_clear)
        bcol.addStretch(1)
        bot.addLayout(bcol)
        outer.addLayout(bot)

        self.pick_stage = 1  # 1 = From, 2 = To

        self.btn_add.clicked.connect(self.do_add)
        self.btn_remove.clicked.connect(self.do_remove)
        self.btn_clear.clicked.connect(self.do_clear)
        self.ed_wl1.valueChanged.connect(self._reset_pick_stage)
        self.ed_wl2.valueChanged.connect(self._reset_pick_stage)

        # Matplotlib click handler
        self._cid = self.canvas.mpl_connect('button_press_event',
                                            self.on_map_click)

        self.draw_map()
        self.refresh_table()

    def closeEvent(self, ev):
        self.app.mask_fig = None
        super().closeEvent(ev)

    # ---- helpers ----
    def _reset_pick_stage(self):
        self.pick_stage = 1
        self.lbl_pick.setText('Click map to pick From / To')

    def draw_map(self):
        app = self.app
        # Rebuild the figure fresh — same robust strategy as CropDialog
        # to avoid colorbar-stacking issues on matplotlib >= 3.8.
        fig = self.canvas.fig
        fig.clear()
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.04], wspace=0.03)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        self.canvas.ax = ax
        self._cax = cax
        self.canvas.axes_list = [ax]
        self._cb = None
        cl = np.nanmax(np.abs(app.deltaA))
        if not np.isfinite(cl) or cl == 0:
            cl = 1.0
        im = ax.pcolormesh(app.wavelength, app.delay, app.deltaA.T,
                           cmap='turbo', vmin=-cl, vmax=cl,
                           shading='nearest')
        try:
            self._cb = self.canvas.fig.colorbar(im, cax=self._cax)
            self._cb.set_label(r'$\Delta$A')
        except Exception:
            pass
        ax.set_xlim(app.wavelength.min(), app.wavelength.max())
        ax.set_ylim(app.delay.min(), app.delay.max())
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel(f'Delay time ({app.t_unit_ax()})')
        ax.grid(True)
        ax.set_title('Masked regions preview (click to pick wavelengths)')
        # Draw current masks
        y1, y2 = app.delay.min(), app.delay.max()
        for wl1, wl2, mode in app.masked_regions:
            col = (0.8, 0.2, 0.2) if mode == 'nan' else (0.2, 0.5, 0.8)
            ax.axvspan(wl1, wl2, color=col, alpha=0.18,
                       edgecolor=col, linewidth=1.2)
        self.canvas.draw_idle()

    def on_map_click(self, event):
        if event.inaxes is not self.canvas.ax:
            return
        if event.xdata is None:
            return
        wl_click = float(event.xdata)
        if self.pick_stage == 1:
            self.ed_wl1.setValue(wl_click)
            self.pick_stage = 2
            self.lbl_pick.setText('Next click sets the To (end) wavelength')
        else:
            self.ed_wl2.setValue(wl_click)
            self.pick_stage = 1
            self.lbl_pick.setText('Next click restarts picking from the From edge')

    def refresh_table(self):
        app = self.app
        self.tbl.setRowCount(0)
        for wl1, wl2, mode in app.masked_regions:
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
            self.tbl.setItem(r, 0, QtWidgets.QTableWidgetItem(f'{wl1:.2f}'))
            self.tbl.setItem(r, 1, QtWidgets.QTableWidgetItem(f'{wl2:.2f}'))
            self.tbl.setItem(r, 2, QtWidgets.QTableWidgetItem(
                'Set to NaN' if mode == 'nan' else 'Set to 0'))

    def do_add(self):
        app = self.app
        wl1, wl2 = self.ed_wl1.value(), self.ed_wl2.value()
        if wl1 > wl2:
            wl1, wl2 = wl2, wl1
        if not (np.isfinite(wl1) and np.isfinite(wl2)) or wl1 >= wl2:
            warn_box(self, 'Input error',
                     'Invalid wavelength range. From must be < To.')
            return
        wl_lo, wl_hi = float(np.min(app.wavelength)), float(np.max(app.wavelength))
        if wl2 < wl_lo or wl1 > wl_hi:
            warn_box(self, 'Input error',
                     f'Region [{wl1:.1f}, {wl2:.1f}] nm is outside '
                     f'data range [{wl_lo:.1f}, {wl_hi:.1f}] nm.')
            return
        mode = self.dd_mode.currentData()
        app.masked_regions.append((wl1, wl2, mode))
        self.refresh_table()
        app.update_all()
        self.draw_map()
        self.pick_stage = 1
        self.lbl_pick.setText(
            f'Added region {len(app.masked_regions)}. '
            'Click map to pick the next From / To.')

    def do_remove(self):
        rows = sorted({ix.row() for ix in self.tbl.selectedIndexes()},
                      reverse=True)
        if not rows:
            warn_box(self, 'Nothing selected',
                     'Select a row in the table first.')
            return
        app = self.app
        for r in rows:
            if 0 <= r < len(app.masked_regions):
                app.masked_regions.pop(r)
        self.refresh_table()
        app.update_all()
        self.draw_map()

    def do_clear(self):
        app = self.app
        if not app.masked_regions:
            return
        app.masked_regions = []
        self.refresh_table()
        app.update_all()
        self.draw_map()


# =====================================================================
# Load & Average window
# =====================================================================
class LoadAverageDialog(QtWidgets.QDialog):
    """Multi-file loader: preview spectrum / kinetics per file and average
    the selected files."""

    def __init__(self, parent, app, path_base: str, filenames: list):
        super().__init__(parent)
        self.app = app
        self.path_base = path_base
        self.setWindowTitle('Load & Average Datasets')
        self.resize(1080, 780)

        # ---- Parse files ----
        self.files = []   # list of dicts
        for name in filenames:
            try:
                wl, t, A = ta_core.parse_data_file(
                    os.path.join(path_base, name))
            except Exception as e:
                warn_box(parent, 'Load Error',
                         f'Failed to read {name}:\n{e}')
                self.reject(); return
            self.files.append({
                'name': name, 'path': os.path.join(path_base, name),
                'wavelength': wl, 'delay': t, 'deltaA': A,
                'include': True,
                'max_abs': float(np.nanmax(np.abs(A))),
            })

        self.sz_ref = self.files[0]['deltaA'].shape
        for fi in self.files[1:]:
            if fi['deltaA'].shape != self.sz_ref:
                warn_box(parent, 'Load Error',
                         f'Shape mismatch: {self.files[0]["name"]} is '
                         f'{self.sz_ref}, {fi["name"]} is {fi["deltaA"].shape}.'
                         '\nAll files must share the same dimensions.')
                self.reject(); return

        outer = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel(
            'Compare files visually, then check those you want to include '
            'in the average. Unchecked files are shown dashed / semi-'
            'transparent.')
        header.setWordWrap(True)
        outer.addWidget(header)

        # File buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_more = QtWidgets.QPushButton('Add more files...')
        self.btn_rem = QtWidgets.QPushButton('Remove selected')
        self.btn_all = QtWidgets.QPushButton('Check all')
        self.btn_none = QtWidgets.QPushButton('Uncheck all')
        btn_row.addWidget(self.btn_add_more)
        btn_row.addWidget(self.btn_rem)
        btn_row.addWidget(self.btn_all)
        btn_row.addWidget(self.btn_none)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        # Table
        self.tbl = QtWidgets.QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(['File', 'Max |dA|', 'Include'])
        self.tbl.horizontalHeader().setStretchLastSection(False)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl.setMaximumHeight(180)
        outer.addWidget(self.tbl)

        # Ref lambda / t
        ref_row = QtWidgets.QHBoxLayout()
        ref_row.addWidget(make_label(u'Ref λ (nm):', 'right'))
        wl0 = self.files[0]['wavelength']
        t0 = self.files[0]['delay']
        wl_ref0 = float(wl0[len(wl0) // 2])
        pos = np.where(t0 >= 1)[0]
        if pos.size:
            t_ref0 = float(t0[pos[0]])
        else:
            t_ref0 = float(t0[len(t0) // 2])
        self.ed_ref_wl = make_double_edit(wl_ref0, minv=float(wl0.min()),
                                          maxv=float(wl0.max()))
        ref_row.addWidget(self.ed_ref_wl)
        ref_row.addWidget(make_label(f'Ref t ({app.t_unit_txt()}):', 'right'))
        self.ed_ref_t = make_double_edit(t_ref0, minv=float(t0.min()),
                                         maxv=float(t0.max()))
        ref_row.addWidget(self.ed_ref_t)
        ref_row.addWidget(make_label('Kinetics scale:', 'right'))
        self.dd_kin_scale = QtWidgets.QComboBox()
        self.dd_kin_scale.addItems(['Linear', 'Log'])
        self.dd_kin_scale.setCurrentText('Log')
        ref_row.addWidget(self.dd_kin_scale)
        ref_row.addStretch(1)
        outer.addLayout(ref_row)

        # Preview axes
        self.canvas = MplCanvas(self, nrows=1, ncols=2, figsize=(10, 4))
        self.ax_s, self.ax_k = self.canvas.axes_list
        outer.addWidget(self.canvas.with_toolbar(self), stretch=1)

        # Action buttons
        act = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton('Save Avg to CSV...')
        act.addWidget(self.btn_save)
        act.addStretch(1)
        self.btn_cancel = QtWidgets.QPushButton('Cancel')
        self.btn_load = QtWidgets.QPushButton('Load Average of Checked')
        style_button(self.btn_load, bg='#4fa35a', fg='white')
        act.addWidget(self.btn_cancel)
        act.addWidget(self.btn_load)
        outer.addLayout(act)

        # Wiring
        self.btn_add_more.clicked.connect(self.on_add_more)
        self.btn_rem.clicked.connect(self.on_remove_selected)
        self.btn_all.clicked.connect(lambda: self.on_set_all(True))
        self.btn_none.clicked.connect(lambda: self.on_set_all(False))
        self.ed_ref_wl.valueChanged.connect(self.draw_preview)
        self.ed_ref_t.valueChanged.connect(self.draw_preview)
        self.dd_kin_scale.currentTextChanged.connect(self.draw_preview)
        self.btn_save.clicked.connect(self.on_save_avg)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_load.clicked.connect(self.on_load_avg)
        self.tbl.itemChanged.connect(self._on_table_item_changed)

        self.refresh_table()
        self.draw_preview()

    def closeEvent(self, ev):
        self.app.load_fig = None
        super().closeEvent(ev)

    # ---- helpers ----
    def refresh_table(self):
        self.tbl.blockSignals(True)
        self.tbl.setRowCount(0)
        for i, fi in enumerate(self.files):
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
            item_name = QtWidgets.QTableWidgetItem(fi['name'])
            item_name.setFlags(item_name.flags() & ~QtCore.Qt.ItemIsEditable)
            self.tbl.setItem(r, 0, item_name)
            item_m = QtWidgets.QTableWidgetItem(f'{fi["max_abs"]:.3e}')
            item_m.setFlags(item_m.flags() & ~QtCore.Qt.ItemIsEditable)
            self.tbl.setItem(r, 1, item_m)
            item_inc = QtWidgets.QTableWidgetItem()
            item_inc.setFlags((item_inc.flags() | QtCore.Qt.ItemIsUserCheckable)
                              & ~QtCore.Qt.ItemIsEditable)
            item_inc.setCheckState(QtCore.Qt.Checked if fi['include']
                                   else QtCore.Qt.Unchecked)
            self.tbl.setItem(r, 2, item_inc)
        self.tbl.blockSignals(False)

    def _on_table_item_changed(self, item):
        if item.column() == 2:
            idx = item.row()
            self.files[idx]['include'] = (item.checkState() == QtCore.Qt.Checked)
            self.draw_preview()

    def on_set_all(self, val: bool):
        for fi in self.files:
            fi['include'] = bool(val)
        self.refresh_table()
        self.draw_preview()

    def on_remove_selected(self):
        rows = sorted({ix.row() for ix in self.tbl.selectedIndexes()},
                      reverse=True)
        if not rows:
            return
        for r in rows:
            if 0 <= r < len(self.files):
                self.files.pop(r)
        if not self.files:
            self.reject(); return
        self.refresh_table()
        self.draw_preview()

    def on_add_more(self):
        paths = ask_open_paths(self, 'Select more TA data file(s)',
                               multi=True)
        if not paths:
            return
        for p in paths:
            name = os.path.basename(p)
            try:
                wl, t, A = ta_core.parse_data_file(p)
            except Exception as e:
                warn_box(self, 'Load Error',
                         f'Failed to read {name}:\n{e}')
                continue
            if A.shape != self.sz_ref:
                warn_box(self, 'Shape Error',
                         f'Shape mismatch with {self.files[0]["name"]}; '
                         f'skipping {name}.')
                continue
            self.files.append({'name': name, 'path': p,
                               'wavelength': wl, 'delay': t, 'deltaA': A,
                               'include': True,
                               'max_abs': float(np.nanmax(np.abs(A)))})
        self.refresh_table()
        self.draw_preview()

    def draw_preview(self):
        app = self.app
        if not self.files:
            return
        wl_ax = self.files[0]['wavelength']
        t_ax = self.files[0]['delay']
        ref_wl = self.ed_ref_wl.value()
        ref_t = self.ed_ref_t.value()
        iW = int(np.argmin(np.abs(wl_ax - ref_wl)))
        iT = int(np.argmin(np.abs(t_ax - ref_t)))
        cmap = ta_core._mpl_cm.get_cmap('tab10', max(10, len(self.files)))

        # Spectrum
        ax = self.ax_s
        ax.clear()
        for i, fi in enumerate(self.files):
            style = '-' if fi['include'] else '--'
            lw = 1.3 if fi['include'] else 0.9
            alpha = 1.0 if fi['include'] else 0.45
            col = cmap(i % 10)
            ax.plot(wl_ax, fi['deltaA'][:, iT], style, color=col,
                    linewidth=lw, alpha=alpha, label=fi['name'])
        ax.axhline(0, color='k', linestyle=':')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel(r'$\Delta$A')
        ax.grid(True)
        ax.set_xlim(wl_ax.min(), wl_ax.max())
        ax.set_title(f'Spectrum at t = {t_ax[iT]:.3g} {app.t_unit_ax()}')
        ax.legend(loc='best', fontsize=7)

        # Kinetics
        ax = self.ax_k
        ax.clear()
        for i, fi in enumerate(self.files):
            style = '-' if fi['include'] else '--'
            lw = 1.3 if fi['include'] else 0.9
            alpha = 1.0 if fi['include'] else 0.45
            col = cmap(i % 10)
            ax.plot(t_ax, fi['deltaA'][iW, :], style, color=col,
                    linewidth=lw, alpha=alpha, label=fi['name'])
        ax.axhline(0, color='k', linestyle=':')
        ax.set_xlabel(f'Delay time ({app.t_unit_ax()})')
        ax.set_ylabel(r'$\Delta$A')
        ax.grid(True)
        if self.dd_kin_scale.currentText() == 'Log':
            pos = t_ax > 0
            if pos.any():
                ax.set_xscale('log')
                ax.set_xlim(t_ax[pos].min(), t_ax.max())
        else:
            ax.set_xlim(t_ax.min(), t_ax.max())
        ax.set_title(rf'Kinetics at $\lambda$ = {wl_ax[iW]:.1f} nm')
        ax.legend(loc='best', fontsize=7)

        self.canvas.draw_idle()

    def compute_average(self):
        incl = [fi for fi in self.files if fi['include']]
        if not incl:
            return None, None, None, 0
        wl = incl[0]['wavelength']
        t = incl[0]['delay']
        stack = np.stack([fi['deltaA'] for fi in incl], axis=2)
        avg = np.nanmean(stack, axis=2)
        return wl, t, avg, len(incl)

    def on_save_avg(self):
        wl, t, avg, N = self.compute_average()
        if avg is None:
            warn_box(self, 'Nothing to save', 'Check at least one file.')
            return
        path, _ = ask_save_path(self, 'Save averaged data',
                                self.app.default_save_path('averaged.csv'))
        if not path:
            return
        try:
            ta_core.write_data_file(path, wl, t, avg, ',')
            info_box(self, 'Saved',
                     f'Saved average of {N} files to:\n{path}')
        except Exception as e:
            warn_box(self, 'Save Error', f'Save failed:\n{e}')

    def on_load_avg(self):
        wl, t, avg, N = self.compute_average()
        if avg is None:
            warn_box(self, 'None selected', 'Check at least one file.')
            return
        incl_names = [fi['name'] for fi in self.files if fi['include']]
        if N == 1:
            desc = f'Loaded: {incl_names[0]}'
        else:
            desc = f'Avg of {N} files ({incl_names[0]}…)'
        self.app.set_loaded_data(wl, t, avg, desc,
                                 source_dir=self.path_base)
        self.accept()
