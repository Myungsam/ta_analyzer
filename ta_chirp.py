"""
TA Analyzer GUI - Chirp correction dialog.
"""
from __future__ import annotations

import numpy as np
from PyQt5 import QtWidgets, QtCore

from ta_widgets import (
    MplCanvas, make_label, make_double_edit, make_int_edit,
    style_button, warn_box, info_box, ask_save_path, ask_open_paths,
)
import ta_core


class ChirpDialog(QtWidgets.QDialog):
    """Chirp correction with click-to-add points and ridge helper."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.setWindowTitle('Chirp Correction')
        self.resize(1180, 780)

        outer = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel(
            r'Click the 2D map to add zero-time (λ, t₀) points '
            '(≥4 required). Fit model: '
            r't₀ = a·sqrt((b·w² − 1)/(c·w² − 1)) + d.')
        outer.addWidget(header)

        main_row = QtWidgets.QHBoxLayout()
        # Left: big 2D map canvas
        self.canvas_map = MplCanvas(self, figsize=(7, 6))
        self.ax_map = self.canvas_map.ax
        main_row.addWidget(self.canvas_map.with_toolbar(self), stretch=3)

        # Right: fit preview + table + kinetics preview
        right = QtWidgets.QVBoxLayout()
        self.canvas_fit = MplCanvas(self, figsize=(5, 2))
        self.ax_fit = self.canvas_fit.ax
        right.addWidget(self.canvas_fit, stretch=1)

        self.tbl = QtWidgets.QTableWidget(0, 2)
        self.tbl.setHorizontalHeaderLabels(
            [u'λ (nm)', f't ({app.t_unit_txt()})'])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        right.addWidget(self.tbl, stretch=1)

        self.canvas_kin = MplCanvas(self, figsize=(5, 2))
        self.ax_kin = self.canvas_kin.ax
        right.addWidget(self.canvas_kin, stretch=1)

        main_row.addLayout(right, stretch=2)
        outer.addLayout(main_row, stretch=1)

        # Ridge helper row
        ridge_box = QtWidgets.QGroupBox('Ridge Helper (reference t₀ curve)')
        r_lay = QtWidgets.QGridLayout(ridge_box)
        self.cb_show_ridge = QtWidgets.QCheckBox('Show ridge overlay')
        r_lay.addWidget(self.cb_show_ridge, 0, 0)
        r_lay.addWidget(make_label('Method:', 'right'), 0, 1)
        self.dd_method = QtWidgets.QComboBox()
        self.dd_method.addItems(['max|dA/dt|', 'max|deltaA|'])
        self.dd_method.setCurrentText(app.ridge_method)
        r_lay.addWidget(self.dd_method, 0, 2)
        r_lay.addWidget(make_label(f'Search t ({app.t_unit_txt()}):', 'right'), 0, 3)
        self.ed_rtmin = make_double_edit(app.ridge_tmin)
        r_lay.addWidget(self.ed_rtmin, 0, 4)
        r_lay.addWidget(make_label('to', 'center'), 0, 5)
        self.ed_rtmax = make_double_edit(app.ridge_tmax)
        r_lay.addWidget(self.ed_rtmax, 0, 6)
        r_lay.addWidget(make_label(u'λ-smooth:', 'right'), 0, 7)
        self.ed_sm = make_int_edit(app.ridge_smooth_n, minv=1, maxv=200)
        r_lay.addWidget(self.ed_sm, 0, 8)
        self.btn_rec_ridge = QtWidgets.QPushButton('Recompute')
        r_lay.addWidget(self.btn_rec_ridge, 0, 9)

        r_lay.addWidget(make_label('Auto-suggest N points:', 'right'), 1, 1)
        self.ed_auto_n = make_int_edit(8, minv=4, maxv=100)
        r_lay.addWidget(self.ed_auto_n, 1, 2)
        self.btn_auto = QtWidgets.QPushButton('Auto-place')
        r_lay.addWidget(self.btn_auto, 1, 3)
        outer.addWidget(ridge_box)

        # Bottom control row
        bot = QtWidgets.QHBoxLayout()
        bot.addWidget(make_label(f'View t_min ({app.t_unit_txt()}):', 'right'))
        self.ed_view_tmin = make_double_edit(
            app.chirp_view_t_min if app.chirp_view_t_min is not None
            else float(app.delay[0]))
        bot.addWidget(self.ed_view_tmin)
        bot.addWidget(make_label(f't_max ({app.t_unit_txt()}):', 'right'))
        self.ed_view_tmax = make_double_edit(app.chirp_view_t_max)
        bot.addWidget(self.ed_view_tmax)
        self.btn_view_reset = QtWidgets.QPushButton('Reset view')
        bot.addWidget(self.btn_view_reset)
        self.btn_undo = QtWidgets.QPushButton('Remove Last')
        bot.addWidget(self.btn_undo)
        self.btn_clr = QtWidgets.QPushButton('Clear Points')
        bot.addWidget(self.btn_clr)
        self.btn_save_pts = QtWidgets.QPushButton('Save Points')
        bot.addWidget(self.btn_save_pts)
        self.btn_load_pts = QtWidgets.QPushButton('Load Points')
        bot.addWidget(self.btn_load_pts)
        bot.addStretch(1)
        self.btn_cancel = QtWidgets.QPushButton('Cancel')
        bot.addWidget(self.btn_cancel)
        self.btn_apply = QtWidgets.QPushButton('Fit && Apply')
        style_button(self.btn_apply, bg='#4fa35a', fg='white')
        bot.addWidget(self.btn_apply)
        outer.addLayout(bot)

        # State
        self.last_click_wl = None
        self.last_click_t = None
        self._ridge_line = None
        self._fit_line_on_map = None
        self._pts_scatter = None

        # Wire
        self._cid = self.canvas_map.mpl_connect(
            'button_press_event', self.add_pt)
        self.cb_show_ridge.toggled.connect(self.toggle_ridge)
        self.dd_method.currentTextChanged.connect(
            lambda v: self.set_and_recompute('method', v))
        self.ed_rtmin.valueChanged.connect(
            lambda v: self.set_and_recompute('tmin', v))
        self.ed_rtmax.valueChanged.connect(
            lambda v: self.set_and_recompute('tmax', v))
        self.ed_sm.valueChanged.connect(
            lambda v: self.set_and_recompute('smooth', int(v)))
        self.btn_rec_ridge.clicked.connect(self.recompute_ridge)
        self.btn_auto.clicked.connect(
            lambda: self.auto_place(int(self.ed_auto_n.value())))
        self.btn_undo.clicked.connect(self.undo_pt)
        self.btn_clr.clicked.connect(self.clear_pts)
        self.btn_save_pts.clicked.connect(self.save_pts)
        self.btn_load_pts.clicked.connect(self.load_pts)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_apply.clicked.connect(self.apply_chirp)
        self.ed_view_tmin.valueChanged.connect(self.on_view_change)
        self.ed_view_tmax.valueChanged.connect(self.on_view_change)
        self.btn_view_reset.clicked.connect(self.reset_view)

        # Initial draw
        self.draw_map_once()
        self.redraw()
        self.apply_view_range()

    def closeEvent(self, ev):
        self.app.chirp_fig = None
        super().closeEvent(ev)

    # ---- Drawing ----
    def draw_map_once(self):
        app = self.app
        base = app.get_chirp_base_data()
        self._base_data = base
        ax = self.ax_map
        ax.clear()
        cl = np.nanmax(np.abs(base))
        if not np.isfinite(cl) or cl == 0:
            cl = 1.0
        self._im = ax.pcolormesh(app.wavelength, app.delay, base.T,
                                 cmap='turbo', vmin=-cl, vmax=cl,
                                 shading='nearest')
        try:
            self.canvas_map.fig.colorbar(
                self._im, ax=ax).set_label(r'$\Delta$A')
        except Exception:
            pass
        if app.bg_applied:
            ax.set_title('Click to add t₀ points — BG-corrected base')
        else:
            ax.set_title('Click to add t₀ points — raw base')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel(f'Delay time ({app.t_unit_ax()})')
        ax.set_xlim(app.wavelength.min(), app.wavelength.max())
        self.canvas_map.draw_idle()

    def add_pt(self, event):
        if event.inaxes is not self.ax_map:
            return
        if event.xdata is None or event.ydata is None:
            return
        pt = (float(event.xdata), float(event.ydata))
        self.app.chirp_pts = np.vstack([self.app.chirp_pts, pt])
        self.last_click_wl, self.last_click_t = pt
        self.redraw()
        self.update_kin_preview()

    def undo_pt(self):
        if len(self.app.chirp_pts) > 0:
            self.app.chirp_pts = self.app.chirp_pts[:-1]
            self.redraw()

    def clear_pts(self):
        self.app.chirp_pts = np.zeros((0, 2))
        self.redraw()

    # ---- Save / Load chirp points ----
    def save_pts(self):
        app = self.app
        pts = app.chirp_pts
        if pts.shape[0] == 0:
            warn_box(self, 'No points', 'No chirp points to save.')
            return
        path, _ = ask_save_path(
            self, 'Save chirp points',
            app.default_save_path('chirp_points.csv'),
            filt='CSV (*.csv);;All files (*)')
        if not path:
            return
        order = np.argsort(pts[:, 0], kind='stable')
        sorted_pts = pts[order]
        hdr = f'wavelength_nm,t0_{app.t_unit_hdr()}'
        try:
            np.savetxt(path, sorted_pts, delimiter=',', header=hdr,
                       comments='', fmt='%.6g')
        except Exception as e:
            warn_box(self, 'Save failed', f'Could not save:\n{e}')
            return
        info_box(self, 'Saved',
                 f'Saved {sorted_pts.shape[0]} chirp points to:\n{path}')

    def load_pts(self):
        path = ask_open_paths(
            self, 'Load chirp points',
            filt='CSV (*.csv *.txt *.dat);;All files (*)',
            multi=False)
        if not path:
            return
        try:
            try:
                arr = np.loadtxt(path, delimiter=',')
            except ValueError:
                arr = np.loadtxt(path, delimiter=',', skiprows=1)
        except Exception as e:
            warn_box(self, 'Load failed', f'Could not read file:\n{e}')
            return
        arr = np.atleast_2d(np.asarray(arr, dtype=float))
        if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
            warn_box(self, 'Bad format',
                     'CSV must have two columns: wavelength, t0.')
            return
        pts = arr[:, :2]
        pts = pts[np.argsort(pts[:, 0], kind='stable')]
        self.app.chirp_pts = pts
        self.redraw()
        info_box(self, 'Loaded',
                 f'Loaded {pts.shape[0]} chirp points from:\n{path}')

    def redraw(self):
        app = self.app
        # Remove old artists
        for attr in ('_pts_scatter', '_fit_line_on_map'):
            art = getattr(self, attr, None)
            if art is not None:
                try:
                    art.remove()
                except Exception:
                    pass
                setattr(self, attr, None)

        # Fill the table
        pts = app.chirp_pts
        self.tbl.setRowCount(0)
        for row in range(pts.shape[0]):
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
            self.tbl.setItem(r, 0,
                QtWidgets.QTableWidgetItem(f'{pts[row, 0]:.2f}'))
            self.tbl.setItem(r, 1,
                QtWidgets.QTableWidgetItem(f'{pts[row, 1]:.4g}'))

        if pts.shape[0] > 0:
            self._pts_scatter = self.ax_map.scatter(
                pts[:, 0], pts[:, 1],
                marker='+', s=140, c='w', linewidths=1.8,
                zorder=5)

        ax = self.ax_fit
        ax.clear()
        n_pts = pts.shape[0]
        if n_pts >= 4:
            try:
                pp, rms_fit = ta_core.fit_chirp_params(pts)
                wl_fine = np.linspace(app.wavelength.min(),
                                      app.wavelength.max(), 400)
                t_fine = ta_core.chirp_model(pp, wl_fine)
                self._fit_line_on_map, = self.ax_map.plot(
                    wl_fine, t_fine, 'k-', linewidth=1.6)
                ax.plot(wl_fine, t_fine, 'b-', linewidth=1.4)
                ax.scatter(pts[:, 0], pts[:, 1], s=50,
                           c='r', edgecolors='r')
                ax.grid(True)
                ax.set_xlabel('Wavelength (nm)')
                ax.set_ylabel(f't₀ ({app.t_unit_ax()})')
                ax.set_title(
                    f'N={n_pts} RMS={rms_fit:.3g} {app.t_unit_ax()}\n'
                    f'a={pp[0]:.3g} b={pp[1]:.3g} '
                    f'c={pp[2]:.3g} d={pp[3]:.3g}')
            except Exception as e:
                ax.set_title(f'Fit failed: {e}')
        else:
            ax.set_title(
                f'Need {max(0, 4 - n_pts)} more point(s) (≥4 required)')
        self.canvas_fit.draw_idle()
        self.canvas_map.draw_idle()

    def update_kin_preview(self):
        app = self.app
        if self.last_click_wl is None:
            return
        base = app.get_chirp_base_data()
        iw = int(np.argmin(np.abs(app.wavelength - self.last_click_wl)))
        ax = self.ax_kin
        ax.clear()
        ax.plot(app.delay, base[iw, :], 'k-', linewidth=1.2)
        ax.axvline(self.last_click_t, color='r', linewidth=1.4,
                   label=f't_clicked = {self.last_click_t:.3g}')
        if app.ridge_wl is not None and app.ridge_t is not None:
            jr = int(np.argmin(np.abs(app.ridge_wl - app.wavelength[iw])))
            ax.axvline(app.ridge_t[jr], color='g', linestyle='--',
                       linewidth=1.2, label=f't_ridge = {app.ridge_t[jr]:.3g}')
        ax.axhline(0, color='k', linestyle=':')
        ax.grid(True)
        lo = max(app.delay[0], self.last_click_t - 2)
        hi = min(app.delay[-1], self.last_click_t + 3)
        if hi > lo:
            ax.set_xlim(lo, hi)
        ax.set_title(rf'$\lambda$ = {app.wavelength[iw]:.1f} nm')
        ax.set_xlabel(f'Delay time ({app.t_unit_ax()})')
        ax.set_ylabel(r'$\Delta$A')
        ax.legend(loc='best', fontsize=7)
        self.canvas_kin.draw_idle()

    # ---- Ridge ----
    def set_and_recompute(self, field, val):
        app = self.app
        if field == 'method':
            app.ridge_method = val
        elif field == 'tmin':
            app.ridge_tmin = float(val)
        elif field == 'tmax':
            app.ridge_tmax = float(val)
        elif field == 'smooth':
            app.ridge_smooth_n = int(val)
        if self.cb_show_ridge.isChecked():
            self.recompute_ridge()

    def recompute_ridge(self):
        app = self.app
        base = app.get_chirp_base_data()
        wl_r, t_r = ta_core.compute_ridge(
            app.wavelength, app.delay, base,
            app.ridge_method, app.ridge_tmin, app.ridge_tmax,
            app.ridge_smooth_n)
        app.ridge_wl = wl_r
        app.ridge_t = t_r
        if self._ridge_line is not None:
            try:
                self._ridge_line.remove()
            except Exception:
                pass
            self._ridge_line = None
        if self.cb_show_ridge.isChecked() and t_r is not None:
            self._ridge_line, = self.ax_map.plot(
                wl_r, t_r, color=(0, 0.7, 0), linewidth=1.2)
        self.canvas_map.draw_idle()

    def toggle_ridge(self):
        if self.cb_show_ridge.isChecked():
            self.recompute_ridge()
        else:
            if self._ridge_line is not None:
                try:
                    self._ridge_line.remove()
                except Exception:
                    pass
                self._ridge_line = None
            self.canvas_map.draw_idle()

    def auto_place(self, N):
        app = self.app
        if app.ridge_t is None:
            self.recompute_ridge()
        if app.ridge_t is None:
            warn_box(self, 'No ridge',
                     'Could not compute ridge (check search window).')
            return
        wls = np.linspace(app.wavelength[0], app.wavelength[-1], N + 2)[1:-1]
        new_pts = np.zeros((N, 2))
        for k in range(N):
            j = int(np.argmin(np.abs(app.ridge_wl - wls[k])))
            new_pts[k, :] = [app.ridge_wl[j], app.ridge_t[j]]
        app.chirp_pts = new_pts
        self.redraw()
        if not self.cb_show_ridge.isChecked():
            self.cb_show_ridge.setChecked(True)  # triggers recompute_ridge

    # ---- View ----
    def apply_view_range(self):
        app = self.app
        tmin = app.chirp_view_t_min
        tmax = app.chirp_view_t_max
        if tmin is None or not np.isfinite(tmin):
            tmin = float(app.delay[0])
        if tmax is None or not np.isfinite(tmax):
            tmax = float(app.delay[-1])
        if tmax <= tmin:
            tmax = tmin + 0.1
        self.ax_map.set_ylim(tmin, tmax)
        self.canvas_map.draw_idle()

    def on_view_change(self):
        self.app.chirp_view_t_min = self.ed_view_tmin.value()
        self.app.chirp_view_t_max = self.ed_view_tmax.value()
        self.apply_view_range()

    def reset_view(self):
        app = self.app
        app.chirp_view_t_min = float(app.delay[0])
        app.chirp_view_t_max = float(min(2.0, app.delay[-1]))
        self.ed_view_tmin.setValue(app.chirp_view_t_min)
        self.ed_view_tmax.setValue(app.chirp_view_t_max)
        self.apply_view_range()

    # ---- Apply ----
    def apply_chirp(self):
        app = self.app
        if app.chirp_pts.shape[0] < 4:
            warn_box(self, 'Not enough points',
                     'At least 4 points are required.')
            return
        try:
            pp, rms_fit = ta_core.fit_chirp_params(app.chirp_pts)
        except Exception as e:
            warn_box(self, 'Fit error', f'Fit failed:\n{e}')
            return
        app.chirp_params = pp
        app.chirp_fit_rms = rms_fit
        app.chirp_applied = True
        app.update_all()
        self.accept()
