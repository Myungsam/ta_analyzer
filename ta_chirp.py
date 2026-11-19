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
import ta_opju_export


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

        # t0 helper: option buttons on the left, one row per method.
        # Row 0 = ridge (argmax) + shared search window / λ-smooth,
        # row 1 = per-λ erf (IRF⊗exp) fit, row 2 = overlay / Auto-place.
        ridge_box = QtWidgets.QGroupBox('t₀ Helper (reference t₀ curve)')
        r_lay = QtWidgets.QGridLayout(ridge_box)
        self.rb_ridge = QtWidgets.QRadioButton('Ridge (argmax)')
        self.rb_erf = QtWidgets.QRadioButton('erf fit (IRF⊗exp)')
        self.rb_ridge.setToolTip(
            'Per wavelength: delay of max |dA/dt| or |ΔA| in the search '
            'window; moving-average over λ-smooth points.')
        self.rb_erf.setToolTip(
            'Per wavelength: fit A·(exp ⊗ Gaussian IRF) + offset in the '
            'search window; t₀ from the fit. Rows with R² below the '
            'minimum or t₀ at a window edge are dropped; λ-smooth applies '
            'a median filter.')
        self._helper_group = QtWidgets.QButtonGroup(self)
        self._helper_group.addButton(self.rb_ridge)
        self._helper_group.addButton(self.rb_erf)
        (self.rb_erf if app.t0_helper == 'erf'
         else self.rb_ridge).setChecked(True)
        r_lay.addWidget(self.rb_ridge, 0, 0)
        r_lay.addWidget(self.rb_erf, 1, 0)

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

        r_lay.addWidget(make_label('R² min:', 'right'), 1, 1)
        self.ed_r2min = make_double_edit(app.erf_r2_min, decimals=3,
                                         minv=0.0, maxv=0.999)
        self.ed_r2min.setSingleStep(0.01)
        r_lay.addWidget(self.ed_r2min, 1, 2)
        r_lay.addWidget(make_label(u'λ stride (0=auto):', 'right'), 1, 3)
        self.ed_stride = make_int_edit(app.erf_stride, minv=0, maxv=1000)
        r_lay.addWidget(self.ed_stride, 1, 4)
        self.lbl_erf = QtWidgets.QLabel('')
        self.lbl_erf.setStyleSheet('color: #555;')
        r_lay.addWidget(self.lbl_erf, 1, 5, 1, 5)

        self.cb_show_ridge = QtWidgets.QCheckBox('Show t₀ overlay')
        r_lay.addWidget(self.cb_show_ridge, 2, 0)
        r_lay.addWidget(make_label('Auto-suggest N points:', 'right'), 2, 1)
        self.ed_auto_n = make_int_edit(8, minv=4, maxv=100)
        r_lay.addWidget(self.ed_auto_n, 2, 2)
        self.btn_auto = QtWidgets.QPushButton('Auto-place')
        r_lay.addWidget(self.btn_auto, 2, 3)
        outer.addWidget(ridge_box)
        self._erf_cache = None       # (key, fit_erf_rows result)
        # erf refits take seconds: debounce window/stride edits.
        self._refit_timer = QtCore.QTimer(self)
        self._refit_timer.setSingleShot(True)
        self._refit_timer.setInterval(400)
        self._refit_timer.timeout.connect(self.recompute_ridge)
        self._update_helper_enabled()

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
        self.rb_erf.toggled.connect(
            lambda on: self.set_and_recompute(
                'helper', 'erf' if on else 'ridge'))
        self.ed_r2min.valueChanged.connect(
            lambda v: self.set_and_recompute('r2min', v))
        self.ed_stride.valueChanged.connect(
            lambda v: self.set_and_recompute('stride', int(v)))
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
        # Redrawn when the data changes while open: replace, don't stack,
        # the colorbar.
        old_cbar = getattr(self, '_cbar', None)
        if old_cbar is not None:
            try:
                old_cbar.remove()
            except Exception:
                pass
            self._cbar = None
        try:
            self._cbar = self.canvas_map.fig.colorbar(self._im, ax=ax)
            self._cbar.set_label(r'$\Delta$A')
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
        ta_opju_export.stage(app, 'chirp', path,
                             f'{sorted_pts.shape[0]} points')
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
        else:
            jr = None
        # Skip the helper line when its nearest row is far away (a
        # rejected erf band): it would describe a different wavelength.
        if jr is not None and app.ridge_wl.size > 1:
            gap = 2.0 * float(np.median(np.abs(np.diff(app.ridge_wl))))
            if abs(app.ridge_wl[jr] - app.wavelength[iw]) > gap:
                jr = None
        if jr is not None:
            ax.axvline(app.ridge_t[jr], color='g', linestyle='--',
                       linewidth=1.2,
                       label=f't_{self._helper_name()} = {app.ridge_t[jr]:.3g}')
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
        elif field == 'helper':
            app.t0_helper = val
            self._update_helper_enabled()
        elif field == 'r2min':
            app.erf_r2_min = float(val)
        elif field == 'stride':
            app.erf_stride = int(val)
        # A change of helper/parameters makes the old curve stale.
        app.ridge_wl = None
        app.ridge_t = None
        self.lbl_erf.setText('')
        if self.cb_show_ridge.isChecked():
            slow = (app.t0_helper == 'erf'
                    and field in ('tmin', 'tmax', 'stride', 'helper'))
            if slow:
                self._refit_timer.start()
            else:
                self.recompute_ridge()
        else:
            self.update_kin_preview()

    def _update_helper_enabled(self):
        erf = self.rb_erf.isChecked()
        self.dd_method.setEnabled(not erf)
        if not erf:
            self.lbl_erf.setText('')
        for w in (self.ed_r2min, self.ed_stride, self.lbl_erf):
            w.setEnabled(erf)

    def _helper_name(self):
        return 'erf' if self.app.t0_helper == 'erf' else 'ridge'

    def _compute_erf(self, base):
        """Per-λ erf fits, cached on (window, stride, base data)."""
        app = self.app
        # Content-based key: a t0 shift, BG change or same-shape reload
        # must refit, not reuse old fits.
        key = (float(app.ridge_tmin), float(app.ridge_tmax),
               ta_core.erf_stride(app.wavelength.size, app.erf_stride),
               base.shape, hash(np.ascontiguousarray(base).tobytes()),
               hash(np.asarray(app.delay, float).tobytes()),
               hash(np.asarray(app.wavelength, float).tobytes()))
        if self._erf_cache is None or self._erf_cache[0] != key:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                fits = ta_core.fit_erf_rows(
                    app.wavelength, app.delay, base,
                    app.ridge_tmin, app.ridge_tmax, app.erf_stride)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            self._erf_cache = (key, fits)
        fits = self._erf_cache[1]
        if fits is None:
            return None
        return ta_core.erf_t0_accept(fits, app.erf_r2_min,
                                     app.ridge_smooth_n)

    def _current_base(self):
        """Fresh BG-corrected base; redraw the map if it changed while the
        (modeless) dialog was open, e.g. after a crop, BG or t0 shift."""
        app = self.app
        base = app.get_chirp_base_data()
        old = self._base_data
        if (base.shape != old.shape
                or not np.array_equal(base, old, equal_nan=True)):
            self._ridge_line = None     # cleared by draw_map_once
            self.draw_map_once()
            self.redraw()
            self.apply_view_range()
        return self._base_data

    def recompute_ridge(self):
        app = self.app
        self._refit_timer.stop()
        base = self._current_base()
        if app.t0_helper == 'erf':
            res = self._compute_erf(base)
            if res is None or res['t0'].size == 0:
                wl_r = t_r = None
            else:
                wl_r, t_r = res['wl'], res['t0']
            if res is None:
                self.lbl_erf.setText(
                    f'window holds < {ta_core.ERF_MIN_POINTS} delays')
            else:
                n_ok, n = int(res['ok'].sum()), res['ok'].size
                fw = (np.nanmedian(res['fwhm_all'][res['ok']])
                      if n_ok else float('nan'))
                self.lbl_erf.setText(
                    f'{n_ok}/{n} λ accepted, median FWHM = {fw:.3g} '
                    f'{app.t_unit_ax()}')
        else:
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
            if app.t0_helper == 'erf':
                self._ridge_line, = self.ax_map.plot(
                    wl_r, t_r, color=(0, 0.7, 0), linewidth=1.2,
                    marker='.', markersize=3)
            else:
                self._ridge_line, = self.ax_map.plot(
                    wl_r, t_r, color=(0, 0.7, 0), linewidth=1.2)
        self.canvas_map.draw_idle()
        self.update_kin_preview()

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
            if app.t0_helper == 'erf':
                warn_box(self, 'No t₀ curve',
                         'No wavelength passed the erf fit (check the '
                         'search window or lower R² min).')
            else:
                warn_box(self, 'No ridge',
                         'Could not compute ridge (check search window).')
            return
        wls = np.linspace(app.wavelength[0], app.wavelength[-1], N + 2)[1:-1]
        # erf rows can have gaps (rejected λ): drop targets with no row
        # within half a target spacing, and keep each source row once.
        half = 0.5 * abs(wls[1] - wls[0]) if N > 1 else np.inf
        near = [int(np.argmin(np.abs(app.ridge_wl - w))) for w in wls]
        idx = np.unique([j for j, w in zip(near, wls)
                         if abs(app.ridge_wl[j] - w) <= half])
        if idx.size < 4:
            warn_box(self, 'Too few points',
                     f'Only {idx.size} of {N} target wavelengths have a '
                     f'nearby {self._helper_name()} t₀ (≥4 needed). '
                     'Points were not changed; widen the search window, '
                     'lower R² min or place points by hand.')
            return
        new_pts = np.column_stack([app.ridge_wl[idx], app.ridge_t[idx]])
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
