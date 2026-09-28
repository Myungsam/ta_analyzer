"""
TA Analyzer GUI - Vibrational Coherence dialog.

Two analyses are exposed in tabs on the same dialog:
    1) |FFT|² 2D map of the residual after kinetic subtraction
       (backend: ta_core.compute_coherence).
    2) LPSVD mode decomposition of a 1D residual trace
       (backend: ta_lpsvd.run_lpsvd_analysis) with Lorentzian or Gaussian
       peak models, configurable preprocessing, and an optional FFT window.
"""
from __future__ import annotations

import os

import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.colors import ListedColormap

from ta_widgets import (
    MplCanvas, make_label, make_double_edit, make_int_edit,
    style_button, info_box, warn_box, ask_save_path,
)
import ta_core
import ta_lpsvd
import ta_residual_store


class CoherenceDialog(QtWidgets.QDialog):
    """Compute and explore the |FFT|² oscillation map of the residual,
    plus LPSVD mode decomposition on a chosen 1D residual trace."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.setWindowTitle('Vibrational Coherence (FFT of residual)')
        self.resize(1480, 880)

        outer = QtWidgets.QHBoxLayout(self)

        # =====================================================================
        # LEFT panel
        # =====================================================================
        left = QtWidgets.QGroupBox('Setup')
        lL = QtWidgets.QVBoxLayout(left)

        # ---- Common: residual source ----
        gb_src = QtWidgets.QGroupBox('Residual source')
        sg = QtWidgets.QVBoxLayout(gb_src)
        self.rb_use_ga = QtWidgets.QRadioButton('Use last Global Analysis residual')
        self.rb_use_lda = QtWidgets.QRadioButton('Use last LDA residual')
        self.rb_use_data = QtWidgets.QRadioButton('Use raw data (no kinetic subtraction)')
        self.rb_use_file = QtWidgets.QRadioButton('Use saved residual file')
        self.rb_use_file.setToolTip(
            'Load a residual file (CSV from Kinetic Fit, or xlsx from '
            'GA/LDA) via the file browser.  Clicking this option opens '
            'the picker; use Browse… to change the file later.')
        self.rb_use_ga.setChecked(True)
        sg.addWidget(self.rb_use_ga)
        sg.addWidget(self.rb_use_lda)
        sg.addWidget(self.rb_use_data)
        sg.addWidget(self.rb_use_file)

        # File-source controls: clicking "Use saved residual file"
        # opens a file browser (see _prompt_file_residual).  The
        # Browse… button lets the user pick a different file without
        # unchecking + re-checking the radio.
        row_btn = QtWidgets.QHBoxLayout()
        self.btn_browse_file = QtWidgets.QPushButton('Browse…')
        self.btn_browse_file.setToolTip(
            'Pick a saved residual file (CSV from Kfit, or xlsx from '
            'GA/LDA).')
        row_btn.addWidget(self.btn_browse_file)
        row_btn.addStretch(1)
        sg.addLayout(row_btn)
        self.lbl_loaded_file = QtWidgets.QLabel('(no file loaded)')
        self.lbl_loaded_file.setStyleSheet('color:#666; font-size:10px;')
        self.lbl_loaded_file.setWordWrap(True)
        sg.addWidget(self.lbl_loaded_file)
        lL.addWidget(gb_src)
        # Loaded-file cache, populated by _prompt_file_residual().
        self._loaded_residual: dict | None = None
        self._loaded_residual_path: str | None = None

        # ---- Common: time window ----
        gb_t = QtWidgets.QGroupBox('Time window')
        tg = QtWidgets.QGridLayout(gb_t)
        d = app.delay
        t_default_lo = float(max(d[0], 0.5))
        t_default_hi = float(d[0] + (d[-1] - d[0]) * 0.5)
        tg.addWidget(make_label(f'From ({app.t_unit_txt()}):'), 0, 0)
        self.ed_t_min = make_double_edit(t_default_lo)
        tg.addWidget(self.ed_t_min, 0, 1)
        tg.addWidget(make_label(f'To ({app.t_unit_txt()}):'), 0, 2)
        self.ed_t_max = make_double_edit(t_default_hi)
        tg.addWidget(self.ed_t_max, 0, 3)
        lL.addWidget(gb_t)

        # ---- Tabs: FFT / LPSVD ----
        self.tabs_setup = QtWidgets.QTabWidget()
        self.tabs_setup.addTab(self._build_fft_setup_tab(), 'FFT')
        self.tabs_setup.addTab(self._build_lpsvd_setup_tab(), 'LPSVD')
        lL.addWidget(self.tabs_setup, stretch=1)

        # ---- Common: status + info ----
        self.lbl_status = QtWidgets.QLabel('Ready.')
        lL.addWidget(self.lbl_status)
        self.txt_info = QtWidgets.QTextEdit()
        self.txt_info.setReadOnly(True)
        self.txt_info.setMinimumHeight(140)
        lL.addWidget(self.txt_info)

        left.setFixedWidth(440)
        outer.addWidget(left)

        # =====================================================================
        # RIGHT panel: results tabs (FFT canvas / LPSVD canvas)
        # =====================================================================
        right = QtWidgets.QGroupBox('Results')
        rL = QtWidgets.QVBoxLayout(right)
        self.tabs_res = QtWidgets.QTabWidget()
        # FFT canvas (existing 2x2 plots)
        self.canvas = MplCanvas(self, nrows=2, ncols=2, figsize=(11, 8))
        self.ax_map, self.ax_spec, self.ax_prof, self.ax_R = \
            self.canvas.axes_list
        self.tabs_res.addTab(self.canvas.with_toolbar(self), 'FFT Map')
        # LPSVD canvas (new 2x2 plots)
        self.canvas_lp = MplCanvas(self, nrows=2, ncols=2, figsize=(11, 8))
        (self.ax_lp_t, self.ax_lp_modes,
         self.ax_lp_fft, self.ax_lp_fftmodes) = self.canvas_lp.axes_list
        self.tabs_res.addTab(self.canvas_lp.with_toolbar(self), 'LPSVD')
        rL.addWidget(self.tabs_res, stretch=1)
        outer.addWidget(right, stretch=1)

        # State
        self._last = None        # FFT result bundle
        self._last_lp = None     # LPSVD result bundle
        self._cur_freq = 0.0

        # Wiring
        self.btn_run.clicked.connect(self.do_run)
        self.btn_reset.clicked.connect(self.do_reset)
        self.btn_exp_map.clicked.connect(self._export_map)
        self.btn_exp_spec.clicked.connect(self._export_spec)
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.btn_run_lp.clicked.connect(self.do_run_lpsvd)
        self.btn_reset_lp.clicked.connect(self.do_reset_lpsvd)
        self.btn_exp_lp.clicked.connect(self._export_lpsvd)
        # Enable/disable LPSVD widgets based on dependent checkboxes
        self.chk_window.toggled.connect(self._sync_lpsvd_widgets)
        self.dd_win_type.currentIndexChanged.connect(self._sync_lpsvd_widgets)
        self.chk_polyfit.toggled.connect(self._sync_lpsvd_widgets)
        self.cmb_wl_mode.currentIndexChanged.connect(self._sync_lpsvd_widgets)
        self.rb_lps_gauss.toggled.connect(self._sync_lpsvd_widgets)
        self._sync_lpsvd_widgets()

        # File-source wiring: click the radio → open a picker; Browse…
        # lets the user swap files without touching the radio.
        self.btn_browse_file.clicked.connect(self._prompt_file_residual)
        self.rb_use_file.toggled.connect(self._on_rb_use_file_toggled)

    # =====================================================================
    # Setup tab builders
    # =====================================================================
    def _build_fft_setup_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        L = QtWidgets.QVBoxLayout(w)
        L.setContentsMargins(4, 4, 4, 4)

        gb_f = QtWidgets.QGroupBox('FFT options')
        fg = QtWidgets.QGridLayout(gb_f)
        fg.addWidget(make_label('Detrend:'), 0, 0)
        self.dd_det = QtWidgets.QComboBox()
        self.dd_det.addItem('Linear', 'linear')
        self.dd_det.addItem('Mean', 'mean')
        self.dd_det.addItem('None', 'none')
        fg.addWidget(self.dd_det, 0, 1)
        fg.addWidget(make_label('Apod:'), 0, 2)
        self.dd_ap = QtWidgets.QComboBox()
        for k in ('hann', 'hamming', 'blackman', 'rect'):
            self.dd_ap.addItem(k, k)
        fg.addWidget(self.dd_ap, 0, 3)
        fg.addWidget(make_label('Zero pad:'), 1, 0)
        self.ed_zp = make_int_edit(2, minv=1, maxv=8)
        fg.addWidget(self.ed_zp, 1, 1)
        fg.addWidget(make_label('Norm:'), 1, 2)
        self.dd_n = QtWidgets.QComboBox()
        self.dd_n.addItem('Peak', 'peak')
        self.dd_n.addItem('Per λ', 'perWl')
        self.dd_n.addItem('None', 'none')
        fg.addWidget(self.dd_n, 1, 3)
        fg.addWidget(make_label('Freq unit:'), 2, 0)
        self.dd_fu = QtWidgets.QComboBox()
        self.dd_fu.addItem('cm⁻¹', 'cm-1')
        self.dd_fu.addItem('THz', 'THz')
        self.dd_fu.addItem('Hz', 'Hz')
        fg.addWidget(self.dd_fu, 2, 1, 1, 3)
        L.addWidget(gb_f)

        ar = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton('Run FFT')
        style_button(self.btn_run, bg='#4fa35a', fg='white')
        self.btn_reset = QtWidgets.QPushButton('Reset')
        ar.addWidget(self.btn_run); ar.addWidget(self.btn_reset)
        L.addLayout(ar)

        ex = QtWidgets.QHBoxLayout()
        ex.addWidget(make_label('Export:'))
        self.btn_exp_map = QtWidgets.QPushButton('|FFT|² map')
        self.btn_exp_spec = QtWidgets.QPushButton('Mean spectrum')
        ex.addWidget(self.btn_exp_map); ex.addWidget(self.btn_exp_spec)
        L.addLayout(ex)
        L.addStretch(1)
        return w

    def _build_lpsvd_setup_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        L = QtWidgets.QVBoxLayout(w)
        L.setContentsMargins(4, 4, 4, 4)

        # ---- Wavelength source for the 1D trace ----
        gb_wl = QtWidgets.QGroupBox('Wavelength source (1D trace)')
        wg = QtWidgets.QGridLayout(gb_wl)
        self.cmb_wl_mode = QtWidgets.QComboBox()
        self.cmb_wl_mode.addItem('Mean over all λ', 'mean')
        self.cmb_wl_mode.addItem('Mean over λ range', 'range')
        self.cmb_wl_mode.addItem('Single λ (nearest)', 'single')
        wg.addWidget(make_label('Mode:'), 0, 0)
        wg.addWidget(self.cmb_wl_mode, 0, 1, 1, 3)
        wl = self.app.wavelength
        self.ed_wl_lo = make_double_edit(float(wl.min()), decimals=3)
        self.ed_wl_hi = make_double_edit(float(wl.max()), decimals=3)
        self.ed_wl_one = make_double_edit(float(wl[len(wl)//2]), decimals=3)
        wg.addWidget(make_label('λ from (nm):'), 1, 0)
        wg.addWidget(self.ed_wl_lo, 1, 1)
        wg.addWidget(make_label('to:'), 1, 2)
        wg.addWidget(self.ed_wl_hi, 1, 3)
        wg.addWidget(make_label('Single λ (nm):'), 2, 0)
        wg.addWidget(self.ed_wl_one, 2, 1, 1, 3)
        L.addWidget(gb_wl)

        # ---- Preprocessing ----
        gb_pp = QtWidgets.QGroupBox('Preprocessing')
        pg = QtWidgets.QGridLayout(gb_pp)
        self.chk_polyfit = QtWidgets.QCheckBox('Polynomial baseline removal')
        self.chk_polyfit.setChecked(True)
        pg.addWidget(self.chk_polyfit, 0, 0, 1, 2)
        pg.addWidget(make_label('Degree:'), 0, 2)
        self.ed_poly_deg = make_int_edit(6, minv=0, maxv=12)
        pg.addWidget(self.ed_poly_deg, 0, 3)
        self.chk_detrend = QtWidgets.QCheckBox('Linear detrend')
        self.chk_detrend.setChecked(True)
        pg.addWidget(self.chk_detrend, 1, 0, 1, 2)
        self.chk_center = QtWidgets.QCheckBox('Mean centering')
        self.chk_center.setChecked(True)
        pg.addWidget(self.chk_center, 1, 2, 1, 2)
        pg.addWidget(make_label('Append copies:'), 2, 0)
        self.ed_append = make_int_edit(0, minv=0, maxv=8)
        pg.addWidget(self.ed_append, 2, 1)
        L.addWidget(gb_pp)

        # ---- Fitting model + number of modes ----
        gb_m = QtWidgets.QGroupBox('Fitting model')
        mg = QtWidgets.QGridLayout(gb_m)
        self.rb_lps_lor = QtWidgets.QRadioButton('Lorentzian (LPSVD)')
        self.rb_lps_gauss = QtWidgets.QRadioButton('Gaussian (curve_fit)')
        self.rb_lps_lor.setChecked(True)
        mg.addWidget(self.rb_lps_lor, 0, 0)
        mg.addWidget(self.rb_lps_gauss, 0, 1)
        mg.addWidget(make_label('Number of modes:'), 1, 0)
        self.ed_lps_modes = make_int_edit(6, minv=1, maxv=64)
        mg.addWidget(self.ed_lps_modes, 1, 1)
        L.addWidget(gb_m)

        # ---- Windowing (for FFT comparison only) ----
        gb_w = QtWidgets.QGroupBox('FFT window (for comparison spectrum)')
        wgw = QtWidgets.QGridLayout(gb_w)
        self.chk_window = QtWidgets.QCheckBox('Apply window before FFT')
        self.chk_window.setChecked(True)
        wgw.addWidget(self.chk_window, 0, 0, 1, 4)
        wgw.addWidget(make_label('Type:'), 1, 0)
        self.dd_win_type = QtWidgets.QComboBox()
        self.dd_win_type.addItem('Kaiser', 'kaiser')
        self.dd_win_type.addItem('Gaussian', 'gaussian')
        self.dd_win_type.addItem('None', 'none')
        wgw.addWidget(self.dd_win_type, 1, 1)
        wgw.addWidget(make_label('β:'), 1, 2)
        self.ed_win_beta = make_double_edit(8.0, decimals=2, minv=0.1, maxv=30.0)
        wgw.addWidget(self.ed_win_beta, 1, 3)
        wgw.addWidget(make_label(f'σ ({self.app.t_unit_txt()}):'), 2, 0)
        self.ed_win_sigma = make_double_edit(2.5, decimals=3, minv=0.01, maxv=1e4)
        wgw.addWidget(self.ed_win_sigma, 2, 1, 1, 3)
        L.addWidget(gb_w)

        # ---- Run / reset / export ----
        ar = QtWidgets.QHBoxLayout()
        self.btn_run_lp = QtWidgets.QPushButton('Run LPSVD')
        style_button(self.btn_run_lp, bg='#4fa35a', fg='white')
        self.btn_reset_lp = QtWidgets.QPushButton('Reset')
        ar.addWidget(self.btn_run_lp); ar.addWidget(self.btn_reset_lp)
        L.addLayout(ar)

        ex = QtWidgets.QHBoxLayout()
        ex.addWidget(make_label('Export:'))
        self.btn_exp_lp = QtWidgets.QPushButton('CSV bundle')
        ex.addWidget(self.btn_exp_lp); ex.addStretch(1)
        L.addLayout(ex)
        L.addStretch(1)
        return w

    # =====================================================================
    # Common helpers
    # =====================================================================
    def closeEvent(self, ev):
        self.app.coh_fig = None
        super().closeEvent(ev)

    def _sync_lpsvd_widgets(self):
        """Enable/disable LPSVD detail widgets based on dependent checkboxes."""
        on_window = self.chk_window.isChecked()
        wt = self.dd_win_type.currentData()
        self.dd_win_type.setEnabled(on_window)
        self.ed_win_beta.setEnabled(on_window and wt == 'kaiser')
        self.ed_win_sigma.setEnabled(on_window and wt == 'gaussian')

        self.ed_poly_deg.setEnabled(self.chk_polyfit.isChecked())

        m = self.cmb_wl_mode.currentData()
        self.ed_wl_lo.setEnabled(m == 'range')
        self.ed_wl_hi.setEnabled(m == 'range')
        self.ed_wl_one.setEnabled(m == 'single')

    # ---- File-source panel ----
    def _on_rb_use_file_toggled(self, checked: bool):
        """Open the file picker on radio-on when nothing is loaded yet.

        If a file has already been loaded (user unchecked and re-checked
        the radio), keep the existing selection — they can use Browse…
        to swap.
        """
        if checked and self._loaded_residual is None:
            self._prompt_file_residual()

    def _prompt_file_residual(self):
        """Open a QFileDialog to pick a saved residual file, then load it.

        Default folder: the currently loaded data's folder (falls back
        to Qt cwd), so residuals sit alongside the raw data.
        """
        default_dir = getattr(self.app, 'data_source_dir', '') or ''
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select residual file',
            default_dir,
            'Residual (*.csv *.xlsx);;CSV (*.csv);;Excel (*.xlsx);;'
            'All files (*)')
        if not path:
            # Cancelled: if the radio was just switched on without a
            # prior file, revert so we don't leave the UI in a broken
            # "use file" state with no file.
            if self.rb_use_file.isChecked() and self._loaded_residual is None:
                self.rb_use_ga.setChecked(True)
            return
        try:
            rec = ta_residual_store.load_residual(path)
        except Exception as e:
            warn_box(self, 'Load residual', f'Could not read file:\n{e}')
            return
        self._loaded_residual = rec
        self._loaded_residual_path = path
        self.rb_use_file.setChecked(True)
        meta = rec.get('meta', {})
        kind = rec.get('kind', '?')
        anal = meta.get('analysis', '?')
        wl_tag = meta.get('wavelength_nm', '')
        wl_part = f', λ={wl_tag} nm' if wl_tag else ''
        base = os.path.basename(path)
        self.lbl_loaded_file.setText(f'File: {base}   ({anal}, {kind}{wl_part})')
        self.lbl_status.setText(
            f'Loaded residual: {base}  ({anal}, {kind}{wl_part})')

    def _file_residual_2d(self):
        """Return (wl, t, R) for a 2D loaded file, mapped to the app grid."""
        if self._loaded_residual is None:
            raise RuntimeError(
                'No saved residual is loaded. Check "Use saved '
                'residual file" and pick a file in the browser.')
        rec = self._loaded_residual
        if rec.get('kind') != '2D':
            raise RuntimeError(
                'FFT needs a 2D residual map (saved by Global Analysis '
                'or LDA). The loaded file is 1D.')
        return rec['wl'], rec['t'], rec['R']

    def _file_residual_1d(self):
        """Return (t, r, label) for a 1D trace built from the loaded file."""
        if self._loaded_residual is None:
            raise RuntimeError(
                'No saved residual is loaded. Check "Use saved '
                'residual file" and pick a file in the browser.')
        rec = self._loaded_residual
        if rec.get('kind') == '1D':
            wl_tag = rec.get('meta', {}).get('wavelength_nm', '')
            label = (f'file λ = {wl_tag} nm' if wl_tag
                     else 'file 1D trace')
            return rec['t'], rec['R'], label
        # 2D residual — collapse to a 1D trace per the existing LPSVD
        # wavelength-source UI (mean / range / single).
        wl = rec['wl']
        R = rec['R']
        mode = self.cmb_wl_mode.currentData()
        if mode == 'mean':
            return rec['t'], np.nanmean(R, axis=0), 'file mean over all λ'
        if mode == 'range':
            lo = float(self.ed_wl_lo.value())
            hi = float(self.ed_wl_hi.value())
            if lo > hi:
                lo, hi = hi, lo
            sel = (wl >= lo) & (wl <= hi)
            if int(sel.sum()) == 0:
                raise RuntimeError('Wavelength range selects no points '
                                   'in the loaded file.')
            return (rec['t'], np.nanmean(R[sel, :], axis=0),
                    f'file mean over {lo:g}…{hi:g} nm')
        target = float(self.ed_wl_one.value())
        idx = int(np.argmin(np.abs(wl - target)))
        return rec['t'], R[idx, :].copy(), f'file λ = {wl[idx]:.2f} nm'

    def _build_residual(self):
        """Build the residual map according to the source choice.

        Returns ``(wl, t, R)`` so file-loaded residuals can run on their
        own grid (which may differ from the currently displayed data
        set).  The in-memory sources (GA / LDA / raw) just return the
        app grid.
        """
        app = self.app
        if self.rb_use_file.isChecked():
            wl, t, R = self._file_residual_2d()
            return wl, t, R
        if self.rb_use_ga.isChecked():
            if app.ga_result_fit is None:
                raise RuntimeError(
                    'No Global Analysis result is available yet. Run a '
                    'global fit first, or pick a different residual source.')
            R = app.deltaA - self._broadcast_ga_fit()
            return app.wavelength, app.delay, R
        if self.rb_use_lda.isChecked():
            if not hasattr(app, '_last_lda') or app._last_lda is None:
                raise RuntimeError(
                    'No LDA result available yet. Run an LDA fit first, '
                    'or pick a different residual source.')
            t_lda = app._last_lda['t_win']
            Drec = app._last_lda['res']['Drec']
            Rfull = app.deltaA.copy()
            for j, tj in enumerate(t_lda):
                ix = int(np.argmin(np.abs(app.delay - tj)))
                Rfull[:, ix] = app.deltaA[:, ix] - Drec[:, j]
            return app.wavelength, app.delay, Rfull
        return app.wavelength, app.delay, app.deltaA.copy()

    def _broadcast_ga_fit(self):
        """Return GA fit on the full delay grid (other rows = data so
        residual is zero outside the GA window)."""
        app = self.app
        full = app.deltaA.copy()
        if (app.ga_result_delay is not None
                and len(app.ga_result_delay) < len(app.delay)):
            for j, tj in enumerate(app.ga_result_delay):
                ix = int(np.argmin(np.abs(app.delay - tj)))
                full[:, ix] = app.ga_result_fit[:, j]
        else:
            full = app.ga_result_fit
        return full

    # =====================================================================
    # FFT path
    # =====================================================================
    def do_reset(self):
        self.rb_use_ga.setChecked(True)
        d = self.app.delay
        self.ed_t_min.setValue(float(max(d[0], 0.5)))
        self.ed_t_max.setValue(float(d[0] + (d[-1] - d[0]) * 0.5))
        self.dd_det.setCurrentIndex(0)
        self.dd_ap.setCurrentIndex(0)
        self.ed_zp.setValue(2)
        self.dd_n.setCurrentIndex(0)
        self.dd_fu.setCurrentIndex(0)
        self._last = None
        for a in (self.ax_map, self.ax_spec, self.ax_prof, self.ax_R):
            a.clear()
        self.canvas.draw_idle()
        self.txt_info.setPlainText('')
        self.lbl_status.setText('Reset to defaults.')

    def do_run(self):
        try:
            wl_src, t_src, R = self._build_residual()
        except Exception as e:
            warn_box(self, 'No residual', str(e))
            return
        app = self.app
        t_lo = float(self.ed_t_min.value())
        t_hi = float(self.ed_t_max.value())
        if t_lo > t_hi:
            t_lo, t_hi = t_hi, t_lo
        m = (t_src >= t_lo) & (t_src <= t_hi)
        if int(m.sum()) < 8:
            warn_box(self, 'Window too narrow', 'Need ≥ 8 delay points.')
            return

        self.lbl_status.setText('Computing FFT…')
        QtWidgets.QApplication.processEvents()
        try:
            res = ta_core.compute_coherence(
                R, t_src,
                t_min=t_lo, t_max=t_hi,
                apod=self.dd_ap.currentData(),
                zero_pad=int(self.ed_zp.value()),
                detrend=self.dd_det.currentData(),
                freq_unit=self.dd_fu.currentData(),
                norm_mode=self.dd_n.currentData(),
                time_unit=app.time_unit)
        except Exception as e:
            warn_box(self, 'FFT error', str(e))
            self.lbl_status.setText('FFT failed.')
            return

        self._last = {'res': res, 'R': R, 't_lo': t_lo, 't_hi': t_hi,
                      'wl': wl_src, 't': t_src}
        info = res['info']
        mean_spec = np.nanmean(res['P'], axis=0)
        order = np.argsort(mean_spec)[::-1][:5]
        u = self.dd_fu.currentData()
        freq = res['freq']
        lines = [f'Window: [{t_lo:.4g}, {t_hi:.4g}] {app.t_unit_txt()}',
                 f'Apod: {info["apod"]}, detrend: {info["detrend"]}, '
                 f'zero-pad: {info["zero_pad"]}',
                 f'Resampled grid: dt = {info["dt_uni"]:.4g} '
                 f'{app.t_unit_txt()}, N = {info["Nwin"]}',
                 f'FFT length: {info["Nfft"]}',
                 f'Norm: {info["norm_mode"]}',
                 '',
                 'Top 5 frequencies (mean over λ):']
        for k in order:
            lines.append(f'  ν = {freq[k]:10.3f} {u}'
                         f'   |FFT|² = {mean_spec[k]:.4g}')
        self.txt_info.setPlainText('\n'.join(lines))
        self._cur_freq = float(freq[int(order[0])])
        self.lbl_status.setText('FFT done.')
        self.tabs_res.setCurrentIndex(0)
        self._plot_results()

    def _plot_results(self):
        if self._last is None:
            return
        app = self.app
        res = self._last['res']
        P = res['P']
        freq = res['freq']
        wl_src = self._last['wl']
        t_src = self._last['t']
        u = self.dd_fu.currentData()
        u_label = {'cm-1': 'cm⁻¹', 'THz': 'THz', 'Hz': 'Hz'}.get(u, u)

        ax = self.ax_map; ax.clear()
        cmap = ListedColormap(app.get_colormap_array())
        f_show = freq[1:]
        P_show = P[:, 1:]
        v_max = float(np.percentile(P_show[np.isfinite(P_show)], 99))
        if not np.isfinite(v_max) or v_max <= 0:
            v_max = float(np.nanmax(P_show)) if P_show.size else 1.0
        ax.pcolormesh(f_show, wl_src, P_show,
                      cmap=cmap, vmin=0, vmax=v_max, shading='nearest')
        ax.set_xlabel(f'Frequency ({u_label})')
        ax.set_ylabel('Wavelength (nm)')
        ax.set_title('|FFT|²  (click to set cursor)')
        ax.axvline(self._cur_freq, color='w', linestyle='--', linewidth=1.0)

        ax = self.ax_spec; ax.clear()
        ax.plot(freq, np.nanmean(P, axis=0), '-',
                color='#2c7fb8', linewidth=1.2)
        ax.set_xlabel(f'Frequency ({u_label})')
        ax.set_ylabel('mean |FFT|² over λ')
        ax.set_title('Mean spectrum')
        ax.grid(True, alpha=0.3)
        ax.axvline(self._cur_freq, color='red', linestyle='--', linewidth=1.0)

        self._draw_profile_panel()

        ax = self.ax_R; ax.clear()
        R = self._last['R']
        t_lo = self._last['t_lo']; t_hi = self._last['t_hi']
        m = (t_src >= t_lo) & (t_src <= t_hi)
        ax.plot(t_src[m], np.nanmean(R[:, m], axis=0),
                '-', color=(0.3, 0.3, 0.3), linewidth=0.8)
        ax.axhline(0, color='k', linestyle=':', linewidth=0.5)
        ax.set_xlabel(f'Delay ({app.t_unit_ax()})')
        ax.set_ylabel(r'$\langle$residual$\rangle_\lambda$')
        ax.set_title(f'Time-domain residual (window {t_lo:.2g}…{t_hi:.2g} '
                     f'{app.t_unit_ax()})')
        ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def _draw_profile_panel(self):
        if self._last is None:
            return
        ax = self.ax_prof; ax.clear()
        P = self._last['res']['P']
        freq = self._last['res']['freq']
        wl_src = self._last.get('wl', self.app.wavelength)
        ifreq = int(np.argmin(np.abs(freq - self._cur_freq)))
        ax.plot(wl_src, P[:, ifreq], '-',
                color='#e34a33', linewidth=1.2)
        u = self.dd_fu.currentData()
        u_label = {'cm-1': 'cm⁻¹', 'THz': 'THz', 'Hz': 'Hz'}.get(u, u)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('|FFT|²')
        ax.set_title(f'λ-profile at ν = {freq[ifreq]:.2f} {u_label}')
        ax.grid(True, alpha=0.3)

    def _on_click(self, event):
        if self._last is None or event.inaxes not in (self.ax_map, self.ax_spec):
            return
        if event.xdata is None:
            return
        self._cur_freq = float(event.xdata)
        for ax in (self.ax_map, self.ax_spec):
            for line in list(ax.lines):
                if line.get_linestyle() == '--':
                    try: line.remove()
                    except Exception: pass
        self.ax_map.axvline(self._cur_freq, color='w', linestyle='--', linewidth=1.0)
        self.ax_spec.axvline(self._cur_freq, color='red', linestyle='--', linewidth=1.0)
        self._draw_profile_panel()
        self.canvas.draw_idle()

    # ---- FFT export ----
    def _export_map(self):
        if self._last is None:
            warn_box(self, 'No result', 'Run FFT first.'); return
        path, _ = ask_save_path(self, 'Export |FFT|² map',
                                self.app.default_save_path('coh_map.csv'))
        if not path: return
        wl = self._last.get('wl', self.app.wavelength)
        freq = self._last['res']['freq']
        P = self._last['res']['P']
        out = np.zeros((len(wl) + 1, len(freq) + 1))
        out[0, 0] = 0.0
        out[0, 1:] = freq
        out[1:, 0] = wl
        out[1:, 1:] = P
        delim = ',' if path.lower().endswith('.csv') else '\t'
        try:
            np.savetxt(path, out, delimiter=delim, fmt='%.10g')
            info_box(self, 'Saved',
                     f'Wrote {path}\n(layout: top row = freqs, '
                     f'left col = wl)')
        except Exception as e:
            warn_box(self, 'Export error', str(e))

    def _export_spec(self):
        if self._last is None:
            warn_box(self, 'No result', 'Run FFT first.'); return
        path, _ = ask_save_path(self, 'Export mean |FFT|² spectrum',
                                self.app.default_save_path('coh_spec.csv'))
        if not path: return
        freq = self._last['res']['freq']
        P = self._last['res']['P']
        m = np.nanmean(P, axis=0)
        delim = ',' if path.lower().endswith('.csv') else '\t'
        try:
            mat = np.column_stack([freq, m])
            np.savetxt(path, mat, delimiter=delim, fmt='%.10g',
                       header=delim.join([
                           f'freq_{self.dd_fu.currentData()}',
                           'mean_|FFT|^2']),
                       comments='')
            info_box(self, 'Saved', f'Wrote {path}')
        except Exception as e:
            warn_box(self, 'Export error', str(e))

    # =====================================================================
    # LPSVD path
    # =====================================================================
    def do_reset_lpsvd(self):
        self.cmb_wl_mode.setCurrentIndex(0)
        wl = self.app.wavelength
        self.ed_wl_lo.setValue(float(wl.min()))
        self.ed_wl_hi.setValue(float(wl.max()))
        self.ed_wl_one.setValue(float(wl[len(wl) // 2]))
        self.chk_polyfit.setChecked(True)
        self.ed_poly_deg.setValue(6)
        self.chk_detrend.setChecked(True)
        self.chk_center.setChecked(True)
        self.ed_append.setValue(0)
        self.rb_lps_lor.setChecked(True)
        self.ed_lps_modes.setValue(6)
        self.chk_window.setChecked(True)
        self.dd_win_type.setCurrentIndex(0)
        self.ed_win_beta.setValue(8.0)
        self.ed_win_sigma.setValue(2.5)
        self._last_lp = None
        for a in (self.ax_lp_t, self.ax_lp_modes,
                  self.ax_lp_fft, self.ax_lp_fftmodes):
            a.clear()
        self.canvas_lp.draw_idle()
        self.lbl_status.setText('LPSVD reset.')

    def _build_lpsvd_trace(self, wl: np.ndarray, R: np.ndarray):
        """Reduce the (λ, t) residual matrix R to a 1D trace per UI choice."""
        mode = self.cmb_wl_mode.currentData()
        if mode == 'mean':
            return np.nanmean(R, axis=0), 'mean over all λ'
        if mode == 'range':
            lo = float(self.ed_wl_lo.value())
            hi = float(self.ed_wl_hi.value())
            if lo > hi: lo, hi = hi, lo
            sel = (wl >= lo) & (wl <= hi)
            if int(sel.sum()) == 0:
                raise RuntimeError('Wavelength range selects no points.')
            return np.nanmean(R[sel, :], axis=0), f'mean over {lo:g}…{hi:g} nm'
        # single
        target = float(self.ed_wl_one.value())
        idx = int(np.argmin(np.abs(wl - target)))
        return R[idx, :].copy(), f'λ = {wl[idx]:.2f} nm'

    def do_run_lpsvd(self):
        # File-mode + 1D file → use the stored trace directly.  All
        # other modes (and 2D files) reduce to a 1D trace via
        # _build_lpsvd_trace on the residual map.
        use_file_1d = (self.rb_use_file.isChecked()
                       and self._loaded_residual is not None
                       and self._loaded_residual.get('kind') == '1D')
        try:
            if use_file_1d:
                t_src, y, wl_tag = self._file_residual_1d()
            else:
                wl_src, t_src, R = self._build_residual()
                y, wl_tag = self._build_lpsvd_trace(wl_src, R)
        except Exception as e:
            warn_box(self, 'No residual', str(e))
            return
        t_lo = float(self.ed_t_min.value())
        t_hi = float(self.ed_t_max.value())
        if t_lo > t_hi: t_lo, t_hi = t_hi, t_lo

        self.lbl_status.setText('Running LPSVD…')
        QtWidgets.QApplication.processEvents()
        try:
            res = ta_lpsvd.run_lpsvd_analysis(
                t_src, y, time_unit=self.app.time_unit,
                t_start=t_lo, t_end=t_hi,
                do_polyfit=self.chk_polyfit.isChecked(),
                polyfit_deg=int(self.ed_poly_deg.value()),
                do_detrend=self.chk_detrend.isChecked(),
                do_center=self.chk_center.isChecked(),
                append_count=int(self.ed_append.value()),
                model=('gaussian' if self.rb_lps_gauss.isChecked()
                       else 'lorentzian'),
                num_modes=int(self.ed_lps_modes.value()),
                window_type=self.dd_win_type.currentData(),
                beta=float(self.ed_win_beta.value()),
                gaussian_sigma=float(self.ed_win_sigma.value()),
                apply_window_to_fft=self.chk_window.isChecked(),
            )
        except Exception as e:
            warn_box(self, 'LPSVD error', str(e))
            self.lbl_status.setText('LPSVD failed.')
            return

        res['wl_tag'] = wl_tag
        self._last_lp = res
        self._write_lpsvd_info(t_lo, t_hi)
        self.lbl_status.setText('LPSVD done.')
        self.tabs_res.setCurrentIndex(1)
        self._plot_lpsvd_results()

    def _write_lpsvd_info(self, t_lo, t_hi):
        res = self._last_lp
        if res is None:
            return
        u_t = self.app.t_unit_txt()
        model = res['model']
        modes = res['modes']
        std_r = res['std_residual']
        src = getattr(self.app, 'data_source_desc', None) or '(unnamed dataset)'
        if self.rb_use_file.isChecked() and self._loaded_residual is not None:
            meta = self._loaded_residual.get('meta', {})
            anal = meta.get('analysis', '?')
            res_src = f'loaded residual file ({anal})'
        elif self.rb_use_ga.isChecked():
            res_src = 'Global Analysis residual'
        elif self.rb_use_lda.isChecked():
            res_src = 'LDA residual'
        else:
            res_src = 'raw ΔA (no kinetic subtraction)'
        lines = [
            f'LPSVD ({model})',
            f'Dataset: {src}',
            f'Residual: {res_src}    Trace: {res["wl_tag"]}',
            f'Window: [{t_lo:.4g}, {t_hi:.4g}] {u_t}    '
            f'dt = {res["dt"]:.4g} {u_t}    N = {res["y_centered"].size}',
            f'Window fn: {res["window_label"]}   '
            f'std(residual) = {std_r:.3e}',
            '',
            'Modes (sorted by amplitude):',
            '   #   ν (cm⁻¹)    f (THz)     amp        damping    rel',
        ]
        for i, m in enumerate(modes):
            f_THz = m['freq_Hz'] * 1e-12
            wn = m['freq_Hz'] / ta_lpsvd.C_CM
            if model == 'gaussian':
                damp_str = f'Γ={m["damp_per_s2"]*1e-24:.3g}/ps²'
            else:
                damp_str = f'γ={m["damp_Hz"]*1e-12:.3g}/ps'
            rel = 'Y' if m.get('reliable') else 'n'
            lines.append(
                f'  {i+1:2d}  {wn:9.2f}  {f_THz:8.3f}  {m["amp"]:9.3e}'
                f'   {damp_str:>12s}   {rel}')
        self.txt_info.setPlainText('\n'.join(lines))

    def _plot_lpsvd_results(self):
        if self._last_lp is None:
            return
        res = self._last_lp
        app = self.app
        u_t = app.t_unit_ax()
        t_late = res['t_late']
        modes = res['modes']

        # (0,0) Time domain: data + total fit + window envelope
        ax = self.ax_lp_t; ax.clear()
        ax.plot(t_late, res['y_raw'], color='lightgray', linewidth=1.6, label='Data')
        ax.plot(t_late, res['total_with_bg'], color='red', linestyle='--',
                linewidth=1.2, label=f"{res['model'].capitalize()} fit")
        ax.plot(t_late, res['window_envelope'], 'k:', alpha=0.5,
                label=res['window_label'])
        ax.set_xlabel(f'Delay ({u_t})')
        ax.set_title(f"Time-domain reconstruction — {res['wl_tag']}")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.25)

        # (0,1) Residual + decomposed modes (offset)
        ax = self.ax_lp_modes; ax.clear()
        ax.plot(t_late, res['residual'], color='black', alpha=0.7,
                linestyle=':', linewidth=1.2, label='Residual')
        offset = float(np.max(np.abs(res['residual']))) * 1.5 if res['residual'].size else 0.0
        cmap = ta_core._mpl_cm.get_cmap('tab10')
        for i, (rec, m) in enumerate(zip(res['recons'], modes)):
            wn = m['freq_Hz'] / ta_lpsvd.C_CM
            color = cmap(i % 10)
            ax.plot(t_late, rec - offset, color=color, linewidth=1.0,
                    label=f'{wn:.1f} cm⁻¹')
            if rec.size:
                offset += float(np.max(np.abs(rec))) * 1.5
        ax.set_xlabel(f'Delay ({u_t})')
        ax.set_yticks([])
        ax.set_title('Residual & decomposed modes')
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.25)

        # (1,0) Frequency domain: orig FFT + fit FFT
        ax = self.ax_lp_fft; ax.clear()
        fcm = res['freq_hz'] / ta_lpsvd.C_CM
        pos = fcm > 0
        ax.plot(fcm[pos], np.abs(res['fft_orig'])[pos],
                color='lightgray', linewidth=1.4, label='Original')
        ax.plot(fcm[pos], np.abs(res['fft_fit'])[pos],
                color='#1c5fbf', linewidth=1.1, label=f"{res['model'].capitalize()} fit")
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_title('Frequency spectrum')
        # Auto-zoom: highlight 0…max(mode freq)·1.5 if modes are present
        if modes:
            fmax_cm = max(m['freq_Hz'] / ta_lpsvd.C_CM for m in modes)
            ax.set_xlim(0, max(50.0, fmax_cm * 1.5))
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper right', fontsize=8)

        # (1,1) Decomposed mode FFTs (fill)
        ax = self.ax_lp_fftmodes; ax.clear()
        ax.plot(fcm[pos], np.abs(res['fft_orig'])[pos],
                color='lightgray', linewidth=1.2, alpha=0.5)
        for i, spec in enumerate(res['fft_modes']):
            color = cmap(i % 10)
            ax.fill_between(fcm[pos], np.abs(spec)[pos], alpha=0.35,
                            color=color, label=f'Mode {i+1}')
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_title('Decomposed mode FFTs')
        if modes:
            fmax_cm = max(m['freq_Hz'] / ta_lpsvd.C_CM for m in modes)
            ax.set_xlim(0, max(50.0, fmax_cm * 1.5))
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper right', fontsize=7, ncol=2)

        self.canvas_lp.draw_idle()

    def _export_lpsvd(self):
        if self._last_lp is None:
            warn_box(self, 'No result', 'Run LPSVD first.'); return
        path, _ = ask_save_path(self, 'Export LPSVD bundle',
                                self.app.default_save_path('lpsvd_bundle.csv'))
        if not path: return
        res = self._last_lp
        delim = ',' if path.lower().endswith('.csv') else '\t'

        # 1) Modes table
        base, ext = (path.rsplit('.', 1) + [''])[:2] if '.' in path else (path, '')
        modes_path = f'{base}_modes.{ext}' if ext else f'{base}_modes.csv'
        rows = []
        for i, m in enumerate(res['modes']):
            wn = m['freq_Hz'] / ta_lpsvd.C_CM
            f_THz = m['freq_Hz'] * 1e-12
            damp = (m['damp_per_s2'] * 1e-24 if res['model'] == 'gaussian'
                    else m['damp_Hz'] * 1e-12)
            rows.append([i + 1, wn, f_THz, m['amp'], damp, m['phase'],
                         1 if m.get('reliable') else 0])
        hdr_damp = ('damping_/ps2' if res['model'] == 'gaussian'
                    else 'damping_/ps')
        np.savetxt(modes_path,
                   np.asarray(rows, dtype=float),
                   delimiter=delim, fmt='%.10g',
                   header=delim.join(['mode', 'wavenumber_cm-1', 'freq_THz',
                                      'amplitude', hdr_damp, 'phase_rad',
                                      'reliable']),
                   comments='')

        # 2) Time-domain table.  The polynomial / detrend / centering
        # baselines are exported as explicit columns so the user can
        # see exactly how the background line was fitted in the time
        # domain.  bg_total = bg_poly + bg_detrend + bg_center, and
        # data = fit + residual = (total_centered + bg_total).
        time_path = f'{base}_time.{ext}' if ext else f'{base}_time.csv'
        N = res['t_late'].size
        bg_poly = res.get('bg_poly', np.zeros(N))
        bg_detrend = res.get('bg_detrend', np.zeros(N))
        bg_center_arr = np.full(N, float(res.get('bg_center', 0.0)))
        cols = [res['t_late'], res['y_raw'], res['total_with_bg'],
                res['residual'], res['window_envelope'],
                res['bg'], bg_poly, bg_detrend, bg_center_arr,
                res['y_centered'], res['total_centered']]
        hdr = ['t', 'data', 'fit', 'residual', 'window',
               'bg_total', 'bg_poly', 'bg_detrend', 'bg_center',
               'y_centered', 'fit_centered']
        for i, rec in enumerate(res['recons']):
            cols.append(rec); hdr.append(f'mode{i+1}')
        mat = np.column_stack(cols)
        np.savetxt(time_path, mat, delimiter=delim, fmt='%.10g',
                   header=delim.join(hdr), comments='')

        # 3) Frequency-domain table (positive freqs only)
        freq_path = f'{base}_freq.{ext}' if ext else f'{base}_freq.csv'
        fcm = res['freq_hz'] / ta_lpsvd.C_CM
        fth = res['freq_hz'] * 1e-12
        pos = fcm > 0
        cols = [fcm[pos], fth[pos],
                np.abs(res['fft_orig'])[pos], np.abs(res['fft_fit'])[pos]]
        hdr = ['wavenumber_cm-1', 'freq_THz', 'orig_mag', 'fit_mag']
        for i, spec in enumerate(res['fft_modes']):
            cols.append(np.abs(spec)[pos]); hdr.append(f'mode{i+1}_mag')
        mat = np.column_stack(cols)
        np.savetxt(freq_path, mat, delimiter=delim, fmt='%.10g',
                   header=delim.join(hdr), comments='')

        # 4) Metadata sidecar — records every condition needed to
        # reproduce this run: time window, polynomial baseline (degree
        # + coefficients + how the line sits in the time domain),
        # detrend/centering flags, fitting model and mode count, and
        # the FFT window settings (window function, β/σ, zero-padding).
        meta_path = f'{base}_meta.txt'
        self._write_lpsvd_meta_file(meta_path, res)

        info_box(self, 'Saved',
                 f'Wrote:\n  {modes_path}\n  {time_path}\n'
                 f'  {freq_path}\n  {meta_path}')

    def _write_lpsvd_meta_file(self, meta_path, res):
        """Write a human-readable processing-conditions sidecar next to
        the LPSVD CSV bundle."""
        p = res.get('params', {})
        u_t = self.app.t_unit_txt()
        # Background source / dataset identification
        src = (getattr(self.app, 'data_source_desc', None)
               or '(unnamed dataset)')
        if self.rb_use_file.isChecked() and self._loaded_residual is not None:
            meta = self._loaded_residual.get('meta', {})
            anal = meta.get('analysis', '?')
            res_src = f'loaded residual file ({anal})'
        elif self.rb_use_ga.isChecked():
            res_src = 'Global Analysis residual'
        elif self.rb_use_lda.isChecked():
            res_src = 'LDA residual'
        else:
            res_src = 'raw ΔA (no kinetic subtraction)'

        def _fmt_opt(x):
            return '(default)' if x is None else f'{x:g}'

        lines = []
        lines.append('# LPSVD processing-conditions log')
        lines.append('# Generated by TA Analyzer (CSV bundle export)')
        lines.append('')
        lines.append('[dataset]')
        lines.append(f'dataset = {src}')
        lines.append(f'residual_source = {res_src}')
        lines.append(f'wavelength_reduction = {res.get("wl_tag", "")}')
        lines.append('')
        lines.append('[time_window]')
        lines.append(f'unit = {u_t}')
        lines.append(f't_start_requested = {_fmt_opt(p.get("t_start_req"))}')
        lines.append(f't_end_requested   = {_fmt_opt(p.get("t_end_req"))}')
        lines.append(f't_start_used      = {p.get("t_start_used", float("nan")):g}')
        lines.append(f't_end_used        = {p.get("t_end_used", float("nan")):g}')
        lines.append(f'dt_median         = {res.get("dt", float("nan")):g}')
        lines.append(f'N_window          = {p.get("N_window", 0)}')
        lines.append(f'append_count      = {p.get("append_count", 0)}')
        lines.append('')
        lines.append('[processing.polyfit]')
        lines.append(f'enabled         = {p.get("do_polyfit", False)}')
        lines.append(f'degree_request  = {p.get("polyfit_deg_requested", 0)}')
        lines.append(f'degree_used     = {p.get("polyfit_deg_used", 0)}')
        coefs = p.get('poly_coefs')
        if coefs:
            lines.append(f'# coefficients in highest-power-first order '
                         f'(np.polyfit convention); independent variable '
                         f'is t in {u_t}.')
            lines.append('coefficients = '
                         + ', '.join(f'{c:.10g}' for c in coefs))
            terms = []
            n = len(coefs) - 1
            for k, c in enumerate(coefs):
                power = n - k
                if power == 0:
                    terms.append(f'{c:.6g}')
                elif power == 1:
                    terms.append(f'{c:.6g}*t')
                else:
                    terms.append(f'{c:.6g}*t^{power}')
            lines.append('expression   = bg_poly(t) = ' + ' + '.join(terms))
        else:
            lines.append('coefficients = (none — polynomial step skipped)')
        # Quick descriptive stats of the actual baseline line in time
        # domain — easy to glance at and confirm the fit was sensible.
        bg_poly = res.get('bg_poly')
        if bg_poly is not None and getattr(bg_poly, 'size', 0):
            lines.append(f'bg_poly_first = {float(bg_poly[0]):.6g}')
            lines.append(f'bg_poly_last  = {float(bg_poly[-1]):.6g}')
            lines.append(f'bg_poly_min   = {float(np.min(bg_poly)):.6g}')
            lines.append(f'bg_poly_max   = {float(np.max(bg_poly)):.6g}')
            lines.append(f'bg_poly_mean  = {float(np.mean(bg_poly)):.6g}')
        lines.append('')
        lines.append('[processing.detrend]')
        lines.append(f'linear_detrend = {p.get("do_detrend", False)}')
        bg_dt = res.get('bg_detrend')
        if (p.get('do_detrend') and bg_dt is not None
                and getattr(bg_dt, 'size', 0)):
            lines.append(f'detrend_first = {float(bg_dt[0]):.6g}')
            lines.append(f'detrend_last  = {float(bg_dt[-1]):.6g}')
        lines.append('')
        lines.append('[processing.center]')
        lines.append(f'mean_center  = {p.get("do_center", False)}')
        lines.append(f'mean_removed = {p.get("bg_center", 0.0):.6g}')
        lines.append('')
        lines.append('[fit_model]')
        lines.append(f'model        = {p.get("model", "?")}')
        lines.append(f'num_modes    = {p.get("num_modes", 0)}')
        lines.append(f'std_residual = {res.get("std_residual", 0.0):.6g}')
        lines.append('')
        lines.append('[fft_window]')
        lines.append(f'window_type        = {p.get("window_type", "none")}')
        lines.append(f'kaiser_beta        = {p.get("beta", 0.0):g}')
        lines.append(f'gaussian_sigma     = {p.get("gaussian_sigma", 0.0):g}')
        lines.append(f'apply_window_to_fft= {p.get("apply_window_to_fft", False)}')
        lines.append(f'window_label       = {res.get("window_label", "")}')
        lines.append(f'fft_pad_factor     = {p.get("fft_pad_factor", 0)}')
        lines.append(f'fft_pad_min        = {p.get("fft_pad_min", 0)}')
        lines.append(f'fft_N_pad          = {p.get("fft_N_pad", 0)}')
        lines.append('')

        with open(meta_path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines))
