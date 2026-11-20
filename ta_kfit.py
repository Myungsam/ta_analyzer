"""
TA Analyzer GUI - Kinetic Fit dialog.

Fits kinetic traces to a sum of (possibly stretched) exponentials
convolved with a Gaussian IRF.  Traces are brought in with **Import**:
either the main window's pinned kinetics ("Import pinned kinetics") or,
as before, one λ-cut (optionally averaged over ± half-width).

One trace  → single-trace fit (ta_core.fit_single_trace).
Several    → global fit with shared τ / β / t₀ / FWHM and per-trace
             amplitudes (ta_core.fit_global_analysis).
Both report 1σ errors, population ratios |Aᵢ| / Σ|Aⱼ| and fit statistics.
"""
from __future__ import annotations

import os

import numpy as np
from PyQt5 import QtWidgets, QtCore

from ta_widgets import (
    MplCanvas, make_label, make_double_edit, make_int_edit,
    style_button, info_box, warn_box, ask_save_path,
)
import ta_core
import ta_residual_store
from ta_ga import (
    ARTIFACT_CHOICES, ARTIFACT_LABELS, ARTIFACT_NAMES, ARTIFACT_TOOLTIP,
    ARTIFACT_SKIP_WARN, ART_SEP_TOOLTIP, FROM_CHIRP_TOOLTIP,
    seed_artifact_from_chirp,
)

# Basis column index → params-CSV key suffix (B_G / B_dG / B_d2G).
_ART_KEYS = {0: 'G', 1: 'dG', 2: 'd2G'}


class KineticFitDialog(QtWidgets.QDialog):
    """Fit one kinetic trace y(t), or several globally.

    Supports an arbitrary mix of standard- and stretched-exponential
    components plus an optional offset (τ = ∞).
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.setWindowTitle('Kinetic Fit')
        self.resize(1180, 760)

        outer = QtWidgets.QHBoxLayout(self)

        # ---------------- Left panel: inputs ----------------
        left = QtWidgets.QGroupBox('Setup')
        lL = QtWidgets.QVBoxLayout(left)

        # Wavelength / averaging window
        row_wl = QtWidgets.QHBoxLayout()
        row_wl.addWidget(make_label('Centre λ (nm):', 'right'))
        wl0 = float(app.selWL) if app.selWL else float(app.wavelength[len(app.wavelength)//2])
        self.ed_wl = make_double_edit(wl0, decimals=2,
                                      minv=float(app.wavelength.min()),
                                      maxv=float(app.wavelength.max()))
        row_wl.addWidget(self.ed_wl)
        row_wl.addWidget(make_label('± half-width (nm):', 'right'))
        self.ed_wl_hw = make_double_edit(0.0, decimals=2, minv=0.0,
                                         maxv=float(np.ptp(app.wavelength)))
        self.ed_wl_hw.setToolTip(
            'Average ΔA over [λ − hw, λ + hw]. Set to 0 to use the '
            'single nearest pixel.')
        row_wl.addWidget(self.ed_wl_hw)
        self.btn_use_main = QtWidgets.QPushButton('Use main λ')
        self.btn_use_main.setToolTip(
            'Copy the main window crosshair wavelength here.')
        row_wl.addWidget(self.btn_use_main)
        lL.addLayout(row_wl)

        # Import: pinned kinetics (global fit) or the λ window above
        row_imp = QtWidgets.QHBoxLayout()
        self.btn_import = QtWidgets.QPushButton('Import')
        style_button(self.btn_import, bg='#4c8cca', fg='white')
        self.btn_import.setToolTip(
            'Load the trace(s) to fit.  Changes to λ / half-width or to '
            'the main-window pins take effect on the next Import.')
        row_imp.addWidget(self.btn_import)
        self.cb_pinned = QtWidgets.QCheckBox('Import pinned kinetics')
        self.cb_pinned.setToolTip(
            'Checked: import every kinetics pinned in the main window '
            '(Integrate BW pins keep their band) and fit them globally.\n'
            'Unchecked, or no pins: import the λ ± half-width trace above.')
        self.cb_pinned.setChecked(bool(app.kinOverlays))
        row_imp.addWidget(self.cb_pinned)
        row_imp.addStretch(1)
        lL.addLayout(row_imp)
        self.lbl_import = QtWidgets.QLabel('No data imported — press Import.')
        self.lbl_import.setWordWrap(True)
        lL.addWidget(self.lbl_import)

        # Number of components + has_inf
        row_n = QtWidgets.QHBoxLayout()
        row_n.addWidget(make_label('Components:', 'right'))
        self.dd_N = QtWidgets.QComboBox()
        self.dd_N.addItems(['1', '2', '3', '4', '5'])
        self.dd_N.setCurrentText('2')
        row_n.addWidget(self.dd_N)
        self.cb_inf = QtWidgets.QCheckBox(u'τ = ∞ offset')
        row_n.addWidget(self.cb_inf)
        row_n.addStretch(1)
        lL.addLayout(row_n)

        # Per-component table: tau / fixed / stretched / beta / fixed
        lL.addWidget(make_label(
            f'Components — initial values ({app.t_unit_txt()}):'))
        self.tbl = QtWidgets.QTableWidget(0, 5)
        self.tbl.setHorizontalHeaderLabels(
            [f'τ_init ({app.t_unit_txt()})', 'τ fixed',
             'Stretched', 'β_init', 'β fixed'])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        lL.addWidget(self.tbl)

        # IRF row
        irf_box = QtWidgets.QGroupBox('IRF (Gaussian)')
        irf_lay = QtWidgets.QGridLayout(irf_box)
        irf_lay.addWidget(make_label(f't₀ ({app.t_unit_txt()}):'), 0, 0)
        self.ed_t0 = make_double_edit(0.0)
        irf_lay.addWidget(self.ed_t0, 0, 1)
        self.cb_t0_fix = QtWidgets.QCheckBox('Fixed')
        self.cb_t0_fix.setChecked(True)
        irf_lay.addWidget(self.cb_t0_fix, 0, 2)
        irf_lay.addWidget(make_label(f'FWHM ({app.t_unit_txt()}):'), 1, 0)
        self.ed_fw = make_double_edit(0.15, minv=1e-6)
        irf_lay.addWidget(self.ed_fw, 1, 1)
        self.cb_fw_fix = QtWidgets.QCheckBox('Fixed')
        self.cb_fw_fix.setChecked(True)
        irf_lay.addWidget(self.cb_fw_fix, 1, 2)
        irf_lay.addWidget(make_label('Stretched-IRF mode:'), 2, 0)
        self.dd_irf = QtWidgets.QComboBox()
        self.dd_irf.addItem('Skip (mask 3σ around t₀)', 'skip')
        self.dd_irf.addItem('Numerical convolution', 'numerical')
        self.dd_irf.setCurrentIndex(0)
        irf_lay.addWidget(self.dd_irf, 2, 1, 1, 2)
        # Coherent artifact (G / G′ / G″ at t₀), fitted jointly; dialog-local
        irf_lay.addWidget(make_label('Artifact:'), 3, 0)
        self.dd_artifact = QtWidgets.QComboBox()
        for lbl, key in ARTIFACT_CHOICES:
            self.dd_artifact.addItem(lbl, key)
        self.dd_artifact.setCurrentIndex(self.dd_artifact.findData('off'))
        self.dd_artifact.setToolTip(
            ARTIFACT_TOOLTIP + u'\n\nWith stretched rows use '
            u'"Numerical convolution" as the Stretched-IRF mode.')
        irf_lay.addWidget(self.dd_artifact, 3, 1)
        self.btn_from_chirp = QtWidgets.QPushButton('From Chirp')
        self.btn_from_chirp.setToolTip(FROM_CHIRP_TOOLTIP)
        irf_lay.addWidget(self.btn_from_chirp, 3, 2)
        self.cb_art_sep = QtWidgets.QCheckBox('Separate artifact FWHM')
        self.cb_art_sep.setToolTip(ART_SEP_TOOLTIP)
        irf_lay.addWidget(self.cb_art_sep, 4, 0)
        self.ed_art_fw = make_double_edit(0.15, minv=1e-6)
        irf_lay.addWidget(self.ed_art_fw, 4, 1)
        self.cb_art_fw_fix = QtWidgets.QCheckBox('Fixed')
        self.cb_art_fw_fix.setChecked(True)
        irf_lay.addWidget(self.cb_art_fw_fix, 4, 2)
        self.lbl_art_warn = QtWidgets.QLabel('')
        self.lbl_art_warn.setWordWrap(True)
        self.lbl_art_warn.setStyleSheet('color: #b35a00;')
        self.lbl_art_warn.setVisible(False)
        irf_lay.addWidget(self.lbl_art_warn, 5, 0, 1, 3)
        lL.addWidget(irf_box)

        # Fit time-range
        tr_box = QtWidgets.QGroupBox('Fit window (delay)')
        tr_lay = QtWidgets.QGridLayout(tr_box)
        tr_lay.addWidget(make_label('From:'), 0, 0)
        self.ed_t_min = make_double_edit(float(app.delay[0]))
        tr_lay.addWidget(self.ed_t_min, 0, 1)
        tr_lay.addWidget(make_label('To:'), 0, 2)
        self.ed_t_max = make_double_edit(float(app.delay[-1]))
        tr_lay.addWidget(self.ed_t_max, 0, 3)
        self.btn_t_full = QtWidgets.QPushButton('Full')
        tr_lay.addWidget(self.btn_t_full, 0, 4)
        lL.addWidget(tr_box)

        # Run / reset / export
        ar = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton('Run Fit')
        style_button(self.btn_run, bg='#4fa35a', fg='white')
        self.btn_reset = QtWidgets.QPushButton('Reset')
        self.btn_save_resid = QtWidgets.QPushButton('Save Residual')
        self.btn_save_resid.setToolTip(
            'Write the residual (data − fit) as a two-column CSV — one '
            'file per trace for a global fit — so it can be re-loaded '
            'later from the Coherence dialog for LPSVD analysis.')
        self.btn_save_resid.setEnabled(False)
        ar.addWidget(self.btn_run); ar.addWidget(self.btn_reset)
        ar.addWidget(self.btn_save_resid)
        lL.addLayout(ar)
        self.btn_artsub_resid = QtWidgets.QPushButton(
            u'Art.-sub → Residual store')
        self.btn_artsub_resid.setToolTip(
            'Write data − artifact (fit window) to the Residual store — '
            'one file per trace for a global fit.  Needs a fit with the '
            'Artifact option on.')
        self.btn_artsub_resid.setEnabled(False)
        lL.addWidget(self.btn_artsub_resid)

        ex = QtWidgets.QHBoxLayout()
        ex.addWidget(make_label('Export:'))
        self.btn_exp_trace = QtWidgets.QPushButton('Trace+Fit (CSV)')
        self.btn_exp_param = QtWidgets.QPushButton('Params (CSV)')
        ex.addWidget(self.btn_exp_trace); ex.addWidget(self.btn_exp_param)
        lL.addLayout(ex)

        self.lbl_status = QtWidgets.QLabel('Ready.')
        lL.addWidget(self.lbl_status)
        self.txt_result = QtWidgets.QTextEdit()
        self.txt_result.setReadOnly(True)
        self.txt_result.setPlainText(
            'Set up the fit on the left, then press "Run Fit". '
            'Results will appear here.')
        lL.addWidget(self.txt_result, stretch=1)

        left.setFixedWidth(440)
        outer.addWidget(left)

        # ---------------- Right panel: plots ----------------
        right = QtWidgets.QGroupBox('Trace, fit, residual')
        rL = QtWidgets.QVBoxLayout(right)

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(make_label('Time scale:'))
        self.dd_scale = QtWidgets.QComboBox()
        self.dd_scale.addItems(['Log', 'Linear'])
        ctrl.addWidget(self.dd_scale)
        self.cb_artsub = QtWidgets.QCheckBox('Show artifact-subtracted')
        self.cb_artsub.setToolTip(
            'Draw data − artifact and fit − artifact (fit window).  '
            'Needs a fit with the Artifact option on; the main data is '
            'not changed.')
        self.cb_artsub.setEnabled(False)
        ctrl.addWidget(self.cb_artsub)
        ctrl.addStretch(1)
        rL.addLayout(ctrl)

        self.canvas = MplCanvas(self, nrows=2, ncols=1, figsize=(7, 6))
        self.ax_main, self.ax_res = self.canvas.axes_list
        rL.addWidget(self.canvas.with_toolbar(self), stretch=1)

        outer.addWidget(right, stretch=1)

        # State
        self._t = None         # delay axis of the imported traces
        self._traces = []      # [{'label', 'wl', 'y', 'n_avg'}, ...]
        self._last_fit = None  # normalised fit result, see do_run

        # Wiring
        self.dd_N.currentTextChanged.connect(self._on_n_change)
        self.cb_inf.toggled.connect(lambda _: self._refresh_plot())
        self.btn_use_main.clicked.connect(
            lambda: self.ed_wl.setValue(float(self.app.selWL)))
        self.btn_t_full.clicked.connect(self._set_t_full)
        self.btn_import.clicked.connect(self.do_import)
        self.btn_run.clicked.connect(self.do_run)
        self.btn_reset.clicked.connect(self.do_reset)
        self.btn_exp_trace.clicked.connect(self._export_trace)
        self.btn_exp_param.clicked.connect(self._export_params)
        self.btn_save_resid.clicked.connect(self.save_residual)
        self.dd_scale.currentTextChanged.connect(lambda _: self._refresh_plot())
        self.cb_artsub.toggled.connect(lambda _: self._refresh_plot())
        self.btn_artsub_resid.clicked.connect(self.save_artsub_residual)
        self.btn_from_chirp.clicked.connect(
            lambda: seed_artifact_from_chirp(self.app, self.dd_artifact,
                                             self.ed_fw, self))
        self.cb_art_sep.toggled.connect(self._update_art_controls)
        self.dd_artifact.currentIndexChanged.connect(self._update_art_warn)
        self.dd_irf.currentIndexChanged.connect(self._update_art_warn)
        self.tbl.itemChanged.connect(self._update_art_warn)

        self._populate_table(2)
        self._update_art_controls()
        self._refresh_plot()

    def closeEvent(self, ev):
        self.app.kfit_fig = None
        super().closeEvent(ev)

    # ----------------- Component table -----------------
    def _on_n_change(self, val):
        self._populate_table(int(val))

    def _populate_table(self, N: int):
        cur = self._read_table()
        self.tbl.setRowCount(0)
        defaults = [1.0, 10.0, 100.0, 1000.0, 10000.0]
        for i in range(N):
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
            tau0 = cur[i]['tau'] if i < len(cur) else defaults[i % 5]
            self.tbl.setItem(r, 0, QtWidgets.QTableWidgetItem(f'{tau0:.4g}'))
            chk1 = QtWidgets.QTableWidgetItem()
            chk1.setFlags((chk1.flags() | QtCore.Qt.ItemIsUserCheckable)
                          & ~QtCore.Qt.ItemIsEditable)
            chk1.setCheckState(
                QtCore.Qt.Checked if (i < len(cur) and cur[i]['tau_fixed'])
                else QtCore.Qt.Unchecked)
            self.tbl.setItem(r, 1, chk1)
            chk2 = QtWidgets.QTableWidgetItem()
            chk2.setFlags((chk2.flags() | QtCore.Qt.ItemIsUserCheckable)
                          & ~QtCore.Qt.ItemIsEditable)
            chk2.setCheckState(
                QtCore.Qt.Checked if (i < len(cur) and cur[i]['stretched'])
                else QtCore.Qt.Unchecked)
            self.tbl.setItem(r, 2, chk2)
            beta0 = cur[i]['beta'] if i < len(cur) else 1.0
            self.tbl.setItem(r, 3, QtWidgets.QTableWidgetItem(f'{beta0:.3g}'))
            chk3 = QtWidgets.QTableWidgetItem()
            chk3.setFlags((chk3.flags() | QtCore.Qt.ItemIsUserCheckable)
                          & ~QtCore.Qt.ItemIsEditable)
            chk3.setCheckState(
                QtCore.Qt.Checked if (i < len(cur) and cur[i]['beta_fixed'])
                else QtCore.Qt.Unchecked)
            self.tbl.setItem(r, 4, chk3)

    def _read_table(self):
        rows = []
        for r in range(self.tbl.rowCount()):
            try:
                tau = float(self.tbl.item(r, 0).text())
            except Exception:
                tau = 1.0
            try:
                beta = float(self.tbl.item(r, 3).text())
            except Exception:
                beta = 1.0
            rows.append({
                'tau': tau,
                'tau_fixed': self.tbl.item(r, 1).checkState() == QtCore.Qt.Checked,
                'stretched': self.tbl.item(r, 2).checkState() == QtCore.Qt.Checked,
                'beta': beta,
                'beta_fixed': self.tbl.item(r, 4).checkState() == QtCore.Qt.Checked,
            })
        return rows

    # ----------------- Artifact controls -----------------
    def _update_art_controls(self, *_):
        """Separate-FWHM edit / Fixed follow the checkbox."""
        sep = self.cb_art_sep.isChecked()
        self.ed_art_fw.setEnabled(sep)
        self.cb_art_fw_fix.setEnabled(sep)
        self._update_art_warn()

    def _art_skip_conflict(self):
        """Artifact On while a stretched row uses the 'skip' IRF mode."""
        stretched = False
        for r in range(self.tbl.rowCount()):
            it = self.tbl.item(r, 2)
            if it is not None and it.checkState() == QtCore.Qt.Checked:
                stretched = True
        return (self.dd_artifact.currentData() != 'off' and stretched
                and self.dd_irf.currentData() == 'skip')

    def _update_art_warn(self, *_):
        on = self._art_skip_conflict()
        self.lbl_art_warn.setText(ARTIFACT_SKIP_WARN if on else '')
        self.lbl_art_warn.setVisible(on)

    def _has_art_result(self):
        res = self._last_fit
        return res is not None and res.get('B') is not None

    def _update_art_buttons(self):
        on = self._has_art_result()
        self.cb_artsub.setEnabled(on)
        self.btn_artsub_resid.setEnabled(on)
        if not on and self.cb_artsub.isChecked():
            self.cb_artsub.blockSignals(True)
            self.cb_artsub.setChecked(False)
            self.cb_artsub.blockSignals(False)

    # ----------------- Helpers -----------------
    def _set_t_full(self):
        self.ed_t_min.setValue(float(self.app.delay[0]))
        self.ed_t_max.setValue(float(self.app.delay[-1]))

    def _get_trace(self):
        """Return (t, y, wl_actual, n_avg) for the λ ± half-width inputs."""
        app = self.app
        wl = float(self.ed_wl.value())
        hw = float(self.ed_wl_hw.value())
        if hw <= 0:
            iw = int(np.argmin(np.abs(app.wavelength - wl)))
            return app.delay.copy(), app.deltaA[iw, :].copy(), \
                   float(app.wavelength[iw]), 1
        m = (app.wavelength >= wl - hw) & (app.wavelength <= wl + hw)
        if not m.any():
            iw = int(np.argmin(np.abs(app.wavelength - wl)))
            return app.delay.copy(), app.deltaA[iw, :].copy(), \
                   float(app.wavelength[iw]), 1
        y_avg = np.nanmean(app.deltaA[m, :], axis=0)
        wl_centre = float(np.mean(app.wavelength[m]))
        return app.delay.copy(), y_avg, wl_centre, int(m.sum())

    def _pinned_traces(self):
        """Main-window pinned kinetics, each with its stored BW band."""
        app = self.app
        out = []
        for w in app.kinOverlays:
            i = int(np.argmin(np.abs(app.wavelength - w)))
            bw = app.kinOverlayBW.get(w, 0.0)
            y, label = app._kin_trace(i, bw)
            n = (int(ta_core.wavelength_gate(
                app.wavelength, app.wavelength[i], bw).sum()) if bw > 0
                 else 1)
            out.append({'label': label, 'wl': float(app.wavelength[i]),
                        'y': np.asarray(y, float).copy(), 'n_avg': n})
        return out

    # ----------------- Import -----------------
    def do_import(self):
        """Load the traces to fit (pinned kinetics or the λ window)."""
        app = self.app
        if app.deltaA is None:
            warn_box(self, 'No data', 'Load data in the main window first.')
            return
        note = ''
        if self.cb_pinned.isChecked() and app.kinOverlays:
            traces = self._pinned_traces()
            src = 'pinned kinetics'
        else:
            _, y, wl, n = self._get_trace()
            label = f'λ = {wl:.2f} nm' + (f' (avg {n} px)' if n > 1 else '')
            traces = [{'label': label, 'wl': wl, 'y': y, 'n_avg': n}]
            src = 'λ window'
            if self.cb_pinned.isChecked():
                note = '  (no pinned kinetics in the main window)'
        if getattr(app, 'art_main', None) is not None:
            note += '  (main data is artifact-subtracted)'
        self._t = app.delay.copy()
        self._traces = traces
        self._last_fit = None
        self.btn_save_resid.setEnabled(False)
        self._update_art_buttons()
        kind = 'global fit' if len(traces) > 1 else 'single-trace fit'
        self.lbl_import.setText(
            f'{len(traces)} trace(s) from {src} → {kind}{note}')
        self.lbl_status.setText('Data imported.')
        self._refresh_plot()

    # ----------------- Run / Reset -----------------
    def do_reset(self):
        self.dd_N.setCurrentText('2')
        self.cb_inf.setChecked(False)
        self.ed_t0.setValue(0.0); self.cb_t0_fix.setChecked(True)
        self.ed_fw.setValue(0.15); self.cb_fw_fix.setChecked(True)
        self.dd_irf.setCurrentIndex(0)
        self.dd_artifact.setCurrentIndex(self.dd_artifact.findData('off'))
        self.cb_art_sep.setChecked(False)
        self.ed_art_fw.setValue(0.15); self.cb_art_fw_fix.setChecked(True)
        self._set_t_full()
        self._populate_table(2)
        self._last_fit = None
        self.btn_save_resid.setEnabled(False)
        self._update_art_buttons()
        self._refresh_plot()
        self.lbl_status.setText('Reset to defaults.')
        self.txt_result.setPlainText('Results will appear here after fitting.')

    def do_run(self):
        if not self._traces:
            warn_box(self, 'No data', 'Press Import first.')
            return
        rows = self._read_table()
        if not rows:
            warn_box(self, 'No components', 'Add at least one component.')
            return
        tau_init = np.asarray([r['tau'] for r in rows], float)
        tau_fixed = np.asarray([r['tau_fixed'] for r in rows], bool)
        beta_init = np.asarray([r['beta'] for r in rows], float)
        beta_fixed = np.asarray([r['beta_fixed'] for r in rows], bool)
        stretch_on = np.asarray([r['stretched'] for r in rows], bool)
        if np.any(tau_init <= 0):
            warn_box(self, 'Invalid input', 'τ must be > 0.')
            return
        if np.any(stretch_on & (beta_init <= 0)):
            warn_box(self, 'Invalid input', 'β must be > 0.')
            return
        art_mode = ta_core._artifact_mode(self.dd_artifact.currentData())
        if art_mode != 'off' and self._art_skip_conflict():
            ans = QtWidgets.QMessageBox.question(
                self, 'Artifact with stretched rows',
                u'Stretched rows with the "skip" IRF mode mask the t₀ '
                u'region, so the artifact would be ignored.\n\n'
                u'Switch IRF mode to numerical?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Yes)
            if ans == QtWidgets.QMessageBox.Cancel:
                return
            if ans == QtWidgets.QMessageBox.Yes:
                self.dd_irf.setCurrentIndex(self.dd_irf.findData('numerical'))
        kw_art = {}
        if art_mode != 'off':
            sep = self.cb_art_sep.isChecked()
            kw_art = dict(
                artifact=art_mode,
                art_fwhm=float(self.ed_art_fw.value()) if sep else None,
                art_fwhm_fixed=self.cb_art_fw_fix.isChecked() if sep else True)

        t_full = self._t
        Y_full = np.vstack([tr['y'] for tr in self._traces])
        t_lo = float(self.ed_t_min.value())
        t_hi = float(self.ed_t_max.value())
        if t_lo > t_hi:
            t_lo, t_hi = t_hi, t_lo
        m = (t_full >= t_lo) & (t_full <= t_hi)
        n_in = int(m.sum())
        has_inf = self.cb_inf.isChecked()
        n_min = (int(tau_init.size) + (1 if has_inf else 0) + 1
                 + ta_core.artifact_min_extra(art_mode))
        if n_in < n_min:
            warn_box(self, 'Window too narrow',
                     f'Need ≥ {n_min} delay points; got {n_in}.')
            return
        t_fit = t_full[m]
        Y_fit = Y_full[:, m]

        kw = dict(beta_init=beta_init, beta_fixed=beta_fixed,
                  stretch_on=stretch_on, irf_mode=self.dd_irf.currentData())
        t0_fixed = self.cb_t0_fix.isChecked()
        fw_fixed = self.cb_fw_fix.isChecked()
        is_global = Y_fit.shape[0] > 1
        self.lbl_status.setText('Fitting…')
        QtWidgets.QApplication.processEvents()
        try:
            if is_global:
                # Shared τ / β / t₀ / FWHM, per-trace amplitudes
                r = ta_core.fit_global_analysis(
                    Y_fit, t_fit, tau_init,
                    float(self.ed_t0.value()), float(self.ed_fw.value()),
                    tau_fixed, t0_fixed, fw_fixed, has_inf,
                    method='trf', **kw, **kw_art)
                A, fit = r['A'], r['fit']
                B, art_fit = r.get('B'), r.get('artifact_fit')
                resid = Y_fit - fit
                if stretch_on.any() and kw['irf_mode'] == 'skip':
                    sig = r['fwhm'] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                    skip = t_fit <= r['t0'] + 3.0 * sig
                    if int((~skip).sum()) > A.shape[1]:
                        resid[:, skip] = np.nan
            else:
                r = ta_core.fit_single_trace(
                    t_fit, Y_fit[0], tau_init=tau_init, tau_fixed=tau_fixed,
                    t0_init=float(self.ed_t0.value()), t0_fixed=t0_fixed,
                    fwhm_init=float(self.ed_fw.value()), fwhm_fixed=fw_fixed,
                    has_inf=has_inf, **kw, **kw_art)
                A, fit = r['A'][None, :], r['fit'][None, :]
                resid = r['residual'][None, :]
                B, art_fit = r.get('B'), r.get('artifact_fit')
                if B is not None:
                    B, art_fit = B[None, :], art_fit[None, :]
        except Exception as e:
            warn_box(self, 'Fit error', f'Fit failed:\n{e}')
            self.lbl_status.setText('Fit failed.')
            return

        # Normalised result: A / fit / residual are always (n_traces, …)
        res = {
            'tau': r['tau'], 'beta': r['beta'], 't0': r['t0'],
            'fwhm': r['fwhm'], 'A': A, 'fit': fit, 'residual': resid,
            'ratio': ta_core.population_ratio(A),
            'info': r['info'], 'stats': r['info'].get('stats'),
            '_global': is_global,
            '_t_fit': t_fit, '_Y_fit': Y_fit,
            '_labels': [tr['label'] for tr in self._traces],
            '_wls': [tr['wl'] for tr in self._traces],
            '_n_avg': [tr['n_avg'] for tr in self._traces],
            '_wl': self._traces[0]['wl'],
            '_stretch_on': stretch_on, '_has_inf': has_inf,
            '_tau_fixed': tau_fixed, '_beta_fixed': beta_fixed,
            '_t0_fixed': t0_fixed, '_fw_fixed': fw_fixed,
            '_window': (t_lo, t_hi, n_in),
            # Coherent artifact: B (n_traces, n_terms) / artifact_fit
            # (n_traces, N) are None when Off; A excludes B.
            'B': B, 'artifact_fit': art_fit,
            'art_fwhm': r.get('art_fwhm'), 'artifact': art_mode,
            'art_ignored': r['info'].get('artifact_ignored'),
            '_art_sep': bool(kw_art) and kw_art['art_fwhm'] is not None,
            '_art_fw_fixed': kw_art.get('art_fwhm_fixed', True),
        }
        self._last_fit = res
        self.txt_result.setPlainText('\n'.join(self._report_lines(res)))
        done = (f'{"Global" if is_global else "Single-trace"} fit done — '
                f'RMS = {res["info"]["rms"]:.3g}')
        if res['art_ignored']:
            done += '\n' + self._art_ignored_msg(res['art_ignored'])
        self.lbl_status.setText(done)
        # A fresh fit is now available — enable the explicit save
        # button.  The user clicks it to write the residual file.
        self.btn_save_resid.setEnabled(True)
        self._update_art_buttons()
        self._refresh_plot()

    @staticmethod
    def _art_ignored_msg(why):
        if why == 'skip-mask':
            return (u'Artifact ignored (skip-mask): stretched rows with '
                    u'the "skip" IRF mode — use "Numerical convolution".')
        return (u'Artifact ignored (outside fit window): fewer than n+1 '
                u'delays within t₀_init ± 2σ_art — check the t₀ initial '
                u'value and the fit window.')

    def _report_lines(self, res):
        u = self.app.t_unit_txt()
        st = res['stats']

        def pm(val, sd):
            if sd is None or not np.isfinite(sd):
                return f'{val:.4g}'
            return f'{val:.4g} ± {sd:.2g}'

        def sd_of(key, i=None):
            if st is None:
                return None
            return st[key] if i is None else st[key][i]

        t_lo, t_hi, n_in = res['_window']
        n_tr = len(res['_labels'])
        lines = [(f'Global fit: {n_tr} traces, shared τ / β / t₀ / FWHM'
                  if res['_global'] else 'Single-trace fit'),
                 f'Window: [{t_lo:.4g}, {t_hi:.4g}] {u}, n = {n_in}',
                 f'IRF mode: {res["info"]["irf_mode"]}', '',
                 'Time constants (± 1σ):']
        for i in range(res['tau'].size):
            fx = ' (fixed)' if res['_tau_fixed'][i] else ''
            txt = (f'  τ{i + 1} = {pm(res["tau"][i], sd_of("tau_std", i))}'
                   f' {u}{fx}')
            if res['_stretch_on'][i]:
                bfx = ' (fixed)' if res['_beta_fixed'][i] else ''
                txt += (f'   β = {pm(res["beta"][i], sd_of("beta_std", i))}'
                        f'{bfx}  [stretched]')
            lines.append(txt)
        lines.append(f't₀ = {pm(res["t0"], sd_of("t0_std"))} {u}'
                     + ('  (fixed)' if res['_t0_fixed'] else ''))
        lines.append(f'FWHM = {pm(res["fwhm"], sd_of("fwhm_std"))} {u}'
                     + ('  (fixed)' if res['_fw_fixed'] else ''))
        lines.append('')
        lines.append('Amplitudes (± 1σ) and population ratio |Aᵢ| / Σ|Aⱼ|:')
        names = [f'A{i + 1}' for i in range(res['tau'].size)]
        if res['_has_inf']:
            names.append('A∞')
        for j, label in enumerate(res['_labels']):
            head = f'  [{label}]'
            if n_tr > 1 and st is not None:
                head += f'  R² = {st["r2_rows"][j]:.4f}'
            lines.append(head)
            for c, nm in enumerate(names):
                sd = None if st is None else st['A_std'][j, c]
                lines.append(f'    {nm} = {pm(res["A"][j, c], sd)}'
                             f'   ({res["ratio"][j, c]:.1f} %)')
            if res.get('B') is not None:
                b_std = None if st is None else st.get('B_std')
                for c, k in enumerate(ta_core.ARTIFACT_MODES[res['artifact']]):
                    sd = None if b_std is None else b_std[j, c]
                    lines.append(f'    B_{ARTIFACT_LABELS[k]} = '
                                 f'{pm(res["B"][j, c], sd)}')
        if res.get('B') is not None:
            lines.append(f'Artifact: {ARTIFACT_NAMES[res["artifact"]]}')
            fw_a = res['art_fwhm']
            if not res['_art_sep']:
                lines.append(f'FWHM_art = {fw_a:.4g} {u} (= IRF FWHM)')
            else:
                lines.append(
                    f'FWHM_art = {pm(fw_a, sd_of("art_fwhm_std"))} {u}'
                    + ('  (fixed)' if res['_art_fw_fixed'] else ''))
            if res['art_ignored']:
                lines.append(self._art_ignored_msg(res['art_ignored']))
        lines.append('')
        lines.extend(ta_core.format_fit_stats(st))
        return lines

    def save_residual(self):
        """Prompt for a save path and write the residual as two-column
        CSV(s) (delay, residual).

        Default filename: ``kfit_{wl}nm_residual.csv``.  A global fit
        writes one file per trace, ``kfit_{wl}nm_<chosen name>.csv``.  Loaded back by
        the Vibrational Coherence dialog via its "Browse..." picker.
        """
        res = self._last_fit
        if res is None:
            warn_box(self, 'No fit', 'Run the fit first.')
            return
        if res['_global']:
            default_name = 'kfit_global_residual.csv'
        else:
            default_name = f'kfit_{res["_wl"]:.1f}nm_residual.csv'
        path, _ = ask_save_path(
            self, 'Save Kfit residual',
            self.app.default_save_path(default_name),
            'CSV (*.csv);;All files (*)')
        if not path:
            return
        if not path.lower().endswith('.csv'):
            path += '.csv'
        if res['_global']:
            folder, stem = os.path.split(path[:-4])
            if stem.startswith('kfit_'):
                stem = stem[len('kfit_'):]
            paths = [os.path.join(folder, f'kfit_{wl:.1f}nm_{stem}.csv')
                     for wl in res['_wls']]
        else:
            paths = [path]
        try:
            t_col = np.asarray(res['_t_fit'], dtype=float).ravel()
            header = f'delay_{self.app.t_unit_hdr()},residual_dA'
            for p, r_row in zip(paths, res['residual']):
                mat = np.column_stack([t_col, np.asarray(r_row, float)])
                np.savetxt(p, mat, delimiter=',', header=header,
                           comments='', fmt='%.10g')
            self.lbl_status.setText(
                f'Residual saved: {os.path.basename(paths[0])}'
                + (f' (+{len(paths) - 1})' if len(paths) > 1 else ''))
            info_box(self, 'Residual saved',
                     'Wrote\n' + '\n'.join(paths) + '\n\n'
                     'In Vibrational Coherence, select "Use saved '
                     'residual file" and browse to this CSV.')
        except Exception as e:
            warn_box(self, 'Save residual', f'Save failed:\n{e}')

    def save_artsub_residual(self):
        """Write data − artifact (fit window) to the Residual store —
        one 1D file per trace."""
        res = self._last_fit
        if not self._has_art_result():
            warn_box(self, 'No artifact',
                     'Run a fit with the Artifact option on first.')
            return
        app = self.app
        t_fit = np.asarray(res['_t_fit'], dtype=float)
        try:
            paths = []
            for j, wl in enumerate(res['_wls']):
                y_sub = (np.asarray(res['_Y_fit'][j], float)
                         - np.asarray(res['artifact_fit'][j], float))
                paths.append(ta_residual_store.save_residual_1d(
                    'KFit_artsub',
                    getattr(app, 'data_source_desc', '') or 'data',
                    t_fit, y_sub, wavelength=float(wl),
                    time_unit=getattr(app, 'time_unit', ''),
                    extra={
                        'artifact_mode': res['artifact'],
                        'art_fwhm': f'{float(res["art_fwhm"]):.6g}',
                        't0': f'{float(res["t0"]):.6g}',
                        't_window': f'[{float(t_fit[0]):.6g},'
                                    f'{float(t_fit[-1]):.6g}]',
                    }))
            self.lbl_status.setText(
                f'Artifact-subtracted data saved: '
                f'{os.path.basename(paths[0])}'
                + (f' (+{len(paths) - 1})' if len(paths) > 1 else ''))
            info_box(self, 'Artifact-subtracted data saved',
                     'Wrote\n' + '\n'.join(paths) + '\n\n'
                     'Re-open the Coherence dialog and click "Load" to '
                     'make it appear in the file list.')
        except Exception as e:
            warn_box(self, 'Save artifact-subtracted', f'Save failed:\n{e}')

    # ----------------- Plotting -----------------
    def _refresh_plot(self):
        app = self.app
        ax = self.ax_main
        axR = self.ax_res
        ax.clear(); axR.clear()
        if not self._traces:
            ax.text(0.5, 0.5, 'Press Import to load data',
                    ha='center', va='center', transform=ax.transAxes,
                    color='0.4')
            self.canvas.draw_idle()
            return

        t = self._t
        single = len(self._traces) == 1
        cmap = ta_core._mpl_cm.get_cmap('tab10')
        fit = self._last_fit
        art = fit is not None and fit.get('B') is not None
        sub = art and self.cb_artsub.isChecked()
        for j, tr in enumerate(self._traces):
            col = 'k' if single else cmap(j % 10)
            if sub:
                # data − artifact / fit − artifact on the fit window
                ax.plot(fit['_t_fit'],
                        fit['_Y_fit'][j] - fit['artifact_fit'][j], '.',
                        color=col, markersize=4,
                        label=('data − artifact' if single
                               else tr['label']))
            else:
                ax.plot(t, tr['y'], '.', color=col, markersize=4,
                        label='data' if single else tr['label'])
            if fit is not None:
                y_fit = fit['fit'][j]
                if sub:
                    y_fit = y_fit - fit['artifact_fit'][j]
                ax.plot(fit['_t_fit'], y_fit, '-',
                        color='r' if single else col, linewidth=1.4,
                        label='fit' if single else None)
                if art:
                    ax.plot(fit['_t_fit'], fit['artifact_fit'][j], '--',
                            color='0.45' if single else col, linewidth=1.0,
                            label='artifact' if single else None)
                axR.plot(fit['_t_fit'], fit['residual'][j], '.',
                         color=(0.4, 0.4, 0.4) if single else col,
                         markersize=3)

        ax.axhline(0, color='k', linestyle=':', linewidth=0.5)
        axR.axhline(0, color='k', linestyle=':', linewidth=0.5)
        if self.dd_scale.currentText() == 'Log' and np.any(t > 0):
            ax.set_xscale('log'); axR.set_xscale('log')
            pos = t[t > 0]
            ax.set_xlim(pos.min(), t.max())
            axR.set_xlim(pos.min(), t.max())
        else:
            ax.set_xscale('linear'); axR.set_xscale('linear')
            ax.set_xlim(t.min(), t.max())
            axR.set_xlim(t.min(), t.max())
        ax.set_ylabel(r'$\Delta$A')
        axR.set_ylabel('residual')
        axR.set_xlabel(f'Delay ({app.t_unit_ax()})')
        if single:
            tr = self._traces[0]
            ax.set_title(rf'$\lambda$ = {tr["wl"]:.2f} nm   '
                         f'(avg {tr["n_avg"]} px)')
        else:
            ax.set_title(f'Global fit — {len(self._traces)} pinned traces')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True); axR.grid(True)
        self.canvas.draw_idle()

    # ----------------- Export -----------------
    def _export_trace(self):
        res = self._last_fit
        if res is None:
            warn_box(self, 'No fit', 'Run the fit first.')
            return
        name = ('kfit_global.csv' if res['_global']
                else f'kfit_{res["_wl"]:.1f}nm.csv')
        path, _ = ask_save_path(self, 'Export trace + fit',
                                self.app.default_save_path(name))
        if not path:
            return
        cols = [res['_t_fit']]
        hdr = [f'delay_{self.app.t_unit_hdr()}']
        for j, wl in enumerate(res['_wls']):
            sfx = f'_{wl:.1f}nm' if res['_global'] else ''
            cols += [res['_Y_fit'][j], res['fit'][j], res['residual'][j]]
            hdr += [f'data{sfx}', f'fit{sfx}', f'residual{sfx}']
            if res.get('B') is not None:
                a = res['artifact_fit'][j]
                cols += [a, res['_Y_fit'][j] - a]
                hdr += [f'artifact{sfx}', f'data_minus_artifact{sfx}']
        delim = ',' if path.lower().endswith('.csv') else '\t'
        try:
            np.savetxt(path, np.column_stack(cols), delimiter=delim,
                       fmt='%.10g', header=delim.join(hdr), comments='')
            info_box(self, 'Saved', f'Wrote {path}')
        except Exception as e:
            warn_box(self, 'Export error', str(e))

    def _export_params(self):
        r = self._last_fit
        if r is None:
            warn_box(self, 'No fit', 'Run the fit first.')
            return
        path, _ = ask_save_path(self, 'Export fit parameters',
                                self.app.default_save_path('kfit_params.csv'))
        if not path:
            return
        st = r['stats'] or {}
        n_tau, n_tr = len(r['tau']), len(r['_labels'])
        nan = float('nan')

        def g(v):
            return f'{v:.6g}'

        rows = [['fit_mode', 'global' if r['_global'] else 'single'],
                ['n_traces', str(n_tr)],
                ['rms', g(r['info']['rms'])]]
        for key in ('r2', 'adj_r2', 'chi2_red', 'rss', 'aic', 'bic',
                    'n_obs', 'n_par', 'dof'):
            rows.append([key, g(st.get(key, nan))])
        rows += [['t0', g(r['t0'])], ['t0_std', g(st.get('t0_std', nan))],
                 ['fwhm', g(r['fwhm'])],
                 ['fwhm_std', g(st.get('fwhm_std', nan))],
                 ['irf_mode', str(r['info']['irf_mode'])]]
        art_on = r.get('B') is not None
        if art_on:
            rows += [['artifact_mode', r['artifact']],
                     ['art_fwhm', g(r['art_fwhm'])],
                     ['art_fwhm_std', g(st.get('art_fwhm_std', nan))],
                     ['artifact_ignored', str(r['art_ignored'] or '')]]
            art_keys = [_ART_KEYS[k]
                        for k in ta_core.ARTIFACT_MODES[r['artifact']]]
            B_std = st.get('B_std', np.full(r['B'].shape, nan))
        tau_std = st.get('tau_std', np.full(n_tau, nan))
        beta_std = st.get('beta_std', np.full(n_tau, nan))
        for i in range(n_tau):
            rows += [[f'tau_{i+1}', g(r['tau'][i])],
                     [f'tau_{i+1}_std', g(tau_std[i])],
                     [f'beta_{i+1}', g(r['beta'][i])],
                     [f'beta_{i+1}_std', g(beta_std[i])],
                     [f'type_{i+1}',
                      'stretched' if r['_stretch_on'][i] else 'exp']]
        names = [str(i + 1) for i in range(n_tau)]
        if r['_has_inf']:
            names.append('inf')
        A_std = st.get('A_std', np.full(r['A'].shape, nan))
        r2_rows = st.get('r2_rows', np.full(n_tr, nan))
        for j, label in enumerate(r['_labels']):
            pre = f'T{j+1}_' if r['_global'] else ''
            rows += [[f'{pre}lambda_nm', f'{r["_wls"][j]:.4g}'],
                     [f'{pre}n_avg_px', str(r['_n_avg'][j])]]
            if r['_global']:
                rows += [[f'{pre}label', label.replace(',', ';')],
                         [f'{pre}r2', g(r2_rows[j])]]
            for c, nm in enumerate(names):
                rows += [[f'{pre}A_{nm}', g(r['A'][j, c])],
                         [f'{pre}A_{nm}_std', g(A_std[j, c])],
                         [f'{pre}ratio_{nm}_pct', g(r['ratio'][j, c])]]
            if art_on:
                for c, k in enumerate(art_keys):
                    rows += [[f'{pre}B_{k}', g(r['B'][j, c])],
                             [f'{pre}B_{k}_std', g(B_std[j, c])]]
        try:
            with open(path, 'w', encoding='utf-8') as f:
                for k, v in rows:
                    f.write(f'{k},{v}\n')
            info_box(self, 'Saved', f'Wrote {path}')
        except Exception as e:
            warn_box(self, 'Export error', str(e))
