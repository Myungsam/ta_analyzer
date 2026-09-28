"""
TA Analyzer GUI - Single-Trace Kinetic Fit dialog.

Fits a single λ-cut of the 2D map (or an averaged window of cuts) to a
sum of (possibly stretched) exponentials convolved with a Gaussian IRF.
Uses ta_core.fit_single_trace.
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


class KineticFitDialog(QtWidgets.QDialog):
    """Fit a single kinetic trace y(t) at a chosen wavelength.

    Supports an arbitrary mix of standard- and stretched-exponential
    components plus an optional offset (τ = ∞).
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.setWindowTitle('Kinetic Fit (single trace)')
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
            'Write the current single-trace residual (data − fit) to '
            'the residuals folder so it can be re-loaded later from '
            'the Coherence dialog for LPSVD analysis.')
        self.btn_save_resid.setEnabled(False)
        ar.addWidget(self.btn_run); ar.addWidget(self.btn_reset)
        ar.addWidget(self.btn_save_resid)
        lL.addLayout(ar)

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
        ctrl.addStretch(1)
        rL.addLayout(ctrl)

        self.canvas = MplCanvas(self, nrows=2, ncols=1, figsize=(7, 6))
        self.ax_main, self.ax_res = self.canvas.axes_list
        rL.addWidget(self.canvas.with_toolbar(self), stretch=1)

        outer.addWidget(right, stretch=1)

        # State
        self._last_fit = None  # dict from ta_core.fit_single_trace

        # Wiring
        self.dd_N.currentTextChanged.connect(self._on_n_change)
        self.cb_inf.toggled.connect(lambda _: self._refresh_plot())
        self.btn_use_main.clicked.connect(
            lambda: self.ed_wl.setValue(float(self.app.selWL)))
        self.btn_t_full.clicked.connect(self._set_t_full)
        self.btn_run.clicked.connect(self.do_run)
        self.btn_reset.clicked.connect(self.do_reset)
        self.btn_exp_trace.clicked.connect(self._export_trace)
        self.btn_exp_param.clicked.connect(self._export_params)
        self.btn_save_resid.clicked.connect(self.save_residual)
        self.dd_scale.currentTextChanged.connect(lambda _: self._refresh_plot())
        for ed in (self.ed_wl, self.ed_wl_hw):
            ed.valueChanged.connect(lambda _: self._refresh_plot(replot_data=True))

        self._populate_table(2)
        self._refresh_plot(replot_data=True)

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

    # ----------------- Helpers -----------------
    def _set_t_full(self):
        self.ed_t_min.setValue(float(self.app.delay[0]))
        self.ed_t_max.setValue(float(self.app.delay[-1]))

    def _get_trace(self):
        """Return (t, y, wl_actual, n_avg) — the trace to fit/plot."""
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

    # ----------------- Run / Reset -----------------
    def do_reset(self):
        self.dd_N.setCurrentText('2')
        self.cb_inf.setChecked(False)
        self.ed_t0.setValue(0.0); self.cb_t0_fix.setChecked(True)
        self.ed_fw.setValue(0.15); self.cb_fw_fix.setChecked(True)
        self.dd_irf.setCurrentIndex(0)
        self._set_t_full()
        self._populate_table(2)
        self._last_fit = None
        self.btn_save_resid.setEnabled(False)
        self._refresh_plot(replot_data=True)
        self.lbl_status.setText('Reset to defaults.')
        self.txt_result.setPlainText('Results will appear here after fitting.')

    def do_run(self):
        app = self.app
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

        t_full, y_full, wl_actual, n_avg = self._get_trace()
        t_lo = float(self.ed_t_min.value())
        t_hi = float(self.ed_t_max.value())
        if t_lo > t_hi:
            t_lo, t_hi = t_hi, t_lo
        m = (t_full >= t_lo) & (t_full <= t_hi)
        n_in = int(m.sum())
        n_min = int(tau_init.size) + (1 if self.cb_inf.isChecked() else 0) + 1
        if n_in < n_min:
            warn_box(self, 'Window too narrow',
                     f'Need ≥ {n_min} delay points; got {n_in}.')
            return
        t_fit = t_full[m]
        y_fit = y_full[m]

        irf_mode = self.dd_irf.currentData()
        try:
            res = ta_core.fit_single_trace(
                t_fit, y_fit,
                tau_init=tau_init, tau_fixed=tau_fixed,
                beta_init=beta_init, beta_fixed=beta_fixed,
                stretch_on=stretch_on,
                t0_init=float(self.ed_t0.value()),
                t0_fixed=self.cb_t0_fix.isChecked(),
                fwhm_init=float(self.ed_fw.value()),
                fwhm_fixed=self.cb_fw_fix.isChecked(),
                has_inf=self.cb_inf.isChecked(),
                irf_mode=irf_mode)
        except Exception as e:
            warn_box(self, 'Fit error', f'Fit failed:\n{e}')
            self.lbl_status.setText('Fit failed.')
            return

        # Stash for plotting / export
        res['_t_fit'] = t_fit
        res['_y_fit'] = y_fit
        res['_t_full'] = t_full
        res['_y_full'] = y_full
        res['_wl'] = wl_actual
        res['_n_avg'] = n_avg
        res['_stretch_on'] = stretch_on
        res['_has_inf'] = self.cb_inf.isChecked()
        self._last_fit = res

        # Build text report
        lines = [f'Fit converged ({res["info"]["iters"]} iters), '
                 f'RMS = {res["info"]["rms"]:.4g}']
        lines.append(f'λ centre = {wl_actual:.2f} nm  '
                     f'(avg of {n_avg} pixels)')
        lines.append(f'Window: [{t_lo:.4g}, {t_hi:.4g}] '
                     f'{app.t_unit_txt()}, n = {n_in}')
        lines.append(f'IRF mode: {res["info"]["irf_mode"]}')
        lines.append('')
        lines.append('  i   τ              β       A          (type)')
        for i in range(tau_init.size):
            tag = 'stretched' if stretch_on[i] else 'exp'
            lines.append(f'  {i+1}   {res["tau"][i]:10.4g} '
                         f' {res["beta"][i]:6.3g}  '
                         f'{res["A"][i]:10.4g}  ({tag})')
        if self.cb_inf.isChecked():
            lines.append(f'  ∞   {"":10}  {"":6}  {res["A"][-1]:10.4g}  (offset)')
        lines.append('')
        lines.append(f't₀ = {res["t0"]:.4g} {app.t_unit_txt()}'
                     + ('  (fixed)' if self.cb_t0_fix.isChecked() else ''))
        lines.append(f'FWHM = {res["fwhm"]:.4g} {app.t_unit_txt()}'
                     + ('  (fixed)' if self.cb_fw_fix.isChecked() else ''))
        self.txt_result.setPlainText('\n'.join(lines))
        self.lbl_status.setText(
            f'Fit done — RMS = {res["info"]["rms"]:.3g}')
        # A fresh fit is now available — enable the explicit save
        # button.  The user clicks it to write the residual file.
        self.btn_save_resid.setEnabled(True)
        self._refresh_plot(replot_data=False)

    def save_residual(self):
        """Prompt for a save path and write the single-trace residual
        as a two-column CSV (delay, residual).

        Default filename: ``kfit_{wl}nm_residual.csv``.  Loaded back by
        the Vibrational Coherence dialog via its "Browse..." picker.
        """
        res = self._last_fit
        if res is None:
            warn_box(self, 'No fit', 'Run the fit first.')
            return
        wl = float(res.get('_wl', float('nan')))
        if np.isfinite(wl):
            default_name = f'kfit_{wl:.1f}nm_residual.csv'
        else:
            default_name = 'kfit_residual.csv'
        path, _ = ask_save_path(
            self, 'Save Kfit residual',
            self.app.default_save_path(default_name),
            'CSV (*.csv);;All files (*)')
        if not path:
            return
        if not path.lower().endswith('.csv'):
            path += '.csv'
        try:
            t_col = np.asarray(res['_t_fit'], dtype=float).ravel()
            r_col = np.asarray(res['residual'], dtype=float).ravel()
            mat = np.column_stack([t_col, r_col])
            header = f'delay_{self.app.t_unit_hdr()},residual_dA'
            np.savetxt(path, mat, delimiter=',', header=header,
                       comments='', fmt='%.10g')
            self.lbl_status.setText(
                f'Residual saved: {os.path.basename(path)}')
            info_box(self, 'Residual saved',
                     f'Wrote\n{path}\n\n'
                     f'In Vibrational Coherence, select "Use saved '
                     f'residual file" and browse to this CSV.')
        except Exception as e:
            warn_box(self, 'Save residual', f'Save failed:\n{e}')

    # ----------------- Plotting -----------------
    def _refresh_plot(self, replot_data: bool = False):
        app = self.app
        ax = self.ax_main
        axR = self.ax_res
        ax.clear(); axR.clear()

        if replot_data or self._last_fit is None:
            t, y, wl_actual, n_avg = self._get_trace()
        else:
            t = self._last_fit['_t_full']
            y = self._last_fit['_y_full']
            wl_actual = self._last_fit['_wl']
            n_avg = self._last_fit['_n_avg']

        ax.plot(t, y, '.', color='k', markersize=4, label='data')
        if self._last_fit is not None:
            t_fit = self._last_fit['_t_fit']
            fit_v = self._last_fit['fit']
            res_v = self._last_fit['residual']
            ax.plot(t_fit, fit_v, 'r-', linewidth=1.4, label='fit')
            axR.plot(t_fit, res_v, '.', color=(0.4, 0.4, 0.4), markersize=3)

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
        ax.set_title(rf'$\lambda$ = {wl_actual:.2f} nm   (avg {n_avg} px)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True); axR.grid(True)
        self.canvas.draw_idle()

    # ----------------- Export -----------------
    def _export_trace(self):
        if self._last_fit is None:
            warn_box(self, 'No fit', 'Run the fit first.')
            return
        path, _ = ask_save_path(
            self, 'Export trace + fit',
            self.app.default_save_path(
                f'kfit_{self._last_fit["_wl"]:.1f}nm.csv'))
        if not path:
            return
        t = self._last_fit['_t_fit']
        y = self._last_fit['_y_fit']
        fit_v = self._last_fit['fit']
        res_v = self._last_fit['residual']
        mat = np.column_stack([t, y, fit_v, res_v])
        delim = ',' if path.lower().endswith('.csv') else '\t'
        try:
            np.savetxt(path, mat, delimiter=delim, fmt='%.10g',
                       header=delim.join([
                           f'delay_{self.app.t_unit_hdr()}',
                           'data', 'fit', 'residual']),
                       comments='')
            info_box(self, 'Saved', f'Wrote {path}')
        except Exception as e:
            warn_box(self, 'Export error', str(e))

    def _export_params(self):
        if self._last_fit is None:
            warn_box(self, 'No fit', 'Run the fit first.')
            return
        path, _ = ask_save_path(self, 'Export fit parameters',
                                self.app.default_save_path('kfit_params.csv'))
        if not path:
            return
        r = self._last_fit
        rows = []
        rows.append(['lambda_nm', f'{r["_wl"]:.4g}'])
        rows.append(['n_avg_px', str(r['_n_avg'])])
        rows.append(['rms', f'{r["info"]["rms"]:.6g}'])
        rows.append(['t0', f'{r["t0"]:.6g}'])
        rows.append(['fwhm', f'{r["fwhm"]:.6g}'])
        rows.append(['irf_mode', str(r['info']['irf_mode'])])
        for i in range(len(r['tau'])):
            tag = 'stretched' if r['_stretch_on'][i] else 'exp'
            rows.append([f'tau_{i+1}', f'{r["tau"][i]:.6g}'])
            rows.append([f'beta_{i+1}', f'{r["beta"][i]:.6g}'])
            rows.append([f'A_{i+1}', f'{r["A"][i]:.6g}'])
            rows.append([f'type_{i+1}', tag])
        if r['_has_inf']:
            rows.append(['A_inf', f'{r["A"][-1]:.6g}'])
        try:
            with open(path, 'w', encoding='utf-8') as f:
                for k, v in rows:
                    f.write(f'{k},{v}\n')
            info_box(self, 'Saved', f'Wrote {path}')
        except Exception as e:
            warn_box(self, 'Export error', str(e))
