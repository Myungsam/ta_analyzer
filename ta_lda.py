"""
TA Analyzer GUI - Lifetime Distribution / Density Analysis (LDA).

Decomposes the 2D ΔA(λ, t) data onto a fixed log-spaced lifetime grid
with Tikhonov regularization.  Backend: ta_core.compute_lda /
ta_core.compute_lcurve.
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
import ta_residual_store


class LDADialog(QtWidgets.QDialog):
    """Lifetime Density Analysis with an L-curve helper."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.setWindowTitle('Lifetime Density Analysis (LDA / LDS)')
        self.resize(1360, 840)

        outer = QtWidgets.QHBoxLayout(self)

        # ---------------- Left panel ----------------
        left = QtWidgets.QGroupBox('Setup')
        lL = QtWidgets.QVBoxLayout(left)

        # Tau grid
        gb_grid = QtWidgets.QGroupBox(f'Lifetime grid ({app.t_unit_txt()})')
        g = QtWidgets.QGridLayout(gb_grid)
        g.addWidget(make_label('τ_min:'), 0, 0)
        d = app.delay
        dt_min = float(np.diff(np.unique(d)).min()) if len(d) > 1 else 0.1
        t_span = float(d.max() - d.min())
        self.ed_tau_min = make_double_edit(max(dt_min * 0.5, 1e-3),
                                           decimals=4, minv=1e-9)
        g.addWidget(self.ed_tau_min, 0, 1)
        g.addWidget(make_label('τ_max:'), 0, 2)
        self.ed_tau_max = make_double_edit(t_span * 5.0, decimals=2,
                                           minv=1e-3)
        g.addWidget(self.ed_tau_max, 0, 3)
        g.addWidget(make_label('# τ points:'), 1, 0)
        self.ed_K = make_int_edit(50, minv=5, maxv=500)
        g.addWidget(self.ed_K, 1, 1)
        g.addWidget(make_label('Spacing:'), 1, 2)
        self.dd_space = QtWidgets.QComboBox()
        self.dd_space.addItem('Log', 'log')
        self.dd_space.addItem('Linear', 'linear')
        g.addWidget(self.dd_space, 1, 3)
        lL.addWidget(gb_grid)

        # IRF
        gb_irf = QtWidgets.QGroupBox('IRF')
        i = QtWidgets.QGridLayout(gb_irf)
        i.addWidget(make_label(f't₀ ({app.t_unit_txt()}):'), 0, 0)
        self.ed_t0 = make_double_edit(0.0)
        i.addWidget(self.ed_t0, 0, 1)
        i.addWidget(make_label(f'FWHM ({app.t_unit_txt()}):'), 1, 0)
        self.ed_fw = make_double_edit(0.15, minv=1e-6)
        i.addWidget(self.ed_fw, 1, 1)
        lL.addWidget(gb_irf)

        # Regularization
        gb_reg = QtWidgets.QGroupBox('Regularization')
        r = QtWidgets.QGridLayout(gb_reg)
        r.addWidget(make_label('Type:'), 0, 0)
        self.dd_reg = QtWidgets.QComboBox()
        self.dd_reg.addItem('L₂ derivative (smooth)', 'l2deriv')
        self.dd_reg.addItem('L₂ amplitude', 'l2')
        r.addWidget(self.dd_reg, 0, 1, 1, 3)
        r.addWidget(make_label('α:'), 1, 0)
        self.ed_alpha = make_double_edit(0.01, decimals=6, minv=0.0)
        r.addWidget(self.ed_alpha, 1, 1)
        self.btn_lcurve = QtWidgets.QPushButton('L-curve...')
        self.btn_lcurve.setToolTip(
            'Sweep α over a log range and pick the corner of the '
            'residual-vs-solution-norm curve.')
        r.addWidget(self.btn_lcurve, 1, 2)
        lL.addWidget(gb_reg)

        # Time window (use only a slice of t)
        gb_tr = QtWidgets.QGroupBox('Fit window (delay)')
        tr = QtWidgets.QGridLayout(gb_tr)
        tr.addWidget(make_label('From:'), 0, 0)
        self.ed_t_min = make_double_edit(float(d[0]))
        tr.addWidget(self.ed_t_min, 0, 1)
        tr.addWidget(make_label('To:'), 0, 2)
        self.ed_t_max = make_double_edit(float(d[-1]))
        tr.addWidget(self.ed_t_max, 0, 3)
        self.btn_t_full = QtWidgets.QPushButton('Full')
        tr.addWidget(self.btn_t_full, 0, 4)
        lL.addWidget(gb_tr)

        ar = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton('Run LDA')
        style_button(self.btn_run, bg='#4fa35a', fg='white')
        self.btn_reset = QtWidgets.QPushButton('Reset')
        self.btn_save_resid = QtWidgets.QPushButton('Save Residual')
        self.btn_save_resid.setToolTip(
            'Write the current LDA residual (D − Drec) to the '
            'residuals folder so it can be re-loaded later from the '
            'Coherence dialog for FFT / LPSVD analysis.')
        self.btn_save_resid.setEnabled(False)
        ar.addWidget(self.btn_run); ar.addWidget(self.btn_reset)
        ar.addWidget(self.btn_save_resid)
        lL.addLayout(ar)

        ex = QtWidgets.QHBoxLayout()
        ex.addWidget(make_label('Export:'))
        self.btn_exp_amap = QtWidgets.QPushButton('A(λ,τ) (CSV)')
        self.btn_exp_rec = QtWidgets.QPushButton('Drec (CSV)')
        ex.addWidget(self.btn_exp_amap); ex.addWidget(self.btn_exp_rec)
        lL.addLayout(ex)

        self.lbl_status = QtWidgets.QLabel('Ready.')
        lL.addWidget(self.lbl_status)
        self.txt_info = QtWidgets.QTextEdit()
        self.txt_info.setReadOnly(True)
        self.txt_info.setMinimumHeight(140)
        lL.addWidget(self.txt_info)

        left.setFixedWidth(400)
        outer.addWidget(left)

        # ---------------- Right panel ----------------
        right = QtWidgets.QGroupBox('Results')
        rL = QtWidgets.QVBoxLayout(right)
        self.canvas = MplCanvas(self, nrows=2, ncols=2, figsize=(10, 8))
        # axes_list returns flat row-major:  [Amap, |A|.sum,  Drec, kinetic]
        self.ax_amap, self.ax_amp, self.ax_rec, self.ax_kin = \
            self.canvas.axes_list
        rL.addWidget(self.canvas.with_toolbar(self), stretch=1)
        outer.addWidget(right, stretch=1)

        # State
        self._last = None
        self._cur_kin_wl = float(app.selWL) if app.selWL else \
            float(app.wavelength[len(app.wavelength) // 2])

        # Wiring
        self.btn_run.clicked.connect(self.do_run)
        self.btn_reset.clicked.connect(self.do_reset)
        self.btn_t_full.clicked.connect(self._t_full)
        self.btn_lcurve.clicked.connect(self._show_lcurve)
        self.btn_exp_amap.clicked.connect(self._export_amap)
        self.btn_exp_rec.clicked.connect(self._export_rec)
        self.btn_save_resid.clicked.connect(self.save_residual)
        self.canvas.mpl_connect('button_press_event', self._on_click)

    def closeEvent(self, ev):
        self.app.lda_fig = None
        super().closeEvent(ev)

    def _t_full(self):
        self.ed_t_min.setValue(float(self.app.delay[0]))
        self.ed_t_max.setValue(float(self.app.delay[-1]))

    def _make_tau_grid(self) -> np.ndarray:
        K = int(self.ed_K.value())
        a = float(self.ed_tau_min.value())
        b = float(self.ed_tau_max.value())
        if a >= b or a <= 0:
            return np.logspace(-1, 3, K)
        if self.dd_space.currentData() == 'log':
            return np.logspace(np.log10(a), np.log10(b), K)
        return np.linspace(a, b, K)

    def do_reset(self):
        d = self.app.delay
        dt_min = float(np.diff(np.unique(d)).min()) if len(d) > 1 else 0.1
        self.ed_tau_min.setValue(max(dt_min * 0.5, 1e-3))
        self.ed_tau_max.setValue(float(d.max() - d.min()) * 5.0)
        self.ed_K.setValue(50)
        self.dd_space.setCurrentIndex(0)
        self.ed_t0.setValue(0.0)
        self.ed_fw.setValue(0.15)
        self.dd_reg.setCurrentIndex(0)
        self.ed_alpha.setValue(0.01)
        self._t_full()
        self._last = None
        self.btn_save_resid.setEnabled(False)
        for a in (self.ax_amap, self.ax_amp, self.ax_rec, self.ax_kin):
            a.clear()
        self.canvas.draw_idle()
        self.txt_info.setPlainText('')
        self.lbl_status.setText('Reset to defaults.')

    def do_run(self):
        app = self.app
        tau_grid = self._make_tau_grid()
        if tau_grid.size < 5:
            warn_box(self, 'Bad grid', 'Need at least 5 τ points.')
            return
        t_lo = float(self.ed_t_min.value())
        t_hi = float(self.ed_t_max.value())
        if t_lo > t_hi:
            t_lo, t_hi = t_hi, t_lo
        m = (app.delay >= t_lo) & (app.delay <= t_hi)
        if int(m.sum()) < tau_grid.size + 2:
            warn_box(self, 'Window too narrow',
                     f'Need ≥ {tau_grid.size + 2} delay points; '
                     f'got {int(m.sum())}.')
            return
        t_win = app.delay[m]
        D_win = app.deltaA[:, m]

        self.lbl_status.setText('Running LDA…')
        QtWidgets.QApplication.processEvents()
        try:
            res = ta_core.compute_lda(
                D_win, t_win, tau_grid,
                t0=float(self.ed_t0.value()),
                fwhm=float(self.ed_fw.value()),
                alpha=float(self.ed_alpha.value()),
                reg_type=self.dd_reg.currentData())
        except Exception as e:
            warn_box(self, 'LDA error', str(e))
            self.lbl_status.setText('LDA failed.')
            return

        self._last = {'res': res, 't_win': t_win, 'D_win': D_win,
                      'tau_grid': tau_grid}
        # Publish onto app so the Coherence dialog's "Use last LDA
        # residual" mode can pull it out without re-running.
        app._last_lda = self._last
        i = res['info']
        lines = [f"K = {i['K']} τ points, M = {i['M']}, N = {i['N']}",
                 f"α = {i['alpha']:.4g}  ({i['reg_type']})",
                 f"RMS = {i['rms']:.4g}",
                 f"Rel. residual = {i['ratio']:.4g}",
                 '']
        # Identify dominant lifetimes
        amp = np.nansum(np.abs(res['A_map']), axis=0)
        order = np.argsort(amp)[::-1][:5]
        lines.append('Top 5 τ peaks (by integrated |A|):')
        for k in order:
            lines.append(f'  τ = {tau_grid[k]:10.4g} {app.t_unit_txt()}  '
                         f'|A|_sum = {amp[k]:.3g}')
        self.txt_info.setPlainText('\n'.join(lines))
        self.lbl_status.setText(f'LDA done — RMS={i["rms"]:.3g}')
        # A fresh result is now available — enable the explicit save
        # button.  The user clicks it to write the residual file.
        self.btn_save_resid.setEnabled(True)
        self._plot_results()

    def save_residual(self):
        """Write the current LDA residual (D − Drec) to the residuals folder.

        Called from the Save Residual button.  The button is only
        enabled when self._last holds a successful run.
        """
        if self._last is None:
            warn_box(self, 'No fit', 'Run an LDA fit first.')
            return
        try:
            t_win = self._last['t_win']
            D_win = self._last['D_win']
            Drec = self._last['res']['Drec']
            R = np.asarray(D_win, dtype=float) - np.asarray(Drec, dtype=float)
            path = ta_residual_store.save_residual_2d(
                analysis='LDA',
                dataset=getattr(self.app, 'data_source_desc', '') or 'data',
                wl=self.app.wavelength, t=t_win, R=R,
                time_unit=getattr(self.app, 'time_unit', ''),
                extra={
                    'alpha': f"{float(self.ed_alpha.value()):.6g}",
                    't_window':
                        f"[{float(t_win[0]):.6g},{float(t_win[-1]):.6g}]",
                })
            self.lbl_status.setText(
                f'Residual saved: {os.path.basename(path)}')
            info_box(self, 'Residual saved',
                     f'Wrote\n{path}\n\n'
                     f'Re-open the Coherence dialog and click '
                     f'"Load" to make it appear in the file list.')
        except Exception as e:
            warn_box(self, 'Save residual', f'Save failed:\n{e}')

    def _plot_results(self):
        if self._last is None:
            return
        app = self.app
        res = self._last['res']
        tau = self._last['tau_grid']
        t = self._last['t_win']
        D = self._last['D_win']
        A_map = res['A_map']
        Drec = res['Drec']

        # 1) A(λ, τ) heatmap
        ax = self.ax_amap; ax.clear()
        cl = float(np.nanmax(np.abs(A_map)))
        if not np.isfinite(cl) or cl == 0:
            cl = 1.0
        cmap = ListedColormap(app.get_colormap_array())
        im = ax.pcolormesh(tau, app.wavelength, A_map,
                           cmap=cmap, vmin=-cl, vmax=cl, shading='nearest')
        ax.set_xscale('log')
        ax.set_xlabel(f'τ ({app.t_unit_ax()})')
        ax.set_ylabel('Wavelength (nm)')
        ax.set_title('LDA amplitude  A(λ, τ)')
        try:
            self.canvas.fig.colorbar(im, ax=ax).set_label(r'$\Delta$A')
        except Exception:
            pass

        # 2) Integrated |A|
        ax = self.ax_amp; ax.clear()
        amp = np.nansum(np.abs(A_map), axis=0)
        ax.plot(tau, amp, '-', color='#2c7fb8', linewidth=1.4)
        ax.set_xscale('log')
        ax.set_xlabel(f'τ ({app.t_unit_ax()})')
        ax.set_ylabel(r'$\Sigma_\lambda |A(\lambda,\tau)|$')
        ax.set_title('Lifetime spectrum')
        ax.grid(True, which='both', alpha=0.3)

        # 3) Reconstruction
        ax = self.ax_rec; ax.clear()
        clr = float(np.nanmax(np.abs(Drec)))
        if not np.isfinite(clr) or clr == 0:
            clr = 1.0
        ax.pcolormesh(app.wavelength, t, Drec.T, cmap=cmap,
                      vmin=-clr, vmax=clr, shading='nearest')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel(f'Delay ({app.t_unit_ax()})')
        ax.set_title('Reconstruction  D̃(λ, t)')
        # Crosshair for kinetics panel
        ax.axvline(self._cur_kin_wl, color='w', linestyle='--', linewidth=1.0)

        # 4) Kinetics at selected λ
        self._draw_kin_panel()
        self.canvas.draw_idle()

    def _draw_kin_panel(self):
        if self._last is None:
            return
        ax = self.ax_kin; ax.clear()
        t = self._last['t_win']
        D = self._last['D_win']
        Drec = self._last['res']['Drec']
        iw = int(np.argmin(np.abs(self.app.wavelength - self._cur_kin_wl)))
        ax.plot(t, D[iw, :], 'k.', markersize=4, label='data')
        ax.plot(t, Drec[iw, :], 'r-', linewidth=1.4, label='LDA fit')
        ax.set_xlabel(f'Delay ({self.app.t_unit_ax()})')
        ax.set_ylabel(r'$\Delta$A')
        if np.any(t > 0):
            ax.set_xscale('log'); ax.set_xlim(t[t > 0].min(), t.max())
        ax.set_title(f'λ = {self.app.wavelength[iw]:.1f} nm')
        ax.grid(True); ax.legend(loc='best', fontsize=8)
        ax.axhline(0, color='k', linestyle=':', linewidth=0.5)

    def _on_click(self, event):
        if self._last is None or event.inaxes not in (self.ax_rec,):
            return
        if event.xdata is None:
            return
        self._cur_kin_wl = float(event.xdata)
        # Update vline
        for line in self.ax_rec.lines:
            try: line.remove()
            except Exception: pass
        self.ax_rec.axvline(self._cur_kin_wl, color='w',
                            linestyle='--', linewidth=1.0)
        self._draw_kin_panel()
        self.canvas.draw_idle()

    # ---- L-curve ----
    def _show_lcurve(self):
        app = self.app
        tau_grid = self._make_tau_grid()
        t_lo = float(self.ed_t_min.value()); t_hi = float(self.ed_t_max.value())
        m = (app.delay >= t_lo) & (app.delay <= t_hi)
        if int(m.sum()) < tau_grid.size + 2:
            warn_box(self, 'Window too narrow', 'Widen the fit window.')
            return
        self.lbl_status.setText('Computing L-curve…')
        QtWidgets.QApplication.processEvents()
        try:
            alpha_list = np.logspace(-6, 2, 18)
            a_list, res_n, sol_n, a_corner = ta_core.compute_lcurve(
                app.deltaA[:, m], app.delay[m], tau_grid,
                t0=float(self.ed_t0.value()),
                fwhm=float(self.ed_fw.value()),
                reg_type=self.dd_reg.currentData(),
                alpha_list=alpha_list)
        except Exception as e:
            warn_box(self, 'L-curve error', str(e))
            self.lbl_status.setText('L-curve failed.')
            return

        # Inline pop-up plot
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('L-curve')
        dlg.resize(640, 520)
        dl = QtWidgets.QVBoxLayout(dlg)
        cv = MplCanvas(dlg)
        ax = cv.ax
        ax.loglog(res_n, sol_n, 'o-', color='#e34a33')
        ic = int(np.argmin(np.abs(a_list - a_corner)))
        ax.plot(res_n[ic], sol_n[ic], 'o', color='#2c7fb8',
                markersize=12, fillstyle='none',
                label=f'corner α = {a_corner:.3g}')
        for x, y, a in zip(res_n[::3], sol_n[::3], a_list[::3]):
            ax.annotate(f'{a:.1e}', (x, y), fontsize=7,
                        textcoords='offset points', xytext=(5, 5))
        ax.set_xlabel('||D − D̃||')
        ax.set_ylabel('||A||')
        ax.set_title('L-curve  (corner = recommended α)')
        ax.legend(loc='best')
        ax.grid(True, which='both', alpha=0.3)
        dl.addWidget(cv.with_toolbar(dlg), stretch=1)
        bb = QtWidgets.QHBoxLayout()
        bb.addStretch(1)
        b_use = QtWidgets.QPushButton('Use this α')
        b_close = QtWidgets.QPushButton('Close')
        bb.addWidget(b_use); bb.addWidget(b_close)
        dl.addLayout(bb)
        b_use.clicked.connect(lambda: (
            self.ed_alpha.setValue(float(a_corner)), dlg.accept()))
        b_close.clicked.connect(dlg.reject)
        self.lbl_status.setText(f'L-curve done — corner α ≈ {a_corner:.3g}')
        dlg.exec_()

    # ---- Export ----
    def _export_amap(self):
        if self._last is None:
            warn_box(self, 'No result', 'Run LDA first.'); return
        path, _ = ask_save_path(self, 'Export A(λ, τ)',
                                self.app.default_save_path('lda_Amap.csv'))
        if not path: return
        # Matrix layout: row = λ, col = τ
        wl = self.app.wavelength
        tau = self._last['tau_grid']
        A = self._last['res']['A_map']
        out = np.zeros((len(wl) + 1, len(tau) + 1))
        out[0, 0] = 0.0
        out[0, 1:] = tau
        out[1:, 0] = wl
        out[1:, 1:] = A
        delim = ',' if path.lower().endswith('.csv') else '\t'
        try:
            np.savetxt(path, out, delimiter=delim, fmt='%.10g')
            info_box(self, 'Saved', f'Wrote {path}')
        except Exception as e:
            warn_box(self, 'Export error', str(e))

    def _export_rec(self):
        if self._last is None:
            warn_box(self, 'No result', 'Run LDA first.'); return
        path, _ = ask_save_path(self, 'Export Drec(λ, t)',
                                self.app.default_save_path('lda_Drec.csv'))
        if not path: return
        delim = ',' if path.lower().endswith('.csv') else '\t'
        try:
            ta_core.write_data_file(
                path, self.app.wavelength, self._last['t_win'],
                self._last['res']['Drec'], delim)
            info_box(self, 'Saved', f'Wrote {path}')
        except Exception as e:
            warn_box(self, 'Export error', str(e))
