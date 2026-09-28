"""
TA Analyzer GUI - MCR-ALS dialog.

Multivariate Curve Resolution by Alternating Least Squares.  Splits the
2D ΔA matrix into N pure spectra S(λ) and concentration profiles C(t).
Backend: ta_core.compute_mcr.
"""
from __future__ import annotations

import numpy as np
from PyQt5 import QtWidgets, QtCore

from ta_widgets import (
    MplCanvas, make_label, make_double_edit, make_int_edit,
    style_button, info_box, warn_box, ask_save_path,
)
import ta_core


class MCRDialog(QtWidgets.QDialog):
    """MCR-ALS factorization with optional non-negativity / unimodality."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.setWindowTitle('MCR-ALS')
        self.resize(1280, 820)

        outer = QtWidgets.QHBoxLayout(self)

        # ---- Left ----
        left = QtWidgets.QGroupBox('Setup')
        lL = QtWidgets.QVBoxLayout(left)

        row_n = QtWidgets.QHBoxLayout()
        row_n.addWidget(make_label('Components N:'))
        self.ed_N = make_int_edit(2, minv=1, maxv=10)
        row_n.addWidget(self.ed_N)
        self.btn_svd = QtWidgets.QPushButton('SVD scree...')
        self.btn_svd.setToolTip(
            'Show the singular-value scree plot to help pick N.')
        row_n.addWidget(self.btn_svd)
        row_n.addStretch(1)
        lL.addLayout(row_n)

        gb_init = QtWidgets.QGroupBox('Initialization')
        ig = QtWidgets.QGridLayout(gb_init)
        ig.addWidget(make_label('Mode:'), 0, 0)
        self.dd_init = QtWidgets.QComboBox()
        self.dd_init.addItem('SVD (truncated)', 'svd')
        self.dd_init.addItem('Random', 'random')
        ig.addWidget(self.dd_init, 0, 1, 1, 3)
        lL.addWidget(gb_init)

        gb_con = QtWidgets.QGroupBox('Constraints')
        cg = QtWidgets.QGridLayout(gb_con)
        self.cb_nn_C = QtWidgets.QCheckBox('Non-negative C(t)')
        self.cb_nn_C.setChecked(True)
        self.cb_nn_C.setToolTip(
            'Force concentration profiles ≥ 0.  Usually appropriate '
            'for population kinetics.')
        cg.addWidget(self.cb_nn_C, 0, 0)
        self.cb_uni_C = QtWidgets.QCheckBox('Unimodal C(t)')
        self.cb_uni_C.setToolTip('Single-peak constraint per component.')
        cg.addWidget(self.cb_uni_C, 0, 1)
        self.cb_nn_S = QtWidgets.QCheckBox('Non-negative S(λ)')
        self.cb_nn_S.setToolTip(
            'Force pure spectra ≥ 0. Off by default — TA spectra '
            'normally contain negative GSB / SE features.')
        cg.addWidget(self.cb_nn_S, 1, 0)
        lL.addWidget(gb_con)

        gb_iter = QtWidgets.QGroupBox('Iteration')
        it_g = QtWidgets.QGridLayout(gb_iter)
        it_g.addWidget(make_label('Max iter:'), 0, 0)
        self.ed_max = make_int_edit(200, minv=10, maxv=5000)
        it_g.addWidget(self.ed_max, 0, 1)
        it_g.addWidget(make_label('Tol (ΔLOF):'), 0, 2)
        self.ed_tol = make_double_edit(1e-4, decimals=8, minv=1e-12)
        it_g.addWidget(self.ed_tol, 0, 3)
        lL.addWidget(gb_iter)

        # Time window
        gb_tr = QtWidgets.QGroupBox('Fit window (delay)')
        tg = QtWidgets.QGridLayout(gb_tr)
        tg.addWidget(make_label('From:'), 0, 0)
        self.ed_t_min = make_double_edit(float(app.delay[0]))
        tg.addWidget(self.ed_t_min, 0, 1)
        tg.addWidget(make_label('To:'), 0, 2)
        self.ed_t_max = make_double_edit(float(app.delay[-1]))
        tg.addWidget(self.ed_t_max, 0, 3)
        self.btn_t_full = QtWidgets.QPushButton('Full')
        tg.addWidget(self.btn_t_full, 0, 4)
        lL.addWidget(gb_tr)

        ar = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton('Run MCR-ALS')
        style_button(self.btn_run, bg='#4fa35a', fg='white')
        self.btn_reset = QtWidgets.QPushButton('Reset')
        ar.addWidget(self.btn_run); ar.addWidget(self.btn_reset)
        lL.addLayout(ar)

        ex = QtWidgets.QHBoxLayout()
        ex.addWidget(make_label('Export:'))
        self.btn_exp_C = QtWidgets.QPushButton('C(t)')
        self.btn_exp_S = QtWidgets.QPushButton('S(λ)')
        self.btn_exp_rec = QtWidgets.QPushButton('Drec')
        ex.addWidget(self.btn_exp_C); ex.addWidget(self.btn_exp_S)
        ex.addWidget(self.btn_exp_rec)
        lL.addLayout(ex)

        self.lbl_status = QtWidgets.QLabel('Ready.')
        lL.addWidget(self.lbl_status)
        self.txt_info = QtWidgets.QTextEdit()
        self.txt_info.setReadOnly(True)
        self.txt_info.setMinimumHeight(140)
        lL.addWidget(self.txt_info, stretch=1)

        left.setFixedWidth(380)
        outer.addWidget(left)

        # ---- Right (4 panels: C, S, LOF history, Drec) ----
        right = QtWidgets.QGroupBox('Results')
        rL = QtWidgets.QVBoxLayout(right)
        self.canvas = MplCanvas(self, nrows=2, ncols=2, figsize=(10, 8))
        self.ax_C, self.ax_S, self.ax_lof, self.ax_rec = \
            self.canvas.axes_list
        rL.addWidget(self.canvas.with_toolbar(self), stretch=1)
        outer.addWidget(right, stretch=1)

        # State
        self._last = None

        # Wiring
        self.btn_run.clicked.connect(self.do_run)
        self.btn_reset.clicked.connect(self.do_reset)
        self.btn_t_full.clicked.connect(self._t_full)
        self.btn_svd.clicked.connect(self._show_svd)
        self.btn_exp_C.clicked.connect(self._export_C)
        self.btn_exp_S.clicked.connect(self._export_S)
        self.btn_exp_rec.clicked.connect(self._export_rec)

    def closeEvent(self, ev):
        self.app.mcr_fig = None
        super().closeEvent(ev)

    def _t_full(self):
        self.ed_t_min.setValue(float(self.app.delay[0]))
        self.ed_t_max.setValue(float(self.app.delay[-1]))

    def do_reset(self):
        self.ed_N.setValue(2)
        self.dd_init.setCurrentIndex(0)
        self.cb_nn_C.setChecked(True)
        self.cb_uni_C.setChecked(False)
        self.cb_nn_S.setChecked(False)
        self.ed_max.setValue(200)
        self.ed_tol.setValue(1e-4)
        self._t_full()
        self._last = None
        for a in (self.ax_C, self.ax_S, self.ax_lof, self.ax_rec):
            a.clear()
        self.canvas.draw_idle()
        self.txt_info.setPlainText('')
        self.lbl_status.setText('Reset to defaults.')

    def _show_svd(self):
        app = self.app
        t_lo = float(self.ed_t_min.value()); t_hi = float(self.ed_t_max.value())
        m = (app.delay >= t_lo) & (app.delay <= t_hi)
        D = app.deltaA[:, m]
        D_clean = np.where(np.isfinite(D), D, 0.0)
        try:
            s = np.linalg.svd(D_clean, compute_uv=False)
        except Exception as e:
            warn_box(self, 'SVD error', str(e))
            return
        s = s[:30]
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('SVD scree plot')
        dlg.resize(560, 460)
        dl = QtWidgets.QVBoxLayout(dlg)
        cv = MplCanvas(dlg)
        ax = cv.ax
        ax.semilogy(np.arange(1, len(s) + 1), s, 'o-',
                    color='#2c7fb8')
        ax.set_xlabel('Component index')
        ax.set_ylabel('Singular value')
        ax.set_title('SVD scree (first 30)')
        ax.grid(True, which='both', alpha=0.3)
        dl.addWidget(cv.with_toolbar(dlg), stretch=1)
        b = QtWidgets.QPushButton('Close')
        b.clicked.connect(dlg.accept)
        dl.addWidget(b)
        dlg.exec_()

    def do_run(self):
        app = self.app
        N = int(self.ed_N.value())
        t_lo = float(self.ed_t_min.value()); t_hi = float(self.ed_t_max.value())
        if t_lo > t_hi:
            t_lo, t_hi = t_hi, t_lo
        m = (app.delay >= t_lo) & (app.delay <= t_hi)
        if int(m.sum()) < max(N + 1, 5):
            warn_box(self, 'Window too narrow', 'Widen the fit window.')
            return
        t_win = app.delay[m]
        # Need orientation (Nt, Nwl) for compute_mcr
        D_win = app.deltaA[:, m].T  # (Nt, Nwl)

        self.lbl_status.setText('Running MCR-ALS…')
        QtWidgets.QApplication.processEvents()
        try:
            res = ta_core.compute_mcr(
                D_win, n_comp=N,
                nn_C=self.cb_nn_C.isChecked(),
                uni_C=self.cb_uni_C.isChecked(),
                nn_S=self.cb_nn_S.isChecked(),
                tol=float(self.ed_tol.value()),
                max_iter=int(self.ed_max.value()),
                init_mode=self.dd_init.currentData())
        except Exception as e:
            warn_box(self, 'MCR error', str(e))
            self.lbl_status.setText('MCR failed.')
            return

        self._last = {'res': res, 't_win': t_win, 'D_win': D_win}
        i = res['info']
        lines = [f'N = {N},  iter = {i["iter"]},  '
                 f'converged = {i["converged"]}',
                 f'LOF (final) = {i["lof"]:.4g} %',
                 f'Window: [{t_lo:.4g}, {t_hi:.4g}] {app.t_unit_txt()}',
                 f'Constraints: nn_C={i["options"]["nn_C"]}, '
                 f'uni_C={i["options"]["uni_C"]}, '
                 f'nn_S={i["options"]["nn_S"]}',
                 f'Init mode: {i["options"]["init_mode"]}']
        self.txt_info.setPlainText('\n'.join(lines))
        self.lbl_status.setText(
            f'MCR done — LOF = {i["lof"]:.3f}%, iter = {i["iter"]}')
        self._plot_results()

    def _plot_results(self):
        if self._last is None:
            return
        app = self.app
        res = self._last['res']
        t = self._last['t_win']
        D = self._last['D_win']  # (Nt, Nwl)
        C = res['C']             # (Nt, N)
        S = res['S']             # (Nwl, N)

        # 1) C(t)
        ax = self.ax_C; ax.clear()
        cmap = ta_core._mpl_cm.get_cmap('tab10', max(10, C.shape[1]))
        for j in range(C.shape[1]):
            ax.plot(t, C[:, j], '-', color=cmap(j),
                    linewidth=1.4, label=f'comp {j + 1}')
        ax.axhline(0, color='k', linestyle=':', linewidth=0.5)
        ax.set_xlabel(f'Delay ({app.t_unit_ax()})')
        ax.set_ylabel('Concentration')
        ax.set_title('Concentration profiles  C(t)')
        ax.grid(True); ax.legend(loc='best', fontsize=8)
        if np.any(t > 0):
            ax.set_xscale('log'); ax.set_xlim(t[t > 0].min(), t.max())

        # 2) S(λ)
        ax = self.ax_S; ax.clear()
        for j in range(S.shape[1]):
            ax.plot(app.wavelength, S[:, j], '-', color=cmap(j),
                    linewidth=1.4, label=f'comp {j + 1}')
        ax.axhline(0, color='k', linestyle=':', linewidth=0.5)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Pure spectrum')
        ax.set_title('Pure spectra  S(λ)')
        ax.grid(True); ax.legend(loc='best', fontsize=8)

        # 3) LOF history
        ax = self.ax_lof; ax.clear()
        h = res['info']['lof_history']
        ax.plot(np.arange(1, len(h) + 1), h, 'o-',
                color='#e34a33', linewidth=1.2, markersize=4)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('LOF (%)')
        ax.set_title('Lack-of-fit history')
        ax.grid(True, which='both', alpha=0.3)

        # 4) Reconstruction
        ax = self.ax_rec; ax.clear()
        Drec = C @ S.T  # (Nt, Nwl)
        # Plot in (λ, t) layout for consistency with main map
        cl = float(np.nanmax(np.abs(Drec)))
        if not np.isfinite(cl) or cl == 0:
            cl = 1.0
        from matplotlib.colors import ListedColormap
        cm = ListedColormap(app.get_colormap_array())
        ax.pcolormesh(app.wavelength, t, Drec, cmap=cm,
                      vmin=-cl, vmax=cl, shading='nearest')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel(f'Delay ({app.t_unit_ax()})')
        ax.set_title(r'Reconstruction  $\widetilde D = C S^T$')

        self.canvas.draw_idle()

    # ---- Export ----
    def _export_C(self):
        if self._last is None:
            warn_box(self, 'No result', 'Run MCR first.'); return
        path, _ = ask_save_path(self, 'Export C(t)',
                                self.app.default_save_path('mcr_C.csv'))
        if not path: return
        t = self._last['t_win']
        C = self._last['res']['C']
        mat = np.column_stack([t, C])
        delim = ',' if path.lower().endswith('.csv') else '\t'
        hdr = delim.join([f'delay_{self.app.t_unit_hdr()}']
                         + [f'C_{j+1}' for j in range(C.shape[1])])
        try:
            np.savetxt(path, mat, delimiter=delim, fmt='%.10g',
                       header=hdr, comments='')
            info_box(self, 'Saved', f'Wrote {path}')
        except Exception as e:
            warn_box(self, 'Export error', str(e))

    def _export_S(self):
        if self._last is None:
            warn_box(self, 'No result', 'Run MCR first.'); return
        path, _ = ask_save_path(self, 'Export S(λ)',
                                self.app.default_save_path('mcr_S.csv'))
        if not path: return
        S = self._last['res']['S']
        mat = np.column_stack([self.app.wavelength, S])
        delim = ',' if path.lower().endswith('.csv') else '\t'
        hdr = delim.join(['wavelength_nm']
                         + [f'S_{j+1}' for j in range(S.shape[1])])
        try:
            np.savetxt(path, mat, delimiter=delim, fmt='%.10g',
                       header=hdr, comments='')
            info_box(self, 'Saved', f'Wrote {path}')
        except Exception as e:
            warn_box(self, 'Export error', str(e))

    def _export_rec(self):
        if self._last is None:
            warn_box(self, 'No result', 'Run MCR first.'); return
        path, _ = ask_save_path(self, 'Export Drec(λ, t)',
                                self.app.default_save_path('mcr_Drec.csv'))
        if not path: return
        Drec = (self._last['res']['C'] @ self._last['res']['S'].T).T  # (λ, t)
        delim = ',' if path.lower().endswith('.csv') else '\t'
        try:
            ta_core.write_data_file(
                path, self.app.wavelength, self._last['t_win'], Drec, delim)
            info_box(self, 'Saved', f'Wrote {path}')
        except Exception as e:
            warn_box(self, 'Export error', str(e))
