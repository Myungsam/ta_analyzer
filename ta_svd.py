"""
ta_svd.py — Singular Value Decomposition explorer for the TA Analyzer.

Decomposes ΔA(λ, t) ≈ U Σ Vᵀ and lets the user inspect:
  * the singular-value spectrum σ_i  (log scale)
  * the first few left  singular vectors  U(:, k)  vs wavelength
  * the first few right singular vectors  V(:, k)  vs delay
  * a rank-N reconstruction and its residual map

NaN cells (typically from wavelength masks) are replaced with zero
before the decomposition.  This is the same convention MATLAB's
``svd(D, 'econ')`` uses implicitly when the input is dense.
"""
from __future__ import annotations
import os
from PyQt5 import QtCore, QtWidgets
import numpy as np

import ta_core
from ta_widgets import (MplCanvas, make_label, make_int_edit,
                        style_button, info_box, warn_box,
                        ask_save_path)


class SVDDialog(QtWidgets.QDialog):
    """Modeless SVD explorer."""

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.setWindowTitle('SVD Analysis')
        self.resize(1200, 760)
        self.setWindowFlags(self.windowFlags()
                            | QtCore.Qt.WindowMaximizeButtonHint
                            | QtCore.Qt.WindowMinimizeButtonHint)

        # ---- Compute SVD up front (it's cheap: <100 ms even on
        # 2136 × 192).  Refreshes when the user reopens. ----
        D = app.deltaA
        if D is None:
            raise RuntimeError('Load data before opening SVD.')

        nan_mask = ~np.isfinite(D)
        n_nan = int(nan_mask.sum())
        Dz = D.copy()
        Dz[nan_mask] = 0.0
        try:
            self._U, self._sigma, self._VT = np.linalg.svd(
                Dz, full_matrices=False)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f'SVD failed: {e}')

        self._D = D
        self._n_nan = n_nan

        self._build_ui()
        self.refresh_all()

    # ----------------------------------------------------------------
    # UI construction
    # ----------------------------------------------------------------
    def _build_ui(self):
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        # ===== Left panel: controls + sigma plot =====
        left = QtWidgets.QWidget()
        left.setMaximumWidth(360)
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(6, 6, 6, 6)

        # Components to keep (for reconstruction)
        row_n = QtWidgets.QHBoxLayout()
        row_n.addWidget(make_label('Components to keep (N):', 'right'))
        max_k = int(min(50, len(self._sigma)))
        self.ed_nkeep = make_int_edit(min(3, max_k), minv=1, maxv=max_k)
        self.ed_nkeep.setToolTip(
            'Rank of the reconstruction.  Setting this above the true '
            'number of independent components only fits noise.')
        self.ed_nkeep.valueChanged.connect(self.refresh_all)
        row_n.addWidget(self.ed_nkeep)
        row_n.addStretch(1)
        ll.addLayout(row_n)

        # Number of basis vectors to plot
        row_s = QtWidgets.QHBoxLayout()
        row_s.addWidget(make_label('Basis vectors to show:', 'right'))
        self.ed_nshow = make_int_edit(min(3, max_k),
                                      minv=1, maxv=min(12, max_k))
        self.ed_nshow.setToolTip(
            'How many U_k / V_k traces to overlay in the basis plots.')
        self.ed_nshow.valueChanged.connect(self.refresh_all)
        row_s.addWidget(self.ed_nshow)
        row_s.addStretch(1)
        ll.addLayout(row_s)

        # Temporal axis scale
        row_t = QtWidgets.QHBoxLayout()
        row_t.addWidget(make_label('Temporal axis scale:', 'right'))
        self.dd_tscale = QtWidgets.QComboBox()
        self.dd_tscale.addItems(['Linear', 'Log'])
        # Default to log if there is a significant late-time tail
        t = self.app.delay
        default_log = (t is not None and len(t) > 1
                       and (t[-1] / max(abs(t[0]), 1e-9)) > 50)
        self.dd_tscale.setCurrentText('Log' if default_log else 'Linear')
        self.dd_tscale.currentTextChanged.connect(self.refresh_all)
        row_t.addWidget(self.dd_tscale)
        row_t.addStretch(1)
        ll.addLayout(row_t)

        # Info label
        self.lbl_info = QtWidgets.QLabel('')
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet(
            'color: #444; font-style: italic;'
            'border: 1px solid #d9d9d9; border-radius: 4px;'
            'padding: 6px; background: #f8f8f8;')
        ll.addWidget(self.lbl_info)

        # Sigma plot
        gb_sig = QtWidgets.QGroupBox('Singular values  (log scale)')
        gs = QtWidgets.QVBoxLayout(gb_sig)
        self.canvas_sig = MplCanvas(self, figsize=(4, 3))
        gs.addWidget(self.canvas_sig.with_toolbar(self), stretch=1)
        ll.addWidget(gb_sig, stretch=1)

        # Export row
        row_exp = QtWidgets.QHBoxLayout()
        btn_exp_usv = QtWidgets.QPushButton('Export U / Σ / V')
        btn_exp_usv.setToolTip(
            'Saves three files (U, Σ, V) with axis headers.')
        btn_exp_usv.clicked.connect(self.export_usv)
        style_button(btn_exp_usv, bg='#4c8cca', fg='white')
        row_exp.addWidget(btn_exp_usv)
        btn_exp_rec = QtWidgets.QPushButton('Export reconstruction')
        btn_exp_rec.setToolTip(
            'Saves the rank-N reconstruction as a 2-D ΔA matrix '
            '(same layout as the main "Export 2D Data" button).')
        btn_exp_rec.clicked.connect(self.export_reconstruction)
        row_exp.addWidget(btn_exp_rec)
        ll.addLayout(row_exp)

        outer.addWidget(left)

        # ===== Right panel: 2x2 grid + residual at bottom =====
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(2, 2, 2, 2)

        top_row = QtWidgets.QHBoxLayout()
        gb_u = QtWidgets.QGroupBox('Spectral basis  U(:, k)  vs wavelength')
        gu = QtWidgets.QVBoxLayout(gb_u)
        self.canvas_U = MplCanvas(self, figsize=(5, 3))
        gu.addWidget(self.canvas_U.with_toolbar(self), stretch=1)
        top_row.addWidget(gb_u, stretch=1)

        gb_v = QtWidgets.QGroupBox('Temporal basis  V(:, k)  vs delay')
        gv = QtWidgets.QVBoxLayout(gb_v)
        self.canvas_V = MplCanvas(self, figsize=(5, 3))
        gv.addWidget(self.canvas_V.with_toolbar(self), stretch=1)
        top_row.addWidget(gb_v, stretch=1)
        rl.addLayout(top_row, stretch=1)

        mid_row = QtWidgets.QHBoxLayout()
        gb_orig = QtWidgets.QGroupBox('Original data')
        go = QtWidgets.QVBoxLayout(gb_orig)
        self.canvas_orig = MplCanvas(self, figsize=(5, 3))
        go.addWidget(self.canvas_orig.with_toolbar(self), stretch=1)
        mid_row.addWidget(gb_orig, stretch=1)

        gb_rec = QtWidgets.QGroupBox('Reconstruction  (rank N)')
        gr = QtWidgets.QVBoxLayout(gb_rec)
        self.canvas_rec = MplCanvas(self, figsize=(5, 3))
        gr.addWidget(self.canvas_rec.with_toolbar(self), stretch=1)
        mid_row.addWidget(gb_rec, stretch=1)
        rl.addLayout(mid_row, stretch=1)

        gb_res = QtWidgets.QGroupBox(
            'Residual  =  original − reconstruction')
        gres = QtWidgets.QVBoxLayout(gb_res)
        self.canvas_res = MplCanvas(self, figsize=(8, 2.5))
        gres.addWidget(self.canvas_res.with_toolbar(self), stretch=1)
        rl.addWidget(gb_res, stretch=1)

        outer.addWidget(right, stretch=1)

    # ----------------------------------------------------------------
    # Refresh
    # ----------------------------------------------------------------
    def _draw_heatmap(self, ax, data, title, clim=None):
        ax.clear()
        wl = self.app.wavelength
        t = self.app.delay
        if clim is None:
            cl = float(np.nanmax(np.abs(data))) if np.isfinite(
                np.nanmax(np.abs(data))) else 1.0
            if cl == 0:
                cl = 1.0
            clim = (-cl, cl)
        im = ax.pcolormesh(wl, t, data.T, cmap='turbo',
                           vmin=clim[0], vmax=clim[1], shading='nearest')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel(f'Delay time ({self.app.t_unit_ax()})')
        ax.set_title(title)
        if self.dd_tscale.currentText().lower() == 'log':
            pos = t > 0
            if pos.any():
                ax.set_yscale('log')
                ax.set_ylim(t[pos].min(), t.max())
        try:
            ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        except Exception:
            pass

    def refresh_all(self):
        n_keep = int(self.ed_nkeep.value())
        n_show = int(self.ed_nshow.value())
        n_keep = min(n_keep, len(self._sigma))
        n_show = min(n_show, len(self._sigma))

        # ---- Singular values ----
        ax = self.canvas_sig.ax
        ax.clear()
        idx = np.arange(1, len(self._sigma) + 1)
        ax.semilogy(idx, self._sigma, 'o-', color='#3a3a3a',
                    markersize=4, linewidth=1)
        ax.semilogy(idx[:n_keep], self._sigma[:n_keep], 'o',
                    color='#c83737', markersize=7,
                    label=f'kept (N={n_keep})')
        ax.set_xlabel('Component index k')
        ax.set_ylabel(r'$\sigma_k$')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
        if len(self._sigma) > 30:
            ax.set_xlim(0.5, 30.5)
        self.canvas_sig.draw_idle()

        # ---- Info text ----
        sig_kept = float(np.sum(self._sigma[:n_keep] ** 2))
        sig_total = float(np.sum(self._sigma ** 2))
        var_kept = sig_kept / sig_total if sig_total > 0 else 0
        nan_msg = (f'  (≈ {self._n_nan} masked / NaN cells set to 0)'
                   if self._n_nan else '')
        if n_keep > 1:
            self.lbl_info.setText(
                f'Data shape: {self._D.shape[0]} λ × {self._D.shape[1]} t'
                f'{nan_msg}\n'
                f'Top-{n_keep} components capture '
                f'{100*var_kept:.3f}% of the squared signal energy.\n'
                f'σ₁ / σ_N = {self._sigma[0]/self._sigma[n_keep-1]:.2f}')
        else:
            self.lbl_info.setText(
                f'Data shape: {self._D.shape[0]} λ × {self._D.shape[1]} t'
                f'{nan_msg}\n'
                f'Top-1 component captures '
                f'{100*var_kept:.3f}% of squared signal energy.')

        # ---- U(λ, k) basis ----
        ax = self.canvas_U.ax
        ax.clear()
        wl = self.app.wavelength
        cmap = ta_core._mpl_cm.get_cmap('tab10', max(10, n_show))
        for k in range(n_show):
            ax.plot(wl, self._U[:, k], '-',
                    color=cmap(k), linewidth=1.2,
                    label=f'k={k+1} (σ={self._sigma[k]:.3g})')
        ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel(r'$U_k(\lambda)$')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        self.canvas_U.draw_idle()

        # ---- V(t, k) basis ----
        ax = self.canvas_V.ax
        ax.clear()
        t = self.app.delay
        for k in range(n_show):
            ax.plot(t, self._VT[k, :], '-',
                    color=cmap(k), linewidth=1.2,
                    label=f'k={k+1}')
        ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
        ax.set_xlabel(f'Delay time ({self.app.t_unit_ax()})')
        ax.set_ylabel(r'$V_k(t)$')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        if self.dd_tscale.currentText().lower() == 'log':
            pos = t > 0
            if pos.any():
                ax.set_xscale('log')
                ax.set_xlim(t[pos].min(), t.max())
        self.canvas_V.draw_idle()

        # ---- Original / reconstruction / residual heatmaps ----
        D_clean = self._D.copy()
        D_clean[~np.isfinite(D_clean)] = 0.0
        D_rec = (self._U[:, :n_keep]
                 * self._sigma[:n_keep]) @ self._VT[:n_keep, :]
        R = D_clean - D_rec
        cl = float(max(np.nanmax(np.abs(D_clean)),
                       np.nanmax(np.abs(D_rec))))
        if not np.isfinite(cl) or cl == 0:
            cl = 1.0

        # Each canvas has only its main axes; clear/reuse colorbars by
        # recreating them.
        self.canvas_orig.fig.clear()
        ax_o = self.canvas_orig.fig.add_subplot(1, 1, 1)
        self.canvas_orig.ax = ax_o
        self.canvas_orig.axes_list = [ax_o]
        self._draw_heatmap(ax_o, D_clean, 'Original  ΔA(λ, t)',
                           clim=(-cl, cl))
        self.canvas_orig.draw_idle()

        self.canvas_rec.fig.clear()
        ax_r = self.canvas_rec.fig.add_subplot(1, 1, 1)
        self.canvas_rec.ax = ax_r
        self.canvas_rec.axes_list = [ax_r]
        self._draw_heatmap(ax_r, D_rec,
                           f'Rank-{n_keep} reconstruction',
                           clim=(-cl, cl))
        self.canvas_rec.draw_idle()

        cl_r = float(np.nanmax(np.abs(R)))
        if not np.isfinite(cl_r) or cl_r == 0:
            cl_r = 1.0
        self.canvas_res.fig.clear()
        ax_res = self.canvas_res.fig.add_subplot(1, 1, 1)
        self.canvas_res.ax = ax_res
        self.canvas_res.axes_list = [ax_res]
        rms = float(np.sqrt(np.nanmean(R ** 2)))
        self._draw_heatmap(
            ax_res, R,
            f'Residual  (RMS = {rms:.3g}, max |R| = {cl_r:.3g})',
            clim=(-cl_r, cl_r))
        self.canvas_res.draw_idle()

        self._D_rec = D_rec

    # ----------------------------------------------------------------
    # Export (GA-style: ask_save_path + short default filename)
    # ----------------------------------------------------------------
    def export_usv(self):
        path, _ = ask_save_path(
            self, 'Export SVD bases (U / Σ / V)',
            self.app.default_save_path('SVD_U.csv'),
            'CSV (*.csv);;TSV (*.tsv);;Excel (*.xlsx)')
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        stem, _ = os.path.splitext(path)
        if stem.endswith('_U'):
            stem = stem[:-2]
        try:
            wl = self.app.wavelength
            t = self.app.delay
            n = len(self._sigma)
            if ext == '.xlsx':
                ta_core.write_data_excel(
                    f'{stem}_U.xlsx', wl, np.arange(1, n+1, dtype=float),
                    self._U)
                ta_core.write_data_excel(
                    f'{stem}_S.xlsx', np.array([0.0]),
                    np.arange(1, n+1, dtype=float),
                    self._sigma[None, :])
                ta_core.write_data_excel(
                    f'{stem}_V.xlsx',
                    np.arange(1, n+1, dtype=float), t, self._VT)
                ext_used = '.xlsx'
            else:
                delim = ',' if ext != '.tsv' else '\t'
                ext_used = '.csv' if delim == ',' else '.tsv'
                ta_core.write_data_file(
                    f'{stem}_U{ext_used}', wl,
                    np.arange(1, n+1, dtype=float), self._U, delim)
                ta_core.write_data_file(
                    f'{stem}_S{ext_used}', np.array([0.0]),
                    np.arange(1, n+1, dtype=float),
                    self._sigma[None, :], delim)
                ta_core.write_data_file(
                    f'{stem}_V{ext_used}',
                    np.arange(1, n+1, dtype=float), t, self._VT, delim)
            info_box(
                self, 'Export complete',
                f'Saved three files:\n'
                f'  {stem}_U{ext_used}   ({self._U.shape[0]} λ × '
                f'{self._U.shape[1]} k)\n'
                f'  {stem}_S{ext_used}   (1 × {n})\n'
                f'  {stem}_V{ext_used}   ({n} × {len(t)})')
        except Exception as e:
            warn_box(self, 'Export error', str(e))

    def export_reconstruction(self):
        n_keep = int(self.ed_nkeep.value())
        path, sel = ask_save_path(
            self, 'Export rank-N reconstruction',
            self.app.default_save_path(f'SVD_recN{n_keep}.csv'),
            'CSV (*.csv);;TSV (*.tsv);;Excel (*.xlsx)')
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        try:
            wl = self.app.wavelength
            t = self.app.delay
            if ext == '.xlsx':
                ta_core.write_data_excel(path, wl, t, self._D_rec)
            else:
                delim = ',' if ext != '.tsv' else '\t'
                ta_core.write_data_file(path, wl, t, self._D_rec, delim)
            info_box(self, 'Export complete',
                     f'Rank-{n_keep} reconstruction saved.\n\n{path}')
        except Exception as e:
            warn_box(self, 'Export error', str(e))

    def closeEvent(self, ev):
        if hasattr(self.app, 'svd_fig') and self.app.svd_fig is self:
            self.app.svd_fig = None
        super().closeEvent(ev)
