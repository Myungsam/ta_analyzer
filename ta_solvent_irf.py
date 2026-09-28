"""
ta_solvent_irf.py — Pure-solvent (coherent-artifact / IRF) subtraction
dialog for the TA Analyzer.

Workflow
--------
1. User loads a pure-solvent TA measurement (same instrument, same
   spectrograph as the sample; the grid usually matches but doesn't
   have to).
2. The solvent ΔA is resampled onto the sample's current (λ, t) grid
   via ``ta_core.align_solvent_to_sample``.
3. A scale factor ``s`` is exposed through a slider + spinbox so the
   user can compensate for the (typically small) intensity mismatch
   between sample and reference runs.
4. The "Subtract solvent IRF" checkbox flips ``app.sub_irf_applied``;
   from that point on every ``recompute()`` does
       ΔA_corrected = ΔA_sample − s · ΔA_solvent_aligned
   alongside BG / chirp / masking.

Auto-scale
----------
A small helper estimates ``s`` by least-squares matching of the sample
and solvent kinetics within ±3·σ of t = 0, on a user-chosen wavelength
band (default: the wavelength of the sample's largest |ΔA| at t ≈ 0).
This is a starting point only — the user is expected to tweak it.
"""
from __future__ import annotations
import os
from PyQt5 import QtCore, QtWidgets
import numpy as np

import ta_core
from ta_widgets import (MplCanvas, make_label, make_double_edit,
                        make_int_edit, style_button,
                        info_box, warn_box, ask_open_paths)


class SolventIRFDialog(QtWidgets.QDialog):
    """Modeless dialog: load a pure-solvent ΔA and subtract its IRF."""

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.setWindowTitle('Solvent IRF subtraction')
        self.resize(1180, 720)
        self.setWindowFlags(self.windowFlags()
                            | QtCore.Qt.WindowMaximizeButtonHint
                            | QtCore.Qt.WindowMinimizeButtonHint)

        # Refresh-debounce timer (like the Crop dialog: 2-D pcolormesh
        # rebuilds add up fast on real datasets).
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(120)
        self._refresh_timer.timeout.connect(self._refresh_now)

        self._build_ui()
        self._reflect_state_in_ui()
        self._refresh_now()

    # ----------------------------------------------------------------
    # UI
    # ----------------------------------------------------------------
    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        # ===== Row 1: load + status =====
        load_row = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton('Load solvent TA file…')
        self.btn_load.setToolTip(
            'Open a pure-solvent ΔA file (CSV / TSV / DAT / TXT). '
            'It will be resampled onto the current sample grid '
            'automatically.')
        style_button(self.btn_load, bg='#4c8cca', fg='white')
        self.btn_load.clicked.connect(self.do_load)
        load_row.addWidget(self.btn_load)

        self.btn_clear = QtWidgets.QPushButton('Clear')
        self.btn_clear.setToolTip(
            'Forget the loaded solvent and turn off subtraction.')
        self.btn_clear.clicked.connect(self.do_clear)
        load_row.addWidget(self.btn_clear)

        self.lbl_loaded = QtWidgets.QLabel('No solvent loaded.')
        self.lbl_loaded.setStyleSheet('color: #555; font-style: italic;')
        load_row.addWidget(self.lbl_loaded, stretch=1)
        outer.addLayout(load_row)

        # ===== Row 2: toggle + scale =====
        scale_row = QtWidgets.QHBoxLayout()
        self.cb_apply = QtWidgets.QCheckBox('Subtract solvent IRF')
        self.cb_apply.setToolTip(
            'When on, every recompute() subtracts scale × '
            'solvent_aligned from the (BG/chirp-corrected) sample.')
        self.cb_apply.toggled.connect(self._on_apply_toggled)
        scale_row.addWidget(self.cb_apply)

        scale_row.addSpacing(20)
        scale_row.addWidget(make_label('Scale:', 'right'))
        # Slider: integer 0..300 maps to 0.00..3.00
        self.sl_scale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_scale.setRange(0, 300)
        self.sl_scale.setValue(100)
        self.sl_scale.setFixedWidth(280)
        self.sl_scale.valueChanged.connect(self._on_slider)
        scale_row.addWidget(self.sl_scale)

        self.ed_scale = make_double_edit(1.0, decimals=3,
                                         minv=-5.0, maxv=5.0)
        self.ed_scale.setSingleStep(0.05)
        self.ed_scale.setFixedWidth(80)
        self.ed_scale.valueChanged.connect(self._on_spinbox)
        scale_row.addWidget(self.ed_scale)

        self.btn_auto = QtWidgets.QPushButton('Auto')
        self.btn_auto.setToolTip(
            'Estimate scale by least-squares matching of the sample '
            'and solvent kinetics within ±3·σ of t = 0, using the '
            'wavelength of the sample\'s largest |ΔA| there.')
        self.btn_auto.clicked.connect(self.do_auto_scale)
        scale_row.addWidget(self.btn_auto)
        scale_row.addStretch(1)
        outer.addLayout(scale_row)

        # ===== Row 3: preview wavelength selector =====
        preview_row = QtWidgets.QHBoxLayout()
        preview_row.addWidget(make_label('Preview kinetics at λ (nm):',
                                          'right'))
        wl = self.app.wavelength
        wl_default = (float(wl[len(wl) // 2]) if wl is not None and len(wl) > 0
                      else 500.0)
        wl_min = float(wl.min()) if wl is not None and len(wl) > 0 else 200.0
        wl_max = float(wl.max()) if wl is not None and len(wl) > 0 else 1100.0
        self.ed_wl = make_double_edit(wl_default, decimals=2,
                                      minv=wl_min, maxv=wl_max)
        self.ed_wl.setFixedWidth(90)
        self.ed_wl.valueChanged.connect(lambda _: self._refresh_timer.start())
        preview_row.addWidget(self.ed_wl)

        preview_row.addWidget(make_label('t window:', 'right'))
        t = self.app.delay
        if t is not None and len(t) > 0:
            t_default_max = min(2.0, float(t.max()))
            t_default_min = max(-2.0, float(t.min()))
        else:
            t_default_min, t_default_max = -2.0, 2.0
        self.ed_t_min = make_double_edit(t_default_min, decimals=3,
                                         minv=-1e9, maxv=1e9)
        self.ed_t_max = make_double_edit(t_default_max, decimals=3,
                                         minv=-1e9, maxv=1e9)
        for ed in (self.ed_t_min, self.ed_t_max):
            ed.setFixedWidth(80)
            ed.valueChanged.connect(lambda _: self._refresh_timer.start())
        preview_row.addWidget(self.ed_t_min)
        preview_row.addWidget(make_label('to'))
        preview_row.addWidget(self.ed_t_max)
        preview_row.addStretch(1)
        outer.addLayout(preview_row)

        # Plot panels: 1 row, 3 columns.  We turn off constrained_layout
        # on these canvases because we recreate colorbars on every refresh
        # and the layout engine throws layout errors when its bookkeeping
        # gets out of sync between calls.  Manual margins are good enough
        # here since the figure size is fixed by the GroupBox.
        plots = QtWidgets.QHBoxLayout()
        gb_left = QtWidgets.QGroupBox('Kinetics at λ — before vs after')
        l_left = QtWidgets.QVBoxLayout(gb_left)
        self.canvas_kin = MplCanvas(self, figsize=(5, 4), tight=False)
        l_left.addWidget(self.canvas_kin.with_toolbar(self), stretch=1)
        plots.addWidget(gb_left, stretch=1)

        gb_mid = QtWidgets.QGroupBox('Aligned solvent  (sample grid)')
        l_mid = QtWidgets.QVBoxLayout(gb_mid)
        self.canvas_solv = MplCanvas(self, figsize=(5, 4), tight=False)
        l_mid.addWidget(self.canvas_solv.with_toolbar(self), stretch=1)
        plots.addWidget(gb_mid, stretch=1)

        gb_right = QtWidgets.QGroupBox('Corrected ΔA  (= sample − s·solvent)')
        l_right = QtWidgets.QVBoxLayout(gb_right)
        self.canvas_corr = MplCanvas(self, figsize=(5, 4), tight=False)
        l_right.addWidget(self.canvas_corr.with_toolbar(self), stretch=1)
        plots.addWidget(gb_right, stretch=1)

        outer.addLayout(plots, stretch=1)

        # ===== Bottom buttons =====
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_close = QtWidgets.QPushButton('Close')
        self.btn_close.clicked.connect(self.close)
        btn_row.addWidget(self.btn_close)
        outer.addLayout(btn_row)

    # ----------------------------------------------------------------
    # State <-> UI sync
    # ----------------------------------------------------------------
    def _reflect_state_in_ui(self):
        """Pull current app state into the UI widgets without firing
        more refreshes than necessary."""
        app = self.app
        has_solv = app.sub_irf_solv_data is not None
        # Loaded label
        if has_solv:
            shape = app.sub_irf_solv_data.shape
            self.lbl_loaded.setText(
                f'Loaded:  {os.path.basename(app.sub_irf_solv_path) or "(unknown)"}  '
                f'({shape[0]} λ × {shape[1]} t)')
        else:
            self.lbl_loaded.setText('No solvent loaded.')
        # Toggle
        self.cb_apply.blockSignals(True)
        self.cb_apply.setChecked(bool(app.sub_irf_applied))
        self.cb_apply.blockSignals(False)
        self.cb_apply.setEnabled(has_solv)
        self.btn_auto.setEnabled(has_solv)
        # Scale
        self._set_scale_widgets(app.sub_irf_scale, fire=False)

    def _set_scale_widgets(self, value: float, fire: bool = True):
        """Update the slider+spinbox without recursion."""
        value = max(-5.0, min(5.0, float(value)))
        self.ed_scale.blockSignals(True)
        self.sl_scale.blockSignals(True)
        self.ed_scale.setValue(value)
        # Slider uses 0..300 = 0.00..3.00; clamp negatives to 0 for the
        # slider but allow them in the spinbox.
        sl_val = int(round(max(0.0, min(3.0, value)) * 100))
        self.sl_scale.setValue(sl_val)
        self.ed_scale.blockSignals(False)
        self.sl_scale.blockSignals(False)
        if fire:
            self.app.sub_irf_scale = value
            self._refresh_timer.start()

    # ----------------------------------------------------------------
    # User interactions
    # ----------------------------------------------------------------
    def _on_slider(self, val):
        self._set_scale_widgets(val / 100.0, fire=True)

    def _on_spinbox(self, val):
        self._set_scale_widgets(val, fire=True)

    def _on_apply_toggled(self, checked: bool):
        app = self.app
        if checked and app.sub_irf_solv_data is None:
            # Refuse to turn on without data
            warn_box(self, 'No solvent loaded',
                     'Load a pure-solvent TA file first.')
            self.cb_apply.blockSignals(True)
            self.cb_apply.setChecked(False)
            self.cb_apply.blockSignals(False)
            return
        app.sub_irf_applied = bool(checked)
        # Keep the toolbar checkbox in sync, but don't fire its own
        # toggled() handler (which would just toggle us back).
        if hasattr(app, 'cb_sub_irf'):
            app.cb_sub_irf.blockSignals(True)
            app.cb_sub_irf.setChecked(bool(checked))
            app.cb_sub_irf.blockSignals(False)
        # Recompute the WHOLE app pipeline (BG/chirp/etc + subtraction)
        app.update_all()
        self._refresh_now()

    def do_load(self):
        # NB: ``ask_open_paths`` returns a *string* when called with the
        # default ``multi=False``, and a *list* only when ``multi=True``.
        # The previous code treated the return value as a list
        # unconditionally, so ``paths[0]`` actually took the first
        # *character* of the path — meaning a real path like
        # ``C:/data/solvent.csv`` was silently truncated to ``C``,
        # and the file was never found unless the current working
        # directory happened to contain a one-character filename.
        # We now explicitly use ``multi=True`` and normalize the result
        # to a list so multi-selection also works.
        result = ask_open_paths(self, 'Open pure-solvent TA file',
                                multi=True)
        if not result:
            return
        # Defensive: accept either a str or a list of str regardless of
        # the helper's behaviour, so a future change to ask_open_paths
        # cannot reintroduce this bug.
        if isinstance(result, str):
            paths = [result]
        else:
            paths = list(result)
        path = paths[0]
        if len(paths) > 1:
            info_box(self, 'Multiple files',
                     f'Using only the first file:\n  {path}')
        try:
            wl, t, A = ta_core.parse_data_file(path)
        except Exception as e:
            warn_box(self, 'Load failed', f'Could not parse:\n{e}')
            return
        app = self.app
        app.sub_irf_solv_wl = wl
        app.sub_irf_solv_t = t
        app.sub_irf_solv_data = A
        app.sub_irf_solv_path = path
        # Resample onto current sample grid
        app.realign_solvent()
        if app.sub_irf_aligned is None:
            warn_box(self, 'Align failed',
                     'Could not resample the solvent onto the current '
                     'sample grid.  Check the file.')
            return
        # First load: don't auto-enable subtraction — let user verify
        # the scale first.  But DO show the panels so they can see it.
        self._reflect_state_in_ui()
        self._refresh_now()
        info_box(self, 'Solvent loaded',
                 f'{wl.size} λ × {t.size} t resampled onto the sample '
                 f'grid.\n\nNow tick "Subtract solvent IRF" and tune the '
                 f'scale, or click "Auto" for an initial estimate.')

    def do_clear(self):
        app = self.app
        app.sub_irf_applied = False
        app.sub_irf_solv_wl = None
        app.sub_irf_solv_t = None
        app.sub_irf_solv_data = None
        app.sub_irf_solv_path = ''
        app.sub_irf_aligned = None
        app.update_all()
        self._reflect_state_in_ui()
        self._refresh_now()

    def do_auto_scale(self):
        """Pick a sensible starting scale by LS-matching kinetics."""
        app = self.app
        if app.sub_irf_aligned is None:
            return
        wl = app.wavelength
        t = app.delay
        # Find a wavelength where the sample has a strong t≈0 feature.
        # Without an explicit IRF model, use the wavelength of the
        # largest |ΔA| inside a narrow ± window around t = 0.
        near_zero = np.abs(t) < 0.5
        if not near_zero.any():
            near_zero = np.abs(t) < (0.1 * (t.max() - t.min()))
        sample_zero = app.deltaA[:, near_zero]
        wl_pick = int(np.argmax(np.nanmax(np.abs(sample_zero), axis=1)))
        # Match kinetics in the same window
        s_kin = app.deltaA[wl_pick, near_zero]
        v_kin = app.sub_irf_aligned[wl_pick, near_zero]
        mask = np.isfinite(s_kin) & np.isfinite(v_kin)
        s_kin = s_kin[mask]
        v_kin = v_kin[mask]
        denom = float(np.dot(v_kin, v_kin))
        if denom < 1e-30:
            warn_box(self, 'Auto-scale',
                     'Solvent reference has no signal near t = 0; '
                     'cannot auto-scale.')
            return
        scale = float(np.dot(s_kin, v_kin) / denom)
        # Clamp to the slider's positive range; users can manually
        # override if a negative scale is genuinely wanted.
        scale = max(-5.0, min(5.0, scale))
        # Also remember the picked wavelength for the preview
        self.ed_wl.blockSignals(True)
        self.ed_wl.setValue(float(wl[wl_pick]))
        self.ed_wl.blockSignals(False)
        self._set_scale_widgets(scale, fire=True)
        info_box(self, 'Auto-scale',
                 f'Estimated scale = {scale:.3f} at '
                 f'λ = {wl[wl_pick]:.1f} nm (largest |ΔA| near t=0).\n\n'
                 'Inspect the kinetics and adjust manually if needed.')

    # ----------------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------------
    def _refresh_now(self):
        """Single redraw step.  Fast: only line plots, no pcolormesh
        for the kinetics panel; pcolormesh for the two map panels but
        on a downsampled grid if the data is large."""
        self._draw_kinetics()
        self._draw_solvent_map()
        self._draw_corrected_map()

    def _t_window(self):
        t1, t2 = self.ed_t_min.value(), self.ed_t_max.value()
        if t1 > t2:
            t1, t2 = t2, t1
        return t1, t2

    def _draw_kinetics(self):
        app = self.app
        ax = self.canvas_kin.ax
        ax.clear()
        if app.delay is None or app.wavelength is None:
            self.canvas_kin.draw_idle()
            return
        wl_pick = float(self.ed_wl.value())
        iw = int(np.argmin(np.abs(app.wavelength - wl_pick)))

        t = app.delay
        t1, t2 = self._t_window()
        mask = (t >= t1) & (t <= t2)
        if not mask.any():
            self.canvas_kin.draw_idle()
            return
        t_sub = t[mask]

        # 1) Sample kinetics BEFORE any solvent subtraction.  We pull
        #    this from a fresh recompute with sub_irf_applied=False so
        #    the user always sees the unsubtracted curve as reference.
        was_on = app.sub_irf_applied
        app.sub_irf_applied = False
        app.recompute()
        sample_kin = app.deltaA[iw, mask].copy()
        app.sub_irf_applied = was_on
        app.recompute()  # restore current pipeline

        ax.plot(t_sub, sample_kin, 'b-', linewidth=1.4,
                label=f'sample (λ={app.wavelength[iw]:.1f} nm)')

        # 2) Scaled solvent (what would be subtracted)
        if app.sub_irf_aligned is not None:
            solv_kin = app.sub_irf_aligned[iw, mask]
            ax.plot(t_sub, app.sub_irf_scale * solv_kin, 'g--',
                    linewidth=1.2,
                    label=f'{app.sub_irf_scale:.3f} × solvent')

            # 3) Corrected (only if checkbox is on)
            corr_kin = sample_kin - app.sub_irf_scale * solv_kin
            ax.plot(t_sub, corr_kin, 'r-', linewidth=1.6,
                    label='corrected')

        ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
        ax.axvline(0, color='k', linestyle=':', linewidth=0.8)
        ax.set_xlabel(f'Delay time ({app.t_unit_ax()})')
        ax.set_ylabel(r'$\Delta$A')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Kinetics at λ = {app.wavelength[iw]:.1f} nm  '
                     f'({"ON" if app.sub_irf_applied else "OFF"})')
        self.canvas_kin.draw_idle()

    def _draw_solvent_map(self):
        app = self.app
        # Rebuild the figure cleanly each call.  Trying to keep one
        # axes alive while removing the colorbar axes leaves a stale
        # reference inside matplotlib's constrained_layout engine,
        # which then crashes on the next draw with
        #   AttributeError: 'NoneType' object has no attribute 'transSubfigure'
        # The Crop dialog uses the same fresh-figure strategy.
        fig = self.canvas_solv.fig
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        self.canvas_solv.ax = ax
        self.canvas_solv.axes_list = [ax]
        if app.sub_irf_aligned is None:
            ax.text(0.5, 0.5,
                    'Load a solvent file to preview.',
                    ha='center', va='center',
                    transform=ax.transAxes, color='#777')
            self.canvas_solv.draw_idle()
            return
        t1, t2 = self._t_window()
        t = app.delay
        mask = (t >= t1) & (t <= t2)
        if not mask.any():
            self.canvas_solv.draw_idle()
            return
        D = app.sub_irf_aligned[:, mask] * app.sub_irf_scale
        cl = float(np.nanmax(np.abs(D)))
        if not np.isfinite(cl) or cl == 0:
            cl = 1.0
        im = ax.pcolormesh(app.wavelength, t[mask], D.T,
                           cmap='turbo', vmin=-cl, vmax=cl,
                           shading='nearest')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel(f'Delay time ({app.t_unit_ax()})')
        ax.set_title(f'{app.sub_irf_scale:.3f} × aligned solvent')
        try:
            fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        except Exception:
            pass
        self.canvas_solv.draw_idle()

    def _draw_corrected_map(self):
        app = self.app
        # Same fresh-figure pattern as _draw_solvent_map.
        fig = self.canvas_corr.fig
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        self.canvas_corr.ax = ax
        self.canvas_corr.axes_list = [ax]
        if app.deltaA is None:
            self.canvas_corr.draw_idle()
            return
        t1, t2 = self._t_window()
        t = app.delay
        mask = (t >= t1) & (t <= t2)
        if not mask.any():
            self.canvas_corr.draw_idle()
            return
        if app.sub_irf_applied:
            D = app.deltaA[:, mask]
            title = 'Corrected  (subtraction ON)'
        elif app.sub_irf_aligned is not None:
            # Preview the hypothetical correction without actually
            # flipping the app pipeline — just so the user can decide.
            D = (app.deltaA[:, mask]
                 - app.sub_irf_scale * app.sub_irf_aligned[:, mask])
            title = 'Preview  (subtraction OFF — toggle to apply)'
        else:
            D = app.deltaA[:, mask]
            title = 'Sample ΔA  (no solvent loaded yet)'
        cl = float(np.nanmax(np.abs(D)))
        if not np.isfinite(cl) or cl == 0:
            cl = 1.0
        im = ax.pcolormesh(app.wavelength, t[mask], D.T,
                           cmap='turbo', vmin=-cl, vmax=cl,
                           shading='nearest')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel(f'Delay time ({app.t_unit_ax()})')
        ax.set_title(title)
        try:
            fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        except Exception:
            pass
        self.canvas_corr.draw_idle()

    # ----------------------------------------------------------------
    def closeEvent(self, ev):
        app = self.app
        # Keep the main-window toolbar checkbox in sync with the real
        # state.  If the user closed the dialog without loading a
        # solvent, the checkbox shouldn't stay ticked.
        if hasattr(app, 'cb_sub_irf'):
            should_be_checked = bool(
                app.sub_irf_applied and app.sub_irf_solv_data is not None)
            if app.cb_sub_irf.isChecked() != should_be_checked:
                app.cb_sub_irf.blockSignals(True)
                app.cb_sub_irf.setChecked(should_be_checked)
                app.cb_sub_irf.blockSignals(False)
        if hasattr(app, 'sub_irf_fig') and app.sub_irf_fig is self:
            app.sub_irf_fig = None
        super().closeEvent(ev)
