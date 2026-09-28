"""
TA Analyzer GUI - Global Analysis dialog (largest sub-window).

Fits a sum of exponentials convolved with a Gaussian IRF to the whole
(M x N) TA matrix; shows the fit maps, DADS, EADS, and kinetics at
the clicked wavelength.
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
import ta_device
import ta_residual_store


class GlobalAnalysisDialog(QtWidgets.QDialog):
    """Global-analysis fit / DADS / EADS / kinetics visualizer."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.setWindowTitle('Global Analysis')
        # Widened from 1420 → 1680 to make room for the rightmost
        # 'Compute Device' panel (~240px) without squeezing the plots.
        self.resize(1680, 1040)
        self.cur_kin_wl = app.selWL
        self._map_vlines = [None, None, None]

        # ---- Fit-cancellation plumbing -------------------------------
        # ``_stop_requested`` flips to True when the user clicks Stop.
        # The GA objective polls ``_on_stop_check`` at the start of each
        # evaluation and raises ta_core.GlobalAnalysisStopped when the
        # flag is set, aborting the scipy Nelder-Mead loop cleanly.
        # ``_fit_running`` gates the button enable/disable logic and
        # prevents the user from launching two fits in parallel via
        # nested QApplication.processEvents() calls.
        self._stop_requested = False
        self._fit_running = False
        # Cached compute-device list + currently selected device dict
        # (populated by ``_populate_devices``).
        self._device_entries: list[dict] = []
        self._device_checkboxes: list[QtWidgets.QCheckBox] = []
        self._selected_device: dict | None = None

        outer = QtWidgets.QHBoxLayout(self)

        # ---------- Left panel ----------
        left = QtWidgets.QGroupBox('Fit setup')
        lL = QtWidgets.QVBoxLayout(left)

        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(make_label('Components:'))
        self.dd_N = QtWidgets.QComboBox()
        self.dd_N.addItems(['1', '2', '3', '4', '5'])
        self.dd_N.setCurrentText(str(app.ga_n_comp))
        row1.addWidget(self.dd_N)
        self.cb_inf = QtWidgets.QCheckBox(u'Include τ = ∞')
        self.cb_inf.setChecked(app.ga_has_inf)
        row1.addWidget(self.cb_inf)
        row1.addStretch(1)
        lL.addLayout(row1)

        lL.addWidget(make_label(
            f'Initial time constants ({app.t_unit_txt()}, positive):'))
        # 5-column table: tau / Fixed / Stretched / β init / β Fixed.
        # The last three columns are only used when "Stretched" is on
        # for that row; otherwise the row is a plain exp like before.
        self.tau_table = QtWidgets.QTableWidget(0, 5)
        self.tau_table.setHorizontalHeaderLabels([
            f'tau_init ({app.t_unit_txt()})',
            'Fixed',
            'Stretched',
            'β init',
            'β Fixed',
        ])
        hdr = self.tau_table.horizontalHeader()
        hdr.setStretchLastSection(True)
        # Make the boolean / β columns narrow
        self.tau_table.setColumnWidth(1, 60)
        self.tau_table.setColumnWidth(2, 80)
        self.tau_table.setColumnWidth(3, 80)
        self.tau_table.setColumnWidth(4, 70)
        lL.addWidget(self.tau_table)

        # IRF mode for stretched components
        row_irf_mode = QtWidgets.QHBoxLayout()
        row_irf_mode.addWidget(make_label('Stretched IRF mode:', 'right'))
        self.dd_irf_mode = QtWidgets.QComboBox()
        self.dd_irf_mode.addItems(['numerical', 'skip'])
        self.dd_irf_mode.setCurrentText(getattr(app, 'ga_irf_mode',
                                                'numerical'))
        self.dd_irf_mode.setToolTip(
            '"numerical": convolve stretched component with the IRF on a '
            'fine grid (recommended).\n'
            '"skip":  ignore the IRF for stretched components and mask '
            'data within ~3·σ of t₀.\n'
            'Plain-exp rows always use the closed-form IRF treatment '
            'regardless of this choice.')
        self.dd_irf_mode.currentTextChanged.connect(
            lambda s: setattr(self.app, 'ga_irf_mode', s))
        row_irf_mode.addWidget(self.dd_irf_mode)
        row_irf_mode.addStretch(1)
        lL.addLayout(row_irf_mode)

        # Optimizer selection: TRF (Levenberg-Marquardt via
        # scipy.optimize.least_squares) is the default because it
        # converges in ~5-10× fewer objective evaluations than the
        # legacy Nelder-Mead loop for our nonlinear-least-squares
        # structure — see fit_global_analysis docstring.  The old
        # Nelder-Mead path is kept as a fallback for datasets whose
        # loss surface has sharp discontinuities.
        row_opt = QtWidgets.QHBoxLayout()
        row_opt.addWidget(make_label('Optimizer:', 'right'))
        self.dd_optimizer = QtWidgets.QComboBox()
        # (display_label, method_string) tuples
        self.dd_optimizer.addItem('TRF (fast, Levenberg-Marquardt)', 'trf')
        self.dd_optimizer.addItem('Nelder-Mead (legacy)', 'nm')
        _cur_opt = getattr(app, 'ga_optimizer', 'trf')
        for i in range(self.dd_optimizer.count()):
            if self.dd_optimizer.itemData(i) == _cur_opt:
                self.dd_optimizer.setCurrentIndex(i)
                break
        self.dd_optimizer.setToolTip(
            'TRF (default): scipy.optimize.least_squares with '
            'Trust-Region-Reflective Levenberg-Marquardt.  Uses the\n'
            'residual-vector structure of the GA fit to build\n'
            'Gauss-Newton steps, converging in typically 5-10\n'
            'objective evaluations vs 250-500 for Nelder-Mead.\n\n'
            'Nelder-Mead: derivative-free scalar minimisation.\n'
            'Slower but robust to loss surfaces with sharp\n'
            'discontinuities that trip up gradient-based methods.')
        self.dd_optimizer.currentIndexChanged.connect(
            lambda i: setattr(self.app, 'ga_optimizer',
                              self.dd_optimizer.itemData(i)))
        row_opt.addWidget(self.dd_optimizer)
        row_opt.addStretch(1)
        lL.addLayout(row_opt)

        # IRF t0 row
        row_t0 = QtWidgets.QHBoxLayout()
        row_t0.addWidget(make_label(f'IRF t₀ ({app.t_unit_txt()}):'))
        self.ed_t0 = make_double_edit(app.ga_t0)
        row_t0.addWidget(self.ed_t0)
        self.cb_t0_fix = QtWidgets.QCheckBox('Fixed')
        self.cb_t0_fix.setChecked(app.ga_t0_fixed)
        row_t0.addWidget(self.cb_t0_fix)
        row_t0.addStretch(1)
        lL.addLayout(row_t0)

        # IRF FWHM row
        row_fw = QtWidgets.QHBoxLayout()
        row_fw.addWidget(make_label(f'IRF FWHM ({app.t_unit_txt()}):'))
        self.ed_fw = make_double_edit(app.ga_fwhm, minv=1e-6)
        row_fw.addWidget(self.ed_fw)
        self.cb_fw_fix = QtWidgets.QCheckBox('Fixed')
        self.cb_fw_fix.setChecked(app.ga_fwhm_fixed)
        row_fw.addWidget(self.cb_fw_fix)
        row_fw.addStretch(1)
        lL.addLayout(row_fw)

        # ---- Fit time-range -------------------------------------------
        # Restrict the fit to a user-chosen [t_min, t_max] window.
        # Useful when the late-time tail is dominated by another process
        # or when the very-early-time data is unreliable (chirp residual,
        # detector ringing, …).  Default = full delay range.
        row_tr_lbl = QtWidgets.QHBoxLayout()
        row_tr_lbl.addWidget(make_label(
            f'Fit t-range ({app.t_unit_txt()}):'))
        row_tr_lbl.addStretch(1)
        lL.addLayout(row_tr_lbl)

        row_tr = QtWidgets.QHBoxLayout()
        # Pull saved values from the app if they exist; otherwise fall
        # back to full extent of the current data (or 0..1 if no data
        # has been loaded yet, which happens only in unit tests).
        if app.delay is not None and len(app.delay) > 0:
            d_lo_full = float(app.delay[0])
            d_hi_full = float(app.delay[-1])
        else:
            d_lo_full, d_hi_full = 0.0, 1.0
        t_lo_default = (app.ga_t_min if app.ga_t_min is not None
                        else d_lo_full)
        t_hi_default = (app.ga_t_max if app.ga_t_max is not None
                        else d_hi_full)
        self.ed_t_min = make_double_edit(t_lo_default, decimals=4)
        self.ed_t_min.setToolTip(
            'Lower bound of the delay window used for fitting. '
            'Data outside this window is ignored by the global fit.')
        self.ed_t_max = make_double_edit(t_hi_default, decimals=4)
        self.ed_t_max.setToolTip(
            'Upper bound of the delay window used for fitting.')
        row_tr.addWidget(make_label('From:', 'right'))
        row_tr.addWidget(self.ed_t_min)
        row_tr.addWidget(make_label('To:', 'right'))
        row_tr.addWidget(self.ed_t_max)
        self.btn_t_full = QtWidgets.QPushButton('Full')
        self.btn_t_full.setToolTip(
            'Reset the fit window to the full delay range.')
        row_tr.addWidget(self.btn_t_full)
        lL.addLayout(row_tr)

        # Live read-out of how many delay points are inside the window
        self.lbl_t_count = QtWidgets.QLabel('')
        self.lbl_t_count.setStyleSheet('color: #555; font-style: italic;')
        lL.addWidget(self.lbl_t_count)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton('Run Fit')
        style_button(self.btn_run, bg='#4fa35a', fg='white')
        # Stop button sits immediately right of Run Fit; it stays
        # disabled outside of an active fit and only lights up while the
        # objective loop is executing.
        self.btn_stop = QtWidgets.QPushButton('Stop')
        style_button(self.btn_stop, bg='#d05a5a', fg='white')
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip(
            'Abort the running global fit at the next objective '
            'evaluation. Enabled only while a fit is in progress.')
        self.btn_reset = QtWidgets.QPushButton('Reset')
        self.btn_save_resid = QtWidgets.QPushButton('Save Residual')
        self.btn_save_resid.setToolTip(
            'Write the current GA residual (D − fit) to the residuals '
            'folder so it can be re-loaded later from the Coherence '
            'dialog for FFT / LPSVD analysis.')
        self.btn_save_resid.setEnabled(app.ga_result_fit is not None)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_reset)
        btn_row.addWidget(self.btn_save_resid)
        lL.addLayout(btn_row)

        # Export buttons
        exp_row = QtWidgets.QHBoxLayout()
        exp_row.addWidget(make_label('Export:', 'right'))
        self.btn_exp_dads = QtWidgets.QPushButton('DADS')
        self.btn_exp_eads = QtWidgets.QPushButton('EADS')
        self.btn_exp_fit = QtWidgets.QPushButton('Fit')
        self.btn_exp_res = QtWidgets.QPushButton('Resid.')
        self.btn_exp_kin = QtWidgets.QPushButton(u'Kin λ')
        for b in (self.btn_exp_dads, self.btn_exp_eads, self.btn_exp_fit,
                  self.btn_exp_res, self.btn_exp_kin):
            exp_row.addWidget(b)
        lL.addLayout(exp_row)

        self.lbl_status = QtWidgets.QLabel('Ready.')
        lL.addWidget(self.lbl_status)
        self.txt_result = QtWidgets.QTextEdit()
        self.txt_result.setReadOnly(True)
        self.txt_result.setFont(
            QtWidgets.QApplication.font().__class__('Monospace'))
        self.txt_result.setPlainText('Results will appear here after fitting.')
        lL.addWidget(self.txt_result, stretch=1)

        left.setFixedWidth(340)
        outer.addWidget(left)

        # ---------- Right panel ----------
        right = QtWidgets.QGroupBox('Results')
        rL = QtWidgets.QVBoxLayout(right)

        # Row 1: three 2D maps (exp / fit / residual)
        self.canvas_maps = MplCanvas(self, nrows=1, ncols=3, figsize=(14, 4))
        self.ax_exp, self.ax_fit_map, self.ax_res = self.canvas_maps.axes_list
        rL.addWidget(self.canvas_maps.with_toolbar(self), stretch=2)

        # Row 2: DADS + DADS-normalized
        self.canvas_dads = MplCanvas(self, nrows=1, ncols=2, figsize=(14, 3))
        self.ax_dads, self.ax_dads_n = self.canvas_dads.axes_list
        rL.addWidget(self.canvas_dads, stretch=1)

        # Row 3: EADS + EADS-normalized
        self.canvas_eads = MplCanvas(self, nrows=1, ncols=2, figsize=(14, 3))
        self.ax_eads, self.ax_eads_n = self.canvas_eads.axes_list
        rL.addWidget(self.canvas_eads, stretch=1)

        # Row 4: component visibility checkboxes (built after fit)
        self._cb_host = QtWidgets.QWidget()
        self._cb_host_lay = QtWidgets.QHBoxLayout(self._cb_host)
        self._cb_host_lay.setContentsMargins(0, 0, 0, 0)
        rL.addWidget(self._cb_host)
        self.cb_comps = []

        # Row 5: kinetics controls + trace
        kin_ctrl = QtWidgets.QHBoxLayout()
        kin_ctrl.addWidget(make_label(u'λ (nm):', 'right'))
        self.ed_kin_wl = make_double_edit(app.selWL, decimals=2)
        kin_ctrl.addWidget(self.ed_kin_wl)
        self.btn_use_main = QtWidgets.QPushButton('Use main selection')
        kin_ctrl.addWidget(self.btn_use_main)
        kin_ctrl.addWidget(make_label('Scale:', 'right'))
        self.dd_kin_scale = QtWidgets.QComboBox()
        self.dd_kin_scale.addItems(['Log', 'Linear'])
        kin_ctrl.addWidget(self.dd_kin_scale)
        kin_ctrl.addStretch(1)
        rL.addLayout(kin_ctrl)

        self.canvas_kin_fit = MplCanvas(self, figsize=(14, 3))
        self.ax_kin_fit = self.canvas_kin_fit.ax
        rL.addWidget(self.canvas_kin_fit, stretch=2)

        outer.addWidget(right, stretch=1)

        # ---------- Rightmost: Compute Device panel ----------
        # Detects the available CPU + GPU devices via TensorFlow and
        # exposes them as an exclusive checkbox group.  The selected
        # entry drives which ``tf.device(...)`` context the lstsq /
        # matmul kernels of the GA objective run in.
        dev_panel = QtWidgets.QGroupBox('Compute Device')
        dL = QtWidgets.QVBoxLayout(dev_panel)

        # TF version / status line (populated after enumeration)
        self.lbl_dev_status = QtWidgets.QLabel('Detecting devices…')
        self.lbl_dev_status.setWordWrap(True)
        self.lbl_dev_status.setStyleSheet('color: #555; font-style: italic;')
        dL.addWidget(self.lbl_dev_status)

        # Exclusive checkbox group: acts as a radio group but presented
        # as checkboxes per the requested UI.
        self._dev_btn_group = QtWidgets.QButtonGroup(self)
        self._dev_btn_group.setExclusive(True)

        # Host widget for the checkbox rows so we can rebuild the list
        # when the user clicks Refresh.
        self._dev_list_host = QtWidgets.QWidget()
        self._dev_list_lay = QtWidgets.QVBoxLayout(self._dev_list_host)
        self._dev_list_lay.setContentsMargins(0, 0, 0, 0)
        self._dev_list_lay.setSpacing(2)
        dL.addWidget(self._dev_list_host)

        self.btn_dev_refresh = QtWidgets.QPushButton('Refresh')
        self.btn_dev_refresh.setToolTip(
            'Re-enumerate available CPU / GPU devices via TensorFlow.')
        dL.addWidget(self.btn_dev_refresh)

        # ---- GPU status readout -------------------------------------
        # Poll pynvml (or nvidia-smi) once a second while a GPU device
        # is selected and show live memory / utilisation / fan / temp.
        # Hidden / greyed when the user is on CPU.
        gb_gpu = QtWidgets.QGroupBox('GPU Status')
        gL = QtWidgets.QVBoxLayout(gb_gpu)
        gL.setSpacing(2)
        self.lbl_gpu_name = QtWidgets.QLabel('—')
        self.lbl_gpu_name.setWordWrap(True)
        self.lbl_gpu_name.setStyleSheet('font-weight: 600;')
        gL.addWidget(self.lbl_gpu_name)

        self.lbl_gpu_mem = QtWidgets.QLabel('Memory: —')
        gL.addWidget(self.lbl_gpu_mem)
        # A thin progress bar visualising memory occupancy.
        self.bar_gpu_mem = QtWidgets.QProgressBar()
        self.bar_gpu_mem.setRange(0, 100)
        self.bar_gpu_mem.setValue(0)
        self.bar_gpu_mem.setTextVisible(False)
        self.bar_gpu_mem.setFixedHeight(8)
        gL.addWidget(self.bar_gpu_mem)

        self.lbl_gpu_util = QtWidgets.QLabel('Utilisation: —')
        gL.addWidget(self.lbl_gpu_util)
        self.bar_gpu_util = QtWidgets.QProgressBar()
        self.bar_gpu_util.setRange(0, 100)
        self.bar_gpu_util.setValue(0)
        self.bar_gpu_util.setTextVisible(False)
        self.bar_gpu_util.setFixedHeight(8)
        gL.addWidget(self.bar_gpu_util)

        # Fan speed: horizontal bar (0-100% of max RPM) + numeric label.
        # Assumes the driver reports fan speed as a duty-cycle percentage
        # where 100 = max design RPM (this matches NVML's
        # nvmlDeviceGetFanSpeed / nvidia-smi 'fan.speed' convention).
        self.lbl_gpu_fan = QtWidgets.QLabel('Fan: —')
        gL.addWidget(self.lbl_gpu_fan)
        self.bar_gpu_fan = QtWidgets.QProgressBar()
        self.bar_gpu_fan.setRange(0, 100)
        self.bar_gpu_fan.setValue(0)
        self.bar_gpu_fan.setTextVisible(False)
        self.bar_gpu_fan.setFixedHeight(8)
        # Give the fan bar a distinct colour so it's not confused with
        # memory / utilisation at a glance.
        self.bar_gpu_fan.setStyleSheet(
            'QProgressBar { background: #eee; border: 1px solid #ccc; '
            'border-radius: 2px; }'
            'QProgressBar::chunk { background-color: #4a8fdd; }')
        gL.addWidget(self.bar_gpu_fan)

        self.lbl_gpu_temp = QtWidgets.QLabel('Temp: —')
        gL.addWidget(self.lbl_gpu_temp)

        self.lbl_gpu_source = QtWidgets.QLabel('')
        self.lbl_gpu_source.setStyleSheet('color: #888; font-size: 10px;')
        gL.addWidget(self.lbl_gpu_source)

        # ---- Administrator-privilege indicator ---------------------
        # NVML fan control requires elevation on Windows.  Show the
        # current state clearly + offer a one-click UAC relaunch when
        # the process is unprivileged so the user isn't left guessing
        # why fan-90 keeps failing.
        self.lbl_admin_status = QtWidgets.QLabel('')
        self.lbl_admin_status.setWordWrap(True)
        gL.addWidget(self.lbl_admin_status)
        self.btn_elevate = QtWidgets.QPushButton(
            'Restart as Administrator…')
        self.btn_elevate.setToolTip(
            'Re-launch the TA Analyzer through a Windows UAC prompt.\n'
            'The current window will close as soon as UAC is accepted;\n'
            'the elevated process starts with fresh state, so save any\n'
            'unsaved analysis first.')
        gL.addWidget(self.btn_elevate)
        # Populated by _refresh_admin_status() (called once now and
        # again from _run_fan_test after a diagnostic).
        self._refresh_admin_status()

        # Manual diagnostic: verify the driver actually honours a
        # fan-speed override on this card by round-tripping through
        # set → wait → read → restore and reporting each phase.
        self.btn_test_fan = QtWidgets.QPushButton('Test fan control…')
        self.btn_test_fan.setToolTip(
            'Runs a ~5 s diagnostic on the selected GPU:\n'
            '  1) Read current fan speed (baseline)\n'
            '  2) Request manual override at 70%\n'
            '  3) Wait 4 s and read again\n'
            '  4) Restore driver auto, read once more\n\n'
            'The dialog then shows all three readings so you can see\n'
            'whether the fan physically moved — not just whether the\n'
            'NVML API accepted the request.')
        gL.addWidget(self.btn_test_fan)

        # ---- Automatic GPU load / thermal boost during fit ----------
        # No opt-in: whenever a GPU device is selected and Run Fit is
        # clicked, we (a) start a keep-warm matmul loop so utilisation
        # stays ≥ 80% and (b) request 90% fan duty via NVML.  Both are
        # unwound in every exit path (success / user-stop / error /
        # dialog close) so idle-time state is always the driver default.
        self.lbl_gpu_auto = QtWidgets.QLabel(
            'Auto during fit:\n'
            '  · keep-warm (util ≥ 80%)\n'
            '  · fan → 90%\n'
            'Both restored to normal on fit end.')
        self.lbl_gpu_auto.setWordWrap(True)
        self.lbl_gpu_auto.setStyleSheet(
            'color: #444; font-size: 11px; '
            'background: #f4f4ef; padding: 4px; border: 1px solid #ddd;')
        self.lbl_gpu_auto.setToolTip(
            'While a GPU device is selected AND a Global Analysis fit '
            'is running, the dialog:\n\n'
            '  1) Runs a continuous background matmul on the GPU to '
            'raise utilisation above 80%.\n'
            '  2) Requests the driver to pin the fan(s) at 90% duty '
            'via NVML.\n\n'
            'Both are automatically undone when the fit finishes, is '
            'cancelled, errors out, or the dialog closes — so the '
            'GPU and fan return to their normal auto state.\n\n'
            'Note: keep-warm typically INCREASES fit wall time (1.5-'
            '3×) because the objective matmul queues behind the '
            'keep-warm ones on the CUDA stream.  NVML fan control may '
            'be refused on some GeForce Windows setups; failure '
            'reasons are reported in the status area.')
        gL.addWidget(self.lbl_gpu_auto)

        dL.addWidget(gb_gpu)
        self._gb_gpu = gb_gpu

        # Runtime handles for the overrides (populated during do_run).
        self._keepwarm: 'ta_device.GPUKeepwarm | None' = None
        self._fan_applied_index: int | None = None

        # 1 Hz refresh timer.  The callback bails out cheaply when the
        # user is on a CPU device, so it's fine to leave it running for
        # the lifetime of the dialog.
        self._gpu_status_timer = QtCore.QTimer(self)
        self._gpu_status_timer.setInterval(1000)
        self._gpu_status_timer.timeout.connect(self._update_gpu_status)

        dL.addStretch(1)
        # Hint at the bottom so the user knows what backend is doing
        # the heavy lifting for the selected device.
        self.lbl_dev_hint = QtWidgets.QLabel('')
        self.lbl_dev_hint.setWordWrap(True)
        self.lbl_dev_hint.setStyleSheet('color: #444; font-size: 11px;')
        dL.addWidget(self.lbl_dev_hint)

        dev_panel.setFixedWidth(240)
        outer.addWidget(dev_panel)

        # Wiring
        self.dd_N.currentTextChanged.connect(self.on_ncomp_change)
        self.btn_run.clicked.connect(self.do_run)
        self.btn_stop.clicked.connect(self.do_stop)
        self.btn_dev_refresh.clicked.connect(self._populate_devices)
        self.btn_test_fan.clicked.connect(self._run_fan_test)
        self.btn_elevate.clicked.connect(self._do_elevate)
        self.btn_reset.clicked.connect(self.do_reset)
        self.btn_exp_dads.clicked.connect(self.export_dads)
        self.btn_exp_eads.clicked.connect(self.export_eads)
        self.btn_exp_fit.clicked.connect(
            lambda: self.export_matrix_2d('fit', 'Global fit', 'fit'))
        self.btn_exp_res.clicked.connect(
            lambda: self.export_matrix_2d('residual', 'Residual', 'residual'))
        self.btn_exp_kin.clicked.connect(self.export_kinetics)
        self.btn_save_resid.clicked.connect(self.save_residual)
        self.btn_use_main.clicked.connect(self.sync_kin_wl_from_main)
        self.ed_kin_wl.valueChanged.connect(self.on_kin_wl_change)
        self.dd_kin_scale.currentTextChanged.connect(self.plot_kinetics)
        self.tau_table.cellChanged.connect(self._on_tau_edit)
        # t-range edits: keep the live point-count label fresh; the
        # "Full" button restores the full delay extent.
        self.ed_t_min.valueChanged.connect(self._update_t_count_label)
        self.ed_t_max.valueChanged.connect(self._update_t_count_label)
        self.btn_t_full.clicked.connect(self._set_t_range_full)

        self._cid_map = self.canvas_maps.mpl_connect(
            'button_press_event', self.on_map_click)

        self.fill_table()
        self._update_t_count_label()

        # Defer device enumeration until after the dialog has appeared,
        # so the (potentially multi-second) TensorFlow import doesn't
        # block the initial paint.
        QtCore.QTimer.singleShot(0, self._populate_devices)
        # Kick off the GPU-status refresh loop (the callback is a no-op
        # while a CPU device is selected, so this is cheap).
        self._gpu_status_timer.start()

    def closeEvent(self, ev):
        # If the user closes the dialog mid-fit, ask the objective to
        # bail out on its next evaluation.  ``do_run`` is a synchronous
        # loop punctuated by processEvents(), so we don't need to join a
        # thread here — flipping the flag is enough.
        if self._fit_running:
            self._stop_requested = True
        # Unconditionally undo any GPU overrides (keep-warm, fan-90)
        # so a background matmul thread doesn't outlive the dialog and
        # the fan isn't left pinned when the window disappears.
        self._teardown_gpu_overrides()
        # Stop polling the GPU so the timer signal doesn't fire on a
        # partially-destroyed dialog.
        try:
            self._gpu_status_timer.stop()
        except Exception:
            pass
        self.app.ga_fig = None
        super().closeEvent(ev)

    # ================================================================
    # Compute device panel
    # ================================================================
    def _populate_devices(self):
        """Enumerate CPU / GPU devices via TensorFlow and rebuild the
        checkbox list.  Called on dialog open and from the Refresh
        button.
        """
        # Clear any existing checkboxes
        for cb in self._device_checkboxes:
            self._dev_btn_group.removeButton(cb)
            cb.setParent(None)
            cb.deleteLater()
        self._device_checkboxes = []
        # Drain the layout too (in case anything else was inserted)
        while self._dev_list_lay.count():
            item = self._dev_list_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        # Enumerate — this triggers the lazy TF import on first call.
        try:
            entries = ta_device.list_devices()
        except Exception as e:
            entries = [{
                'id': '/CPU:0',
                'label': f'CPU (device enumeration failed: {e})',
                'backend': 'numpy',
            }]
        self._device_entries = entries

        # Try to preserve the previously selected device id across a
        # refresh so the user's choice isn't clobbered.
        prev_id = (self._selected_device or {}).get('id')

        # Rebuild the checkbox rows (exclusive → single selection)
        for i, ent in enumerate(entries):
            cb = QtWidgets.QCheckBox(ent['label'])
            cb.setToolTip(f"tf.device('{ent['id']}')  ·  backend={ent['backend']}")
            self._dev_btn_group.addButton(cb, i)
            self._dev_list_lay.addWidget(cb)
            self._device_checkboxes.append(cb)
            cb.toggled.connect(self._on_device_toggled)

        # Choose the default: previously-selected id if still present,
        # otherwise the first GPU if any, else the first entry (CPU).
        default_idx = 0
        if prev_id is not None:
            for i, ent in enumerate(entries):
                if ent['id'] == prev_id:
                    default_idx = i
                    break
        else:
            for i, ent in enumerate(entries):
                if ent['id'].startswith('/GPU'):
                    default_idx = i
                    break
        if self._device_checkboxes:
            self._device_checkboxes[default_idx].setChecked(True)

        # Status line
        if ta_device.tf_available():
            ver = ta_device.tf_version()
            n_gpu = sum(1 for e in entries if e['id'].startswith('/GPU'))
            self.lbl_dev_status.setText(
                f'TensorFlow {ver}  ·  {n_gpu} GPU(s) detected')
        else:
            err = ta_device.tf_import_error() or 'TensorFlow not installed'
            self.lbl_dev_status.setText(
                f'TF unavailable — {err}\nFalling back to NumPy on CPU.')

    def _on_device_toggled(self, checked: bool):
        """Store the currently-checked device entry and update the hint."""
        if not checked:
            # Exclusive group: the *un*-toggled callback also fires.
            # Only act on the one that ended up checked.
            return
        idx = self._dev_btn_group.checkedId()
        if 0 <= idx < len(self._device_entries):
            self._selected_device = self._device_entries[idx]
            ent = self._selected_device
            lines = [f"Selected: {ent['label']}"]
            if ent['id'].startswith('/GPU'):
                # Be honest about the split: matmul rides the GPU
                # (cuBLAS is stable) but lstsq goes through NumPy on
                # CPU to sidestep the cuSOLVER init crash observed on
                # TF 2.10 + Ada Lovelace GPUs.
                lines.append(f"matmul → GPU ({ent['id']}, cuBLAS)")
                lines.append("lstsq → CPU (NumPy) — cuSOLVER workaround")
            else:
                lines.append(
                    f"lstsq + matmul → {ent['backend']} @ {ent['id']}")
            self.lbl_dev_hint.setText("\n".join(lines))
            # Refresh the GPU-status widgets immediately so the user
            # sees the newly-selected card without waiting a full second
            # for the periodic timer.
            self._update_gpu_status()

    def _make_backend(self):
        """Instantiate a TFBackend for the currently-selected device."""
        ent = self._selected_device
        if ent is None:
            # Nothing chosen yet (shouldn't happen after _populate_devices,
            # but be defensive) — use CPU/NumPy.
            return ta_device.TFBackend(device='/CPU:0', backend='numpy')
        return ta_device.TFBackend(device=ent['id'], backend=ent['backend'])

    # ---- GPU load / thermal overrides ---------------------------------
    def _apply_gpu_overrides(self) -> list[str]:
        """Apply the fit-time GPU boost (keep-warm + fan-90).

        Called at the start of ``do_run`` once we've committed to
        fitting.  Unconditional when the selected device is a GPU —
        the user asked for this to happen automatically for every fit,
        not as an opt-in.  ``_teardown_gpu_overrides`` reverses both on
        every fit-exit path so idle-time behaviour is untouched.

        Returns a list of one-line status strings (successes and
        failures) to fold into the status label.
        """
        msgs: list[str] = []
        ent = self._selected_device
        if ent is None or not ent['id'].startswith('/GPU'):
            # No GPU in use → nothing to boost.
            return msgs

        gpu_idx = ta_device.gpu_index_from_device_id(ent['id'])

        # 1) Fan-90 first, so the fan is already ramping up by the time
        #    the keep-warm loop starts to raise temperature.
        # If the process isn't elevated, NVML will refuse — skip the
        # attempt with a single one-liner instead of printing the same
        # NVML permission error on every fit.  The admin banner in
        # the device panel is the persistent, actionable indicator.
        if gpu_idx is not None:
            if not ta_device.is_admin():
                msgs.append(
                    'Fan-90 SKIPPED: not running as Administrator '
                    '(see banner in device panel).')
            else:
                ok, msg = ta_device.set_gpu_fan_speed(gpu_idx, 90)
                msgs.append(
                    ('Fan-90: ' if ok else 'Fan-90 FAILED: ') + msg)
                if ok:
                    self._fan_applied_index = gpu_idx

        # 2) Keep-warm background matmul loop → GPU util ≥ 80%
        kw = ta_device.GPUKeepwarm(ent['id'])
        ok, msg = kw.start()
        msgs.append(('Keep-warm: ' if ok else 'Keep-warm FAILED: ') + msg)
        if ok:
            self._keepwarm = kw

        return msgs

    def _teardown_gpu_overrides(self):
        """Undo whatever ``_apply_gpu_overrides`` set up.

        Safe to call unconditionally — no-op when nothing was applied.
        Invoked on every fit-exit path (success, user-stop, exception,
        dialog close) so a killed fit can't leave the fan pinned at
        90% indefinitely.
        """
        if self._keepwarm is not None:
            try:
                self._keepwarm.stop()
            except Exception:
                pass
            self._keepwarm = None
        if self._fan_applied_index is not None:
            try:
                ta_device.restore_gpu_fan_auto(self._fan_applied_index)
            except Exception:
                pass
            self._fan_applied_index = None

    # ================================================================
    # GPU live status
    # ================================================================
    def _update_gpu_status(self):
        """Poll the currently-selected GPU (if any) and update the
        status widgets.  Called by the 1 Hz QTimer and on device change.

        Cheap when a CPU device is selected: exits after one branch
        without hitting pynvml / nvidia-smi.
        """
        ent = self._selected_device
        if ent is None or not ent['id'].startswith('/GPU'):
            # No GPU currently in use — show a placeholder and skip the
            # query so we don't spin up nvidia-smi for nothing.
            self._gb_gpu.setEnabled(False)
            self.lbl_gpu_name.setText('(no GPU selected)')
            self.lbl_gpu_mem.setText('Memory: —')
            self.bar_gpu_mem.setValue(0)
            self.lbl_gpu_util.setText('Utilisation: —')
            self.bar_gpu_util.setValue(0)
            self.lbl_gpu_fan.setText('Fan: —')
            self.bar_gpu_fan.setValue(0)
            self.lbl_gpu_temp.setText('Temp: —')
            self.lbl_gpu_source.setText('')
            return

        idx = ta_device.gpu_index_from_device_id(ent['id'])
        if idx is None:
            return
        try:
            st = ta_device.get_gpu_status(idx)
        except Exception:
            st = None

        self._gb_gpu.setEnabled(True)
        if st is None:
            self.lbl_gpu_name.setText(ent['label'])
            self.lbl_gpu_mem.setText('Memory: (unavailable)')
            self.bar_gpu_mem.setValue(0)
            self.lbl_gpu_util.setText('Utilisation: (unavailable)')
            self.bar_gpu_util.setValue(0)
            self.lbl_gpu_fan.setText('Fan: (unavailable)')
            self.bar_gpu_fan.setValue(0)
            self.lbl_gpu_temp.setText('Temp: (unavailable)')
            self.lbl_gpu_source.setText(
                'Install pynvml or ensure nvidia-smi is on PATH '
                'for live GPU metrics.')
            return

        self.lbl_gpu_name.setText(st['name'])
        self.lbl_gpu_mem.setText(
            f"Memory: {st['memory_used_mb']:.0f} / "
            f"{st['memory_total_mb']:.0f} MB "
            f"({st['memory_pct']:.1f}%)")
        self.bar_gpu_mem.setValue(int(round(st['memory_pct'])))

        util = st['utilization_pct']
        if util is None:
            self.lbl_gpu_util.setText('Utilisation: (n/a)')
            self.bar_gpu_util.setValue(0)
        else:
            self.lbl_gpu_util.setText(f'Utilisation: {util}%')
            self.bar_gpu_util.setValue(int(util))

        fan = st['fan_speed_pct']
        if fan is None:
            self.lbl_gpu_fan.setText('Fan: (n/a — no sensor)')
            self.bar_gpu_fan.setValue(0)
            self.bar_gpu_fan.setEnabled(False)
        else:
            # Fan speed is already a % of max design RPM in NVML /
            # nvidia-smi's convention, so we drive the 0-100 bar
            # directly.
            self.lbl_gpu_fan.setText(f'Fan: {fan}%')
            self.bar_gpu_fan.setValue(max(0, min(100, int(fan))))
            self.bar_gpu_fan.setEnabled(True)

        temp = st['temperature_c']
        temp_str = f'{temp} °C' if temp is not None else 'n/a'
        self.lbl_gpu_temp.setText(f'Temp: {temp_str}')

        self.lbl_gpu_source.setText(f"source: {st['source']}")

    def _run_fan_test(self):
        """Run the ta_device.test_fan_control() diagnostic on the
        currently-selected GPU and show a readable report.

        Blocks the UI for ~5 s during the set → wait → read cycle;
        we disable the button and pump the event loop so the user
        sees a "Testing…" state instead of a frozen dialog.
        """
        ent = self._selected_device
        if ent is None or not ent['id'].startswith('/GPU'):
            info_box(self, 'Fan test',
                     'Select a GPU device first (a CPU device has no '
                     'controllable fan).')
            return
        idx = ta_device.gpu_index_from_device_id(ent['id'])
        if idx is None:
            warn_box(self, 'Fan test',
                     f"Cannot parse GPU index from '{ent['id']}'.")
            return
        # Guard against running the test mid-fit (would conflict with
        # the automatic fan-90 override).
        if self._fit_running:
            warn_box(self, 'Fan test',
                     'A Global Analysis fit is currently running. '
                     'Stop it before testing fan control.')
            return

        self.btn_test_fan.setEnabled(False)
        self.btn_test_fan.setText('Testing… (5 s)')
        QtWidgets.QApplication.processEvents()
        try:
            rep = ta_device.test_fan_control(
                gpu_index=idx, target_pct=70, wait_seconds=4.0)
        except Exception as e:
            self.btn_test_fan.setEnabled(True)
            self.btn_test_fan.setText('Test fan control…')
            warn_box(self, 'Fan test', f'Diagnostic crashed:\n{e}')
            return
        self.btn_test_fan.setEnabled(True)
        self.btn_test_fan.setText('Test fan control…')

        # Build a human-readable report
        def _fmt(v, unit='%'):
            return f'{v}{unit}' if v is not None else '—'

        verdict = ('WORKING' if rep['appears_to_work']
                   else 'NOT DETECTABLE')
        lines = [
            f'Fan control on GPU {idx}: {verdict}',
            '',
            f'Baseline fan speed  : {_fmt(rep["baseline_pct"])}',
            f'Target requested    : {rep["target_pct"]}%',
            f'After manual set    : {_fmt(rep["after_set_pct"])}   '
            f'({"OK" if rep["set_ok"] else "SET FAILED"})',
            f'After auto restore  : {_fmt(rep["restored_pct"])}   '
            f'({"OK" if rep["restore_ok"] else "RESTORE FAILED"})',
            f'Reader source       : {rep["source"] or "n/a"}',
            '',
            f'Set  message : {rep["set_msg"]}',
            f'Restore message : {rep["restore_msg"]}',
        ]
        if not rep['appears_to_work']:
            lines += [
                '',
                'Interpretation:',
                '  The NVML calls may have returned OK, but the fan',
                '  read-back did not move meaningfully toward the',
                '  target.  Typical causes on consumer GeForce cards:',
                '    • Program not running as Administrator',
                '    • Driver silently ignores manual policy on',
                '      GeForce (Quadro / Tesla / Ada workstation',
                '      cards usually honour it)',
                '    • Fan already at auto-curve equilibrium near',
                '      the target speed (unlikely at 70%)',
                '',
                '  Consider MSI Afterburner or EVGA Precision X1 for',
                '  reliable manual fan control on GeForce Windows.',
            ]
        else:
            lines += [
                '',
                f'Verified: manual fan-speed control works on this GPU.',
                'The GA dialog will automatically pin fan to 90% at',
                'the start of each fit and restore auto on completion.',
            ]

        # Nice symmetric box (Info / Warning based on verdict)
        report_text = '\n'.join(lines)
        if rep['appears_to_work']:
            info_box(self, 'Fan control diagnostic', report_text)
        else:
            warn_box(self, 'Fan control diagnostic', report_text)

    # ---------- Administrator privileges ------------------------------
    def _refresh_admin_status(self):
        """Update the admin-status label + elevate button visibility."""
        admin = ta_device.is_admin()
        if admin:
            self.lbl_admin_status.setText(
                '✓ Running with Administrator privileges — fan '
                'control available.')
            self.lbl_admin_status.setStyleSheet(
                'color: #1c6e2b; font-size: 11px; font-weight: 600;')
            self.btn_elevate.setVisible(False)
        else:
            self.lbl_admin_status.setText(
                '⚠ Not running as Administrator.\n'
                'NVML fan control is disabled — the driver will '
                'refuse fan-90 requests until the app is elevated.')
            self.lbl_admin_status.setStyleSheet(
                'color: #a55a00; font-size: 11px; font-weight: 600; '
                'background: #fff6e5; padding: 4px; border: 1px '
                'solid #f0c979;')
            self.btn_elevate.setVisible(True)

    def _do_elevate(self):
        """Trigger a UAC prompt to re-launch the application as
        Administrator, then close the current session so the elevated
        replacement takes over.
        """
        # Explicit confirmation because this closes the app.
        reply = QtWidgets.QMessageBox.question(
            self, 'Restart as Administrator',
            'This will close the current TA Analyzer window and '
            're-launch it through a Windows UAC prompt.\n\n'
            'Click Yes when Windows asks for permission.\n\n'
            '⚠ Any unsaved analysis in this session will be lost. '
            'Continue?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if reply != QtWidgets.QMessageBox.Yes:
            return

        ok, msg = ta_device.relaunch_as_admin()
        if not ok:
            warn_box(self, 'Elevation failed', msg)
            return
        # UAC prompt was accepted → the elevated process is already
        # starting.  Shut this session down cleanly so we don't leave
        # two copies of the analyser running.
        try:
            self._teardown_gpu_overrides()
        except Exception:
            pass
        QtWidgets.QApplication.instance().quit()

    # ---------- Tau table ----------
    def fill_table(self):
        """Populate the (tau, fixed, stretched, β, β-fixed) table from
        app state.

        β-init and β-fixed cells are visually disabled (text muted) when
        the row is not stretched, but they keep their stored value so
        toggling Stretched on / off doesn't lose user input.
        """
        app = self.app
        N = app.ga_n_comp
        # ---- Resize tau arrays first (pad with new geometric guesses
        #      so different decade tau seeds appear) ----
        if len(app.ga_tau_init) != N:
            if len(app.ga_tau_init) < N:
                extra = N - len(app.ga_tau_init)
                last_tau = app.ga_tau_init[-1] if len(app.ga_tau_init) else 1.0
                new = last_tau * 10.0 ** np.arange(1, extra + 1)
                app.ga_tau_init = np.concatenate([app.ga_tau_init, new])
                app.ga_tau_fixed = np.concatenate(
                    [app.ga_tau_fixed, np.zeros(extra, bool)])
            else:
                app.ga_tau_init = app.ga_tau_init[:N]
                app.ga_tau_fixed = app.ga_tau_fixed[:N]

        # ---- Resize the β / stretched arrays similarly ----
        for name, default in (('ga_beta_init', 1.0),
                              ('ga_beta_fixed', False),
                              ('ga_stretch_on', False)):
            arr = getattr(app, name)
            if len(arr) != N:
                pad_dtype = bool if ('fixed' in name or 'stretch' in name) else float
                if len(arr) < N:
                    pad = np.full(N - len(arr), default, dtype=pad_dtype)
                    arr = np.concatenate([np.asarray(arr, pad_dtype), pad])
                else:
                    arr = arr[:N]
                setattr(app, name, arr)

        self.tau_table.blockSignals(True)
        self.tau_table.setRowCount(0)
        for ii in range(N):
            r = self.tau_table.rowCount()
            self.tau_table.insertRow(r)
            # Col 0: tau_init
            self.tau_table.setItem(r, 0, QtWidgets.QTableWidgetItem(
                f'{app.ga_tau_init[ii]:.4g}'))
            # Col 1: Fixed (tau)
            chk = QtWidgets.QTableWidgetItem()
            chk.setFlags((chk.flags() | QtCore.Qt.ItemIsUserCheckable)
                         & ~QtCore.Qt.ItemIsEditable)
            chk.setCheckState(QtCore.Qt.Checked if app.ga_tau_fixed[ii]
                              else QtCore.Qt.Unchecked)
            self.tau_table.setItem(r, 1, chk)
            # Col 2: Stretched
            chk2 = QtWidgets.QTableWidgetItem()
            chk2.setFlags((chk2.flags() | QtCore.Qt.ItemIsUserCheckable)
                          & ~QtCore.Qt.ItemIsEditable)
            chk2.setCheckState(QtCore.Qt.Checked if app.ga_stretch_on[ii]
                               else QtCore.Qt.Unchecked)
            self.tau_table.setItem(r, 2, chk2)
            # Col 3: β init
            self.tau_table.setItem(r, 3, QtWidgets.QTableWidgetItem(
                f'{app.ga_beta_init[ii]:.3g}'))
            # Col 4: β Fixed
            chk3 = QtWidgets.QTableWidgetItem()
            chk3.setFlags((chk3.flags() | QtCore.Qt.ItemIsUserCheckable)
                          & ~QtCore.Qt.ItemIsEditable)
            chk3.setCheckState(QtCore.Qt.Checked if app.ga_beta_fixed[ii]
                               else QtCore.Qt.Unchecked)
            self.tau_table.setItem(r, 4, chk3)
        self.tau_table.blockSignals(False)
        self._refresh_beta_row_states()

    def _refresh_beta_row_states(self):
        """Grey out β columns for rows that are NOT stretched."""
        for r in range(self.tau_table.rowCount()):
            it_str = self.tau_table.item(r, 2)
            if it_str is None:
                continue
            stretched = (it_str.checkState() == QtCore.Qt.Checked)
            for c in (3, 4):
                it = self.tau_table.item(r, c)
                if it is None:
                    continue
                if stretched:
                    it.setForeground(QtCore.Qt.black)
                else:
                    # Mute the disabled cells visually
                    it.setForeground(QtCore.Qt.gray)

    def _on_tau_edit(self, row, col):
        """Live-write each table edit back into app state.

        Stretched-toggle (col 2) also re-greys the β cells so the user
        immediately sees which columns are active for which row.
        """
        app = self.app
        if col == 0:
            try:
                v = float(self.tau_table.item(row, col).text())
                app.ga_tau_init[row] = v
            except Exception:
                pass
        elif col == 1:
            item = self.tau_table.item(row, col)
            app.ga_tau_fixed[row] = (item.checkState() == QtCore.Qt.Checked)
        elif col == 2:                    # Stretched toggle
            item = self.tau_table.item(row, col)
            app.ga_stretch_on[row] = (item.checkState() == QtCore.Qt.Checked)
            # Re-grey the β cells right away
            self._refresh_beta_row_states()
        elif col == 3:                    # β init
            try:
                v = float(self.tau_table.item(row, col).text())
                # Clamp to a physically meaningful range (0, 2]
                v = max(min(v, 2.0), 1e-3)
                app.ga_beta_init[row] = v
            except Exception:
                pass
        elif col == 4:                    # β Fixed
            item = self.tau_table.item(row, col)
            app.ga_beta_fixed[row] = (item.checkState() == QtCore.Qt.Checked)

    def read_table(self):
        """Copy the table values back into app state.

        Reads all 5 columns (tau / Fixed / Stretched / β / β-Fixed).
        """
        app = self.app
        N = self.tau_table.rowCount()
        tau = np.zeros(N)
        fix = np.zeros(N, bool)
        stretched = np.zeros(N, bool)
        beta = np.ones(N)
        bfix = np.zeros(N, bool)
        for r in range(N):
            try:
                tau[r] = float(self.tau_table.item(r, 0).text())
            except Exception:
                tau[r] = 1.0
            fix[r] = (self.tau_table.item(r, 1).checkState()
                      == QtCore.Qt.Checked)
            stretched[r] = (self.tau_table.item(r, 2).checkState()
                            == QtCore.Qt.Checked)
            try:
                beta[r] = float(self.tau_table.item(r, 3).text())
                beta[r] = max(min(beta[r], 2.0), 1e-3)
            except Exception:
                beta[r] = 1.0
            bfix[r] = (self.tau_table.item(r, 4).checkState()
                       == QtCore.Qt.Checked)
        app.ga_tau_init = tau
        app.ga_tau_fixed = fix
        app.ga_stretch_on = stretched
        app.ga_beta_init = beta
        app.ga_beta_fixed = bfix

    def on_ncomp_change(self, val):
        self.app.ga_n_comp = int(val)
        self.fill_table()

    def do_reset(self):
        app = self.app
        app.ga_n_comp = 3
        app.ga_tau_init = np.array([1.0, 10.0, 100.0])
        app.ga_tau_fixed = np.array([False, False, False])
        app.ga_has_inf = False
        app.ga_t0 = 0.0; app.ga_t0_fixed = True
        app.ga_fwhm = 0.15; app.ga_fwhm_fixed = True
        # Also reset the fit window to the full delay extent.
        app.ga_t_min = None
        app.ga_t_max = None
        self.dd_N.setCurrentText(str(app.ga_n_comp))
        self.cb_inf.setChecked(app.ga_has_inf)
        self.ed_t0.setValue(app.ga_t0)
        self.cb_t0_fix.setChecked(app.ga_t0_fixed)
        self.ed_fw.setValue(app.ga_fwhm)
        self.cb_fw_fix.setChecked(app.ga_fwhm_fixed)
        self._set_t_range_full()
        self.fill_table()
        self.btn_save_resid.setEnabled(False)
        self.lbl_status.setText('Reset to defaults.')

    def _set_t_range_full(self):
        """Restore the fit window to the full delay extent."""
        d = self.app.delay
        if d is None or len(d) == 0:
            return
        for ed in (self.ed_t_min, self.ed_t_max):
            ed.blockSignals(True)
        self.ed_t_min.setValue(float(d[0]))
        self.ed_t_max.setValue(float(d[-1]))
        for ed in (self.ed_t_min, self.ed_t_max):
            ed.blockSignals(False)
        self._update_t_count_label()

    def _update_t_count_label(self):
        """Show how many delay points are inside the current t-range."""
        d = self.app.delay
        if d is None:
            self.lbl_t_count.setText('')
            return
        t_lo = float(self.ed_t_min.value())
        t_hi = float(self.ed_t_max.value())
        if t_lo > t_hi:
            t_lo, t_hi = t_hi, t_lo
        n_in = int(((d >= t_lo) & (d <= t_hi)).sum())
        self.lbl_t_count.setText(
            f'  → fit will use {n_in} of {len(d)} delay points')

    # ---------- Fit ----------
    def do_stop(self):
        """Request cancellation of the running global fit.

        Just flips a flag; the actual abort happens inside the objective
        the next time it polls ``_on_stop_check`` (typically ~ms later).
        """
        if not self._fit_running:
            return
        self._stop_requested = True
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText('Stopping…')

    def _on_stop_check(self) -> bool:
        """Called by the GA objective every iteration.

        Pumps the Qt event loop so Stop-button clicks are picked up
        while the (synchronous) scipy Nelder-Mead loop is running, then
        returns True iff the user asked to abort.
        """
        # Throttle processEvents to keep overhead well under 1% for
        # short objective calls, while still guaranteeing sub-second
        # responsiveness of the Stop button.
        self._stop_poll_ctr = getattr(self, '_stop_poll_ctr', 0) + 1
        if self._stop_poll_ctr % 8 == 0:
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.AllEvents, 5)
        return bool(self._stop_requested)

    def _set_fit_running(self, running: bool):
        """Toggle button enable-state around a fit run."""
        self._fit_running = running
        self.btn_run.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        # Prevent the user from resetting or launching sub-actions
        # halfway through a fit.
        self.btn_reset.setEnabled(not running)
        self.tau_table.setEnabled(not running)
        self.dd_N.setEnabled(not running)
        self.dd_irf_mode.setEnabled(not running)
        self.dd_optimizer.setEnabled(not running)
        # Device selection is frozen for the duration of one fit —
        # switching mid-fit would corrupt the tf.device context (and
        # would also strand the keep-warm loop on the old device).
        for cb in self._device_checkboxes:
            cb.setEnabled(not running)
        self.btn_dev_refresh.setEnabled(not running)
        # Fan diagnostic also disabled during a fit — it would clash
        # with the automatic fan-90 override we apply in _apply_gpu_overrides.
        self.btn_test_fan.setEnabled(not running)

    def do_run(self):
        app = self.app
        self.read_table()
        app.ga_has_inf = self.cb_inf.isChecked()
        app.ga_t0 = self.ed_t0.value()
        app.ga_t0_fixed = self.cb_t0_fix.isChecked()
        app.ga_fwhm = self.ed_fw.value()
        app.ga_fwhm_fixed = self.cb_fw_fix.isChecked()

        if np.any(app.ga_tau_init <= 0):
            warn_box(self, 'Invalid input', 'All tau values must be positive.')
            return
        if app.ga_fwhm <= 0:
            warn_box(self, 'Invalid input', 'IRF FWHM must be positive.')
            return

        # Build the fit window from the user's t-range edits
        t_lo = float(self.ed_t_min.value())
        t_hi = float(self.ed_t_max.value())
        if t_lo > t_hi:
            t_lo, t_hi = t_hi, t_lo
        t_mask = (app.delay >= t_lo) & (app.delay <= t_hi)
        n_in = int(t_mask.sum())
        # Need enough delay points to actually constrain the model:
        # roughly, more points than free parameters (at least 3 + N_tau).
        n_min = 3 + len(app.ga_tau_init) + (1 if app.ga_has_inf else 0)
        if n_in < n_min:
            warn_box(self, 'Fit window too narrow',
                     f'The chosen t-range [{t_lo:.4g}, {t_hi:.4g}] '
                     f'{app.t_unit_txt()} contains only {n_in} delay '
                     f'point(s). At least {n_min} are needed for this '
                     'model.\n\nWiden the range or press "Full".')
            return

        # Persist the user's choice so re-opening the dialog shows it
        app.ga_t_min = t_lo
        app.ga_t_max = t_hi

        # Slice once and reuse for fit + display
        t_fit = app.delay[t_mask]
        D_fit = app.deltaA[:, t_mask]

        # Build the backend for the currently-selected device.  Cheap:
        # TF has already been imported by _populate_devices.
        backend = self._make_backend()
        dev_desc = backend.describe()
        # Optimizer chosen by the user (TRF or Nelder-Mead).
        method = self.dd_optimizer.itemData(
            self.dd_optimizer.currentIndex()) or 'trf'

        # Reset the cancellation flag and put the UI into "fitting" mode
        # (Run disabled, Stop enabled).
        self._stop_requested = False
        self._stop_poll_ctr = 0
        self._set_fit_running(True)

        # Apply the optional GPU load / thermal overrides (keep-warm
        # + fan-90).  Their status messages are prepended to the fit
        # status so the user immediately sees whether they took effect.
        override_msgs = self._apply_gpu_overrides()
        base_status = (f'Fitting on {n_in} delay points  ·  '
                       f'device {dev_desc}…')
        if override_msgs:
            self.lbl_status.setText(' | '.join(override_msgs)
                                    + '\n' + base_status)
        else:
            self.lbl_status.setText(base_status)
        QtWidgets.QApplication.processEvents()
        import time as _time
        _fit_t0 = _time.perf_counter()
        try:
            # Pass stretched / β state through.  When ga_stretch_on is
            # all-False the call reduces to the plain-exp behaviour by
            # construction inside fit_global_analysis.
            res = ta_core.fit_global_analysis(
                D_fit, t_fit,
                app.ga_tau_init, app.ga_t0, app.ga_fwhm,
                app.ga_tau_fixed, app.ga_t0_fixed, app.ga_fwhm_fixed,
                app.ga_has_inf,
                beta_init=app.ga_beta_init,
                beta_fixed=app.ga_beta_fixed,
                stretch_on=app.ga_stretch_on,
                irf_mode=app.ga_irf_mode,
                backend=backend,
                stop_check=self._on_stop_check,
                method=method)
        except ta_core.GlobalAnalysisStopped:
            self._teardown_gpu_overrides()
            self._set_fit_running(False)
            self.lbl_status.setText('Fit stopped by user.')
            return
        except TypeError:
            # Older fit_global_analysis without the new kwargs —
            # fall back gracefully (device selection + method choice
            # are lost in that path, but the fit still runs).
            try:
                res = ta_core.fit_global_analysis(
                    D_fit, t_fit,
                    app.ga_tau_init, app.ga_t0, app.ga_fwhm,
                    app.ga_tau_fixed, app.ga_t0_fixed, app.ga_fwhm_fixed,
                    app.ga_has_inf,
                    beta_init=app.ga_beta_init,
                    stretch_on=app.ga_stretch_on,
                    irf_mode=app.ga_irf_mode)
            except Exception as e:
                self._teardown_gpu_overrides()
                self._set_fit_running(False)
                warn_box(self, 'Fit error', f'Fit failed:\n{e}')
                self.lbl_status.setText('Fit failed.')
                return
        except Exception as e:
            self._teardown_gpu_overrides()
            self._set_fit_running(False)
            warn_box(self, 'Fit error', f'Fit failed:\n{e}')
            self.lbl_status.setText('Fit failed.')
            return
        _fit_wall = _time.perf_counter() - _fit_t0

        app.ga_result_tau = res['tau']
        app.ga_result_beta = res.get('beta',
                                     np.ones_like(res['tau']))
        app.ga_result_stretch_on = res.get(
            'stretch_on', np.zeros_like(res['tau'], bool))
        app.ga_result_t0 = res['t0']
        app.ga_result_fwhm = res['fwhm']
        app.ga_result_has_inf = app.ga_has_inf
        app.ga_result_dads = res['A']
        # Keep the (sub-)delay axis used for the fit so the display
        # routines plot fit/residual on the same x-axis as the fit
        # itself rather than on the full data extent.
        app.ga_result_delay = t_fit
        app.ga_result_data = D_fit
        app.ga_result_t_window = (t_lo, t_hi)
        app.ga_result_fit = res['fit']
        app.ga_result_rms = res['info']['rms']
        try:
            EADS, _, _ = ta_core.compute_eads_from_dads(
                res['A'], res['tau'], app.ga_has_inf)
            app.ga_result_eads = EADS
        except Exception:
            app.ga_result_eads = None

        # Build results text
        info = res['info']
        method_used = info.get('method', 'nm')
        method_lbl = {'trf': 'TRF (Levenberg-Marquardt)',
                      'nm':  'Nelder-Mead'}.get(method_used, method_used)
        nfev = info.get('nfev', info['iters'])
        lines = [f"Optimizer   : {method_lbl}"]
        lines.append(f"Wall time   : {_fit_wall*1e3:.1f} ms")
        lines.append(f"Objective evaluations: {nfev}"
                     + (f"  (iters={info['iters']})"
                        if info['iters'] != nfev else ''))
        lines.append(f"Initial RMS : {info['initialRMS']:.4g}")
        lines.append(f"Final   RMS : {info['rms']:.4g}")
        # "Not converged" heuristic: final RMS barely improved on the
        # initial one, and the optimiser did more than a couple of
        # function evaluations (so we didn't just bail out at x0).
        if (info['rms'] >= 0.99 * info['initialRMS']
                and nfev > 2):
            lines.append('')
            lines.append('** NOT CONVERGED — try different')
            lines.append('   initial tau values, or fix IRF/t0.')
        lines.append('')
        lines.append(f"Fit window: [{t_lo:.4g}, {t_hi:.4g}] "
                     f"{app.t_unit_txt()} ({n_in}/{len(app.delay)} pts)")
        if app.ga_stretch_on.any():
            lines.append(f"IRF mode (stretched): {app.ga_irf_mode}")
        lines.append('')
        lines.append('Time constants:')
        for ii, tv in enumerate(res['tau']):
            tag = ' (fixed)' if app.ga_tau_fixed[ii] else ''
            if app.ga_result_stretch_on[ii]:
                bv = app.ga_result_beta[ii]
                btag = ' (β fixed)' if app.ga_beta_fixed[ii] else ''
                lines.append(
                    f'  tau_{ii + 1} = {tv:10.4g} {app.t_unit_txt()}{tag}'
                    f'   β = {bv:.3g}{btag}   [stretched]')
            else:
                lines.append(
                    f'  tau_{ii + 1} = {tv:10.4g} {app.t_unit_txt()}{tag}')
        if app.ga_has_inf:
            lines.append('  tau_inf  (offset component)')
        lines.append('')
        lines.append(f"IRF t0   = {res['t0']:.4g} {app.t_unit_txt()}"
                     + (' (fixed)' if app.ga_t0_fixed else ''))
        lines.append(f"IRF FWHM = {res['fwhm']:.4g} {app.t_unit_txt()}"
                     + (' (fixed)' if app.ga_fwhm_fixed else ''))
        lines.append('')
        lines.append(f'Compute device: {dev_desc}')
        self.txt_result.setPlainText('\n'.join(lines))
        self.lbl_status.setText(
            f"Done. RMS = {info['rms']:.3g}  ·  {method_lbl}  ·  "
            f"{_fit_wall*1e3:.0f} ms  ·  device {dev_desc}")
        # A fresh fit is now available — enable the explicit save
        # button.  The user clicks it to write the residual file.
        self.btn_save_resid.setEnabled(True)
        # Restore fan curve + stop keep-warm loop before releasing the
        # UI so the GPU winds down promptly.
        self._teardown_gpu_overrides()
        # Return the UI to idle state (Run enabled, Stop disabled)
        # before drawing so the buttons update immediately.
        self._set_fit_running(False)
        self.plot_results()

    def save_residual(self):
        """Write the current GA residual (D − fit) to the residuals folder.

        Called from the Save Residual button.  The button is only
        enabled when a successful fit is available, so we don't need a
        defensive check before reading the cached arrays.
        """
        app = self.app
        if app.ga_result_fit is None or app.ga_result_data is None:
            warn_box(self, 'No fit', 'Run a global fit first.')
            return
        try:
            R = (np.asarray(app.ga_result_data, dtype=float)
                 - np.asarray(app.ga_result_fit, dtype=float))
            path = ta_residual_store.save_residual_2d(
                analysis='GA',
                dataset=getattr(app, 'data_source_desc', '') or 'data',
                wl=app.wavelength, t=app.ga_result_delay, R=R,
                time_unit=getattr(app, 'time_unit', ''),
                extra={
                    'rms': f"{app.ga_result_rms:.6g}",
                    't_window':
                        f"[{float(app.ga_result_delay[0]):.6g},"
                        f"{float(app.ga_result_delay[-1]):.6g}]",
                })
            self.lbl_status.setText(
                f'Residual saved: {os.path.basename(path)}')
            info_box(self, 'Residual saved',
                     f'Wrote\n{path}\n\n'
                     f'Re-open the Coherence dialog and click '
                     f'"Load" to make it appear in the file list.')
        except Exception as e:
            warn_box(self, 'Save residual', f'Save failed:\n{e}')

    def plot_results(self):
        if self.app.ga_result_dads is None:
            return
        self.rebuild_comp_checkboxes()
        self.plot_dads()
        self.plot_eads()
        self.plot_maps()
        self.plot_kinetics()

    def on_comp_vis(self):
        self.plot_dads()
        self.plot_eads()

    def sync_kin_wl_from_main(self):
        self.cur_kin_wl = self.app.selWL
        self.ed_kin_wl.setValue(self.cur_kin_wl)
        self.plot_kinetics()

    def on_kin_wl_change(self, val):
        self.cur_kin_wl = float(val)
        self.plot_kinetics()

    # ---------- Checkboxes ----------
    def rebuild_comp_checkboxes(self):
        # Clear old
        while self._cb_host_lay.count():
            item = self._cb_host_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.cb_comps = []
        app = self.app
        if app.ga_result_dads is None:
            return
        nc = app.ga_result_dads.shape[1]
        n_main = len(app.ga_result_tau)
        self._cb_host_lay.addWidget(make_label('Show components:'))
        for ii in range(nc):
            if ii < n_main:
                txt = (rf'τ_{ii+1} = {app.ga_result_tau[ii]:.3g} '
                       + app.t_unit_txt())
            else:
                txt = u'τ = ∞'
            cb = QtWidgets.QCheckBox(txt)
            cb.setChecked(True)
            cb.toggled.connect(self.on_comp_vis)
            self._cb_host_lay.addWidget(cb)
            self.cb_comps.append(cb)
        self._cb_host_lay.addStretch(1)

    def get_comp_visible(self):
        app = self.app
        if not self.cb_comps:
            return np.ones(app.ga_result_dads.shape[1], bool)
        return np.array([cb.isChecked() for cb in self.cb_comps], bool)

    # ---------- Plots ----------
    def plot_dads(self):
        app = self.app
        if app.ga_result_dads is None:
            return
        A = app.ga_result_dads
        nc = A.shape[1]
        n_main = len(app.ga_result_tau)
        cmap = ta_core._mpl_cm.get_cmap('tab10', max(10, nc))
        vis = self.get_comp_visible()

        # Raw
        ax = self.ax_dads
        ax.clear()
        for ii in range(nc):
            if not vis[ii]:
                continue
            lbl = (rf'τ_{ii+1} = {app.ga_result_tau[ii]:.3g} {app.t_unit_ax()}'
                   if ii < n_main else r'τ = ∞')
            ax.plot(app.wavelength, A[:, ii], '-',
                    color=cmap(ii), linewidth=1.4, label=lbl)
        ax.axhline(0, color='k', linestyle=':')
        ax.grid(True)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel(r'Amplitude $(\Delta A)$')
        ax.set_title(f'Decay-associated difference spectra '
                     f'(RMS = {app.ga_result_rms:.3g})')
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(app.wavelength.min(), app.wavelength.max())

        # Normalized
        ax = self.ax_dads_n
        ax.clear()
        for ii in range(nc):
            if not vis[ii]:
                continue
            col = A[:, ii]
            pk = np.nanmax(np.abs(col))
            if not np.isfinite(pk) or pk == 0:
                pk = 1.0
            lbl = (rf'τ_{ii+1}' if ii < n_main else r'τ = ∞')
            ax.plot(app.wavelength, col / pk, '-',
                    color=cmap(ii), linewidth=1.4, label=lbl)
        ax.axhline(0, color='k', linestyle=':')
        ax.grid(True)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Normalized amplitude')
        ax.set_title('DADS (peak-normalized)')
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(app.wavelength.min(), app.wavelength.max())
        ax.set_ylim(-1.1, 1.1)
        self.canvas_dads.draw_idle()

    def plot_eads(self):
        app = self.app
        self.ax_eads.clear()
        self.ax_eads_n.clear()
        if app.ga_result_eads is None:
            self.ax_eads.set_title(
                'Evolution-associated difference spectra '
                '(sequential) — run fit to compute')
            self.ax_eads_n.set_title('EADS (peak-normalized)')
            self.canvas_eads.draw_idle()
            return

        E = app.ga_result_eads
        nc = E.shape[1]
        n_main = len(app.ga_result_tau)
        cmap = ta_core._mpl_cm.get_cmap('tab10', max(10, nc))
        sort_idx = np.argsort(app.ga_result_tau)
        tau_sorted = app.ga_result_tau[sort_idx]
        if app.ga_result_has_inf:
            perm_fit = np.concatenate([sort_idx, [n_main]])
        else:
            perm_fit = sort_idx
        vis = self.get_comp_visible()

        # Raw EADS
        ax = self.ax_eads
        for iS in range(nc):
            fit_idx = perm_fit[iS]
            if fit_idx < len(vis) and not vis[fit_idx]:
                continue
            col = E[:, iS]
            if iS < n_main:
                lbl = (rf'S_{iS+1}  τ = {tau_sorted[iS]:.3g} '
                       + app.t_unit_ax())
            else:
                lbl = rf'S_{iS+1}  τ = ∞'
            ax.plot(app.wavelength, col, '-',
                    color=cmap(fit_idx), linewidth=1.4, label=lbl)
        ax.axhline(0, color='k', linestyle=':')
        ax.grid(True)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel(r'Amplitude $(\Delta A)$')
        ax.set_title(r'Evolution-associated difference spectra '
                     r'(sequential: S₁ → S₂ → …)')
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(app.wavelength.min(), app.wavelength.max())

        # Normalized EADS
        ax = self.ax_eads_n
        for iS in range(nc):
            fit_idx = perm_fit[iS]
            if fit_idx < len(vis) and not vis[fit_idx]:
                continue
            col = E[:, iS]
            pk = np.nanmax(np.abs(col))
            if not np.isfinite(pk) or pk == 0:
                pk = 1.0
            ax.plot(app.wavelength, col / pk, '-',
                    color=cmap(fit_idx), linewidth=1.4,
                    label=rf'S_{iS+1}')
        ax.axhline(0, color='k', linestyle=':')
        ax.grid(True)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Normalized amplitude')
        ax.set_title('EADS (peak-normalized)')
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(app.wavelength.min(), app.wavelength.max())
        ax.set_ylim(-1.1, 1.1)
        self.canvas_eads.draw_idle()

    def plot_maps(self):
        app = self.app
        if app.ga_result_fit is None:
            return
        # Use the same delay window the fit used, not the full data
        # extent.  This keeps the experimental/fit/residual maps
        # comparable on identical axes — and prevents the residual
        # panel from being dominated by data the user excluded.
        t = (app.ga_result_delay if app.ga_result_delay is not None
             else app.delay)
        D_exp = (app.ga_result_data if app.ga_result_data is not None
                 else app.deltaA)
        cmap_arr = app.get_colormap_array()
        cl = np.nanmax(np.abs(D_exp))
        if not np.isfinite(cl) or cl == 0:
            cl = 1.0
        pos = t[t > 0]
        use_log = (pos.size > 0 and t.max() / pos.min() > 50.0)

        axes = [self.ax_exp, self.ax_fit_map, self.ax_res]
        datas = [D_exp, app.ga_result_fit, D_exp - app.ga_result_fit]
        R_fin = np.isfinite(datas[2])
        if R_fin.any():
            res_rms = float(np.sqrt(np.mean(datas[2][R_fin] ** 2)))
        else:
            res_rms = 0.0
        clr = np.nanmax(np.abs(datas[2]))
        if not np.isfinite(clr) or clr == 0:
            clr = cl / 10.0
        win = app.ga_result_t_window
        if win is not None:
            window_str = (f' [t={win[0]:.3g}…{win[1]:.3g} '
                          + app.t_unit_ax() + ']')
        else:
            window_str = ''
        titles = ['Experimental  (click to pick λ)' + window_str,
                  'Global fit',
                  f'Residual  (RMS = {res_rms:.3g})']
        clims = [(-cl, cl), (-cl, cl), (-clr, clr)]
        from matplotlib.colors import ListedColormap
        cmap_obj = ListedColormap(cmap_arr)
        for ax, data, title, clim in zip(axes, datas, titles, clims):
            ax.clear()
            im = ax.pcolormesh(app.wavelength, t, data.T, cmap=cmap_obj,
                               vmin=clim[0], vmax=clim[1], shading='nearest')
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel(f'Delay time ({app.t_unit_ax()})')
            ax.set_title(title)
            ax.set_xlim(app.wavelength.min(), app.wavelength.max())
            if use_log:
                ax.set_yscale('log')
                ax.set_ylim(pos.min(), t.max())
            else:
                ax.set_ylim(t.min(), t.max())

        # Vertical crosshairs at curKinWL
        self._map_vlines = []
        for ax in axes:
            ln = ax.axvline(self.cur_kin_wl, color='w', linestyle='--',
                            linewidth=1.2)
            self._map_vlines.append(ln)
        self.canvas_maps.draw_idle()

    def on_map_click(self, event):
        if event.inaxes not in (self.ax_exp, self.ax_fit_map, self.ax_res):
            return
        if event.xdata is None:
            return
        self.cur_kin_wl = float(event.xdata)
        self.ed_kin_wl.setValue(self.cur_kin_wl)
        for ln in self._map_vlines:
            if ln is not None:
                ln.set_xdata([self.cur_kin_wl, self.cur_kin_wl])
        self.canvas_maps.draw_idle()
        self.plot_kinetics()

    def plot_kinetics(self):
        app = self.app
        if app.ga_result_fit is None:
            return
        iw = int(np.argmin(np.abs(app.wavelength - self.cur_kin_wl)))
        wl_actual = app.wavelength[iw]
        # Use the fit's own delay axis for the fit / residual curves
        t_fit = (app.ga_result_delay if app.ga_result_delay is not None
                 else app.delay)
        D_fit = (app.ga_result_data if app.ga_result_data is not None
                 else app.deltaA)
        ax = self.ax_kin_fit
        ax.clear()
        # Greyed-out data points outside the fit window — provides
        # visual context without confusing the user about what the
        # fit actually saw.
        if (app.ga_result_t_window is not None
                and app.ga_result_delay is not None
                and len(app.ga_result_delay) < len(app.delay)):
            t_lo, t_hi = app.ga_result_t_window
            outside = (app.delay < t_lo) | (app.delay > t_hi)
            if outside.any():
                ax.plot(app.delay[outside], app.deltaA[iw, outside],
                        '.', color='lightgrey', markersize=5,
                        label='Outside fit window')
        ax.plot(t_fit, D_fit[iw, :], 'k.', markersize=7, label='Data')
        ax.plot(t_fit, app.ga_result_fit[iw, :], 'r-',
                linewidth=1.6, label='Fit')
        ax.plot(t_fit, D_fit[iw, :] - app.ga_result_fit[iw, :],
                '-', color=(0.45, 0.45, 0.45), linewidth=0.9,
                label='Residual')
        ax.axhline(0, color='k', linestyle=':')
        ax.grid(True)
        ax.set_xlabel(f'Delay time ({app.t_unit_ax()})')
        ax.set_ylabel(r'$\Delta$A')
        ax.legend(loc='best')
        ax.set_title(rf'Kinetics at  λ = {wl_actual:.2f} nm   '
                     rf'(requested {self.cur_kin_wl:.2f} nm)')
        use_log = (self.dd_kin_scale.currentText() == 'Log'
                   and np.any(app.delay > 0))
        if use_log:
            ax.set_xscale('log')
            pos = app.delay[app.delay > 0]
            ax.set_xlim(pos.min(), app.delay.max())
        else:
            ax.set_xscale('linear')
            ax.set_xlim(app.delay.min(), app.delay.max())
        self.canvas_kin_fit.draw_idle()

    # ---------- Exports ----------
    def export_dads(self):
        app = self.app
        if app.ga_result_dads is None:
            warn_box(self, 'No results', 'Run the fit first.')
            return
        nc = app.ga_result_dads.shape[1]
        n_main = len(app.ga_result_tau)
        header = ['wavelength_nm']
        for ii in range(n_main):
            header.append(
                f'DADS_tau={app.ga_result_tau[ii]:.4g}_{app.t_unit_hdr()}')
        if app.ga_result_has_inf:
            header.append('DADS_tau=inf')
        mat = np.column_stack([app.wavelength, app.ga_result_dads])
        path, _ = ask_save_path(self, 'Export DADS',
                                app.default_save_path('DADS.csv'))
        if not path:
            return
        delim = ',' if path.lower().endswith('.csv') else '\t'
        try:
            np.savetxt(path, mat, delimiter=delim,
                       header=delim.join(header), comments='',
                       fmt='%.10g')
            info_box(self, 'DADS exported', f'Saved: {path}')
        except Exception as e:
            warn_box(self, 'Export error', f'Export failed:\n{e}')

    def export_eads(self):
        app = self.app
        if app.ga_result_eads is None:
            warn_box(self, 'No results',
                     'EADS not available. Either the fit has not been run '
                     'yet, or the B matrix was singular (near-degenerate '
                     'lifetimes).')
            return
        n_decay = len(app.ga_result_tau)
        n_tot = app.ga_result_eads.shape[1]
        tau_sorted = np.sort(app.ga_result_tau)
        header = ['wavelength_nm']
        for ii in range(n_decay):
            header.append(
                f'EADS_S{ii+1}_tau={tau_sorted[ii]:.4g}_{app.t_unit_hdr()}')
        if app.ga_result_has_inf:
            header.append(f'EADS_S{n_tot}_tau=inf')
        mat = np.column_stack([app.wavelength, app.ga_result_eads])
        path, _ = ask_save_path(self, 'Export EADS',
                                app.default_save_path('EADS.csv'))
        if not path:
            return
        delim = ',' if path.lower().endswith('.csv') else '\t'
        try:
            np.savetxt(path, mat, delimiter=delim,
                       header=delim.join(header), comments='', fmt='%.10g')
            info_box(self, 'EADS exported',
                     f'Saved EADS (sequential, {n_tot} species).\n\n'
                     f'File: {path}')
        except Exception as e:
            warn_box(self, 'Export error', f'Export failed:\n{e}')

    def export_matrix_2d(self, which, label, tag):
        app = self.app
        if app.ga_result_fit is None:
            warn_box(self, 'No results', 'Run the fit first.')
            return
        # Use the fit's own (sliced) data so fit and residual matrices
        # have the exact same dimensions as ga_result_fit.
        D_fit = (app.ga_result_data if app.ga_result_data is not None
                 else app.deltaA)
        t_axis = (app.ga_result_delay if app.ga_result_delay is not None
                  else app.delay)
        if which == 'fit':
            M = app.ga_result_fit
        elif which == 'residual':
            M = D_fit - app.ga_result_fit
        else:
            return
        suggested = f'GA_{tag}.csv'
        path, _ = ask_save_path(self, f'Export {label} (2D matrix)',
                                app.default_save_path(suggested))
        if not path:
            return
        delim = ',' if path.lower().endswith('.csv') else '\t'
        try:
            ta_core.write_data_file(path, app.wavelength, t_axis, M, delim)
            info_box(self, f'{label} exported',
                     f'Saved {label} matrix  '
                     f'({len(app.wavelength)} λ x {len(t_axis)} t).\n\n'
                     f'File: {path}')
        except Exception as e:
            warn_box(self, 'Export error', f'Export failed:\n{e}')

    def export_kinetics(self):
        app = self.app
        if app.ga_result_fit is None:
            warn_box(self, 'No results', 'Run the fit first.')
            return
        iw = int(np.argmin(np.abs(app.wavelength - self.cur_kin_wl)))
        wl_actual = app.wavelength[iw]
        # Fit-window data and axis
        D_fit = (app.ga_result_data if app.ga_result_data is not None
                 else app.deltaA)
        t_col = (app.ga_result_delay if app.ga_result_delay is not None
                 else app.delay)
        data_col = D_fit[iw, :]
        fit_col = app.ga_result_fit[iw, :]
        res_col = data_col - fit_col
        mat = np.column_stack([t_col, data_col, fit_col, res_col])
        header = [f'delay_{app.t_unit_hdr()}', 'data_dA', 'fit_dA',
                  'residual_dA']
        suggested = f'kinetics_{wl_actual:.1f}nm.csv'
        path, _ = ask_save_path(
            self, f'Export kinetics @ {wl_actual:.2f} nm',
            app.default_save_path(suggested))
        if not path:
            return
        delim = ',' if path.lower().endswith('.csv') else '\t'
        try:
            np.savetxt(path, mat, delimiter=delim,
                       header=delim.join(header), comments='', fmt='%.10g')
            info_box(self, 'Kinetics exported',
                     f'Saved kinetics at λ = {wl_actual:.2f} nm.\n\n'
                     f'Columns: delay, data, fit, residual.\n'
                     f'File: {path}')
        except Exception as e:
            warn_box(self, 'Export error', f'Export failed:\n{e}')
