"""
TA Analyzer - Main window.

Ties together the numerical core (ta_core), the 2D map / spectrum /
kinetics panels, the toolbar, and all the sub-dialogs.

Run from the command line with:
    python ta_main.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap

import ta_core
from ta_widgets import (
    MplCanvas, make_label, make_double_edit, make_int_edit,
    style_button, info_box, warn_box, ask_save_path, ask_open_paths,
    compute_zlim,
)
from ta_dialogs_a import (
    BackgroundDialog, CropDialog, MaskDialog, LoadAverageDialog,
)
from ta_chirp import ChirpDialog
from ta_ga import GlobalAnalysisDialog
from ta_load_custom import CustomLoadDialog
from ta_accumulate import AccumulationDialog, LoadDataChooserDialog


# =====================================================================
# Right-click menu filter — consumes RMB before matplotlib sees it
# =====================================================================
class _RightClickMenuFilter(QtCore.QObject):
    """Qt event filter that swallows right-mouse-button presses on a
    matplotlib canvas and invokes ``on_right_click`` instead.  This
    keeps matplotlib's zoom tool from starting a zoom-out drag
    (leaving a cursor-tracking rubber band) when the user only wanted
    to open our context menu."""

    def __init__(self, canvas: QtCore.QObject, on_right_click):
        super().__init__(canvas)
        self._on_right_click = on_right_click

    def eventFilter(self, obj, event):
        try:
            et = event.type()
        except Exception:
            return False
        if et in (QtCore.QEvent.MouseButtonPress,
                  QtCore.QEvent.MouseButtonRelease,
                  QtCore.QEvent.MouseButtonDblClick):
            try:
                if event.button() == QtCore.Qt.RightButton:
                    if et == QtCore.QEvent.MouseButtonPress:
                        # Fire the menu; matplotlib never sees this event.
                        self._on_right_click()
                    return True
            except Exception:
                return False
        return False


# =====================================================================
# Axis-range prompt (used by spectrum/kinetics manual-scale mode)
# =====================================================================
def _prompt_axis_range(parent, title: str,
                       cur_min: float, cur_max: float):
    """Modal min/max input for a single axis.  Returns ``(lo, hi)`` on
    Apply or ``None`` on Cancel / invalid entry."""
    dlg = QtWidgets.QDialog(parent)
    dlg.setWindowTitle(title)
    lay = QtWidgets.QFormLayout(dlg)

    # Wide-range spinboxes; matplotlib uses floats so we mirror that.
    sb_min = QtWidgets.QDoubleSpinBox(dlg)
    sb_max = QtWidgets.QDoubleSpinBox(dlg)
    for sb in (sb_min, sb_max):
        sb.setDecimals(6)
        sb.setRange(-1e12, 1e12)
        sb.setSingleStep(abs(cur_max - cur_min) / 20.0 or 0.1)
    sb_min.setValue(float(cur_min))
    sb_max.setValue(float(cur_max))
    lay.addRow('Min:', sb_min)
    lay.addRow('Max:', sb_max)

    btns = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
        parent=dlg)
    btns.accepted.connect(dlg.accept)
    btns.rejected.connect(dlg.reject)
    lay.addRow(btns)

    if dlg.exec_() != QtWidgets.QDialog.Accepted:
        return None
    lo = float(sb_min.value())
    hi = float(sb_max.value())
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return None
    return lo, hi


# =====================================================================
# Main window
# =====================================================================
class TAAnalyzer(QtWidgets.QMainWindow):
    """Main TA Data Analyzer window (Python port of the MATLAB class)."""

    def __init__(self):
        super().__init__()
        self._init_state()
        self._build_ui()
        self._update_status_label()

    # ================================================================
    # State initialization
    # ================================================================
    def _init_state(self):
        # --- Data (current working) ---
        self.wavelength: np.ndarray | None = None
        self.delay: np.ndarray | None = None
        self.deltaA: np.ndarray | None = None
        self.deltaA_raw: np.ndarray | None = None

        # --- Original snapshot (for Crop / Revert) ---
        self.original_wavelength: np.ndarray | None = None
        self.original_delay: np.ndarray | None = None
        self.original_deltaA: np.ndarray | None = None
        self.data_source_desc = ''
        # Folder the currently loaded data came from — used as the
        # default directory for every subsequent save dialog so exports
        # land next to the raw data instead of the process cwd.
        self.data_source_dir = ''

        # --- Background correction ---
        self.bg_applied = False
        self.bg_spectrum: np.ndarray | None = None
        self.bg_n = 0

        # --- Solvent IRF subtraction ---
        # When sub_irf_applied is True the pipeline subtracts
        # ``sub_irf_scale * sub_irf_aligned`` from each (BG-corrected)
        # ΔA frame.  ``sub_irf_aligned`` is the pure-solvent reference
        # already resampled onto the current sample (λ, t) grid, so it
        # can be subtracted with a single broadcast.
        # The raw solvent (its own grid + values) is stored alongside
        # so we can re-align after the sample is cropped or chirped.
        self.sub_irf_applied = False
        self.sub_irf_scale = 1.0
        self.sub_irf_solv_wl: np.ndarray | None = None
        self.sub_irf_solv_t: np.ndarray | None = None
        self.sub_irf_solv_data: np.ndarray | None = None
        self.sub_irf_solv_path = ''
        self.sub_irf_aligned: np.ndarray | None = None

        # --- Chirp correction ---
        self.chirp_applied = False
        self.chirp_pts = np.zeros((0, 2))
        self.chirp_params: np.ndarray | None = None
        self.chirp_fit_rms = float('nan')
        self.chirp_view_t_min: float | None = None
        self.chirp_view_t_max = 2.0

        # --- Ridge helper ---
        self.ridge_method = 'max|dA/dt|'
        self.ridge_tmin = -0.5
        self.ridge_tmax = 1.5
        self.ridge_smooth_n = 5
        self.ridge_wl: np.ndarray | None = None
        self.ridge_t: np.ndarray | None = None

        # --- Current selection (crosshair position) ---
        self.selWL = 0.0
        self.selT = 0.0
        # Index of the currently-selected delay in self.delay.  This is
        # the source of truth driving the spectrum panel's "Delay idx"
        # spinbox; selT is kept consistent with self.delay[_selT_idx].
        self._selT_idx = 0

        # --- Pinned overlays ---
        self.specOverlays: list[float] = []   # pinned delay values
        self.kinOverlays: list[float] = []    # pinned wavelength values

        # --- Delay-axis scale mode ---
        self.delay_scale_mode = 'linear'      # 'linear' | 'log' | 'split'
        self.split_threshold = 1.0

        # --- Crop request + wavelength resampling (set by Crop Data) ---
        # crop_bounds is the (wl_min, wl_max, t_min, t_max) the user asked
        # for in the original frame; None ⇒ full range.  It only seeds the
        # Crop dialog and its "did the crop change?" test — whether the
        # data *is* cropped is always derived from the grid (is_cropped).
        # With resampling on, the working λ axis is binned from the
        # cropped original axis (_crop_wl_pre_resample); resample_info
        # holds the bin grouping so the solvent can reuse it.
        self.crop_bounds: tuple | None = None
        self.resample_enabled = False
        self.resample_dx = 1.0
        self.resample_mode = 'average'     # 'average' | 'decimate'
        self.resample_info: dict | None = None
        self._crop_wl_pre_resample: np.ndarray | None = None

        # --- Main 2D map display range ---
        self.main_view_tmin: float | None = None
        self.main_view_tmax: float | None = None

        # --- 2D map colormap / z-range ---
        self.map_colormap = 'turbo'
        self.map_z_min: float | None = None
        self.map_z_max: float | None = None

        # --- Zero-time shift (for ns-TA, no chirp) ---
        self.tZeroShift = 0.0

        # --- Wavelength region masks ---
        # list of (wl_min, wl_max, 'nan'|'zero')
        self.masked_regions: list[tuple[float, float, str]] = []

        # --- Global analysis setup ---
        self.ga_n_comp = 3
        self.ga_tau_init = np.array([1.0, 10.0, 100.0])
        self.ga_tau_fixed = np.array([False, False, False])
        self.ga_has_inf = False
        self.ga_t0 = 0.0
        self.ga_t0_fixed = True
        self.ga_fwhm = 0.15
        self.ga_fwhm_fixed = True
        # Stretched-exponential per component (added later; defaults
        # match plain-exp behavior so existing code paths keep working).
        # When ga_stretch_on[j] is True, component j fits
        # exp(-((t-t0)/τ_j)^β_j) ⊗ IRF instead of the plain exp.
        self.ga_beta_init = np.array([1.0, 1.0, 1.0])
        self.ga_beta_fixed = np.array([False, False, False])
        self.ga_stretch_on = np.array([False, False, False])
        # IRF-handling for stretched components: 'numerical' does a
        # direct convolution on a fine grid; 'skip' ignores the IRF
        # and masks early times.  Plain-exp components always use
        # the closed-form erfcx path regardless.
        self.ga_irf_mode = 'numerical'
        # User-specified delay window for the global fit.  None ⇒ use
        # the full delay extent.  These are persisted across re-opens
        # of the Global Analysis dialog.
        self.ga_t_min: float | None = None
        self.ga_t_max: float | None = None

        # --- Global analysis results ---
        self.ga_result_tau: np.ndarray | None = None
        self.ga_result_beta: np.ndarray | None = None
        self.ga_result_stretch_on: np.ndarray | None = None
        self.ga_result_t0: float | None = None
        self.ga_result_fwhm: float | None = None
        self.ga_result_has_inf = False
        self.ga_result_dads: np.ndarray | None = None
        self.ga_result_eads: np.ndarray | None = None
        self.ga_result_fit: np.ndarray | None = None
        self.ga_result_rms = float('nan')
        # Delay axis and ΔA matrix that the fit was actually run on
        # (slice of self.delay / self.deltaA when a t-window was set).
        # plot_maps / plot_kinetics / export use these so the fit result
        # is shown / saved on the same axis it was computed on.
        self.ga_result_delay: np.ndarray | None = None
        self.ga_result_data: np.ndarray | None = None
        self.ga_result_t_window: tuple | None = None

        # --- Time-axis unit ---
        self.time_unit = 'ps'   # 'ps' or 'us'

        # --- Child windows ---
        self.bg_fig = None
        self.chirp_fig = None
        self.crop_fig = None
        self.load_fig = None
        self.ga_fig = None
        self.mask_fig = None
        # New (this session): dedicated dialogs for the additional
        # analysis methods.  Same single-instance discipline as the
        # others (close + reopen rather than stacking).
        self.kfit_fig = None
        self.lda_fig = None
        self.mcr_fig = None
        self.coh_fig = None
        self.svd_fig = None
        # Pure-solvent IRF / coherent-artifact subtraction dialog.
        self.sub_irf_fig = None
        # LDA result cache so the Coherence dialog can subtract LDA's
        # reconstruction in its 'Use last LDA residual' mode.
        self._last_lda = None

        # --- 2D axes containers (rebuilt on scale-mode changes) ---
        self._ax2d_list: list = []
        self._hVLine: list = []   # crosshair vertical lines on 2D map
        self._hHLine: list = []   # crosshair horizontal lines on 2D map

        # --- Spectrum / kinetics zoom-state preservation ---
        # When the user pans / zooms in the spectrum or kinetics panel
        # we capture the new view limits here.  Any subsequent redraw
        # caused by changing the selected delay / wavelength then
        # restores these limits, so the user's zoom is not blown away
        # every time they move the crosshair.  None  ⇒  use auto-scale
        # (the default for the very first draw of new data).
        self._spec_xlim: tuple | None = None
        self._spec_ylim: tuple | None = None
        # kinetics can have one or two panels (split mode); index 0 is the
        # left/single panel, index 1 is the right panel in split mode.
        self._kin_xlims: list = [None, None]
        self._kin_ylims: list = [None, None]
        # While we are programmatically redrawing we don't want our own
        # set_xlim / set_ylim calls to be recorded as "user zoom".
        self._suppress_zoom_save = False

        # --- Per-axis auto-scale toggle (right-click menu on each panel) ---
        # Independent flags per axis:
        #   auto_x = True  ⇒ x-range is auto-fit to the data whose y falls
        #                    inside the current ylim (i.e. what is
        #                    actually visible on-screen)
        #   auto_y = True  ⇒ y-range is auto-fit to the data whose x
        #                    falls inside the current xlim
        # When both are True the fit reduces to full data extent.
        # Toolbar rectangle-zoom / pan disables both flags for that panel;
        # re-enabling either flag releases the toolbar mode.
        self._spec_auto_x: bool = True
        self._spec_auto_y: bool = True
        # Kinetics may have one or two panels (split mode) so the flags
        # are per-sub-axes.  Index 0 = left/single, 1 = right (split).
        self._kin_auto_x: list = [True, True]
        self._kin_auto_y: list = [True, True]

        # Colorbar handles for the main 2D map (avoid stacking).
        # Cleared each time the figure is rebuilt in _draw_map_2d.
        self._main_cb = None

        # ---- Incremental-update infrastructure ----
        # Caches the artists from the most recent FULL rebuild of the
        # 2D map / spectrum / kinetics panels, plus a "layout signature"
        # that captures everything that would force a re-build (axis
        # sizes, scale mode, overlay count, …).  When the signature is
        # unchanged, redraws can update artists in place via set_array
        # / set_ydata / set_clim — orders of magnitude faster than
        # tearing down and rebuilding the figure.
        self._map_images: list = []          # one pcolormesh per axes
        self._map_layout_sig = None
        self._spec_lines_overlay: list = []  # overlay Line2D handles
        self._spec_line_current = None       # current-selT Line2D
        self._spec_layout_sig = None
        # kinetics: lists of per-panel artists (one slot per panel,
        # split mode uses both)
        self._kin_lines_overlay: list = [[], []]
        self._kin_line_current: list = [None, None]
        self._kin_layout_sig = None

    # ================================================================
    # UI construction
    # ================================================================
    def _build_ui(self):
        self.setWindowTitle('TA Data Analyzer')
        self.resize(1400, 820)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setSpacing(6)
        outer.setContentsMargins(10, 10, 10, 10)

        # ---- Toolbar row 1 ----
        tb1 = QtWidgets.QHBoxLayout()
        tb1.setSpacing(5)
        btn_load = QtWidgets.QPushButton('Load Data...')
        btn_load.setToolTip(
            'Open the data-loading chooser: Standard (auto-detect), '
            'Custom (manual X/Y/Z region selection), or Accumulation '
            '(folder of repeats, drop bad shots then average).')
        btn_load.clicked.connect(self.load_data)
        tb1.addWidget(btn_load)

        btn_reset = QtWidgets.QPushButton('Reset Corrections')
        btn_reset.clicked.connect(self.reset_corrections)
        tb1.addWidget(btn_reset)

        btn_crop = QtWidgets.QPushButton('Crop Data...')
        btn_crop.clicked.connect(self.open_crop_window)
        tb1.addWidget(btn_crop)

        btn_bg = QtWidgets.QPushButton('Background Correction...')
        btn_bg.clicked.connect(self.open_background_window)
        tb1.addWidget(btn_bg)

        # Subtract solvent IRF — a checkbox (not a button) because the
        # on/off state is the user-visible action.  Opening the dialog
        # is what happens when the box is first checked while no
        # solvent is loaded.
        self.cb_sub_irf = QtWidgets.QCheckBox('Subtract solvent IRF')
        self.cb_sub_irf.setToolTip(
            'Subtract a pure-solvent (coherent artifact / IRF) '
            'reference ΔA from the sample.  Tick to open the '
            'dialog and load the solvent file; the dialog also '
            'controls the subtraction scale factor.')
        self.cb_sub_irf.toggled.connect(self._on_sub_irf_toggled)
        tb1.addWidget(self.cb_sub_irf)

        btn_chirp = QtWidgets.QPushButton('Chirp Correction...')
        btn_chirp.clicked.connect(self.open_chirp_window)
        tb1.addWidget(btn_chirp)

        btn_t0 = QtWidgets.QPushButton('Set t=0 here')
        btn_t0.setToolTip(
            'Make the currently selected crosshair position on the main '
            '2D map the new t=0. Click anywhere on the map first to move '
            'the crosshair to your desired zero-time location.')
        btn_t0.clicked.connect(self.apply_zero_time_at_crosshair)
        tb1.addWidget(btn_t0)

        btn_mask = QtWidgets.QPushButton('Mask Wavelengths...')
        btn_mask.setToolTip(
            'Mask wavelength regions (e.g., pump scattering). Can set them '
            'to NaN (excluded from fits) or to 0.')
        btn_mask.clicked.connect(self.open_mask_window)
        tb1.addWidget(btn_mask)

        btn_ga = QtWidgets.QPushButton('Global Analysis...')
        style_button(btn_ga, bg='#8c4db3', fg='white')
        btn_ga.clicked.connect(self.open_global_analysis_window)
        tb1.addWidget(btn_ga)

        # Advanced analysis menu (4 newer tools live here so the
        # toolbar doesn't grow one button per technique).
        self.btn_analysis = QtWidgets.QPushButton('More Analysis ▾')
        style_button(self.btn_analysis, bg='#5a4d8c', fg='white')
        self.btn_analysis.setToolTip(
            'SVD, single-trace kinetic fit, Lifetime Density Analysis, '
            'MCR-ALS, vibrational coherence (FFT) — open as separate '
            'windows.')
        analysis_menu = QtWidgets.QMenu(self)
        analysis_menu.addAction('Singular Value Decomposition (SVD)…',
                                self.open_svd_window)
        analysis_menu.addAction('Kinetic Fit (single trace)…',
                                self.open_kinetic_fit_window)
        analysis_menu.addAction('Lifetime Density Analysis (LDA)…',
                                self.open_lda_window)
        analysis_menu.addAction('MCR-ALS…', self.open_mcr_window)
        analysis_menu.addSeparator()
        analysis_menu.addAction('Coherence (FFT of residual)…',
                                self.open_coherence_window)
        self.btn_analysis.setMenu(analysis_menu)
        tb1.addWidget(self.btn_analysis)

        tb1.addWidget(make_label('Delay scale:', 'right'))
        self.dd_scale = QtWidgets.QComboBox()
        self.dd_scale.addItems(['Linear', 'Log', 'Split'])
        self.dd_scale.setCurrentText('Linear')
        self.dd_scale.currentTextChanged.connect(self._on_scale_mode_change)
        tb1.addWidget(self.dd_scale)

        self.lbl_split_threshold = make_label(
            f'Split threshold ({self.t_unit_txt()}):', 'right')
        tb1.addWidget(self.lbl_split_threshold)
        self.ed_threshold = make_double_edit(self.split_threshold,
                                             minv=1e-6, decimals=4)
        self.ed_threshold.valueChanged.connect(self._on_threshold_change)
        tb1.addWidget(self.ed_threshold)

        tb1.addStretch(1)
        self.lbl_status = QtWidgets.QLabel('No data loaded')
        self.lbl_status.setStyleSheet('color:#4d4d4d;')
        tb1.addWidget(self.lbl_status)
        outer.addLayout(tb1)

        # ---- Toolbar row 2 ----
        tb2 = QtWidgets.QHBoxLayout()
        tb2.setSpacing(5)

        self.lbl_view_tmin = make_label(
            f'Display t_min ({self.t_unit_txt()}):', 'right')
        tb2.addWidget(self.lbl_view_tmin)
        self.ed_view_tmin = make_double_edit(0.0, decimals=4)
        self.ed_view_tmin.valueChanged.connect(
            lambda v: self._on_main_view_change('min', v))
        tb2.addWidget(self.ed_view_tmin)

        self.lbl_view_tmax = make_label(
            f't_max ({self.t_unit_txt()}):', 'right')
        tb2.addWidget(self.lbl_view_tmax)
        self.ed_view_tmax = make_double_edit(0.0, decimals=4)
        self.ed_view_tmax.valueChanged.connect(
            lambda v: self._on_main_view_change('max', v))
        tb2.addWidget(self.ed_view_tmax)

        btn_reset_view = QtWidgets.QPushButton('Reset display')
        btn_reset_view.clicked.connect(self.reset_main_view)
        tb2.addWidget(btn_reset_view)

        tb2.addWidget(make_label('Time unit:', 'right'))
        self.dd_time_unit = QtWidgets.QComboBox()
        self.dd_time_unit.addItem('ps', 'ps')
        self.dd_time_unit.addItem(u'μs', 'us')
        self.dd_time_unit.setToolTip(
            'Switch time-axis labels between picoseconds and microseconds. '
            'Numeric values in the loaded data are NOT converted; only '
            'display labels and export headers change.')
        self.dd_time_unit.currentIndexChanged.connect(
            lambda _: self._on_time_unit_change(self.dd_time_unit.currentData()))
        tb2.addWidget(self.dd_time_unit)

        tb2.addStretch(1)
        tb2.addWidget(QtWidgets.QLabel(
            '(full range shown when tmin ≥ tmax)'))
        outer.addLayout(tb2)

        # ---- Main content splitter ----
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(splitter, stretch=1)

        # --- 2D map panel ---
        pnl2d = QtWidgets.QGroupBox('2D Map  —  click to select (λ, t)')
        pnl2d_lay = QtWidgets.QVBoxLayout(pnl2d)
        pnl2d_lay.setContentsMargins(5, 5, 5, 5)

        # Colour / Z-range strip
        ctrl2d = QtWidgets.QHBoxLayout()
        ctrl2d.setSpacing(5)
        ctrl2d.addWidget(make_label('Colormap:', 'right'))
        self.dd_colormap = QtWidgets.QComboBox()
        self.dd_colormap.addItems(['turbo', 'parula', 'jet', 'hot', 'cool',
                                    'gray', 'RdBu', 'BWR', 'seismic'])
        self.dd_colormap.setCurrentText(self.map_colormap)
        self.dd_colormap.currentTextChanged.connect(self._on_colormap_change)
        ctrl2d.addWidget(self.dd_colormap)

        ctrl2d.addWidget(make_label('Z min:', 'right'))
        self.ed_z_min = make_double_edit(0.0, decimals=6)
        self.ed_z_min.valueChanged.connect(
            lambda v: self._on_z_change('min', v))
        ctrl2d.addWidget(self.ed_z_min)

        ctrl2d.addWidget(make_label('Z max:', 'right'))
        self.ed_z_max = make_double_edit(0.0, decimals=6)
        self.ed_z_max.valueChanged.connect(
            lambda v: self._on_z_change('max', v))
        ctrl2d.addWidget(self.ed_z_max)

        btn_reset_z = QtWidgets.QPushButton('Reset Z')
        btn_reset_z.clicked.connect(self.reset_z_range)
        ctrl2d.addWidget(btn_reset_z)
        ctrl2d.addStretch(1)
        # "Original" export: cropped raw data with no BG / chirp / IRF /
        # mask applied.  Placed to the left of the corrected export so
        # the user sees pre-processing → corrected reading left to right.
        btn_export_2d_orig = QtWidgets.QPushButton('Export 2D data_original...')
        btn_export_2d_orig.setToolTip(
            'Save the 2D ΔA matrix BEFORE any processing '
            '(BG / chirp / solvent-IRF / masks) but AFTER any active '
            'crop, to CSV / TSV / Excel (.xlsx).')
        style_button(btn_export_2d_orig, bg='#7a7a7a', fg='white')
        btn_export_2d_orig.clicked.connect(self.export_2d_data_original)
        ctrl2d.addWidget(btn_export_2d_orig)
        # Export the (corrected, cropped, masked) 2D matrix as data — not
        # an image. The matplotlib navigation toolbar already handles
        # PNG / SVG export of the figure itself.
        btn_export_2d = QtWidgets.QPushButton('Export 2D Data...')
        btn_export_2d.setToolTip(
            'Save the displayed 2D ΔA matrix to CSV / TSV / Excel '
            '(.xlsx). The file reflects all current corrections '
            '(BG, chirp, masks) and any active crop.')
        style_button(btn_export_2d, bg='#4c8cca', fg='white')
        btn_export_2d.clicked.connect(self.export_2d_data)
        ctrl2d.addWidget(btn_export_2d)
        pnl2d_lay.addLayout(ctrl2d)

        # 2D map canvas
        self.canvas_2d = MplCanvas(self, figsize=(8, 6))
        self.fig_2d = self.canvas_2d.fig
        # Remove default single axes - we'll add our own
        self.fig_2d.clear()
        pnl2d_lay.addWidget(self.canvas_2d.with_toolbar(self), stretch=1)
        self.canvas_2d.mpl_connect('button_press_event', self._on_2d_click)
        splitter.addWidget(pnl2d)

        # --- Right side: Spectrum + Kinetics ---
        right_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # Spectrum panel
        pnl_s = QtWidgets.QGroupBox('Spectrum  (at selected delay)')
        p_s_lay = QtWidgets.QVBoxLayout(pnl_s)
        p_s_lay.setContentsMargins(5, 5, 5, 5)
        self.canvas_spec = MplCanvas(self, figsize=(5, 3))
        p_s_lay.addWidget(self.canvas_spec.with_toolbar(self), stretch=1)
        # Hook the navigation toolbar's Home / Back / Forward actions so
        # that pressing them also clears our application-side zoom
        # cache, otherwise the next redraw would restore the old zoom.
        self._hook_toolbar_reset(self.canvas_spec.toolbar,
                                 self._reset_spec_zoom)
        # Register pan/zoom capture; will be re-registered by every
        # _draw_spectrum (ax.clear() drops callbacks).
        self._wire_spec_zoom_callbacks()

        s_ctrl = QtWidgets.QHBoxLayout()
        # Delay control by INDEX, not by value.  The arrows on the
        # spinbox now step to the actual previous / next delay point
        # recorded during the experiment, with the current delay value
        # shown read-only next to it.  This is what the user asked for:
        # "increment by index, show the corresponding delay time".
        self.lbl_sel_t = make_label('Delay idx:', 'right')
        s_ctrl.addWidget(self.lbl_sel_t)
        self.ed_sel_t_idx = make_int_edit(0, minv=0, maxv=0)
        self.ed_sel_t_idx.setToolTip(
            'Index into the recorded delay array (use the up/down arrows '
            'to step to the previous / next measured delay point).')
        self.ed_sel_t_idx.valueChanged.connect(self._on_sel_t_idx_change)
        s_ctrl.addWidget(self.ed_sel_t_idx)
        # Read-only display of the actual delay value at the chosen index
        self.lbl_sel_t_val = QtWidgets.QLabel('—')
        self.lbl_sel_t_val.setMinimumWidth(110)
        self.lbl_sel_t_val.setStyleSheet(
            'border: 1px solid #b5b5b5; border-radius: 3px; '
            'padding: 2px 6px; background: #f7f7f7;')
        self.lbl_sel_t_val.setToolTip('Recorded delay time at this index.')
        s_ctrl.addWidget(self.lbl_sel_t_val)
        btn_pin_s = QtWidgets.QPushButton('Pin')
        btn_pin_s.clicked.connect(self.pin_spec)
        s_ctrl.addWidget(btn_pin_s)
        btn_clr_s = QtWidgets.QPushButton('Clear pins')
        btn_clr_s.clicked.connect(self.clear_spec_overlays)
        s_ctrl.addWidget(btn_clr_s)
        btn_exp_s = QtWidgets.QPushButton('Export pins')
        btn_exp_s.setToolTip(
            'Save all pinned spectra (one column per pinned delay) as CSV.')
        btn_exp_s.clicked.connect(self.export_spec_overlays)
        s_ctrl.addWidget(btn_exp_s)
        # (Reset-zoom button removed: the Home icon in matplotlib's
        # navigation toolbar above does the same thing.)
        s_ctrl.addStretch(1)
        p_s_lay.addLayout(s_ctrl)
        right_split.addWidget(pnl_s)

        # Kinetics panel
        pnl_k = QtWidgets.QGroupBox('Kinetics  (at selected wavelength)')
        p_k_lay = QtWidgets.QVBoxLayout(pnl_k)
        p_k_lay.setContentsMargins(5, 5, 5, 5)
        self.canvas_kin = MplCanvas(self, figsize=(5, 3))
        self.fig_kin = self.canvas_kin.fig
        self.fig_kin.clear()
        # Kinetics callbacks are wired inside _draw_kinetics, because that
        # method rebuilds the figure (1 or 2 axes depending on scale mode)
        # and the axes objects therefore change between draws.
        p_k_lay.addWidget(self.canvas_kin.with_toolbar(self), stretch=1)
        self._hook_toolbar_reset(self.canvas_kin.toolbar,
                                 self._reset_kin_zoom)
        k_ctrl = QtWidgets.QHBoxLayout()
        k_ctrl.addWidget(make_label('λ (nm):', 'right'))
        self.ed_sel_wl = make_double_edit(0.0, decimals=2)
        self.ed_sel_wl.valueChanged.connect(lambda v: self.set_sel_wl(v))
        k_ctrl.addWidget(self.ed_sel_wl)
        btn_pin_k = QtWidgets.QPushButton('Pin')
        btn_pin_k.clicked.connect(self.pin_kin)
        k_ctrl.addWidget(btn_pin_k)
        btn_clr_k = QtWidgets.QPushButton('Clear pins')
        btn_clr_k.clicked.connect(self.clear_kin_overlays)
        k_ctrl.addWidget(btn_clr_k)
        btn_exp_k = QtWidgets.QPushButton('Export pins')
        btn_exp_k.setToolTip(
            'Save all pinned kinetics (one column per pinned wavelength) '
            'as CSV.')
        btn_exp_k.clicked.connect(self.export_kin_overlays)
        k_ctrl.addWidget(btn_exp_k)
        # (Reset-zoom button removed: the Home icon in matplotlib's
        # navigation toolbar above does the same thing.)
        k_ctrl.addStretch(1)
        p_k_lay.addLayout(k_ctrl)
        right_split.addWidget(pnl_k)

        splitter.addWidget(right_split)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        # Wire right-click + axis-region click on the spectrum & kinetics
        # canvases (auto-scale menu + manual-range dialog).
        self._wire_panel_interactions()

    # ================================================================
    # Time-unit helpers (match MATLAB: tUnitAx / tUnitTxt / tUnitHdr)
    # ================================================================
    def t_unit_ax(self) -> str:
        """Time-unit string for axis labels (matplotlib uses mathtext)."""
        return r'$\mu$s' if self.time_unit == 'us' else 'ps'

    def t_unit_txt(self) -> str:
        """Time-unit string for plain UI labels (Unicode)."""
        return u'μs' if self.time_unit == 'us' else 'ps'

    def t_unit_hdr(self) -> str:
        """Time-unit string for CSV headers / filenames (ASCII)."""
        return 'us' if self.time_unit == 'us' else 'ps'

    def _on_time_unit_change(self, new_unit):
        if new_unit == self.time_unit:
            return
        self.time_unit = new_unit
        # Close all child windows — their static labels would be stale
        for attr in ('bg_fig', 'chirp_fig', 'crop_fig',
                     'load_fig', 'ga_fig', 'mask_fig',
                     'kfit_fig', 'lda_fig', 'mcr_fig', 'coh_fig'):
            d = getattr(self, attr)
            if d is not None:
                try:
                    d.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        # Kinetics x-axis label changes (ps↔μs); discard saved kinetics
        # zoom so the next draw re-fits.  Spectrum's x-axis is wavelength
        # and is unaffected, so we keep that zoom.
        self._kin_xlims = [None, None]
        self._kin_ylims = [None, None]
        self._refresh_time_unit_labels()
        self.update_all()

    def _refresh_time_unit_labels(self):
        u = self.t_unit_txt()
        self.lbl_split_threshold.setText(f'Split threshold ({u}):')
        self.lbl_view_tmin.setText(f'Display t_min ({u}):')
        self.lbl_view_tmax.setText(f't_max ({u}):')
        self.lbl_sel_t.setText(f'Delay ({u}):')

    # ================================================================
    # Colormap / Z-range
    # ================================================================
    def get_colormap_array(self, N: int = 256) -> np.ndarray:
        """Expose for GA dialog.  Uses shared ta_core palette logic."""
        return ta_core.get_colormap_array(self.map_colormap, N)

    def _compute_zlim(self):
        return compute_zlim(self.deltaA, self.map_z_min, self.map_z_max)

    def _on_colormap_change(self, name: str):
        self.map_colormap = name
        # Cmap-only change: skip the heavy full rebuild.
        self._update_map_cmap()

    def _on_z_change(self, which: str, val: float):
        if which == 'min':
            new_min = val
            new_max = self.ed_z_max.value()
        else:
            new_min = self.ed_z_min.value()
            new_max = val
        if new_min >= new_max:
            # Invalid - revert to current limits silently
            c_min, c_max = self._compute_zlim()
            self.ed_z_min.blockSignals(True)
            self.ed_z_max.blockSignals(True)
            self.ed_z_min.setValue(c_min)
            self.ed_z_max.setValue(c_max)
            self.ed_z_min.blockSignals(False)
            self.ed_z_max.blockSignals(False)
            return
        self.map_z_min = new_min
        self.map_z_max = new_max
        # Z-range-only change: skip the heavy full rebuild.
        self._update_map_clim()

    def reset_z_range(self):
        self.map_z_min = None
        self.map_z_max = None
        c_min, c_max = self._compute_zlim()
        self.ed_z_min.blockSignals(True)
        self.ed_z_max.blockSignals(True)
        self.ed_z_min.setValue(c_min)
        self.ed_z_max.setValue(c_max)
        self.ed_z_min.blockSignals(False)
        self.ed_z_max.blockSignals(False)
        self._update_map_clim()

    # ================================================================
    # Data loading
    # ================================================================
    def load_data(self):
        """Entry point for the toolbar's single Load Data button.

        Opens a small chooser dialog and dispatches to one of the three
        actual loading flows (standard file picker + auto-detect, manual
        X/Y/Z region picker, folder-based accumulation).
        """
        chooser = LoadDataChooserDialog(self)
        if chooser.exec_() != QtWidgets.QDialog.Accepted:
            return
        mode = chooser.mode
        if mode == 'standard':
            self.load_data_standard()
        elif mode == 'custom':
            self.load_data_custom()
        elif mode == 'accumulation':
            self.load_data_accumulate()

    def load_data_standard(self):
        """Auto-detected single-file load (or Load & Average when the
        user selects multiple files in the OS dialog)."""
        paths = ask_open_paths(self, 'Select TA data file(s)', multi=True)
        if not paths:
            return
        if len(paths) == 1:
            path = paths[0]
            try:
                wl, t, A = ta_core.parse_data_file(path)
            except Exception as e:
                warn_box(self, 'Load Error',
                         f'Failed to read file:\n{e}')
                return
            self.set_loaded_data(wl, t, A,
                                 f'Loaded: {os.path.basename(path)}',
                                 source_dir=os.path.dirname(path))
        else:
            # Multi-file → Load & Average window
            base = os.path.dirname(paths[0])
            names = [os.path.basename(p) for p in paths]
            self._open_dialog(LoadAverageDialog, 'load_fig', base, names)

    def load_data_custom(self):
        """Open the Excel-like preview dialog so the user can hand-pick the
        X / Y / Z regions of a CSV whose layout the auto-detector does not
        recognise."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select CSV file (custom layout)', '',
            'CSV (*.csv);;All files (*)')
        if not path:
            return
        dlg = CustomLoadDialog(self, file_path=path)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        result = dlg.get_result()
        if result is None:
            return
        wl, t, A = result
        self.set_loaded_data(
            wl, t, A,
            f'Loaded (custom): {os.path.basename(path)}',
            source_dir=os.path.dirname(path))

    def load_data_accumulate(self):
        """Open the folder-based accumulation dialog: preview each repeat
        measurement's 2D map and average only the checked files."""
        self._open_dialog(AccumulationDialog, 'accum_fig', '')

    def set_loaded_data(self, wl: np.ndarray, t: np.ndarray,
                        A: np.ndarray, source_desc: str,
                        source_dir: str = ''):
        # Snapshot as original + current working copies
        self.original_wavelength = wl.copy()
        self.original_delay = t.copy()
        self.original_deltaA = A.copy()
        self.data_source_desc = source_desc
        # Remember where the raw file(s) came from — every export
        # dialog opens here by default (empty string = fallback to
        # Qt's default = process cwd).
        self.data_source_dir = source_dir or ''

        self.wavelength = wl.copy()
        self.delay = t.copy()
        self.deltaA_raw = A.copy()
        self.deltaA = A.copy()

        # Reset all correction state
        self.bg_applied = False
        self.bg_spectrum = None
        self.bg_n = 0
        # Loading a fresh sample means whatever pure-solvent reference
        # was previously loaded most likely doesn't match.  Drop it.
        self.sub_irf_applied = False
        self.sub_irf_scale = 1.0
        self.sub_irf_solv_wl = None
        self.sub_irf_solv_t = None
        self.sub_irf_solv_data = None
        self.sub_irf_solv_path = ''
        self.sub_irf_aligned = None
        self.chirp_applied = False
        self.chirp_pts = np.zeros((0, 2))
        self.chirp_params = None
        self.chirp_fit_rms = float('nan')
        self.chirp_view_t_min = float(t[0])
        self.chirp_view_t_max = float(min(2.0, t[-1]))
        self.ridge_wl = None
        self.ridge_t = None
        self.tZeroShift = 0.0
        self.masked_regions = []
        self.specOverlays = []
        self.kinOverlays = []
        self.crop_bounds = None
        self.resample_enabled = False
        self.resample_dx = 1.0
        self.resample_mode = 'average'
        self.resample_info = None
        self._crop_wl_pre_resample = None
        # New data ⇒ any previous GA result refers to the OLD matrix
        # and would fail shape checks / mislead the user. Clear it.
        self._clear_ga_state()
        # New data → drop any stored spectrum/kinetics pan/zoom so the
        # first draw auto-fits to the new extent.
        self._reset_all_panel_zoom()

        # Default selection: middle wavelength, first non-negative delay
        self.selWL = float(wl[len(wl) // 2])
        idx0 = int(np.argmax(t >= 0)) if np.any(t >= 0) else len(t) // 2
        self.selT = float(t[idx0])
        self._selT_idx = idx0

        self.ed_sel_wl.blockSignals(True)
        self.ed_sel_wl.setRange(float(wl.min()), float(wl.max()))
        self.ed_sel_wl.setValue(self.selWL)
        self.ed_sel_wl.blockSignals(False)
        self._sync_sel_t_widgets()

        # Reset main display range
        self.main_view_tmin = float(t[0])
        self.main_view_tmax = float(t[-1])
        self.ed_view_tmin.blockSignals(True)
        self.ed_view_tmax.blockSignals(True)
        self.ed_view_tmin.setValue(self.main_view_tmin)
        self.ed_view_tmax.setValue(self.main_view_tmax)
        self.ed_view_tmin.blockSignals(False)
        self.ed_view_tmax.blockSignals(False)

        # Reset Z-range to auto
        self.map_z_min = None
        self.map_z_max = None
        cl = float(np.nanmax(np.abs(A)))
        if not np.isfinite(cl) or cl == 0:
            cl = 1.0
        self.ed_z_min.blockSignals(True)
        self.ed_z_max.blockSignals(True)
        self.ed_z_min.setValue(-cl)
        self.ed_z_max.setValue(cl)
        self.ed_z_min.blockSignals(False)
        self.ed_z_max.blockSignals(False)

        self._update_status_label()
        self.update_all()

    def _update_status_label(self):
        if self.wavelength is None:
            self.lbl_status.setText('No data loaded')
            return
        base = self.data_source_desc or 'Data'
        tag = '   [cropped]' if self.is_cropped() else ''
        if self.resample_enabled:
            tag += f'   [{self._resample_tag()}]'
        self.lbl_status.setText(
            f'{base}   ({len(self.wavelength)} λ x {len(self.delay)} t){tag}')

    # ================================================================
    # Correction pipeline (raw -> BG -> chirp -> masks)
    # ================================================================
    def recompute(self):
        """Re-apply the whole correction chain from deltaA_raw → deltaA.

        Order: BG → chirp → solvent-IRF subtraction → wavelength masks.

        Solvent subtraction happens AFTER chirp so the alignment between
        sample and solvent is done in the chirp-corrected time axis.
        The cached ``sub_irf_aligned`` is itself already chirp-corrected
        (see ``realign_solvent``), so both terms in
        ``sample − scale·solvent`` are in the same frame and the IRF
        feature lands at t=0 for every λ.
        """
        if self.deltaA_raw is None:
            return
        data = self.deltaA_raw.copy()
        if self.bg_applied and self.bg_spectrum is not None:
            data = data - self.bg_spectrum[:, None]
        if self.chirp_applied and self.chirp_params is not None:
            data = ta_core.apply_chirp_shift(
                data, self.delay.astype(float),
                self.wavelength.astype(float), self.chirp_params)
        # Solvent (pure-solvent) IRF / coherent-artifact subtraction.
        # ``sub_irf_aligned`` was prepared by ``realign_solvent`` — it's
        # already resampled onto the current (λ, t) grid AND already
        # chirp-corrected if the sample is.  We just scale & subtract.
        # Cells where the solvent didn't cover the sample's grid are
        # left untouched (see apply_solvent_subtraction).
        if (self.sub_irf_applied
                and self.sub_irf_aligned is not None
                and self.sub_irf_aligned.shape == data.shape):
            data = ta_core.apply_solvent_subtraction(
                data, self.sub_irf_aligned, self.sub_irf_scale)
        # Apply wavelength masks
        for wl1, wl2, mode in self.masked_regions:
            m = (self.wavelength >= wl1) & (self.wavelength <= wl2)
            if not m.any():
                continue
            if mode == 'nan':
                data[m, :] = np.nan
            else:
                data[m, :] = 0.0
        self.deltaA = data

    def realign_solvent(self):
        """Resample the stored pure-solvent ΔA onto the current (λ, t),
        then apply the same chirp correction the sample is using.

        Called automatically after any operation that changes the
        sample grid (load / crop / zero-time shift) or the chirp state
        (apply / revert chirp), so the cached ``sub_irf_aligned`` stays
        in lockstep with what ``recompute()`` will subtract from.

        Why apply chirp to the solvent too:
            Chirp is a property of the *optics* (GVD through the
            cell + spectrograph), so for a sample and its pure
            solvent measured back-to-back on the same instrument
            the chirp curve is essentially identical.  The sample
            in ``recompute()`` goes  raw → -BG → chirp_shift →
            -scale·solvent → mask.  For the IRF feature to actually
            land at t≈0 for every λ in BOTH datasets — which is the
            whole point of subtracting them — the solvent must be in
            the chirp-corrected frame as well.  If we only resampled
            it and skipped chirp, the IRF in the solvent reference
            would still sit at t₀(λ) ≠ 0 while the sample's IRF has
            already been moved to t=0, and the subtraction would
            actually *add* artifacts at both positions instead of
            removing them.

        BG correction is intentionally NOT applied to the solvent:
            the bg_spectrum was estimated from pre-trigger frames of
            the *sample* and characterises whatever offset / noise
            floor that scan had.  The solvent scan has its own offset
            (usually negligible since there's no excited-state signal)
            and is not the same number.  Forcing it here would only
            inject a bias.

        Safe to call when no solvent has been loaded — it's a no-op.
        """
        if (self.sub_irf_solv_wl is None or self.sub_irf_solv_data is None
                or self.wavelength is None or self.delay is None):
            self.sub_irf_aligned = None
            return
        try:
            # With resampling on, align the raw solvent onto the cropped
            # grid the sample had *before* resampling, then bin it with
            # exactly the sample's grouping so both are smoothed alike.
            resampled = self.resample_enabled and self.resample_info is not None
            aligned = ta_core.align_solvent_to_sample(
                self.sub_irf_solv_wl, self.sub_irf_solv_t,
                self.sub_irf_solv_data,
                self._crop_wl_pre_resample if resampled else self.wavelength,
                self.delay)
            if resampled:
                aligned = ta_core.apply_bin_groups(aligned,
                                                   self.resample_info)
            # Mirror the sample's chirp correction onto the solvent.
            # Use the same chirp_params + same (wavelength, delay)
            # axes so each row shifts by exactly t₀(λ_i), matching the
            # sample row-for-row.
            if (self.chirp_applied
                    and self.chirp_params is not None
                    and aligned is not None):
                aligned = ta_core.apply_chirp_shift(
                    aligned, self.delay.astype(float),
                    self.wavelength.astype(float),
                    self.chirp_params)
            self.sub_irf_aligned = aligned
        except Exception:
            # Defensive: a corrupted solvent should not crash the pipeline
            self.sub_irf_aligned = None

    def get_chirp_base_data(self):
        """Data after BG but before chirp — what the Chirp dialog draws."""
        base = self.deltaA_raw.copy()
        if self.bg_applied and self.bg_spectrum is not None:
            base = base - self.bg_spectrum[:, None]
        return base

    def is_cropped(self) -> bool:
        """True when the λ/t range is smaller than the loaded data.

        Judged on the grid *before* wavelength resampling, so a
        full-range resample does not count as a crop."""
        if self.original_wavelength is None:
            return False
        wl = (self._crop_wl_pre_resample if self.resample_enabled
              else self.wavelength)
        return (len(wl) != len(self.original_wavelength)
                or len(self.delay) != len(self.original_delay))

    def is_modified(self) -> bool:
        """Cropped or wavelength-resampled — i.e. Revert has an effect."""
        return self.is_cropped() or self.resample_enabled

    def _resample_tag(self) -> str:
        mode = 'avg' if self.resample_mode == 'average' else 'decimate'
        return f'resampled Δλ={self.resample_dx:g} nm, {mode}'

    def update_all(self):
        """Recompute corrections, then redraw all three panels."""
        # Keep the pure-solvent reference resampled onto the current
        # (λ, t) grid; recompute then just multiplies by the scale and
        # subtracts.  realign is a no-op if no solvent is loaded.
        self.realign_solvent()
        self.recompute()
        self._draw_map_2d()
        self._draw_spectrum()
        self._draw_kinetics()

    # ================================================================
    # Drawing: 2D map
    # ================================================================
    def _kinetics_default_xlim(self, t_sub: np.ndarray, scale_str: str):
        """Default x-limits for the kinetics panel: full data extent
        (log-safe).  Deliberately independent of ``main_view_tmin/tmax``
        — the Display t-range controls the 2D map only; the kinetics
        panel always plots the whole time axis unless the user
        explicitly sets a manual range from its own auto-scale menu."""
        data_lo = float(t_sub.min())
        data_hi = float(t_sub.max())
        lo, hi = data_lo, data_hi
        if scale_str == 'log':
            pos = t_sub[t_sub > 0]
            min_pos = float(pos.min()) if pos.size else 1e-12
            lo = max(lo, min_pos)
        if hi <= lo:
            lo, hi = data_lo, data_hi
            if scale_str == 'log':
                pos = t_sub[t_sub > 0]
                if pos.size:
                    lo = float(pos.min())
        return lo, hi

    def _clip_view_to_mask(self, t_sub: np.ndarray, scale_str: str):
        """Clip user's main_view_[tmin/tmax] to data in t_sub + scale limits."""
        tmin = self.main_view_tmin
        tmax = self.main_view_tmax
        use_view = (tmin is not None and tmax is not None and tmin < tmax)
        data_lo = float(t_sub.min())
        data_hi = float(t_sub.max())
        if use_view:
            lo = max(data_lo, tmin)
            hi = min(data_hi, tmax)
        else:
            lo, hi = data_lo, data_hi
        if scale_str == 'log':
            pos = t_sub[t_sub > 0]
            min_pos = float(pos.min()) if pos.size else 1e-12
            lo = max(lo, min_pos)
        if hi <= lo:
            lo, hi = data_lo, data_hi
            if scale_str == 'log':
                pos = t_sub[t_sub > 0]
                if pos.size:
                    lo = float(pos.min())
        if hi <= lo:
            hi = lo + max(abs(lo) * 1e-10, 0.1)
        return lo, hi

    def _map_title(self, scale_str: str = '') -> str:
        parts = []
        if self.bg_applied:
            parts.append(f'BG(N={self.bg_n})')
        if self.chirp_applied:
            parts.append('chirp(Sellmeier)')
        base = ', '.join(parts) if parts else 'raw'
        if self.is_cropped():
            base += ', cropped'
        if self.resample_enabled:
            base += ', resampled'
        if scale_str:
            return rf'2D $\Delta$A  ({base}, {scale_str} scale)'
        return rf'2D $\Delta$A  ({base})'

    def _compute_map_layout_sig(self):
        """A hashable tuple that uniquely identifies the *structure* of
        the 2D map.  When this is unchanged we can skip ``fig.clear()``
        and just update existing artists in place.

        Anything that would change the number of axes, the data extent,
        or the y-axis scale must be part of this signature.
        """
        if self.deltaA is None or self.wavelength is None:
            return None
        return (
            self.delay_scale_mode,
            float(self.split_threshold),
            len(self.wavelength),
            len(self.delay),
            float(self.wavelength[0]), float(self.wavelength[-1]),
            float(self.delay[0]), float(self.delay[-1]),
            self.time_unit,
            tuple(self.masked_regions),  # mask changes data shape of NaN map
            self.bg_applied, self.chirp_applied, self.is_cropped(),
            self.resample_enabled,
        )

    def _draw_map_2d(self):
        """Render the 2D ΔA map.

        Uses incremental update (set_array / set_clim / set_cmap) when
        the figure structure is unchanged since the last full rebuild
        (~1 ms instead of ~85 ms for a 2136 × 192 dataset).  Falls back
        to full rebuild whenever the signature changes (scale mode,
        crop, time unit, …).
        """
        if self.deltaA is None:
            return
        new_sig = self._compute_map_layout_sig()
        if (new_sig is not None and new_sig == self._map_layout_sig
                and self._map_images and self._ax2d_list):
            self._update_map_artists_in_place()
            return
        self._draw_map_2d_full(new_sig)

    def _draw_map_2d_full(self, new_sig):
        """Heavy path: rebuild the entire figure from scratch."""
        t = self.delay
        thr = self.split_threshold
        mode = self.delay_scale_mode

        if mode == 'linear':
            masks = [np.ones(t.shape, bool)]
            scales = ['linear']
        elif mode == 'log':
            pos = t > 0
            if not pos.any():
                pos = np.ones(t.shape, bool)
            masks = [pos]
            scales = ['log']
        else:  # split
            mask_top = t >= thr
            mask_bot = t <= thr
            masks = [mask_top, mask_bot]
            scales = ['log', 'linear']

        c_min, c_max = self._compute_zlim()
        cmap = ListedColormap(self.get_colormap_array())

        # Rebuild the figure fresh each time — simpler than tracking state
        self.fig_2d.clear()
        self._ax2d_list = []
        self._hVLine = []
        self._hHLine = []

        if mode == 'split':
            gs = self.fig_2d.add_gridspec(
                2, 2, height_ratios=[2, 1], width_ratios=[1, 0.04],
                wspace=0.05, hspace=0.05)
            ax_top = self.fig_2d.add_subplot(gs[0, 0])
            ax_bot = self.fig_2d.add_subplot(gs[1, 0])
            cbar_ax = self.fig_2d.add_subplot(gs[:, 1])
            self._ax2d_list = [ax_top, ax_bot]
            self._cbar_ax = cbar_ax
        else:
            self._ax2d_list = [self.fig_2d.add_subplot(1, 1, 1)]
            self._cbar_ax = None

        first_im = None
        self._map_images = []  # filled in by the loop below
        for k, (ax, mask, scale) in enumerate(
                zip(self._ax2d_list, masks, scales)):
            t_sub = t[mask]
            if t_sub.size == 0:
                self._map_images.append(None)
                continue
            d_sub = self.deltaA[:, mask]

            # NaN-aware rendering: make NaN cells transparent
            Z = np.ma.masked_invalid(d_sub.T)
            im = ax.pcolormesh(self.wavelength, t_sub, Z,
                               cmap=cmap, vmin=c_min, vmax=c_max,
                               shading='nearest')
            self._map_images.append(im)
            if first_im is None:
                first_im = im
            ax.set_yscale(scale)
            ylo, yhi = self._clip_view_to_mask(t_sub, scale)
            ax.set_ylim(ylo, yhi)
            ax.set_xlim(float(self.wavelength.min()),
                        float(self.wavelength.max()))

            # Labels: bottom-most panel keeps the x-axis label
            if k == len(self._ax2d_list) - 1:
                ax.set_xlabel('Wavelength (nm)')
            else:
                ax.tick_params(labelbottom=False)
            ax.set_ylabel(f'Delay time ({self.t_unit_ax()})')
            if k == 0:
                ax.set_title(self._map_title(scale))

            # Crosshair
            vln = ax.axvline(self.selWL, color='w', linestyle='--',
                             linewidth=1.0)
            hln = ax.axhline(self.selT, color='w', linestyle='--',
                             linewidth=1.0)
            self._hVLine.append(vln)
            self._hHLine.append(hln)

        # Colorbar
        if first_im is not None:
            if self._cbar_ax is not None:
                cb = self.fig_2d.colorbar(first_im, cax=self._cbar_ax)
            else:
                cb = self.fig_2d.colorbar(first_im, ax=self._ax2d_list[0])
            cb.set_label(r'$\Delta$A')
            self._main_cb = cb

        # Cache the structure so subsequent redraws (Z range, colormap,
        # data tweaks) can take the fast incremental path.
        self._map_layout_sig = new_sig
        self.canvas_2d.draw_idle()

    def _update_map_artists_in_place(self):
        """Fast path for redraws when only data / clim / cmap / view-
        range changed.

        About 80× faster than _draw_map_2d_full for a 2136 × 192 map
        because pcolormesh, colorbar, and axes setup are all reused.
        """
        t = self.delay
        thr = self.split_threshold
        mode = self.delay_scale_mode
        if mode == 'linear':
            masks = [np.ones(t.shape, bool)]
        elif mode == 'log':
            pos = t > 0
            if not pos.any():
                pos = np.ones(t.shape, bool)
            masks = [pos]
        else:
            masks = [t >= thr, t <= thr]

        c_min, c_max = self._compute_zlim()
        cmap = ListedColormap(self.get_colormap_array())
        for im, mask in zip(self._map_images, masks):
            if im is None:
                continue
            d_sub = self.deltaA[:, mask]
            Z = np.ma.masked_invalid(d_sub.T)
            # set_array on a QuadMesh wants a 1-D ravelled array on
            # newer matplotlibs; .ravel() works on either.
            im.set_array(Z.ravel())
            im.set_clim(c_min, c_max)
            im.set_cmap(cmap)
        # Refresh y-limits on each panel — this is what makes
        # "Display t_min / t_max" actually take effect on the 2D map
        # without forcing a full figure rebuild.
        for ax, mask in zip(self._ax2d_list, masks):
            t_sub = t[mask]
            if t_sub.size == 0:
                continue
            scale = ax.get_yscale()
            ylo, yhi = self._clip_view_to_mask(t_sub, scale)
            ax.set_ylim(ylo, yhi)
        # Refresh title (e.g. masked-region count may have shifted) on
        # the top axes only.
        if self._ax2d_list:
            scale_top = self._ax2d_list[0].get_yscale()
            self._ax2d_list[0].set_title(self._map_title(scale_top))
        self.canvas_2d.draw_idle()

    def _update_map_clim(self):
        """Public entry point: only Z-range changed."""
        if not self._map_images:
            self._draw_map_2d()
            return
        c_min, c_max = self._compute_zlim()
        for im in self._map_images:
            if im is not None:
                im.set_clim(c_min, c_max)
        self.canvas_2d.draw_idle()

    def _update_map_cmap(self):
        """Public entry point: only colormap changed."""
        if not self._map_images:
            self._draw_map_2d()
            return
        cmap = ListedColormap(self.get_colormap_array())
        for im in self._map_images:
            if im is not None:
                im.set_cmap(cmap)
        self.canvas_2d.draw_idle()

    # ================================================================
    # Drawing: spectrum panel
    # ================================================================
    def _hook_toolbar_reset(self, toolbar, refit_callback):
        """Wrap toolbar.home / back / forward so that after matplotlib
        restores its cached view, our ``refit_callback`` re-applies the
        auto-scale rules on top of it (fitting auto-enabled axes to the
        valid data within the just-restored viewport)."""
        for action_name in ('home', 'back', 'forward'):
            original = getattr(toolbar, action_name, None)
            if original is None:
                continue

            def make_wrapped(orig=original, cb=refit_callback):
                def wrapped(*args, **kwargs):
                    result = orig(*args, **kwargs)
                    try:
                        cb()
                    except Exception:
                        pass
                    return result
                return wrapped

            try:
                setattr(toolbar, action_name, make_wrapped())
            except Exception:
                pass

    def _wire_spec_zoom_callbacks(self):
        """Register pan/zoom capture on the spectrum axes.

        Note: ``ax.clear()`` discards every callback registered on the
        axes, so this must be called right after each clear, not just
        once at startup.
        """
        ax = self.canvas_spec.ax
        ax.callbacks.connect('xlim_changed', self._on_spec_xlim_changed)
        ax.callbacks.connect('ylim_changed', self._on_spec_ylim_changed)

    def _compute_spec_layout_sig(self):
        """Identifies when the spectrum panel needs a full rebuild
        rather than just updating the current-trace ydata.
        """
        if self.deltaA is None or self.wavelength is None:
            return None
        return (
            len(self.wavelength),
            float(self.wavelength[0]), float(self.wavelength[-1]),
            tuple(self.specOverlays),
            self.time_unit,
        )

    def _draw_spectrum(self):
        if self.deltaA is None:
            return
        new_sig = self._compute_spec_layout_sig()
        if (new_sig is not None and new_sig == self._spec_layout_sig
                and self._spec_line_current is not None):
            self._update_spectrum_in_place()
            return
        self._draw_spectrum_full(new_sig)

    def _update_spectrum_in_place(self):
        """Fast path: only the current-delay trace's ydata changes.

        Overlays were already drawn at fixed delays during the last
        full rebuild and don't move when the user changes self.selT.
        """
        self._suppress_zoom_save = True
        try:
            idx = int(np.argmin(np.abs(self.delay - self.selT)))
            ydata = self.deltaA[:, idx]
            self._spec_line_current.set_ydata(ydata)
            label = (f't = {self.delay[idx]:.3g} {self.t_unit_ax()} '
                     f'(current)')
            self._spec_line_current.set_label(label)

            ax = self.canvas_spec.ax
            # Update title or legend's current-line text
            if not self.specOverlays:
                ax.set_title(label)
            else:
                # Refresh the existing legend so the "current" entry's
                # text changes too (legend is not auto-linked to label).
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()
                ax.legend(loc='best', fontsize=8)

            # Auto-Y ON → refit y within the current xlim (viewport-
            # aware).  Auto-X ON → refit x within the current ylim.
            # OFF for an axis → user's manual limit is preserved.
            self._apply_spec_limits(ax, keep_current_view=True)
            self.canvas_spec.draw_idle()
        finally:
            self._suppress_zoom_save = False

    def _apply_spec_limits(self, ax, keep_current_view: bool = False):
        """Set xlim/ylim on the spectrum axes according to the current
        auto-X / auto-Y flags and stored manual limits.

        ``keep_current_view`` (fast-path use): don't reset a manual axis
        to its stored value — leave whatever matplotlib currently shows
        so that only the auto-fitted axis moves.
        """
        wl_lo = float(self.wavelength.min())
        wl_hi = float(self.wavelength.max())

        ax_x = self._spec_auto_x
        ay_y = self._spec_auto_y

        if ax_x and ay_y:
            # Full data extent both ways
            ax.set_xlim(wl_lo, wl_hi)
            ax.relim(visible_only=False)
            ax.autoscale_view(scalex=False, scaley=True)
            return

        if ax_x and not ay_y:
            # Fix Y first (manual), then fit X within that Y window
            if not keep_current_view:
                if self._spec_ylim is not None:
                    ax.set_ylim(self._spec_ylim)
            xr = self._visible_x_range(ax, ax.get_ylim())
            ax.set_xlim(xr if xr is not None else (wl_lo, wl_hi))
            return

        if not ax_x and ay_y:
            # Fix X first (manual), then fit Y within that X window
            if not keep_current_view:
                if self._spec_xlim is not None:
                    ax.set_xlim(self._spec_xlim)
                else:
                    ax.set_xlim(wl_lo, wl_hi)
            yr = self._visible_y_range(ax, ax.get_xlim())
            if yr is not None:
                ax.set_ylim(yr)
            else:
                ax.relim(visible_only=False)
                ax.autoscale_view(scalex=False, scaley=True)
            return

        # Both manual
        if not keep_current_view:
            if self._spec_xlim is not None:
                ax.set_xlim(self._spec_xlim)
            else:
                ax.set_xlim(wl_lo, wl_hi)
            if self._spec_ylim is not None:
                ax.set_ylim(self._spec_ylim)

    def _draw_spectrum_full(self, new_sig):
        """Heavy path: rebuild axes contents from scratch."""
        # Avoid recording our own programmatic set_xlim/set_ylim calls
        # below as if the user had panned/zoomed.
        self._suppress_zoom_save = True
        try:
            ax = self.canvas_spec.ax
            ax.clear()
            # ax.clear() also wipes all registered callbacks, so re-wire
            # zoom capture before any further set_xlim/set_ylim calls.
            self._wire_spec_zoom_callbacks()

            n_ov = len(self.specOverlays)
            cmap = ta_core._mpl_cm.get_cmap('tab10', max(10, n_ov + 1))

            leg_handles = []
            leg_labels = []
            self._spec_lines_overlay = []
            for k, t_val in enumerate(self.specOverlays):
                idx = int(np.argmin(np.abs(self.delay - t_val)))
                h, = ax.plot(self.wavelength, self.deltaA[:, idx], '-',
                             color=cmap(k % 10), linewidth=1.0)
                leg_handles.append(h)
                self._spec_lines_overlay.append(h)
                leg_labels.append(
                    f't = {self.delay[idx]:.3g} {self.t_unit_ax()}')

            idx = int(np.argmin(np.abs(self.delay - self.selT)))
            h, = ax.plot(self.wavelength, self.deltaA[:, idx], 'b-',
                         linewidth=1.6)
            leg_handles.append(h)
            self._spec_line_current = h
            label_current = (f't = {self.delay[idx]:.3g} '
                             f'{self.t_unit_ax()} (current)')
            h.set_label(label_current)
            leg_labels.append(label_current)

            ax.axhline(0, color='k', linestyle=':')
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel(r'$\Delta$A')
            ax.grid(True)

            # Apply view limits via per-axis auto-scale rules.
            self._apply_spec_limits(ax, keep_current_view=False)

            if len(leg_handles) > 1:
                # Set labels on the overlay handles too (used by legend)
                for h, lbl in zip(self._spec_lines_overlay, leg_labels[:-1]):
                    h.set_label(lbl)
                ax.legend(loc='best', fontsize=8)
            else:
                ax.set_title(leg_labels[-1])

            # Cache structure so subsequent selT changes can use the
            # incremental update path.
            self._spec_layout_sig = new_sig
            self.canvas_spec.draw_idle()
        finally:
            self._suppress_zoom_save = False

    # ================================================================
    # Drawing: kinetics panel (handles linear/log/split modes)
    # ================================================================
    def _compute_kin_layout_sig(self):
        """Identifies when the kinetics panel needs a full rebuild
        rather than just updating the current-trace ydata.
        """
        if self.deltaA is None or self.delay is None:
            return None
        return (
            self.delay_scale_mode,
            float(self.split_threshold) if self.delay_scale_mode == 'split'
                else None,
            len(self.delay),
            float(self.delay[0]), float(self.delay[-1]),
            tuple(self.kinOverlays),
            self.time_unit,
        )

    def _draw_kinetics(self):
        if self.deltaA is None:
            return
        new_sig = self._compute_kin_layout_sig()
        n_panels_expected = 2 if self.delay_scale_mode == 'split' else 1
        cur_axes = self.fig_kin.axes
        if (new_sig is not None and new_sig == self._kin_layout_sig
                and self._kin_line_current[0] is not None
                and len(cur_axes) == n_panels_expected):
            self._update_kinetics_in_place()
            return
        self._draw_kinetics_full(new_sig)

    def _update_kinetics_in_place(self):
        """Fast path: only the current-wavelength trace's ydata changes."""
        self._suppress_zoom_save = True
        try:
            mode = self.delay_scale_mode
            if mode == 'linear':
                masks = [np.ones(self.delay.shape, bool)]
            elif mode == 'log':
                pos = self.delay > 0
                if not pos.any():
                    pos = np.ones(self.delay.shape, bool)
                masks = [pos]
            else:  # split
                thr = self.split_threshold
                masks = [self.delay <= thr, self.delay >= thr]

            idx = int(np.argmin(np.abs(self.wavelength - self.selWL)))
            label_current = (f'λ = {self.wavelength[idx]:.1f} nm '
                             f'(current)')
            axes = self.fig_kin.axes
            for k, (m, line) in enumerate(zip(masks, self._kin_line_current)):
                if line is None:
                    continue
                line.set_ydata(self.deltaA[idx, m])
                line.set_label(label_current)
                if k < len(axes):
                    ax = axes[k]
                    if k == 0:
                        if not self.kinOverlays:
                            ax.set_title(label_current)
                        else:
                            leg = ax.get_legend()
                            if leg is not None:
                                leg.remove()
                            ax.legend(loc='best', fontsize=8)
                    t_sub = self.delay[m]
                    self._apply_kin_limits(
                        ax, k, t_sub, ax.get_xscale(),
                        keep_current_view=True)
            self.canvas_kin.draw_idle()
        finally:
            self._suppress_zoom_save = False

    def _apply_kin_limits(self, ax, panel_idx: int, t_sub,
                          scale_str: str,
                          keep_current_view: bool = False):
        """Kinetics counterpart of ``_apply_spec_limits``: per-sub-axes
        auto-X / auto-Y flags with visible-viewport refit."""
        auto_x = (self._kin_auto_x[panel_idx]
                  if panel_idx < len(self._kin_auto_x) else True)
        auto_y = (self._kin_auto_y[panel_idx]
                  if panel_idx < len(self._kin_auto_y) else True)

        default_xlim = self._kinetics_default_xlim(t_sub, scale_str)

        if auto_x and auto_y:
            ax.set_xlim(default_xlim)
            ax.relim(visible_only=False)
            ax.autoscale_view(scalex=False, scaley=True)
            return

        if auto_x and not auto_y:
            if not keep_current_view:
                if (panel_idx < len(self._kin_ylims)
                        and self._kin_ylims[panel_idx] is not None):
                    ax.set_ylim(self._kin_ylims[panel_idx])
            xr = self._visible_x_range(ax, ax.get_ylim())
            ax.set_xlim(xr if xr is not None else default_xlim)
            return

        if not auto_x and auto_y:
            if not keep_current_view:
                if (panel_idx < len(self._kin_xlims)
                        and self._kin_xlims[panel_idx] is not None):
                    ax.set_xlim(self._kin_xlims[panel_idx])
                else:
                    ax.set_xlim(default_xlim)
            yr = self._visible_y_range(ax, ax.get_xlim())
            if yr is not None:
                ax.set_ylim(yr)
            else:
                ax.relim(visible_only=False)
                ax.autoscale_view(scalex=False, scaley=True)
            return

        # Both manual
        if not keep_current_view:
            if (panel_idx < len(self._kin_xlims)
                    and self._kin_xlims[panel_idx] is not None):
                ax.set_xlim(self._kin_xlims[panel_idx])
            else:
                ax.set_xlim(default_xlim)
            if (panel_idx < len(self._kin_ylims)
                    and self._kin_ylims[panel_idx] is not None):
                ax.set_ylim(self._kin_ylims[panel_idx])

    def _draw_kinetics_full(self, new_sig):
        # Block our own programmatic limit-setting from being recorded
        # as user pan/zoom in the callback below.
        self._suppress_zoom_save = True
        try:
            n_ov = len(self.kinOverlays)
            cmap = ta_core._mpl_cm.get_cmap('tab10', max(10, n_ov + 1))

            mode = self.delay_scale_mode
            if mode == 'linear':
                masks = [np.ones(self.delay.shape, bool)]
                scales = ['linear']
            elif mode == 'log':
                pos = self.delay > 0
                if not pos.any():
                    pos = np.ones(self.delay.shape, bool)
                masks = [pos]
                scales = ['log']
            else:  # split
                thr = self.split_threshold
                masks = [self.delay <= thr, self.delay >= thr]
                scales = ['linear', 'log']

            self.fig_kin.clear()
            n_panels = len(masks)
            if n_panels == 1:
                axes = [self.fig_kin.add_subplot(1, 1, 1)]
            else:
                gs = self.fig_kin.add_gridspec(1, 2, width_ratios=[1, 2],
                                                wspace=0.25)
                axes = [self.fig_kin.add_subplot(gs[0, 0]),
                        self.fig_kin.add_subplot(gs[0, 1])]

            # Re-wire pan/zoom capture for each freshly-built axes.  The
            # axes objects are new every redraw (we use fig.clear()), so
            # callbacks registered in __init__ would be lost.
            for k, ax in enumerate(axes):
                ax.callbacks.connect(
                    'xlim_changed',
                    lambda a, idx=k: self._on_kin_xlim_changed(a, idx))
                ax.callbacks.connect(
                    'ylim_changed',
                    lambda a, idx=k: self._on_kin_ylim_changed(a, idx))

            # Reset the artist caches before re-populating
            self._kin_lines_overlay = [[], []]
            self._kin_line_current = [None, None]

            for k, (ax, m, scale) in enumerate(zip(axes, masks, scales)):
                t_sub = self.delay[m]
                if t_sub.size == 0:
                    continue
                leg_h, leg_l = [], []
                for j, wl_val in enumerate(self.kinOverlays):
                    idx = int(np.argmin(np.abs(self.wavelength - wl_val)))
                    h, = ax.plot(t_sub, self.deltaA[idx, m], '-',
                                 color=cmap(j % 10), linewidth=1.0)
                    self._kin_lines_overlay[k].append(h)
                    label = f'λ = {self.wavelength[idx]:.1f} nm'
                    h.set_label(label)
                    if k == 0:
                        leg_h.append(h)
                        leg_l.append(label)
                idx = int(np.argmin(np.abs(self.wavelength - self.selWL)))
                h, = ax.plot(t_sub, self.deltaA[idx, m], 'r-', linewidth=1.6)
                self._kin_line_current[k] = h
                label_current = (f'λ = {self.wavelength[idx]:.1f} nm '
                                 f'(current)')
                h.set_label(label_current)
                if k == 0:
                    leg_h.append(h)
                    leg_l.append(label_current)
                ax.axhline(0, color='k', linestyle=':')
                ax.set_xlabel(f'Delay time ({self.t_unit_ax()})')
                ax.set_ylabel(r'$\Delta$A')
                ax.grid(True)
                ax.set_xscale(scale)

                # Apply view limits via per-axis (per-sub-axes) auto-scale.
                self._apply_kin_limits(ax, k, t_sub, scale,
                                       keep_current_view=False)

                if k == 0 and len(leg_h) > 1:
                    ax.legend(leg_h, leg_l, loc='best', fontsize=8)
                if k == 0 and n_panels == 1:
                    ax.set_title(f'λ = {self.wavelength[idx]:.1f} nm')

            # Cache structure so subsequent selWL changes can use the
            # incremental update path.
            self._kin_layout_sig = new_sig
            self.canvas_kin.draw_idle()
        finally:
            self._suppress_zoom_save = False

    # ================================================================
    # Crosshair + interaction
    # ================================================================
    def _update_crosshairs(self):
        for vln in self._hVLine:
            try:
                vln.set_xdata([self.selWL, self.selWL])
            except Exception:
                pass
        for hln in self._hHLine:
            try:
                hln.set_ydata([self.selT, self.selT])
            except Exception:
                pass
        self.canvas_2d.draw_idle()

    def _on_2d_click(self, event):
        if self.deltaA is None:
            return
        if event.inaxes not in self._ax2d_list:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.set_sel_wl(float(event.xdata), do_update=False)
        self.set_sel_t(float(event.ydata), do_update=False)
        self._update_crosshairs()
        self._draw_spectrum()
        self._draw_kinetics()

    def set_sel_wl(self, val: float, do_update: bool = True):
        if self.wavelength is None:
            return
        val = float(np.clip(val, self.wavelength.min(),
                            self.wavelength.max()))
        self.selWL = val
        self.ed_sel_wl.blockSignals(True)
        self.ed_sel_wl.setValue(val)
        self.ed_sel_wl.blockSignals(False)
        if do_update:
            self._update_crosshairs()
            self._draw_kinetics()

    def set_sel_t(self, val: float, do_update: bool = True):
        """Move the delay crosshair to the recorded delay nearest to ``val``.

        We always snap to one of the measured delays — the spectrum
        panel cannot show a ΔA(λ) trace for a delay that wasn't
        recorded.  The corresponding index is also updated, which
        refreshes the "Delay idx" spinbox and the value-display label.
        """
        if self.delay is None:
            return
        # Snap to nearest measured delay, store both the index and value.
        idx = int(np.argmin(np.abs(self.delay - val)))
        self._selT_idx = idx
        self.selT = float(self.delay[idx])
        self._sync_sel_t_widgets()
        if do_update:
            self._update_crosshairs()
            self._draw_spectrum()

    def _sync_sel_t_widgets(self):
        """Refresh both the index spinbox and the read-only value label.

        Safe to call whenever ``self.delay`` or ``self._selT_idx`` may
        have changed (load, crop, zero-time shift, …).  Guards the
        spinbox against re-firing valueChanged.
        """
        if self.delay is None:
            self.ed_sel_t_idx.blockSignals(True)
            self.ed_sel_t_idx.setRange(0, 0)
            self.ed_sel_t_idx.setValue(0)
            self.ed_sel_t_idx.blockSignals(False)
            self.lbl_sel_t_val.setText('—')
            return
        n = len(self.delay)
        idx = int(np.clip(self._selT_idx, 0, n - 1))
        self._selT_idx = idx
        self.ed_sel_t_idx.blockSignals(True)
        self.ed_sel_t_idx.setRange(0, n - 1)
        self.ed_sel_t_idx.setValue(idx)
        self.ed_sel_t_idx.blockSignals(False)
        # Show "<idx> / <N-1>  →  <delay value> ps" in the label, so
        # users see both the position in the array and the physical
        # delay it corresponds to.
        self.lbl_sel_t_val.setText(
            f'{self.delay[idx]:.4g} {self.t_unit_txt()}   '
            f'(of {n} pts)')

    def _on_sel_t_idx_change(self, new_idx: int):
        """User edited the index spinbox → update selT and redraw spectrum."""
        if self.delay is None:
            return
        new_idx = int(np.clip(int(new_idx), 0, len(self.delay) - 1))
        self._selT_idx = new_idx
        self.selT = float(self.delay[new_idx])
        # Refresh the value-display label without re-firing the spinbox
        self.lbl_sel_t_val.setText(
            f'{self.selT:.4g} {self.t_unit_txt()}   '
            f'(of {len(self.delay)} pts)')
        self._update_crosshairs()
        self._draw_spectrum()

    # ================================================================
    # Scale / view / threshold
    # ================================================================
    def _on_scale_mode_change(self, val: str):
        self.delay_scale_mode = val.lower()
        self.update_all()

    def _on_threshold_change(self, val: float):
        if not np.isfinite(val) or val <= 0:
            return
        self.split_threshold = float(val)
        if self.delay_scale_mode == 'split':
            self.update_all()

    def _on_main_view_change(self, which: str, val: float):
        if which == 'min':
            self.main_view_tmin = float(val)
        else:
            self.main_view_tmax = float(val)
        # Display t-range only controls the 2D map's y-axis window.
        # Kinetics has its own independent x-axis (auto-scale or manual
        # via the panel's right-click menu).
        self._draw_map_2d()

    def reset_main_view(self):
        if self.delay is None:
            self.main_view_tmin = None
            self.main_view_tmax = None
            return
        self.main_view_tmin = float(self.delay[0])
        self.main_view_tmax = float(self.delay[-1])
        self.ed_view_tmin.blockSignals(True)
        self.ed_view_tmax.blockSignals(True)
        self.ed_view_tmin.setValue(self.main_view_tmin)
        self.ed_view_tmax.setValue(self.main_view_tmax)
        self.ed_view_tmin.blockSignals(False)
        self.ed_view_tmax.blockSignals(False)
        self._draw_map_2d()

    # ================================================================
    # Spectrum / Kinetics zoom-state preservation
    # ================================================================
    def _on_spec_xlim_changed(self, ax):
        """Triggered by user pan/zoom (matplotlib toolbar) in the
        spectrum panel.  If the change came from the zoom-rect or pan
        tool we auto-disable both auto flags — otherwise the next
        redraw would immediately overwrite the user's window."""
        if self._suppress_zoom_save:
            return
        self._spec_xlim = tuple(float(v) for v in ax.get_xlim())
        if self._toolbar_mode_is_navigation(self.canvas_spec.toolbar):
            self._spec_auto_x = False
            self._spec_auto_y = False

    def _on_spec_ylim_changed(self, ax):
        if self._suppress_zoom_save:
            return
        self._spec_ylim = tuple(float(v) for v in ax.get_ylim())
        if self._toolbar_mode_is_navigation(self.canvas_spec.toolbar):
            self._spec_auto_x = False
            self._spec_auto_y = False

    def _on_kin_xlim_changed(self, ax, panel_idx: int):
        """Triggered by user pan/zoom in a kinetics axes (idx 0 or 1)."""
        if self._suppress_zoom_save:
            return
        if 0 <= panel_idx < len(self._kin_xlims):
            self._kin_xlims[panel_idx] = tuple(
                float(v) for v in ax.get_xlim())
        if self._toolbar_mode_is_navigation(self.canvas_kin.toolbar):
            if 0 <= panel_idx < len(self._kin_auto_x):
                self._kin_auto_x[panel_idx] = False
                self._kin_auto_y[panel_idx] = False

    def _on_kin_ylim_changed(self, ax, panel_idx: int):
        if self._suppress_zoom_save:
            return
        if 0 <= panel_idx < len(self._kin_ylims):
            self._kin_ylims[panel_idx] = tuple(
                float(v) for v in ax.get_ylim())
        if self._toolbar_mode_is_navigation(self.canvas_kin.toolbar):
            if 0 <= panel_idx < len(self._kin_auto_x):
                self._kin_auto_x[panel_idx] = False
                self._kin_auto_y[panel_idx] = False

    def _reset_spec_zoom(self):
        """Called AFTER Home / Back / Forward has restored a view.
        Refit auto-enabled axes within the just-restored viewport so
        the user sees the valid data range in that region."""
        ax = self.canvas_spec.ax
        self._suppress_zoom_save = True
        try:
            if self._spec_auto_y:
                yr = self._visible_y_range(ax, ax.get_xlim())
                if yr is not None:
                    ax.set_ylim(yr)
            if self._spec_auto_x:
                xr = self._visible_x_range(ax, ax.get_ylim())
                if xr is not None:
                    ax.set_xlim(xr)
            # If both flags are OFF we leave whatever matplotlib
            # restored; that's the user's stored manual view.
        finally:
            self._suppress_zoom_save = False
        self.canvas_spec.draw_idle()

    def _reset_kin_zoom(self):
        """After Home / Back / Forward on the kinetics toolbar: refit
        auto-enabled axes within the just-restored viewport of each
        sub-axes."""
        self._suppress_zoom_save = True
        try:
            for i, ax in enumerate(list(self.fig_kin.axes)[:2]):
                if (i < len(self._kin_auto_y)
                        and self._kin_auto_y[i]):
                    yr = self._visible_y_range(ax, ax.get_xlim())
                    if yr is not None:
                        ax.set_ylim(yr)
                if (i < len(self._kin_auto_x)
                        and self._kin_auto_x[i]):
                    xr = self._visible_x_range(ax, ax.get_ylim())
                    if xr is not None:
                        ax.set_xlim(xr)
        finally:
            self._suppress_zoom_save = False
        self.canvas_kin.draw_idle()

    def _reset_all_panel_zoom(self):
        """Reset both spectrum and kinetics zoom — used when the
        underlying data changes (load, crop, time-unit switch).  Also
        re-enables both auto flags per panel: fresh data should start
        in a fully auto-scaled view."""
        self._spec_xlim = None
        self._spec_ylim = None
        self._kin_xlims = [None, None]
        self._kin_ylims = [None, None]
        self._spec_auto_x = True
        self._spec_auto_y = True
        self._kin_auto_x = [True, True]
        self._kin_auto_y = [True, True]

    # ================================================================
    # Auto-scale / manual-range interactions
    # (right-click menu + click-on-axis dialog + zoom-tool coupling)
    # ================================================================
    def _wire_panel_interactions(self):
        """Hook LEFT-click on both panels (for axis-region range
        dialogs) via matplotlib, and RIGHT-click via a Qt event filter
        so it is consumed before matplotlib's zoom/pan tools can see it.

        Rationale for the split: when the matplotlib zoom-rect tool is
        active, ``press_zoom`` treats a right-mouse-down as the start
        of a zoom-out drag.  If our mpl callback pops a modal menu, the
        release event never reaches matplotlib and the tool is stuck
        in "drag in progress" — every subsequent mouse move then paints
        a fresh rubber band tracking the cursor.  Consuming the right-
        click at the Qt level (return True from an event filter)
        prevents matplotlib from starting the drag at all.
        """
        self.canvas_spec.mpl_connect(
            'button_press_event', self._on_spec_click)
        self.canvas_kin.mpl_connect(
            'button_press_event', self._on_kin_click)

        self._spec_rmb_filter = _RightClickMenuFilter(
            self.canvas_spec,
            lambda: self._show_panel_menu('spec', None))
        self.canvas_spec.installEventFilter(self._spec_rmb_filter)

        self._kin_rmb_filter = _RightClickMenuFilter(
            self.canvas_kin,
            lambda: self._show_panel_menu('kin', None))
        self.canvas_kin.installEventFilter(self._kin_rmb_filter)

    @staticmethod
    def _hit_axis_region(canvas, event):
        """Return ``(axes, 'x'|'y')`` when the pixel-position of the
        press falls on a tick / label margin next to an axes; else
        ``(None, None)``.  Uses a small pixel margin — clicks well
        outside the figure never match."""
        margin = 55  # pixels — roomy enough to catch tick labels
        for ax in canvas.figure.axes:
            bb = ax.bbox  # pixel bbox of the plot area
            in_x_range = (bb.x0 - 5) <= event.x <= (bb.x1 + 5)
            in_y_range = (bb.y0 - 5) <= event.y <= (bb.y1 + 5)
            # Bottom margin → x-axis
            if in_x_range and (bb.y0 - margin) <= event.y < bb.y0:
                return ax, 'x'
            # Left margin → y-axis
            if in_y_range and (bb.x0 - margin) <= event.x < bb.x0:
                return ax, 'y'
        return None, None

    # ---- Viewport-aware data-range helpers ----
    @staticmethod
    def _visible_y_range(ax, xlim):
        """Return (ymin, ymax) across all Line2D artists on ``ax`` whose
        xdata falls inside ``xlim``.  Skips two-point reference lines
        (axhline / axvline)."""
        xl, xh = float(xlim[0]), float(xlim[1])
        ymin, ymax = None, None
        for line in ax.get_lines():
            xdata = np.asarray(line.get_xdata())
            ydata = np.asarray(line.get_ydata())
            if xdata.size < 3:
                continue  # axhline/axvline
            mask = (xdata >= xl) & (xdata <= xh)
            yvis = ydata[mask]
            yvis = yvis[np.isfinite(yvis)]
            if yvis.size == 0:
                continue
            lo = float(np.min(yvis))
            hi = float(np.max(yvis))
            ymin = lo if ymin is None or lo < ymin else ymin
            ymax = hi if ymax is None or hi > ymax else ymax
        if ymin is None or ymax is None:
            return None
        if ymax == ymin:
            pad = abs(ymin) * 0.05 or 0.01
            return (ymin - pad, ymax + pad)
        pad = (ymax - ymin) * 0.05
        return (ymin - pad, ymax + pad)

    @staticmethod
    def _visible_x_range(ax, ylim):
        """Return (xmin, xmax) across all Line2D artists on ``ax`` whose
        ydata falls inside ``ylim``.  Skips two-point reference lines."""
        yl, yh = float(ylim[0]), float(ylim[1])
        xmin, xmax = None, None
        for line in ax.get_lines():
            xdata = np.asarray(line.get_xdata())
            ydata = np.asarray(line.get_ydata())
            if xdata.size < 3:
                continue
            mask = (ydata >= yl) & (ydata <= yh) & np.isfinite(ydata)
            xvis = xdata[mask]
            xvis = xvis[np.isfinite(xvis)]
            if xvis.size == 0:
                continue
            lo = float(np.min(xvis))
            hi = float(np.max(xvis))
            xmin = lo if xmin is None or lo < xmin else xmin
            xmax = hi if xmax is None or hi > xmax else xmax
        if xmin is None or xmax is None:
            return None
        if xmax == xmin:
            pad = abs(xmin) * 0.05 or 0.01
            return (xmin - pad, xmax + pad)
        pad = (xmax - xmin) * 0.05
        return (xmin - pad, xmax + pad)

    # ---- Toolbar (matplotlib) coupling ----
    @staticmethod
    def _toolbar_mode_is_navigation(toolbar) -> bool:
        """True if the matplotlib toolbar is in zoom-rect or pan mode."""
        mode = getattr(toolbar, 'mode', '') or ''
        return mode.startswith('zoom') or mode.startswith('pan')

    @staticmethod
    def _release_toolbar_mode(toolbar):
        """Toggle any active navigation tool (zoom rect / pan) off and
        clear the rubber-band overlay.

        Matplotlib normally erases the dashed selection rectangle when
        a zoom drag completes, but toggling the tool off from code (as
        we do when the user re-enables auto-scale) can leave the last
        rubber band painted on the canvas.  Explicitly calling
        ``remove_rubberband()`` + a full ``canvas.draw()`` guarantees
        the overlay is gone regardless of matplotlib version / backend.
        """
        mode = getattr(toolbar, 'mode', '') or ''
        try:
            if mode.startswith('zoom'):
                toolbar.zoom()
            elif mode.startswith('pan'):
                toolbar.pan()
        except Exception:
            pass

        canvas = getattr(toolbar, 'canvas', None)
        if canvas is None:
            return
        # Best-effort rubber-band clear (API varies across versions).
        try:
            if hasattr(toolbar, 'remove_rubberband'):
                toolbar.remove_rubberband()
        except Exception:
            pass
        try:
            if hasattr(canvas, 'drawRectangle'):
                # Qt backend: passing None erases the stored rectangle.
                canvas.drawRectangle(None)
        except Exception:
            pass
        # Full immediate repaint — a lingering rectangle is drawn on
        # top of the canvas pixmap and only a full draw wipes it.
        try:
            canvas.draw()
        except Exception:
            try:
                canvas.draw_idle()
            except Exception:
                pass

    # ---- Spectrum panel: LEFT-click handler (RMB handled by Qt filter) ----
    def _on_spec_click(self, event):
        # Right-clicks are consumed by the Qt event filter before
        # matplotlib forwards them; if one leaks through we still open
        # the menu instead of doing nothing.
        if event.button == 3:
            self._show_panel_menu('spec', None)
            return
        if event.button == 1:
            ax, which = self._hit_axis_region(self.canvas_spec, event)
            if ax is None:
                return
            # Clicking an axis label region always opens the dialog for
            # that axis; force manual mode so the typed value is not
            # immediately overwritten by auto-fit.
            if which == 'x':
                self._spec_auto_x = False
            else:
                self._spec_auto_y = False
            self._open_axis_range_dialog('spec', 0, which, ax)

    # ---- Kinetics panel: LEFT-click handler (RMB handled by Qt filter) ----
    def _on_kin_click(self, event):
        if event.button == 3:
            self._show_panel_menu('kin', None)
            return
        if event.button == 1:
            ax, which = self._hit_axis_region(self.canvas_kin, event)
            if ax is None:
                return
            axes = list(self.fig_kin.axes)
            try:
                idx = axes.index(ax)
            except ValueError:
                idx = 0
            if which == 'x':
                if 0 <= idx < len(self._kin_auto_x):
                    self._kin_auto_x[idx] = False
            else:
                if 0 <= idx < len(self._kin_auto_y):
                    self._kin_auto_y[idx] = False
            self._open_axis_range_dialog('kin', idx, which, ax)

    def _show_panel_menu(self, panel: str, event):
        """Right-click context menu for the spectrum / kinetics panels.
        Auto scale X / Y are independent checkable actions; Set X/Y
        range entries are enabled only when the corresponding auto is
        off.  For kinetics split mode the menu shows left / right
        variants."""
        menu = QtWidgets.QMenu(self)

        if panel == 'spec':
            act_ax = menu.addAction('Auto scale X')
            act_ay = menu.addAction('Auto scale Y')
            act_ax.setCheckable(True); act_ay.setCheckable(True)
            act_ax.setChecked(self._spec_auto_x)
            act_ay.setChecked(self._spec_auto_y)
            act_ax.triggered.connect(
                lambda ck: self._set_auto('spec', 0, 'x', bool(ck)))
            act_ay.triggered.connect(
                lambda ck: self._set_auto('spec', 0, 'y', bool(ck)))
            menu.addSeparator()
            act_sx = menu.addAction('Set X range…')
            act_sy = menu.addAction('Set Y range…')
            act_sx.setEnabled(not self._spec_auto_x)
            act_sy.setEnabled(not self._spec_auto_y)
            act_sx.triggered.connect(
                lambda: self._open_axis_range_dialog(
                    'spec', 0, 'x', self.canvas_spec.ax))
            act_sy.triggered.connect(
                lambda: self._open_axis_range_dialog(
                    'spec', 0, 'y', self.canvas_spec.ax))
        else:  # kin
            axes = list(self.fig_kin.axes)
            n = len(axes)
            for i, ax in enumerate(axes[:2]):
                suffix = ('' if n <= 1
                          else f' ({"left" if i == 0 else "right"})')
                a_x = menu.addAction(f'Auto scale X{suffix}')
                a_y = menu.addAction(f'Auto scale Y{suffix}')
                a_x.setCheckable(True); a_y.setCheckable(True)
                a_x.setChecked(self._kin_auto_x[i])
                a_y.setChecked(self._kin_auto_y[i])
                a_x.triggered.connect(
                    lambda ck, i=i: self._set_auto('kin', i, 'x', bool(ck)))
                a_y.triggered.connect(
                    lambda ck, i=i: self._set_auto('kin', i, 'y', bool(ck)))
                menu.addSeparator()
                s_x = menu.addAction(f'Set X range{suffix}…')
                s_y = menu.addAction(f'Set Y range{suffix}…')
                s_x.setEnabled(not self._kin_auto_x[i])
                s_y.setEnabled(not self._kin_auto_y[i])
                s_x.triggered.connect(
                    lambda _=False, i=i, a=ax:
                        self._open_axis_range_dialog('kin', i, 'x', a))
                s_y.triggered.connect(
                    lambda _=False, i=i, a=ax:
                        self._open_axis_range_dialog('kin', i, 'y', a))
                if i == 0 and n > 1:
                    menu.addSeparator()

        menu.exec_(QtGui.QCursor.pos())

    def _set_auto(self, panel: str, sub_idx: int, which: str, enable: bool):
        """Set the auto flag for one axis of one (sub-)panel.

        Turning ON:
          - Release toolbar zoom/pan mode (an active magnifier makes
            auto-fit useless — the next drag would just override).
          - Recompute this axis's range within the CURRENT viewport of
            the other axis (per the user spec: enabling auto-X should
            fit X to the data whose Y is currently on-screen; enabling
            auto-Y fits Y to the data whose X is currently on-screen).
          - Clear the stored manual limit for this axis so subsequent
            redraws remain in auto mode.

        Turning OFF:
          - Capture the current axes limit into the manual-limit slot
            so the value the user was looking at is preserved and
            editable via the range dialog.
        """
        # Locate the target axes
        if panel == 'spec':
            ax_list = [self.canvas_spec.ax]
            toolbar = self.canvas_spec.toolbar
        else:
            ax_list = list(self.fig_kin.axes)
            toolbar = self.canvas_kin.toolbar
        if sub_idx < 0 or sub_idx >= len(ax_list):
            return
        ax = ax_list[sub_idx]

        # Update the flag
        if panel == 'spec':
            if which == 'x':
                self._spec_auto_x = enable
            else:
                self._spec_auto_y = enable
        else:
            if which == 'x':
                self._kin_auto_x[sub_idx] = enable
            else:
                self._kin_auto_y[sub_idx] = enable

        if enable:
            # Release toolbar zoom/pan so it doesn't fight the auto-fit
            self._release_toolbar_mode(toolbar)
            # Clear manual storage for this axis
            if panel == 'spec':
                if which == 'x':
                    self._spec_xlim = None
                else:
                    self._spec_ylim = None
            else:
                if which == 'x':
                    self._kin_xlims[sub_idx] = None
                else:
                    self._kin_ylims[sub_idx] = None
            # Apply auto-fit within the current viewport of the other axis
            self._suppress_zoom_save = True
            try:
                if which == 'x':
                    rng = self._visible_x_range(ax, ax.get_ylim())
                    if rng is not None:
                        ax.set_xlim(rng)
                else:
                    rng = self._visible_y_range(ax, ax.get_xlim())
                    if rng is not None:
                        ax.set_ylim(rng)
            finally:
                self._suppress_zoom_save = False
            (self.canvas_spec if panel == 'spec'
                else self.canvas_kin).draw_idle()
        else:
            # Capture current limit into manual slot
            if which == 'x':
                lim = tuple(float(v) for v in ax.get_xlim())
                if panel == 'spec':
                    self._spec_xlim = lim
                else:
                    self._kin_xlims[sub_idx] = lim
            else:
                lim = tuple(float(v) for v in ax.get_ylim())
                if panel == 'spec':
                    self._spec_ylim = lim
                else:
                    self._kin_ylims[sub_idx] = lim

    def _open_axis_range_dialog(self, panel: str, sub_idx: int,
                                which: str, ax):
        """Small modal dialog that lets the user type a min/max for
        one axis.  Apply → immediately updates the axes + stores the
        value in the panel's persistent limit slots.  Also flips the
        corresponding auto flag off so the value sticks."""
        cur = (ax.get_xlim() if which == 'x' else ax.get_ylim())
        title_axis = 'X (wavelength)' if (panel == 'spec' and which == 'x') \
            else 'Y (ΔA)' if (panel == 'spec' and which == 'y') \
            else 'X (delay)' if (panel == 'kin' and which == 'x') \
            else 'Y (ΔA)'
        title = f'Set {title_axis} range — ' + \
                ('Spectrum' if panel == 'spec' else 'Kinetics')
        result = _prompt_axis_range(self, title, float(cur[0]),
                                    float(cur[1]))
        if result is None:
            return
        lo, hi = result
        # Ensure the manual value is the source of truth for this axis
        if panel == 'spec':
            if which == 'x':
                self._spec_auto_x = False
            else:
                self._spec_auto_y = False
        else:
            if which == 'x':
                self._kin_auto_x[sub_idx] = False
            else:
                self._kin_auto_y[sub_idx] = False
        # Apply + store
        self._suppress_zoom_save = True
        try:
            if which == 'x':
                ax.set_xlim(lo, hi)
            else:
                ax.set_ylim(lo, hi)
            if panel == 'spec':
                if which == 'x':
                    self._spec_xlim = (lo, hi)
                else:
                    self._spec_ylim = (lo, hi)
                self.canvas_spec.draw_idle()
            else:
                if which == 'x':
                    if 0 <= sub_idx < len(self._kin_xlims):
                        self._kin_xlims[sub_idx] = (lo, hi)
                else:
                    if 0 <= sub_idx < len(self._kin_ylims):
                        self._kin_ylims[sub_idx] = (lo, hi)
                self.canvas_kin.draw_idle()
        finally:
            self._suppress_zoom_save = False

    def _clear_ga_state(self):
        """Drop any cached global-analysis fit results.

        Called whenever the underlying wavelength / delay arrays are
        replaced (load, crop), because the cached fit matrices no
        longer have shapes that match the current ``deltaA``.  Also
        forgets any user-set fit window — the new data has its own
        meaningful range.
        """
        self.ga_t_min = None
        self.ga_t_max = None
        self.ga_result_tau = None
        self.ga_result_t0 = None
        self.ga_result_fwhm = None
        self.ga_result_has_inf = False
        self.ga_result_dads = None
        self.ga_result_eads = None
        self.ga_result_fit = None
        self.ga_result_rms = float('nan')
        self.ga_result_delay = None
        self.ga_result_data = None
        self.ga_result_t_window = None

    # ================================================================
    # Overlays (pin / clear)
    # ================================================================
    def pin_spec(self):
        if self.deltaA is None:
            return
        idx = int(np.argmin(np.abs(self.delay - self.selT)))
        val = float(self.delay[idx])
        if val not in self.specOverlays:
            self.specOverlays.append(val)
        self._draw_spectrum()

    def pin_kin(self):
        if self.deltaA is None:
            return
        idx = int(np.argmin(np.abs(self.wavelength - self.selWL)))
        val = float(self.wavelength[idx])
        if val not in self.kinOverlays:
            self.kinOverlays.append(val)
        self._draw_kinetics()

    def clear_spec_overlays(self):
        self.specOverlays = []
        self._draw_spectrum()

    def clear_kin_overlays(self):
        self.kinOverlays = []
        self._draw_kinetics()

    # ---- Save-path defaults ----
    def default_save_path(self, filename: str) -> str:
        """Prepend the source-data folder to a suggested filename so the
        Save-as dialog opens next to where the raw data was loaded.

        Empty ``data_source_dir`` (no data loaded yet, or the loader
        didn't record a folder) falls back to the bare filename — Qt
        then uses the process cwd, matching the pre-existing behavior.
        """
        if not filename:
            filename = 'out.csv'
        if self.data_source_dir and os.path.isdir(self.data_source_dir):
            return os.path.join(self.data_source_dir, filename)
        return filename

    # ---- Export pinned overlays ----
    def _pin_export_basename(self):
        """File-stem suggestion for pin exports — mirrors the
        ``Export 2D Data`` naming logic."""
        base = self.data_source_desc or 'data'
        for pref in ('Loaded: ', 'Avg of '):
            if base.startswith(pref):
                base = base[len(pref):]
                break
        base = base.split(' ')[0]
        stem = os.path.splitext(os.path.basename(base))[0]
        return stem or 'data'

    def export_spec_overlays(self):
        if self.deltaA is None:
            warn_box(self, 'No data', 'Load data first.')
            return
        if not self.specOverlays:
            warn_box(self, 'No pinned spectra',
                     'Pin one or more delays before exporting.')
            return
        # Resolve each pinned delay to the nearest recorded index;
        # de-dup if two pins snap to the same column.
        seen, idxs, vals = set(), [], []
        for t_val in self.specOverlays:
            i = int(np.argmin(np.abs(self.delay - t_val)))
            if i in seen:
                continue
            seen.add(i)
            idxs.append(i)
            vals.append(float(self.delay[i]))
        unit = self.t_unit_hdr()
        headers = ['wavelength_nm'] + [
            f'dA_t={v:.6g}{unit}' for v in vals]
        arr = np.column_stack(
            [self.wavelength] + [self.deltaA[:, i] for i in idxs])

        suggested = f'{self._pin_export_basename()}_pinned_spectra.csv'
        path, _ = ask_save_path(self, 'Export pinned spectra',
                                self.default_save_path(suggested))
        if not path:
            return
        try:
            np.savetxt(path, arr, delimiter=',',
                       header=','.join(headers), comments='', fmt='%.6g')
            info_box(self, 'Export complete',
                     f'Saved {len(idxs)} pinned spectrum(a) '
                     f'({len(self.wavelength)} wavelengths) to:\n{path}')
        except Exception as e:
            warn_box(self, 'Export error', f'Export failed:\n{e}')

    def export_kin_overlays(self):
        if self.deltaA is None:
            warn_box(self, 'No data', 'Load data first.')
            return
        if not self.kinOverlays:
            warn_box(self, 'No pinned kinetics',
                     'Pin one or more wavelengths before exporting.')
            return
        seen, idxs, vals = set(), [], []
        for w_val in self.kinOverlays:
            i = int(np.argmin(np.abs(self.wavelength - w_val)))
            if i in seen:
                continue
            seen.add(i)
            idxs.append(i)
            vals.append(float(self.wavelength[i]))
        unit = self.t_unit_hdr()
        headers = [f'delay_{unit}'] + [
            f'dA_wl={v:.4g}nm' for v in vals]
        arr = np.column_stack(
            [self.delay] + [self.deltaA[i, :] for i in idxs])

        suggested = f'{self._pin_export_basename()}_pinned_kinetics.csv'
        path, _ = ask_save_path(self, 'Export pinned kinetics',
                                self.default_save_path(suggested))
        if not path:
            return
        try:
            np.savetxt(path, arr, delimiter=',',
                       header=','.join(headers), comments='', fmt='%.6g')
            info_box(self, 'Export complete',
                     f'Saved {len(idxs)} pinned kinetic(s) '
                     f'({len(self.delay)} delays) to:\n{path}')
        except Exception as e:
            warn_box(self, 'Export error', f'Export failed:\n{e}')

    # ================================================================
    # Reset / zero-time shift / crop
    # ================================================================
    def reset_corrections(self):
        if self.deltaA_raw is None:
            return
        self.bg_applied = False
        self.bg_spectrum = None
        self.bg_n = 0
        # Switch off solvent IRF subtraction too, but KEEP the loaded
        # solvent file in memory — the user may just want to compare
        # before/after, and re-loading is expensive.  Toggling the
        # checkbox in the Solvent-IRF dialog brings it back instantly.
        self.sub_irf_applied = False
        self.chirp_applied = False
        self.chirp_pts = np.zeros((0, 2))
        self.chirp_params = None
        self.chirp_fit_rms = float('nan')
        self.ridge_wl = None
        self.ridge_t = None
        self.masked_regions = []
        # Undo any accumulated zero-time shift
        if self.tZeroShift != 0.0:
            self.delay = self.delay + self.tZeroShift
            self.selT += self.tZeroShift
            if self.main_view_tmin is not None:
                self.main_view_tmin += self.tZeroShift
                self.main_view_tmax += self.tZeroShift
            self.tZeroShift = 0.0
            # Re-snap selT to the nearest measured delay (it may have
            # drifted off-grid due to the shift) and refresh widgets.
            self._selT_idx = int(np.argmin(np.abs(self.delay - self.selT)))
            self.selT = float(self.delay[self._selT_idx])
            self._sync_sel_t_widgets()
            if self.main_view_tmin is not None:
                self.ed_view_tmin.blockSignals(True)
                self.ed_view_tmax.blockSignals(True)
                self.ed_view_tmin.setValue(self.main_view_tmin)
                self.ed_view_tmax.setValue(self.main_view_tmax)
                self.ed_view_tmin.blockSignals(False)
                self.ed_view_tmax.blockSignals(False)
        # Resetting corrections changes the effective ΔA range and may
        # also have shifted the delay axis (zero-time undo above), so
        # any saved spectrum/kinetics zoom is no longer meaningful.
        self._reset_all_panel_zoom()
        self.update_all()

    def apply_zero_time_shift(self, delta: float):
        """Shift the delay axis so that `delta` in the current axis becomes 0."""
        if not np.isfinite(delta) or delta == 0 or self.delay is None:
            return
        self.delay = self.delay - delta
        self.tZeroShift += delta
        self.selT -= delta
        if self.main_view_tmin is not None:
            self.main_view_tmin -= delta
            self.main_view_tmax -= delta
        # The delay axis was just translated, so any stored kinetics
        # zoom (whose x-axis is delay) no longer points at the same
        # data.  Spectrum is unaffected — its x-axis is wavelength.
        self._kin_xlims = [None, None]
        self._kin_ylims = [None, None]
        # Clamp / update edits — re-snap to nearest measured delay
        self.selT = float(np.clip(self.selT, self.delay.min(),
                                  self.delay.max()))
        self._selT_idx = int(np.argmin(np.abs(self.delay - self.selT)))
        self.selT = float(self.delay[self._selT_idx])
        self._sync_sel_t_widgets()
        if self.main_view_tmin is not None:
            self.ed_view_tmin.blockSignals(True)
            self.ed_view_tmax.blockSignals(True)
            self.ed_view_tmin.setValue(self.main_view_tmin)
            self.ed_view_tmax.setValue(self.main_view_tmax)
            self.ed_view_tmin.blockSignals(False)
            self.ed_view_tmax.blockSignals(False)
        self.update_all()

    def apply_zero_time_at_crosshair(self):
        if self.deltaA is None:
            warn_box(self, 'No data', 'Load data first.')
            return
        delta = self.selT
        if not np.isfinite(delta):
            return
        if delta == 0:
            warn_box(self, 'No shift needed',
                     'Crosshair is already at t = 0. Click somewhere '
                     'else on the 2D map first to pick a new t_0 '
                     'location.')
            return
        self.apply_zero_time_shift(delta)
        self.lbl_status.setText(
            f't_0 shift applied: {delta:.4g} {self.t_unit_txt()}  '
            f'(total: {self.tZeroShift:.4g} {self.t_unit_txt()})')

    def apply_crop_by_range(self, wl_min, wl_max, t_min, t_max,
                            resample: dict | None = None):
        """Rebuild the working grid from the original data: crop to the
        given λ/t range, then (optionally) resample the λ axis.

        ``resample`` is ``{'enabled': bool, 'dx': float, 'mode': str}``.
        Calling without it (Revert, scripts) turns resampling off — the
        working grid always reflects exactly this call's request.
        """
        if self.original_deltaA is None:
            return
        if wl_min > wl_max:
            wl_min, wl_max = wl_max, wl_min
        if t_min > t_max:
            t_min, t_max = t_max, t_min

        wl_mask = ((self.original_wavelength >= wl_min) &
                   (self.original_wavelength <= wl_max))
        t_mask = ((self.original_delay >= t_min) &
                  (self.original_delay <= t_max))
        if not wl_mask.any() or not t_mask.any():
            warn_box(self, 'Invalid range', 'Empty crop range.')
            return

        full = bool(wl_mask.all() and t_mask.all())
        self.crop_bounds = (None if full
                            else (float(wl_min), float(wl_max),
                                  float(t_min), float(t_max)))
        on = bool(resample and resample.get('enabled'))
        if on:
            self.resample_dx = float(resample['dx'])
            self.resample_mode = str(resample['mode'])
        self._rebuild_working_grid(wl_mask, t_mask, on)

        # Reset corrections (they were fit to the old range)
        self.bg_applied = False
        self.bg_spectrum = None
        self.bg_n = 0
        # Crop changes the sample grid; the solvent reference (if any)
        # must be re-aligned onto the new grid before recompute runs.
        # We keep the raw solvent data and just turn off the toggle
        # until realign_solvent() repopulates sub_irf_aligned.
        self.sub_irf_applied = False
        self.sub_irf_aligned = None
        self.chirp_applied = False
        self.chirp_pts = np.zeros((0, 2))
        self.chirp_params = None
        self.chirp_fit_rms = float('nan')
        self.ridge_wl = None
        self.ridge_t = None
        self.tZeroShift = 0.0
        self.masked_regions = []
        # Crop changed the wavelength / delay arrays, so the previous
        # GA result no longer matches the displayed matrix dimensions.
        self._clear_ga_state()
        # Crop changes the data extent, so any saved spectrum/kinetics
        # pan/zoom is no longer meaningful — drop it and let the first
        # post-crop draw auto-fit to the new range.
        self._reset_all_panel_zoom()

        # Clip overlays and selection
        wl_lo, wl_hi = float(self.wavelength.min()), float(self.wavelength.max())
        t_lo, t_hi = float(self.delay.min()), float(self.delay.max())
        self.specOverlays = [t for t in self.specOverlays
                             if t_lo <= t <= t_hi]
        self.kinOverlays = [w for w in self.kinOverlays
                            if wl_lo <= w <= wl_hi]
        self.selWL = float(np.clip(self.selWL, wl_lo, wl_hi))
        self.selT = float(np.clip(self.selT, t_lo, t_hi))
        # Re-snap selT to a measured delay inside the crop and update
        # the index spinbox.  (Crop changed the delay array.)
        self._selT_idx = int(np.argmin(np.abs(self.delay - self.selT)))
        self.selT = float(self.delay[self._selT_idx])

        # Update edits
        for ed in (self.ed_sel_wl, self.ed_view_tmin, self.ed_view_tmax):
            ed.blockSignals(True)
        self.ed_sel_wl.setRange(wl_lo, wl_hi)
        self.ed_sel_wl.setValue(self.selWL)
        self.main_view_tmin = t_lo
        self.main_view_tmax = t_hi
        self.ed_view_tmin.setValue(t_lo)
        self.ed_view_tmax.setValue(t_hi)
        for ed in (self.ed_sel_wl, self.ed_view_tmin, self.ed_view_tmax):
            ed.blockSignals(False)
        self._sync_sel_t_widgets()

        # Reset Z-range to auto for the cropped data
        self.map_z_min = None
        self.map_z_max = None
        cl = float(np.nanmax(np.abs(self.deltaA_raw)))
        if not np.isfinite(cl) or cl == 0:
            cl = 1.0
        self.ed_z_min.blockSignals(True)
        self.ed_z_max.blockSignals(True)
        self.ed_z_min.setValue(-cl)
        self.ed_z_max.setValue(cl)
        self.ed_z_min.blockSignals(False)
        self.ed_z_max.blockSignals(False)

        self._update_status_label()
        self.update_all()

    def _rebuild_working_grid(self, wl_mask, t_mask, resample_on: bool):
        """Slice the originals with the masks and, if requested, bin the
        λ axis with the current ``resample_dx`` / ``resample_mode``."""
        self.wavelength = self.original_wavelength[wl_mask].copy()
        self.delay = self.original_delay[t_mask].copy()
        self.deltaA_raw = self.original_deltaA[np.ix_(wl_mask, t_mask)].copy()
        self._crop_wl_pre_resample = self.wavelength.copy()
        self.resample_enabled = False
        self.resample_info = None
        if not resample_on:
            return
        wl_r, A_r, info = ta_core.resample_wavelength(
            self.wavelength, self.deltaA_raw,
            self.resample_dx, self.resample_mode)
        if info['applied']:
            self.wavelength, self.deltaA_raw = wl_r, A_r
            self.resample_enabled = True
            self.resample_info = info

    def interpolate_delays_at(self, drop_idx, method: str = 'linear'):
        """Overwrite the spectra at ``drop_idx`` with values interpolated
        along the delay axis from the remaining columns.

        The substitution is applied to ``deltaA_raw`` (the post-crop raw
        matrix) so that any subsequent BG / chirp / solvent corrections
        re-run on the cleaned data.  The ``original_deltaA`` source
        remains untouched, so "Revert to Original" still undoes both
        the crop and any interpolation pass.
        """
        if self.deltaA_raw is None or self.delay is None:
            return
        drop_idx = list(dict.fromkeys(int(i) for i in drop_idx))
        drop_idx = [i for i in drop_idx if 0 <= i < len(self.delay)]
        if not drop_idx:
            return
        if len(self.delay) - len(drop_idx) < 2:
            warn_box(self, 'Too many delays selected',
                     'At least two non-deleted delays are required to '
                     'interpolate; nothing was changed.')
            return
        try:
            self.deltaA_raw = ta_core.interpolate_missing_columns(
                self.deltaA_raw, self.delay, drop_idx, method=method)
        except Exception as e:
            warn_box(self, 'Interpolation failed', str(e))
            return
        # GA / LDA / Kfit results were fit to the previous raw matrix.
        self._clear_ga_state()
        self.update_all()

    def interpolate_wavelengths_at(self, drop_idx, method: str = 'linear'):
        """Overwrite the kinetics at ``drop_idx`` (wavelength rows) with
        values interpolated along the wavelength axis from the remaining
        rows.  Mirror of :meth:`interpolate_delays_at`.
        """
        if self.deltaA_raw is None or self.wavelength is None:
            return
        drop_idx = list(dict.fromkeys(int(i) for i in drop_idx))
        drop_idx = [i for i in drop_idx if 0 <= i < len(self.wavelength)]
        if not drop_idx:
            return
        if len(self.wavelength) - len(drop_idx) < 2:
            warn_box(self, 'Too many wavelengths selected',
                     'At least two non-deleted wavelengths are required to '
                     'interpolate; nothing was changed.')
            return
        try:
            self.deltaA_raw = ta_core.interpolate_missing_rows(
                self.deltaA_raw, self.wavelength, drop_idx, method=method)
        except Exception as e:
            warn_box(self, 'Interpolation failed', str(e))
            return
        self._clear_ga_state()
        self.update_all()

    def interpolate_2d_at(self, drop_t_idx, drop_w_idx):
        """Bilinear (2D-surface) interpolation across the union of the
        given delay columns and wavelength rows in ``deltaA_raw``.
        """
        if self.deltaA_raw is None:
            return
        drop_t = [int(i) for i in drop_t_idx
                  if 0 <= int(i) < len(self.delay)]
        drop_w = [int(i) for i in drop_w_idx
                  if 0 <= int(i) < len(self.wavelength)]
        drop_t = sorted(set(drop_t))
        drop_w = sorted(set(drop_w))
        if not drop_t and not drop_w:
            return
        # Guard: need at least 2 kept rows + 2 kept cols.
        if (len(self.delay) - len(drop_t) < 2
                or len(self.wavelength) - len(drop_w) < 2):
            warn_box(self, 'Too many drops selected',
                     'At least two non-deleted wavelengths AND two '
                     'non-deleted delays are required for 2D interpolation; '
                     'nothing was changed.')
            return
        try:
            self.deltaA_raw = ta_core.interpolate_missing_2d(
                self.deltaA_raw, self.wavelength, self.delay,
                drop_w, drop_t)
        except Exception as e:
            warn_box(self, 'Interpolation failed', str(e))
            return
        self._clear_ga_state()
        self.update_all()

    # ================================================================
    # Export
    # ================================================================
    def _export_2d_matrix(self, data, suffix,
                          caption='Export 2D map data'):
        """Shared body for the two 'Export 2D ...' buttons.  Saves
        ``data`` (shape ``(n_lambda, n_t)``) over the current
        ``(wavelength, delay)`` axes to CSV / TSV / Excel.  ``suffix``
        identifies the pipeline state for the suggested filename + the
        confirmation dialog."""
        base_name = self.data_source_desc or 'data'
        for pref in ('Loaded: ', 'Avg of '):
            if base_name.startswith(pref):
                base_name = base_name[len(pref):]
                break
        base_name = base_name.split(' ')[0]
        name_only = os.path.splitext(os.path.basename(base_name))[0]
        if not name_only:
            name_only = 'data'
        suggested = f'{name_only}_2D_{suffix}.csv'

        path, sel = QtWidgets.QFileDialog.getSaveFileName(
            self, caption, self.default_save_path(suggested),
            'CSV (*.csv);;TSV (*.tsv);;Excel (*.xlsx);;Text (*.txt);;'
            'All files (*)')
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        if not ext:
            if 'tsv' in sel.lower():
                ext = '.tsv'; path += '.tsv'
            elif 'xlsx' in sel.lower() or 'excel' in sel.lower():
                ext = '.xlsx'; path += '.xlsx'
            elif 'txt' in sel.lower():
                ext = '.txt'; path += '.txt'
            else:
                ext = '.csv'; path += '.csv'

        try:
            if ext == '.xlsx':
                ta_core.write_data_excel(path, self.wavelength,
                                         self.delay, data)
                fmt = 'Excel (.xlsx)'
            else:
                if ext == '.tsv' or ext == '.txt':
                    delim = '\t'
                    fmt = ('TSV (tab-separated)' if ext == '.tsv'
                           else 'tab-separated text')
                else:
                    delim = ','
                    fmt = 'CSV (comma-separated)'
                ta_core.write_data_file(path, self.wavelength, self.delay,
                                        data, delim)
            info_box(self, 'Export complete',
                     f'Saved 2D ΔA matrix as {fmt}.\n\n'
                     f'Size: {len(self.wavelength)} λ × '
                     f'{len(self.delay)} t\n'
                     f'Pipeline: {suffix}\n\n'
                     f'Layout:  first row = delays, '
                     f'first column = wavelengths.\n\n'
                     f'File: {path}')
        except RuntimeError as e:
            # openpyxl missing
            warn_box(self, 'Export error', str(e))
        except Exception as e:
            warn_box(self, 'Export error', f'Export failed:\n{e}')

    def export_2d_data(self):
        """Save the displayed (fully corrected) 2D ΔA matrix to
        CSV / TSV / Excel / TXT.  Suffix records the pipeline state
        (BG / chirp / cropped / masked)."""
        if self.deltaA is None:
            warn_box(self, 'No data', 'Load data first.')
            return

        parts = []
        if self.bg_applied:
            parts.append(f'BG{self.bg_n}')
        if self.chirp_applied:
            parts.append('chirp')
        if self.is_cropped():
            parts.append('cropped')
        if self.resample_enabled:
            parts.append(f'rs{self.resample_dx:g}nm')
        if self.masked_regions:
            parts.append(f'masked{len(self.masked_regions)}')
        suffix = '_'.join(parts) if parts else 'raw'
        self._export_2d_matrix(self.deltaA, suffix,
                               caption='Export 2D map data')

    def export_2d_data_original(self):
        """Save the cropped *original* 2D matrix — i.e. ``deltaA_raw``
        before BG / chirp / solvent-IRF / mask are applied.  If no crop
        is active this is exactly the loaded file's data."""
        if self.deltaA_raw is None:
            warn_box(self, 'No data', 'Load data first.')
            return
        suffix = 'cropped_original' if self.is_cropped() else 'original'
        if self.resample_enabled:
            suffix += f'_rs{self.resample_dx:g}nm'
        self._export_2d_matrix(
            self.deltaA_raw, suffix,
            caption='Export 2D map (original, pre-processing)')

    # ================================================================
    # Dialog management
    # ================================================================
    def _open_dialog(self, DialogClass, attr_name: str, *args, **kwargs):
        existing = getattr(self, attr_name, None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return existing
        dlg = DialogClass(self, self, *args, **kwargs)
        setattr(self, attr_name, dlg)
        # Clean up the reference whenever the dialog is dismissed
        dlg.finished.connect(
            lambda _=0, a=attr_name: setattr(self, a, None))
        dlg.show()
        return dlg

    def open_background_window(self):
        if self.deltaA_raw is None:
            warn_box(self, 'No data', 'Load data first.')
            return
        self._open_dialog(BackgroundDialog, 'bg_fig')

    def open_solvent_irf_window(self):
        """Open the pure-solvent IRF subtraction dialog.

        Lazy import: only pull ta_solvent_irf in when the user actually
        clicks, to keep startup fast and the dependency chain explicit.
        """
        if self.deltaA_raw is None:
            warn_box(self, 'No data', 'Load data first.')
            return
        from ta_solvent_irf import SolventIRFDialog
        self._open_dialog(SolventIRFDialog, 'sub_irf_fig')

    def _on_sub_irf_toggled(self, checked: bool):
        """Toolbar checkbox toggled.

        On  → open the dialog (if not already open) so the user can
              load a solvent.  Don't actually flip
              ``sub_irf_applied`` here — that's the dialog's job once
              a solvent is loaded; otherwise we'd subtract nothing
              and pretend to be on.
        Off → turn off subtraction immediately.  Keep the loaded
              solvent in memory so re-checking is instant.
        """
        if checked:
            # If we already have a solvent, just enable subtraction and
            # rerun the pipeline; otherwise open the dialog to let the
            # user load one.  The dialog will respect the current state.
            if self.sub_irf_solv_data is not None:
                self.sub_irf_applied = True
                self.update_all()
            self.open_solvent_irf_window()
            # If the user closes the dialog without loading and we still
            # have no solvent, untick the box (no silent state mismatch).
            if (self.sub_irf_solv_data is None
                    and self.sub_irf_fig is None):
                self.cb_sub_irf.blockSignals(True)
                self.cb_sub_irf.setChecked(False)
                self.cb_sub_irf.blockSignals(False)
        else:
            self.sub_irf_applied = False
            self.update_all()

    def open_crop_window(self):
        if self.original_deltaA is None:
            warn_box(self, 'No data', 'Load data first.')
            return
        self._open_dialog(CropDialog, 'crop_fig')

    def open_mask_window(self):
        if self.deltaA_raw is None:
            warn_box(self, 'No data', 'Load data first.')
            return
        self._open_dialog(MaskDialog, 'mask_fig')

    def open_chirp_window(self):
        if self.deltaA is None:
            warn_box(self, 'No data', 'Load data first.')
            return
        self._open_dialog(ChirpDialog, 'chirp_fig')

    def open_global_analysis_window(self):
        if self.deltaA is None:
            warn_box(self, 'No data', 'Load data first.')
            return
        self._open_dialog(GlobalAnalysisDialog, 'ga_fig')

    def open_svd_window(self):
        if self.deltaA is None:
            warn_box(self, 'No data', 'Load data first.')
            return
        from ta_svd import SVDDialog
        self._open_dialog(SVDDialog, 'svd_fig')

    def open_kinetic_fit_window(self):
        if self.deltaA is None:
            warn_box(self, 'No data', 'Load data first.')
            return
        # Lazy import: ta_kfit pulls in ta_core but nothing else, so
        # importing only when the user clicks keeps startup fast and
        # makes the dependency cycle obvious.
        from ta_kfit import KineticFitDialog
        self._open_dialog(KineticFitDialog, 'kfit_fig')

    def open_lda_window(self):
        if self.deltaA is None:
            warn_box(self, 'No data', 'Load data first.')
            return
        from ta_lda import LDADialog
        self._open_dialog(LDADialog, 'lda_fig')

    def open_mcr_window(self):
        if self.deltaA is None:
            warn_box(self, 'No data', 'Load data first.')
            return
        from ta_mcr import MCRDialog
        self._open_dialog(MCRDialog, 'mcr_fig')

    def open_coherence_window(self):
        if self.deltaA is None:
            warn_box(self, 'No data', 'Load data first.')
            return
        from ta_coherence import CoherenceDialog
        self._open_dialog(CoherenceDialog, 'coh_fig')

    # ================================================================
    # Clean shutdown
    # ================================================================
    def closeEvent(self, ev):
        for attr in ('bg_fig', 'chirp_fig', 'crop_fig',
                     'load_fig', 'ga_fig', 'mask_fig',
                     'kfit_fig', 'svd_fig', 'sub_irf_fig',
                     'lda_fig', 'mcr_fig', 'coh_fig'):
            d = getattr(self, attr, None)
            if d is not None:
                try:
                    d.close()
                except Exception:
                    pass
        super().closeEvent(ev)


# =====================================================================
# Entry point
# =====================================================================
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = TAAnalyzer()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
