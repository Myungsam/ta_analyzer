"""
TA Analyzer - Load Data (Accumulation) dialog.

Instead of loading one pre-averaged file, the user points at a folder full
of individual repeat measurements, previews each one as a 2D TA map, and
accumulates only the datasets whose maps look clean.

Workflow:
    1. Pick a folder (top row: path label + "Browse…" button).
    2. All parseable TA files in the folder are listed on the left, each
       with a checkbox on the right of its row.
    3. Clicking a row draws that file's 2D TA map on the right so the
       user can eyeball outliers / bad shots.
    4. "Accumulate & Load" averages the checked files (arithmetic mean
       over the shared λ×t grid) and pushes the result to the main
       window via ``app.set_loaded_data``.
"""
from __future__ import annotations

import os
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

from ta_widgets import (
    MplCanvas, style_button, warn_box,
    draw_heatmap, compute_zlim,
)
import ta_core


# File extensions we attempt to parse.  Everything else is ignored so the
# list isn't polluted by README.txt / notes.md / hidden files.
_TA_EXTENSIONS = ('.csv', '.tsv', '.dat', '.txt')


class LoadDataChooserDialog(QtWidgets.QDialog):
    """Small dialog that picks between the three data-loading modes.

    Result codes returned via ``self.mode`` after ``exec_()``:
        'standard'      → auto-detected CSV/DAT/TXT
        'custom'        → manual X/Y/Z region selection
        'accumulation'  → folder-based multi-file accumulation
    A cancel yields ``self.mode is None``.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode: str | None = None
        self.setWindowTitle('Load Data')
        self.setModal(True)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 12)
        outer.setSpacing(10)

        header = QtWidgets.QLabel('Choose how to load your TA data:')
        header.setStyleSheet('font-weight: bold; font-size: 12pt;')
        outer.addWidget(header)

        # Three big cards.  Each is a QPushButton laid out as a rich label
        # (title + wrapped description) so the user sees the trade-off
        # without reading a tooltip.
        modes = [
            ('standard',
             'Standard',
             'Load a single CSV / DAT / TXT file whose layout the '
             'parser can auto-detect (MATLAB style or 2-column metadata '
             'prefix).  Best for typical measurement outputs.',
             '#4c8cca'),
            ('custom',
             'Custom (manual X / Y / Z regions)',
             'Open a spreadsheet-like preview of a CSV and hand-pick the '
             'wavelength, delay, and ΔA regions.  Use this when the '
             'auto-detector fails on an unusual layout.',
             '#8c4db3'),
            ('accumulation',
             'Accumulation (folder of repeats)',
             'Point at a folder of repeated measurements, preview each '
             "file's 2D TA map, and average only the checked files.  "
             'Use this to drop bad shots before analysis.',
             '#4fa35a'),
        ]

        for mode_id, title, desc, bg in modes:
            btn = QtWidgets.QPushButton()
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setMinimumHeight(84)
            btn.setStyleSheet(
                f'QPushButton {{ text-align: left; padding: 10px 14px; '
                f'background: {bg}; color: white; border-radius: 6px; '
                f'border: none; }} '
                f'QPushButton:hover {{ background: {bg}; '
                f'border: 2px solid #ffffff; }}')
            # Rich title on top, wrapped description underneath.
            lay = QtWidgets.QVBoxLayout(btn)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(3)
            t = QtWidgets.QLabel(title)
            t.setStyleSheet('font-weight: bold; font-size: 11pt; '
                            'color: white; background: transparent;')
            d = QtWidgets.QLabel(desc)
            d.setWordWrap(True)
            d.setStyleSheet('color: rgba(255,255,255,0.92); '
                            'background: transparent;')
            lay.addWidget(t)
            lay.addWidget(d)
            btn.clicked.connect(lambda _=False, m=mode_id: self._choose(m))
            outer.addWidget(btn)

        # Bottom row: single Cancel button
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        btn_cancel = QtWidgets.QPushButton('Cancel')
        btn_cancel.clicked.connect(self.reject)
        row.addWidget(btn_cancel)
        outer.addLayout(row)

        self.setFixedWidth(560)
        self.adjustSize()

    def _choose(self, mode: str):
        self.mode = mode
        self.accept()


class AccumulationDialog(QtWidgets.QDialog):
    """Folder-based multi-file selector with per-file 2D map preview."""

    def __init__(self, parent, app, initial_folder: str = ''):
        super().__init__(parent)
        self.app = app
        self.setWindowTitle('Load Data (Accumulation)')
        self.resize(1280, 800)

        # cached entries: list of dicts with parsed arrays
        # keys: 'name', 'path', 'wavelength', 'delay', 'deltaA',
        #       'include', 'shape', 'max_abs', 'error'
        self.files: list[dict] = []
        self.sz_ref: tuple | None = None
        self._current_row = -1
        self._suppress_item_signal = False

        self._build_ui()

        if initial_folder and os.path.isdir(initial_folder):
            self._load_folder(initial_folder)

    # ---------------------------------------------------------------
    # UI construction
    # ---------------------------------------------------------------
    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # Top row: folder path + browse
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel('Folder:'))
        self.ed_folder = QtWidgets.QLineEdit()
        self.ed_folder.setReadOnly(True)
        self.ed_folder.setPlaceholderText(
            'Choose a folder containing repeated TA measurements…')
        top.addWidget(self.ed_folder, stretch=1)
        self.btn_browse = QtWidgets.QPushButton('Browse Folder…')
        self.btn_browse.clicked.connect(self._on_browse)
        top.addWidget(self.btn_browse)
        outer.addLayout(top)

        # Middle: horizontal splitter -> left table, right preview
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # ---- Left: file table ----
        left = QtWidgets.QWidget()
        left_lay = QtWidgets.QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(4)

        select_row = QtWidgets.QHBoxLayout()
        self.btn_all = QtWidgets.QPushButton('Check all')
        self.btn_none = QtWidgets.QPushButton('Uncheck all')
        self.btn_all.clicked.connect(lambda: self._set_all(True))
        self.btn_none.clicked.connect(lambda: self._set_all(False))
        select_row.addWidget(self.btn_all)
        select_row.addWidget(self.btn_none)
        select_row.addStretch(1)
        left_lay.addLayout(select_row)

        self.tbl = QtWidgets.QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(['File', 'Shape', 'Include'])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self.tbl.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        hdr = self.tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.tbl.itemChanged.connect(self._on_item_changed)
        self.tbl.currentCellChanged.connect(self._on_current_cell_changed)
        left_lay.addWidget(self.tbl, stretch=1)

        self.lbl_summary = QtWidgets.QLabel('No folder loaded.')
        self.lbl_summary.setWordWrap(True)
        left_lay.addWidget(self.lbl_summary)

        splitter.addWidget(left)

        # ---- Right: TA map preview ----
        right = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(4)

        self.lbl_preview = QtWidgets.QLabel(
            'Select a file on the left to preview its TA map.')
        self.lbl_preview.setStyleSheet('font-weight: bold;')
        right_lay.addWidget(self.lbl_preview)

        self.canvas = MplCanvas(self, nrows=1, ncols=1, figsize=(7, 6))
        self.ax = self.canvas.axes_list[0]
        self._cbar = None
        right_lay.addWidget(self.canvas.with_toolbar(self), stretch=1)

        # y-scale toggle for the preview
        opt_row = QtWidgets.QHBoxLayout()
        opt_row.addWidget(QtWidgets.QLabel('Delay scale:'))
        self.dd_yscale = QtWidgets.QComboBox()
        self.dd_yscale.addItems(['Linear', 'Log'])
        self.dd_yscale.currentTextChanged.connect(self._redraw_preview)
        opt_row.addWidget(self.dd_yscale)
        opt_row.addStretch(1)
        right_lay.addLayout(opt_row)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([420, 820])
        outer.addWidget(splitter, stretch=1)

        # Bottom action row
        act = QtWidgets.QHBoxLayout()
        act.addStretch(1)
        self.btn_cancel = QtWidgets.QPushButton('Cancel')
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_accum = QtWidgets.QPushButton('Accumulate && Load')
        style_button(self.btn_accum, bg='#4fa35a', fg='white')
        self.btn_accum.clicked.connect(self._on_accumulate)
        act.addWidget(self.btn_cancel)
        act.addWidget(self.btn_accum)
        outer.addLayout(act)

    # ---------------------------------------------------------------
    # Folder browsing / parsing
    # ---------------------------------------------------------------
    def _on_browse(self):
        start = self.ed_folder.text() or ''
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select folder containing TA data files', start)
        if not folder:
            return
        self._load_folder(folder)

    def _load_folder(self, folder: str):
        self.ed_folder.setText(folder)
        try:
            entries = sorted(os.listdir(folder))
        except OSError as e:
            warn_box(self, 'Folder Error', f'Cannot list folder:\n{e}')
            return

        candidates = [
            os.path.join(folder, n) for n in entries
            if os.path.isfile(os.path.join(folder, n))
            and n.lower().endswith(_TA_EXTENSIONS)
        ]
        if not candidates:
            self.files = []
            self.sz_ref = None
            self._refresh_table()
            self.lbl_summary.setText(
                f'No TA files found in "{folder}".  '
                f'Looking for {", ".join(_TA_EXTENSIONS)}.')
            self._clear_preview('No files.')
            return

        # Parse everything up-front so subsequent row-clicks are instant.
        # We track two orthogonal states per file:
        #   parse_error : file couldn't be read at all (excluded + no preview)
        #   error       : final combined status (parse OR shape mismatch)
        # Anything that parses successfully is previewable, even if its
        # shape rules it out of the accumulation set.
        parsed: list[dict] = []
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for p in candidates:
                name = os.path.basename(p)
                try:
                    wl, t, A = ta_core.parse_data_file(p)
                    parsed.append({
                        'name': name, 'path': p,
                        'wavelength': wl, 'delay': t, 'deltaA': A,
                        'include': True,
                        'shape': A.shape,
                        'max_abs': float(np.nanmax(np.abs(A)))
                            if A.size else float('nan'),
                        'parse_error': None,
                        'error': None,
                    })
                except Exception as e:
                    parsed.append({
                        'name': name, 'path': p,
                        'wavelength': None, 'delay': None, 'deltaA': None,
                        'include': False,
                        'shape': None,
                        'max_abs': float('nan'),
                        'parse_error': str(e),
                        'error': str(e),
                    })
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        # Determine the reference shape from the MAJORITY of successfully
        # parsed files (mode), not the alphabetically first one.  A typical
        # measurement folder contains many raw repeats (same shape) plus a
        # handful of analysis outputs (DADS/EADS/GA_fit/... with a smaller
        # cropped shape). Picking the first alphabetically often lands on
        # one of those outputs and wrongly rejects the real measurements.
        ok = [fi for fi in parsed if fi['error'] is None]
        if ok:
            from collections import Counter
            shape_counts = Counter(fi['shape'] for fi in ok)
            self.sz_ref, _ = shape_counts.most_common(1)[0]
        else:
            self.sz_ref = None

        # Files with a different shape are still "parseable" (previewable)
        # but tagged as shape-mismatch so they stay unchecked. The error
        # message tells the user why they can't be included.
        if self.sz_ref is not None:
            for fi in parsed:
                if fi['error'] is None and fi['shape'] != self.sz_ref:
                    fi['error'] = (
                        f'Shape mismatch (got {fi["shape"][0]}x'
                        f'{fi["shape"][1]}, majority '
                        f'{self.sz_ref[0]}x{self.sz_ref[1]})')
                    fi['include'] = False

        self.files = parsed
        self._refresh_table()
        self._update_summary()

        # Auto-select first parseable file for preview.
        first_ok = next((i for i, fi in enumerate(self.files)
                        if fi['error'] is None), -1)
        if first_ok >= 0:
            self.tbl.selectRow(first_ok)
        else:
            self._clear_preview('No parseable file in this folder.')

    # ---------------------------------------------------------------
    # Table helpers
    # ---------------------------------------------------------------
    def _refresh_table(self):
        self._suppress_item_signal = True
        self.tbl.setRowCount(0)
        for i, fi in enumerate(self.files):
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)

            # Column 0: file name.  Colour code the three states:
            #   red    = parse failed (not loadable at all)
            #   grey   = parsed but shape mismatch (previewable, not includable)
            #   normal = parsed and shape matches reference (fully usable)
            item_name = QtWidgets.QTableWidgetItem(fi['name'])
            if fi['parse_error'] is not None:
                item_name.setForeground(QtCore.Qt.red)
                item_name.setToolTip(
                    f'{fi["path"]}\n\nParse error: {fi["parse_error"]}')
            elif fi['error'] is not None:
                item_name.setForeground(QtGui.QColor('#888'))
                item_name.setToolTip(
                    f'{fi["path"]}\n\n{fi["error"]}')
            else:
                item_name.setToolTip(fi['path'])
            self.tbl.setItem(r, 0, item_name)

            # Column 1: shape
            if fi['shape'] is not None:
                shape_txt = f'{fi["shape"][0]}x{fi["shape"][1]}'
            else:
                shape_txt = '—'
            item_shape = QtWidgets.QTableWidgetItem(shape_txt)
            item_shape.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tbl.setItem(r, 1, item_shape)

            # Column 2: include checkbox
            item_inc = QtWidgets.QTableWidgetItem()
            flags = QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled
            if fi['error'] is not None:
                # Disable checkbox on rows that failed to parse or don't
                # match the reference shape.
                flags = QtCore.Qt.ItemIsUserCheckable
                item_inc.setToolTip(fi['error'])
            item_inc.setFlags(flags)
            item_inc.setCheckState(
                QtCore.Qt.Checked if fi['include'] else QtCore.Qt.Unchecked)
            item_inc.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tbl.setItem(r, 2, item_inc)
        self._suppress_item_signal = False

    def _on_item_changed(self, item):
        if self._suppress_item_signal:
            return
        if item.column() != 2:
            return
        row = item.row()
        if not (0 <= row < len(self.files)):
            return
        fi = self.files[row]
        if fi['error'] is not None:
            # Revert - can't include a broken file.
            self._suppress_item_signal = True
            item.setCheckState(QtCore.Qt.Unchecked)
            self._suppress_item_signal = False
            return
        fi['include'] = (item.checkState() == QtCore.Qt.Checked)
        self._update_summary()

    def _on_current_cell_changed(self, cur_row, cur_col, _prev_row, _prev_col):
        if cur_row == self._current_row:
            return
        self._current_row = cur_row
        if 0 <= cur_row < len(self.files):
            self._draw_preview(cur_row)

    def _set_all(self, checked: bool):
        state = QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
        self._suppress_item_signal = True
        for r, fi in enumerate(self.files):
            if fi['error'] is not None:
                fi['include'] = False
                self.tbl.item(r, 2).setCheckState(QtCore.Qt.Unchecked)
                continue
            fi['include'] = checked
            self.tbl.item(r, 2).setCheckState(state)
        self._suppress_item_signal = False
        self._update_summary()

    def _update_summary(self):
        if not self.files:
            self.lbl_summary.setText('No folder loaded.')
            return
        n_total = len(self.files)
        n_parse_ok = sum(1 for fi in self.files if fi['parse_error'] is None)
        n_match = sum(1 for fi in self.files
                      if fi['parse_error'] is None
                      and (self.sz_ref is None or fi['shape'] == self.sz_ref))
        n_chk = sum(1 for fi in self.files if fi['include'])
        parts = [f'{n_total} file(s) in folder',
                 f'{n_parse_ok} loadable']
        if self.sz_ref is not None and n_match != n_parse_ok:
            parts.append(
                f'{n_match} match majority shape '
                f'{self.sz_ref[0]}x{self.sz_ref[1]}')
        parts.append(f'{n_chk} checked for accumulation')
        self.lbl_summary.setText(' · '.join(parts))

    # ---------------------------------------------------------------
    # Preview drawing
    # ---------------------------------------------------------------
    def _clear_preview(self, msg: str = ''):
        self.ax.clear()
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        if msg:
            self.ax.text(0.5, 0.5, msg, ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=11,
                         color='#666')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw_idle()

    def _redraw_preview(self, *_):
        if 0 <= self._current_row < len(self.files):
            self._draw_preview(self._current_row)

    def _draw_preview(self, row: int):
        fi = self.files[row]
        # If parsing itself failed there is no matrix to draw.  A
        # shape-mismatch entry still has valid ΔA and gets a normal
        # preview so the user can eyeball whether it's an analysis output
        # or a mis-shaped acquisition.
        if fi['deltaA'] is None:
            reason = fi['parse_error'] or fi['error'] or 'no data'
            self.lbl_preview.setText(
                f'{fi["name"]}   (cannot preview: {reason})')
            self._clear_preview(reason)
            return

        wl = fi['wavelength']
        t = fi['delay']
        A = fi['deltaA']

        # Reset axes fully (colorbar handling: rebuild figure state).
        # draw_heatmap creates its own colorbar; remove the previous one
        # first so they don't stack across selections.
        self.ax.clear()
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        c_min, c_max = compute_zlim(A)
        cmap_array = ta_core.get_colormap_array('turbo', 256)
        y_scale = 'log' if self.dd_yscale.currentText() == 'Log' else 'linear'
        try:
            _im, self._cbar = draw_heatmap(
                self.ax, wl, t, A, cmap_array, (c_min, c_max),
                y_scale=y_scale, show_cbar=True)
        except Exception as e:
            self._clear_preview(f'Draw failed: {e}')
            return
        self.ax.set_xlabel('Wavelength (nm)')
        self.ax.set_ylabel(f'Delay time ({self.app.t_unit_ax()})')
        chk_state = 'INCLUDED' if fi['include'] else 'excluded'
        self.ax.set_title(
            f'{fi["name"]}   [{chk_state}]   '
            f'max|ΔA| = {fi["max_abs"]:.3g}',
            fontsize=10)
        self.canvas.draw_idle()
        self.lbl_preview.setText(f'Preview: {fi["name"]}')

    # ---------------------------------------------------------------
    # Accumulate & finish
    # ---------------------------------------------------------------
    def _on_accumulate(self):
        incl = [fi for fi in self.files
                if fi['include'] and fi['error'] is None]
        if not incl:
            warn_box(self, 'Nothing selected',
                     'Check at least one file to accumulate.')
            return

        wl = incl[0]['wavelength']
        t = incl[0]['delay']

        # All shapes are already reference-checked in _load_folder, but the
        # axes themselves (values) could still differ if two files share the
        # (M, N) shape but were measured on different grids. Reject that
        # explicitly rather than silently blending grids.
        for fi in incl[1:]:
            if (fi['wavelength'].shape != wl.shape
                    or not np.allclose(fi['wavelength'], wl,
                                       rtol=1e-4, atol=1e-6)):
                warn_box(self, 'Grid mismatch',
                         f'{fi["name"]}: wavelength axis differs from '
                         f'{incl[0]["name"]}.  Cannot accumulate.')
                return
            if (fi['delay'].shape != t.shape
                    or not np.allclose(fi['delay'], t,
                                       rtol=1e-4, atol=1e-6)):
                warn_box(self, 'Grid mismatch',
                         f'{fi["name"]}: delay axis differs from '
                         f'{incl[0]["name"]}.  Cannot accumulate.')
                return

        stack = np.stack([fi['deltaA'] for fi in incl], axis=2)
        avg = np.nanmean(stack, axis=2)

        n = len(incl)
        first = incl[0]['name']
        if n == 1:
            desc = f'Loaded (accum): {first}'
        else:
            desc = f'Accumulated {n} files ({first}…)'

        folder = self.ed_folder.text().strip()
        self.app.set_loaded_data(wl, t, avg, desc,
                                 source_dir=folder)
        self.accept()
