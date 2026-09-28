"""
TA Analyzer - Custom (manual) CSV loader.

For measurement files whose layout differs from the two formats handled
by ``ta_core.parse_data_file``.  Opens a spreadsheet-like preview of the
CSV and asks the user to drag-select, in order:

    1. X-axis (wavelength)  - must be a 1D row or 1D column
    2. Y-axis (delay)       - must be a 1D row or 1D column
    3. Z-matrix (deltaA)    - 2D block whose shape matches (X, Y) or its
                              transpose.  We auto-transpose on extraction
                              so the rest of the app always receives
                              (M wavelengths) x (N delays).
"""
from __future__ import annotations

import os
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

from ta_widgets import warn_box, info_box
import ta_core


def _col_letter(idx: int) -> str:
    """Excel-style column header for a 0-based column index (A, B, ..., AA)."""
    s = ''
    n = idx + 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        s = chr(ord('A') + rem) + s
    return s


def _detect_delimiter(text: str) -> str | None:
    """Pick comma / tab / semicolon from the first non-empty line, else None
    (whitespace-split)."""
    first = next((ln for ln in text.splitlines() if ln.strip()), '')
    if not first:
        return None
    n_tab = first.count('\t')
    n_comma = first.count(',')
    n_semi = first.count(';')
    best = max(n_tab, n_comma, n_semi)
    if best == 0:
        return None
    if n_tab == best:
        return '\t'
    if n_comma == best:
        return ','
    return ';'


class CustomLoadDialog(QtWidgets.QDialog):
    """Preview + manual region-selection dialog for non-standard CSVs."""

    # Stage constants
    STAGE_X = 'x'
    STAGE_Y = 'y'
    STAGE_Z = 'z'
    STAGE_DONE = 'done'

    def __init__(self, parent=None, file_path: str | None = None):
        super().__init__(parent)
        self.setWindowTitle('Load Data - Custom Selection')
        self.resize(1200, 760)

        self.file_path: str | None = None
        self.raw_data: list[list[str]] = []         # cell strings as displayed
        self.numeric_data: np.ndarray | None = None  # same shape, NaN for non-numeric
        self.delim: str | None = None

        # Selection ranges: (r1, c1, r2, c2) inclusive
        self.x_range: tuple[int, int, int, int] | None = None
        self.y_range: tuple[int, int, int, int] | None = None
        self.z_range: tuple[int, int, int, int] | None = None
        self.stage: str = self.STAGE_X
        # Cached result
        self._result: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        # Anchor cell for Ctrl+Shift+Arrow extension — set on every plain
        # (no-Shift) click, frozen while Shift is held.
        self._sel_anchor: tuple[int, int] = (0, 0)

        self._build_ui()
        if file_path:
            self._load_file(file_path)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)

        # File row
        row_file = QtWidgets.QHBoxLayout()
        row_file.addWidget(QtWidgets.QLabel('File:'))
        self.lbl_file = QtWidgets.QLabel('(no file selected)')
        self.lbl_file.setStyleSheet('color:#444;')
        row_file.addWidget(self.lbl_file, 1)
        btn_browse = QtWidgets.QPushButton('Browse...')
        btn_browse.clicked.connect(self._on_browse)
        row_file.addWidget(btn_browse)
        outer.addLayout(row_file)

        # Stage banner
        self.lbl_stage = QtWidgets.QLabel()
        self.lbl_stage.setStyleSheet(
            'background:#264f8e; color:white; padding:6px 8px; '
            'font-weight:600; border-radius:3px;')
        outer.addWidget(self.lbl_stage)

        # Current-selection readout
        self.lbl_selection = QtWidgets.QLabel('Current selection: (none)')
        self.lbl_selection.setStyleSheet('color:#222; padding:2px;')
        outer.addWidget(self.lbl_selection)

        # Table (Excel-like)
        self.table = QtWidgets.QTableWidget()
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectItems)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.ContiguousSelection)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        # Track anchor on plain (no-Shift) cell presses so Ctrl+Shift+Arrow
        # can extend FROM that anchor.
        self.table.cellPressed.connect(self._on_table_cell_pressed)
        # Catch Ctrl(+Shift)+Arrow before the table's default handler so we
        # can implement Excel-style jump-to-data-edge.
        self.table.installEventFilter(self)
        # Make headers behave like Excel
        self.table.horizontalHeader().setDefaultSectionSize(96)
        self.table.verticalHeader().setDefaultSectionSize(20)
        outer.addWidget(self.table, 1)

        # Help line for shortcuts
        lbl_hint = QtWidgets.QLabel(
            'Tip: drag to select.  Ctrl+Shift+Arrow extends to the end of '
            'the contiguous data run (Excel-style).  Ctrl+Arrow jumps the '
            'cursor without extending.')
        lbl_hint.setStyleSheet('color:#666; font-size:11px;')
        outer.addWidget(lbl_hint)

        # Action buttons (per-stage)
        row_act = QtWidgets.QHBoxLayout()
        self.btn_confirm = QtWidgets.QPushButton('Confirm This Selection')
        self.btn_confirm.setStyleSheet(
            'background:#2c7a3d; color:white; font-weight:600; padding:4px 10px;')
        self.btn_confirm.clicked.connect(self._on_confirm_selection)
        row_act.addWidget(self.btn_confirm)

        self.btn_clear = QtWidgets.QPushButton('Clear Current Selection')
        self.btn_clear.clicked.connect(self._on_clear_selection)
        row_act.addWidget(self.btn_clear)

        self.btn_back = QtWidgets.QPushButton('Back to Previous Step')
        self.btn_back.clicked.connect(self._on_back)
        row_act.addWidget(self.btn_back)

        self.btn_reset = QtWidgets.QPushButton('Reset All')
        self.btn_reset.clicked.connect(self._on_reset_all)
        row_act.addWidget(self.btn_reset)

        self.btn_auto = QtWidgets.QPushButton('Auto-detect')
        self.btn_auto.setToolTip(
            'Try the built-in format detector and pre-fill all three '
            'regions.  You can then adjust them or just hit Load.')
        self.btn_auto.clicked.connect(self._on_auto_detect)
        row_act.addWidget(self.btn_auto)
        row_act.addStretch(1)
        outer.addLayout(row_act)

        # Summary of confirmed ranges
        grp = QtWidgets.QGroupBox('Confirmed selections')
        gl = QtWidgets.QGridLayout(grp)
        self.lbl_x = QtWidgets.QLabel()
        self.lbl_y = QtWidgets.QLabel()
        self.lbl_z = QtWidgets.QLabel()
        for w in (self.lbl_x, self.lbl_y, self.lbl_z):
            w.setStyleSheet('padding:2px 4px;')
        gl.addWidget(self.lbl_x, 0, 0)
        gl.addWidget(self.lbl_y, 1, 0)
        gl.addWidget(self.lbl_z, 2, 0)
        outer.addWidget(grp)

        # OK / Cancel
        bbox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.btn_ok = bbox.button(QtWidgets.QDialogButtonBox.Ok)
        self.btn_ok.setText('Load')
        self.btn_ok.setEnabled(False)
        bbox.accepted.connect(self._on_accept)
        bbox.rejected.connect(self.reject)
        outer.addWidget(bbox)

        self._refresh_state_ui()

    # ------------------------------------------------------------------
    # File parsing / table population
    # ------------------------------------------------------------------
    def _on_browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select CSV file', '',
            'CSV (*.csv);;All files (*)')
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        except Exception as e:
            warn_box(self, 'Read Error', f'Failed to open file:\n{e}')
            return
        if not text.strip():
            warn_box(self, 'Empty File', 'The file is empty.')
            return

        self.delim = _detect_delimiter(text)
        rows: list[list[str]] = []
        for line in text.splitlines():
            if not line and not rows:
                continue
            if self.delim is None:
                cells = line.split()
            else:
                cells = line.split(self.delim)
            rows.append([c.strip() for c in cells])
        if not rows:
            warn_box(self, 'Empty File', 'No data rows found.')
            return
        max_cols = max(len(r) for r in rows)
        for r in rows:
            if len(r) < max_cols:
                r.extend([''] * (max_cols - len(r)))

        # Hard cap to keep the UI responsive — these are big TA files.
        MAX_ROWS = 4000
        MAX_COLS = 1000
        truncated = False
        if len(rows) > MAX_ROWS:
            rows = rows[:MAX_ROWS]
            truncated = True
        if max_cols > MAX_COLS:
            rows = [r[:MAX_COLS] for r in rows]
            max_cols = MAX_COLS
            truncated = True

        self.raw_data = rows
        n_rows = len(rows)
        n_cols = max_cols
        num = np.full((n_rows, n_cols), np.nan)
        for i, r in enumerate(rows):
            for j in range(n_cols):
                s = r[j]
                if not s:
                    continue
                low = s.lower()
                if low in ('nan', 'na', 'n/a', '-'):
                    continue
                try:
                    num[i, j] = float(s)
                except ValueError:
                    pass
        self.numeric_data = num

        self.file_path = path
        self.lbl_file.setText(path)
        self._populate_table(rows)

        # Reset selection state
        self.x_range = None
        self.y_range = None
        self.z_range = None
        self.stage = self.STAGE_X
        self._result = None
        self.table.clearSelection()
        self._refresh_state_ui()

        if truncated:
            info_box(
                self, 'Preview Truncated',
                f'The file is larger than the preview limit; only the first '
                f'{n_rows} rows × {n_cols} columns are shown.  This is fine '
                f'for selecting axis/matrix regions provided your data lies '
                f'inside that window.')

    def _populate_table(self, rows: list[list[str]]):
        self.table.blockSignals(True)
        self.table.clear()
        n_rows = len(rows)
        n_cols = max((len(r) for r in rows), default=0)
        self.table.setRowCount(n_rows)
        self.table.setColumnCount(n_cols)
        self.table.setHorizontalHeaderLabels(
            [_col_letter(j) for j in range(n_cols)])
        self.table.setVerticalHeaderLabels(
            [str(i + 1) for i in range(n_rows)])

        num = self.numeric_data
        color_num = QtGui.QColor(255, 255, 255)
        color_meta = QtGui.QColor(255, 247, 209)   # light yellow = non-numeric
        color_blank = QtGui.QColor(244, 244, 244)
        for i, r in enumerate(rows):
            for j, s in enumerate(r):
                item = QtWidgets.QTableWidgetItem(s)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                if not s:
                    item.setBackground(color_blank)
                elif np.isfinite(num[i, j]):
                    item.setBackground(color_num)
                    item.setTextAlignment(QtCore.Qt.AlignRight
                                          | QtCore.Qt.AlignVCenter)
                else:
                    item.setBackground(color_meta)
                self.table.setItem(i, j, item)
        # Excel-like column widths
        self.table.resizeColumnsToContents()
        # Cap excessively wide columns
        hdr = self.table.horizontalHeader()
        for j in range(n_cols):
            w = hdr.sectionSize(j)
            if w > 180:
                self.table.setColumnWidth(j, 180)
            elif w < 60:
                self.table.setColumnWidth(j, 60)
        self.table.blockSignals(False)

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------
    def _current_selection(self) -> tuple[int, int, int, int] | None:
        ranges = self.table.selectedRanges()
        if not ranges:
            return None
        r1 = min(rng.topRow() for rng in ranges)
        r2 = max(rng.bottomRow() for rng in ranges)
        c1 = min(rng.leftColumn() for rng in ranges)
        c2 = max(rng.rightColumn() for rng in ranges)
        if r1 < 0 or c1 < 0:
            return None
        return (r1, c1, r2, c2)

    def _on_selection_changed(self):
        sel = self._current_selection()
        if sel is None:
            self.lbl_selection.setText('Current selection: (none)')
            return
        r1, c1, r2, c2 = sel
        a1 = f"{_col_letter(c1)}{r1 + 1}"
        a2 = f"{_col_letter(c2)}{r2 + 1}"
        nr = r2 - r1 + 1
        nc = c2 - c1 + 1
        shape = '1D column' if nc == 1 and nr > 1 else (
            '1D row' if nr == 1 and nc > 1 else
            'single cell' if nr == 1 and nc == 1 else
            f'2D ({nr} x {nc})')
        self.lbl_selection.setText(
            f'Current selection: {a1}:{a2}   ({nr} rows x {nc} cols, {shape})')

    @staticmethod
    def _is_1d(sel: tuple[int, int, int, int]) -> bool:
        r1, c1, r2, c2 = sel
        return (r1 == r2) or (c1 == c2)

    @staticmethod
    def _length_1d(sel: tuple[int, int, int, int]) -> int:
        r1, c1, r2, c2 = sel
        return max(r2 - r1 + 1, c2 - c1 + 1)

    def _values_1d(self, sel: tuple[int, int, int, int]) -> np.ndarray:
        r1, c1, r2, c2 = sel
        return self.numeric_data[r1:r2 + 1, c1:c2 + 1].ravel()

    # ------------------------------------------------------------------
    # Excel-style keyboard navigation (Ctrl+Arrow, Ctrl+Shift+Arrow)
    # ------------------------------------------------------------------
    def _on_table_cell_pressed(self, row: int, col: int):
        """Track the selection anchor for Ctrl+Shift+Arrow extension."""
        mods = QtWidgets.QApplication.keyboardModifiers()
        if not (mods & QtCore.Qt.ShiftModifier):
            self._sel_anchor = (row, col)

    def eventFilter(self, source, event):
        if (source is self.table
                and event.type() == QtCore.QEvent.KeyPress
                and self.numeric_data is not None):
            mods = event.modifiers()
            key = event.key()
            arrows = (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down,
                      QtCore.Qt.Key_Left, QtCore.Qt.Key_Right)
            if key in arrows:
                ctrl = bool(mods & QtCore.Qt.ControlModifier)
                shift = bool(mods & QtCore.Qt.ShiftModifier)
                if ctrl and shift:
                    self._extend_to_data_edge(key)
                    return True
                if ctrl and not shift:
                    self._jump_to_data_edge(key)
                    return True
        return super().eventFilter(source, event)

    def _data_edge_target(self, start_r: int, start_c: int,
                          key: int) -> tuple[int, int]:
        """Excel-style 'jump to data edge'.

        - On a data cell whose neighbor in ``key`` direction is also data:
          go to the LAST contiguous data cell before a blank/NaN.
        - On a data cell whose neighbor is blank/NaN: go to the NEXT data
          cell in that direction.
        - On a blank/NaN cell: go to the NEXT data cell in that direction.
        - If no data cell is reachable, land on the boundary of the table.
        """
        num = self.numeric_data
        n_rows, n_cols = num.shape

        if key == QtCore.Qt.Key_Right:
            dr, dc = 0, 1
        elif key == QtCore.Qt.Key_Left:
            dr, dc = 0, -1
        elif key == QtCore.Qt.Key_Down:
            dr, dc = 1, 0
        else:  # Key_Up
            dr, dc = -1, 0

        def in_bounds(r, c):
            return 0 <= r < n_rows and 0 <= c < n_cols

        def is_data(r, c):
            return in_bounds(r, c) and np.isfinite(num[r, c])

        def edge(r, c):
            # Last in-bounds cell when stepping (dr, dc) from (r, c).
            if dc > 0:
                return r, n_cols - 1
            if dc < 0:
                return r, 0
            if dr > 0:
                return n_rows - 1, c
            return 0, c

        if not in_bounds(start_r, start_c):
            return start_r, start_c

        nr, nc = start_r + dr, start_c + dc
        if not in_bounds(nr, nc):
            return start_r, start_c

        if is_data(start_r, start_c) and is_data(nr, nc):
            # Walk to last data before a blank
            while is_data(nr + dr, nc + dc):
                nr, nc = nr + dr, nc + dc
            return nr, nc

        # Otherwise: scan forward to next data cell
        while in_bounds(nr, nc) and not is_data(nr, nc):
            nr, nc = nr + dr, nc + dc
        if in_bounds(nr, nc):
            return nr, nc
        # Nothing found — go to table edge
        return edge(start_r, start_c)

    def _extend_to_data_edge(self, key: int):
        cur = self.table.currentIndex()
        if not cur.isValid():
            return
        target_r, target_c = self._data_edge_target(
            cur.row(), cur.column(), key)
        ar, ac = self._sel_anchor
        r1, r2 = sorted((ar, target_r))
        c1, c2 = sorted((ac, target_c))
        self.table.clearSelection()
        sel = QtWidgets.QTableWidgetSelectionRange(r1, c1, r2, c2)
        self.table.setRangeSelected(sel, True)
        # Move the cursor to the new target without disturbing the
        # selection we just set.
        self.table.setCurrentCell(
            target_r, target_c, QtCore.QItemSelectionModel.NoUpdate)
        item = self.table.item(target_r, target_c)
        if item is not None:
            self.table.scrollToItem(
                item, QtWidgets.QAbstractItemView.EnsureVisible)

    def _jump_to_data_edge(self, key: int):
        cur = self.table.currentIndex()
        if not cur.isValid():
            return
        target_r, target_c = self._data_edge_target(
            cur.row(), cur.column(), key)
        self.table.setCurrentCell(target_r, target_c)  # default: clear+select
        self._sel_anchor = (target_r, target_c)
        item = self.table.item(target_r, target_c)
        if item is not None:
            self.table.scrollToItem(
                item, QtWidgets.QAbstractItemView.EnsureVisible)

    # ------------------------------------------------------------------
    # Stage handlers
    # ------------------------------------------------------------------
    def _on_confirm_selection(self):
        if self.numeric_data is None:
            warn_box(self, 'No File', 'Load a file first (Browse...).')
            return
        sel = self._current_selection()
        if sel is None:
            warn_box(self, 'No Selection',
                     'Please drag-select a region in the table first.')
            return

        if self.stage == self.STAGE_X:
            if not self._is_1d(sel):
                warn_box(self, 'Invalid Selection',
                         'X-axis (wavelength) must be a 1D array.\n'
                         'Please select EXACTLY ONE row or ONE column.')
                return
            vals = self._values_1d(sel)
            if vals.size < 2 or not np.all(np.isfinite(vals)):
                warn_box(self, 'Non-numeric / Too Short',
                         'X-axis selection must contain at least 2 numeric '
                         'cells with no blank or text entries.')
                return
            self.x_range = sel
            self.stage = self.STAGE_Y
            self.table.clearSelection()

        elif self.stage == self.STAGE_Y:
            if not self._is_1d(sel):
                warn_box(self, 'Invalid Selection',
                         'Y-axis (delay) must be a 1D array.\n'
                         'Please select EXACTLY ONE row or ONE column.')
                return
            vals = self._values_1d(sel)
            if vals.size < 2 or not np.all(np.isfinite(vals)):
                warn_box(self, 'Non-numeric / Too Short',
                         'Y-axis selection must contain at least 2 numeric '
                         'cells with no blank or text entries.')
                return
            self.y_range = sel
            self.stage = self.STAGE_Z
            self.table.clearSelection()

        elif self.stage == self.STAGE_Z:
            r1, c1, r2, c2 = sel
            nr = r2 - r1 + 1
            nc = c2 - c1 + 1
            x_len = self._length_1d(self.x_range)
            y_len = self._length_1d(self.y_range)
            # Accept either orientation (we'll transpose on extract).
            if not ((nr == x_len and nc == y_len)
                    or (nr == y_len and nc == x_len)):
                warn_box(
                    self, 'Dimension Mismatch',
                    f'Z-matrix shape ({nr} x {nc}) does not match the X-axis '
                    f'({x_len} points) and Y-axis ({y_len} points).\n\n'
                    f'Expected ({x_len} x {y_len}) or ({y_len} x {x_len}).\n'
                    'Please reselect the Z-matrix region.')
                return
            self.z_range = sel
            self.stage = self.STAGE_DONE

        elif self.stage == self.STAGE_DONE:
            info_box(self, 'All set',
                     'All three regions are confirmed.  Click "Load" to import.')

        self._refresh_state_ui()

    def _on_clear_selection(self):
        # Clear the most recent confirmed range (or just the table selection
        # if nothing has been confirmed yet at the current stage).
        if self.stage == self.STAGE_DONE:
            self.z_range = None
            self.stage = self.STAGE_Z
        elif self.stage == self.STAGE_Z and self.z_range is None:
            # nothing confirmed at z yet -> just clear table highlight
            pass
        elif self.stage == self.STAGE_Y and self.y_range is None:
            pass
        elif self.stage == self.STAGE_X and self.x_range is None:
            pass
        self.table.clearSelection()
        self._refresh_state_ui()

    def _on_back(self):
        if self.stage == self.STAGE_DONE:
            self.stage = self.STAGE_Z
            self.z_range = None
        elif self.stage == self.STAGE_Z:
            self.stage = self.STAGE_Y
            self.y_range = None
        elif self.stage == self.STAGE_Y:
            self.stage = self.STAGE_X
            self.x_range = None
        elif self.stage == self.STAGE_X:
            # already at first step
            self.x_range = None
        self.table.clearSelection()
        self._refresh_state_ui()

    def _on_reset_all(self):
        self.x_range = None
        self.y_range = None
        self.z_range = None
        self.stage = self.STAGE_X
        self.table.clearSelection()
        self._refresh_state_ui()

    def _on_auto_detect(self):
        if self.numeric_data is None or self.file_path is None:
            warn_box(self, 'No File', 'Load a file first.')
            return
        try:
            wl, t, A = ta_core.parse_data_file(self.file_path)
        except Exception as e:
            warn_box(self, 'Auto-detect Failed',
                     f'Could not auto-detect the format:\n{e}\n\n'
                     'Please select the regions manually.')
            return

        # Decide format from the very first cell of the file -- mirrors
        # the logic in ta_core.parse_data_file:
        #   Format A: corner cell numeric or empty; delays in row 0 col
        #     1.., wavelengths in col 0 row 1.., matrix at (1.., 1..)
        #   Format B: corner cell holds metadata text (e.g. "Integration
        #     time:"); the entire column 0 is metadata.  Wavelengths sit
        #     in col 1 row 1.., delays in row 0 col 2.. (col 1 row 0 is
        #     a numeric "corner" placeholder that parse_data_file skips),
        #     matrix at (1.., 2..).
        num = self.numeric_data
        first_cell_numeric = np.isfinite(num[0, 0])
        n_rows, n_cols = num.shape

        if first_cell_numeric:
            # Format A
            x_off_r, x_off_c = 1, 0
            y_off_r, y_off_c = 0, 1
            z_r0, z_c0 = 1, 1
        else:
            # Format B
            x_off_r, x_off_c = 1, 1
            y_off_r, y_off_c = 0, 2
            z_r0, z_c0 = 1, 2

        M, N = len(wl), len(t)
        # Bound to table extent (defensive — auto-detect already succeeded
        # on the same file, so this normally fits exactly).
        M = min(M, n_rows - x_off_r)
        N = min(N, n_cols - y_off_c)

        self.x_range = (x_off_r, x_off_c, x_off_r + M - 1, x_off_c)
        self.y_range = (y_off_r, y_off_c, y_off_r, y_off_c + N - 1)
        self.z_range = (z_r0, z_c0, z_r0 + M - 1, z_c0 + N - 1)
        self.stage = self.STAGE_DONE
        self._refresh_state_ui()
        self._highlight_confirmed(self.z_range)

    def _highlight_confirmed(self, rng: tuple[int, int, int, int]):
        r1, c1, r2, c2 = rng
        self.table.clearSelection()
        sel = QtWidgets.QTableWidgetSelectionRange(r1, c1, r2, c2)
        self.table.setRangeSelected(sel, True)
        self.table.scrollToItem(self.table.item(r1, c1),
                                QtWidgets.QAbstractItemView.PositionAtCenter)

    # ------------------------------------------------------------------
    # UI state refresh
    # ------------------------------------------------------------------
    def _refresh_state_ui(self):
        # Stage banner
        if self.numeric_data is None:
            self.lbl_stage.setText('Select a CSV file to preview (Browse...)')
        elif self.stage == self.STAGE_X:
            self.lbl_stage.setText(
                'STEP 1 of 3 : Drag-select the X-axis (WAVELENGTH).  '
                'Must be a single row OR a single column.')
        elif self.stage == self.STAGE_Y:
            self.lbl_stage.setText(
                'STEP 2 of 3 : Drag-select the Y-axis (DELAY).  '
                'Must be a single row OR a single column.')
        elif self.stage == self.STAGE_Z:
            self.lbl_stage.setText(
                'STEP 3 of 3 : Drag-select the Z-matrix (deltaA).  '
                'Shape must match the X and Y lengths.')
        else:
            self.lbl_stage.setText(
                'All three regions confirmed.  Press "Load" to import the data.')

        def fmt_1d(rng, name):
            if rng is None:
                return f'{name}: not set'
            r1, c1, r2, c2 = rng
            a1 = f'{_col_letter(c1)}{r1 + 1}'
            a2 = f'{_col_letter(c2)}{r2 + 1}'
            n = self._length_1d(rng)
            return f'{name}: {a1}:{a2}   ({n} points)'

        def fmt_2d(rng, name):
            if rng is None:
                return f'{name}: not set'
            r1, c1, r2, c2 = rng
            a1 = f'{_col_letter(c1)}{r1 + 1}'
            a2 = f'{_col_letter(c2)}{r2 + 1}'
            nr = r2 - r1 + 1
            nc = c2 - c1 + 1
            return f'{name}: {a1}:{a2}   ({nr} rows x {nc} cols)'

        self.lbl_x.setText(fmt_1d(self.x_range, 'X-axis (wavelength)'))
        self.lbl_y.setText(fmt_1d(self.y_range, 'Y-axis (delay)     '))
        self.lbl_z.setText(fmt_2d(self.z_range, 'Z-matrix (deltaA)  '))

        # Enable Load only when all three are set
        self.btn_ok.setEnabled(self.stage == self.STAGE_DONE
                               and self.x_range is not None
                               and self.y_range is not None
                               and self.z_range is not None)
        # Confirm/Clear/Back are only meaningful with a file loaded
        has_file = self.numeric_data is not None
        self.btn_confirm.setEnabled(has_file)
        self.btn_clear.setEnabled(has_file)
        self.btn_back.setEnabled(has_file)
        self.btn_reset.setEnabled(has_file)
        self.btn_auto.setEnabled(has_file)

    # ------------------------------------------------------------------
    # Accept / result extraction
    # ------------------------------------------------------------------
    def _on_accept(self):
        try:
            wl, t, A = self._extract()
        except Exception as e:
            warn_box(self, 'Extraction Failed', str(e))
            return
        self._result = (wl, t, A)
        self.accept()

    def _extract(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (self.numeric_data is None
                or self.x_range is None
                or self.y_range is None
                or self.z_range is None):
            raise ValueError('Internal: missing selections.')

        x_vals = self._values_1d(self.x_range).astype(float).copy()
        y_vals = self._values_1d(self.y_range).astype(float).copy()
        zr1, zc1, zr2, zc2 = self.z_range
        Z = self.numeric_data[zr1:zr2 + 1, zc1:zc2 + 1].astype(float).copy()
        x_len = len(x_vals)
        y_len = len(y_vals)
        nr, nc = Z.shape
        if nr == x_len and nc == y_len:
            A = Z
        elif nr == y_len and nc == x_len:
            A = Z.T
        else:
            raise ValueError(
                f'Z-matrix shape {Z.shape} no longer matches X({x_len}) / '
                f'Y({y_len}).  Please reselect.')

        # +/-inf -> NaN (matches ta_core.parse_data_file's behavior)
        A[np.isinf(A)] = np.nan

        # Sort wavelengths and delays ascending — required by the rest of
        # the app.
        if not np.all(x_vals[1:] >= x_vals[:-1]):
            ix = np.argsort(x_vals)
            x_vals = x_vals[ix]
            A = A[ix, :]
        if not np.all(y_vals[1:] >= y_vals[:-1]):
            it = np.argsort(y_vals)
            y_vals = y_vals[it]
            A = A[:, it]
        return x_vals, y_vals, A

    def get_result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return (wavelength, delay, deltaA) after the dialog is accepted."""
        return self._result
